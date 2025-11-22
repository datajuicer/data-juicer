# Some codes here are adapted from
# https://github.com/MegEngine/YOLOX/blob/main/yolox/utils/logger.py

# Copyright 2021 Megvii, Base Detection
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import inspect
import os
import sys
from io import StringIO

from loguru import logger
from loguru._file_sink import FileSink
from tabulate import tabulate

from data_juicer.utils.file_utils import add_suffix_to_filename

try:
    from IPython import get_ipython
except Exception:
    get_ipython = None

LOGGER_SETUP = False


def is_notebook():
    try:
        if get_ipython is None:
            return False
        ip = get_ipython()
        return ip is not None and ip.__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


def get_caller_name(depth=0):
    """
    Get caller name by depth.

    :param depth: depth of caller context, use 0 for caller depth.
    :return: module name of the caller
    """
    # the following logic is a little bit faster than inspect.stack() logic
    frame = inspect.currentframe().f_back
    for _ in range(depth):
        frame = frame.f_back

    return frame.f_globals["__name__"]


class StreamToLoguru:
    """Stream object that redirects writes to a logger instance."""

    def __init__(self, level="INFO", caller_names=("datasets", "logging")):
        """
        Initialization method.

        :param level: log level string of loguru. Default value: "INFO".
        :param caller_names: caller names of redirected module.
                    Default value: (apex, pycocotools).
        """
        self.level = level
        self.caller_names = caller_names
        self.buffer = StringIO()
        self.BUFFER_SIZE = 1024 * 1024

    def fileno(self):
        return self.buffer.fileno()

    def write(self, buf):
        full_name = get_caller_name(depth=1)
        module_name = full_name.rsplit(".", maxsplit=-1)[0]
        self.buffer.write(buf)
        if module_name in self.caller_names:
            for line in buf.rstrip().splitlines():
                # use caller level log
                logger.opt(depth=2).log(self.level, line.rstrip())
        else:
            # sys.__stdout__.write(buf)
            logger.opt(raw=True).info(buf)

        self.buffer.truncate(self.BUFFER_SIZE)

    def getvalue(self):
        return self.buffer.getvalue()

    def flush(self):
        self.buffer.flush()

    def isatty(self):
        return False  # Log streams are usually not terminal devices


def redirect_sys_output(log_level="INFO"):
    """
    Redirect stdout/stderr to loguru with log level.

    :param log_level: log level string of loguru. Default value: "INFO".
    """
    if is_notebook():
        return
    redirect_logger = StreamToLoguru(level=log_level)
    sys.stderr = redirect_logger
    sys.stdout = redirect_logger


def get_log_file_path():
    """
    Get the path to the location of the log file.

    :return: a location of log file.
    """
    for _, handler in logger._core.handlers.items():
        if isinstance(handler._sink, FileSink):
            return handler._sink._file.name


def setup_logger(save_dir, distributed_rank=0, filename="log.txt", mode="o", level="INFO", redirect="auto"):
    """
    Setup logger for training and testing.

    :param save_dir: location to save log file
    :param distributed_rank: device rank when multi-gpu environment
    :param filename: log file name to save
    :param mode: log file write mode, `append` or `override`. default is `o`.
    :param level: log severity level. It's "INFO" in default.
    :param redirect: whether to redirect system output, `auto`, True or False.
    :return: logger instance.
    """
    global LOGGER_SETUP

    if LOGGER_SETUP:
        return

    loguru_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )

    logger.remove()
    save_file = os.path.join(save_dir, filename)
    if mode == "o" and os.path.exists(save_file):
        os.remove(save_file)

    if redirect == "auto":
        redirect = not is_notebook()
    else:
        redirect = bool(redirect)

    # only keep logger in rank0 process
    if distributed_rank == 0:
        logger.add(
            sys.stderr,
            format=loguru_format,
            level=level,
            enqueue=not is_notebook(),
        )
        logger.add(
            save_file,
            format=loguru_format,
            level=level,
            compression="gz",
            enqueue=True,
        )

    # for interest of levels: debug, error, warning
    logger.add(
        add_suffix_to_filename(save_file, "_DEBUG"),
        level="DEBUG",
        filter=lambda x: "DEBUG" == x["level"].name,
        format=loguru_format,
        compression="gz",
        enqueue=True,
        serialize=True,
    )
    logger.add(
        add_suffix_to_filename(save_file, "_ERROR"),
        level="ERROR",
        filter=lambda x: "ERROR" == x["level"].name,
        format=loguru_format,
        compression="gz",
        enqueue=True,
        serialize=True,
    )
    logger.add(
        add_suffix_to_filename(save_file, "_WARNING"),
        level="WARNING",
        filter=lambda x: "WARNING" == x["level"].name,
        format=loguru_format,
        compression="gz",
        enqueue=True,
        serialize=True,
    )

    # redirect stdout/stderr to loguru
    if redirect:
        redirect_sys_output(level)
    LOGGER_SETUP = True


def make_log_summarization(max_show_item=10):
    error_pattern = r"^An error occurred in (.*?) when " r"processing samples? \"(.*?)\" -- (.*?): (.*?) -- (.*?)$"
    log_file = get_log_file_path()
    if log_file is None:
        return
    log_file = log_file.replace("_ERROR", "").replace("_WARNING", "").replace("_DEBUG", "")
    error_log_file = add_suffix_to_filename(log_file, "_ERROR")
    warning_log_file = add_suffix_to_filename(log_file, "_WARNING")
    error_pattern = r"^An error occurred in (.*?) when " r"processing samples? \"(.*?)\" -- (.*?): (.*?) -- (.*?)$"
    log_file = get_log_file_path().replace("_ERROR", "").replace("_WARNING", "").replace("_DEBUG", "")
    error_log_file = add_suffix_to_filename(log_file, "_ERROR")
    warning_log_file = add_suffix_to_filename(log_file, "_WARNING")

    import jsonlines as jl
    import regex as re

    # make error summarization
    error_dict = {}
    if os.path.exists(error_log_file):
        with jl.open(error_log_file) as reader:
            for error_log in reader:
                error_msg = error_log["record"]["message"]
                find_res = re.findall(error_pattern, error_msg, flags=re.DOTALL)
                if len(find_res) > 0:
                    op_name, sample, err_type, err_msg, _ = find_res[0]
                    error = (op_name, err_type, err_msg)
                    error_dict.setdefault(error, 0)
                    error_dict[error] += 1
    total_error_count = sum(error_dict.values())
    # make warning summarization
    warning_count = 0
    if os.path.exists(warning_log_file):
        with jl.open(warning_log_file) as reader:
            for _ in reader:
                warning_count += 1
    # make summary log
    summary = (
        f"Processing finished with:\n"
        f"<yellow>Warnings</yellow>: {warning_count}\n"
        f"<red>Errors</red>: {total_error_count}\n"
    )
    error_items = list(error_dict.items())
    error_items.sort(key=lambda it: it[1], reverse=True)
    error_items = error_items[:max_show_item]
    # convert error items to a table
    if len(error_items) > 0:
        error_table = []
        table_header = ["OP/Method", "Error Type", "Error Message", "Error Count"]
        for key, num in error_items:
            op_name, error_type, error_msg = key
            error_table.append([op_name, error_type, error_msg, num])
        table = tabulate(error_table, table_header, tablefmt="fancy_grid")
        summary += table
    summary += f"\nError/Warning details can be found in the log file " f"[{log_file}] and its related log files."
    logger.opt(ansi=True).info(summary)


class HiddenPrints:
    """Define a range that hide the outputs within this range."""

    def __enter__(self):
        """
        Store the original standard output and redirect the standard output to
        null when entering this range.
        """
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Close the redirected standard output and restore it when exiting from
        this range.
        """
        sys.stdout.close()
        sys.stdout = self._original_stdout

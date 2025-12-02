import ast
import asyncio
import copy
import os
import re
import shutil
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional, Union

import aiohttp
import pandas as pd
from datasets.utils.extract import ZstdExtractor as Extractor

from data_juicer.utils.common_utils import dict_to_hash
from data_juicer.utils.constant import DEFAULT_PREFIX, Fields


class Sizes:
    KiB = 2**10  # 1024
    MiB = 2**20  # 1024*1024
    GiB = 2**30  # 1024*1024*1024
    TiB = 2**40  # 1024*1024*1024*1024


def byte_size_to_size_str(byte_size: int):
    # get the string format of shard size
    if byte_size // Sizes.TiB:
        size_str = "%.2f TiB" % (byte_size / Sizes.TiB)
    elif byte_size // Sizes.GiB:
        size_str = "%.2f GiB" % (byte_size / Sizes.GiB)
    elif byte_size // Sizes.MiB:
        size_str = "%.2f MiB" % (byte_size / Sizes.MiB)
    elif byte_size // Sizes.KiB:
        size_str = "%.2f KiB" % (byte_size / Sizes.KiB)
    else:
        size_str = "%.2f Bytes" % byte_size
    return size_str


async def follow_read(
    logfile_path: str,
    skip_existing_content: bool = False,
) -> AsyncGenerator:
    """Read a file in online and iterative manner

    Args:
        logfile_path (`str`):
            The file path to be read.
        skip_existing_content (`bool`, defaults to `False):
            If True, read from the end, otherwise read from the beginning.

    Returns:
        One line string of the file content.
    """
    # in most unix file systems, the read operation is safe
    # for a file being target file of another "write process"
    with open(logfile_path, "r", encoding="utf-8", errors="ignore") as logfile:
        if skip_existing_content:
            # move to the file's end, similar to `tail -f`
            logfile.seek(0, 2)

        while True:
            line = logfile.readline()
            if not line:
                # no new line, wait to avoid CPU override
                await asyncio.sleep(0.1)
                continue
            yield line


def find_files_with_suffix(
    path: Union[str, Path], suffixes: Union[str, List[str], None] = None
) -> Dict[str, List[str]]:
    """
    Traverse a path to find all files with the specified suffixes.

    :param path: path (str/Path): source path
    :param suffixes: specified file suffixes, '.txt' or ['.txt', '.md']
        etc
    :return: list of all files with the specified suffixes
    """
    path = Path(path)
    file_dict = {}

    if suffixes is None:
        suffixes = []

    if isinstance(suffixes, str):
        suffixes = [suffixes]

    suffixes = [x.lower() if x.startswith(".") else "." + x.lower() for x in suffixes]

    if path.is_file():
        files = [path]
    else:
        searched_files = path.rglob("*")
        files = [file for file in searched_files if file.is_file()]

    extractor = Extractor

    # only keep the file with the specified suffixes
    for file in files:
        suffix = file.suffix.lower()

        if extractor.is_extractable(file):
            # TODO
            # hard code
            # only support zstd-format file now,
            # and use the last 2 sub-suffixes as the final suffix
            # just like '.jsonl.zst'
            file_suffixes = [suffix.lower() for suffix in file.suffixes]
            suffix = "".join(file_suffixes[-2:])

        if not suffixes or (suffix in suffixes):
            if suffix not in file_dict:
                file_dict[suffix] = [str(file)]
            else:
                file_dict[suffix].append(str(file))
    return file_dict


def is_remote_path(path: str):
    """Check if the path is a remote path."""
    return path.startswith(("http://", "https://"))


def is_absolute_path(path: Union[str, Path]) -> bool:
    """
    Check whether input path is a absolute path.

    :param path: input path
    :return: True means input path is absolute path, False means input
        path is a relative path.
    """
    if is_remote_path(str(path)):
        return True

    return Path(path).is_absolute()


def add_suffix_to_filename(filename, suffix):
    """
    Add a suffix to the filename. Only regard the content after the last dot
    as the file extension.
    E.g.
    1. abc.jpg + "_resized" --> abc_resized.jpg
    2. edf.xyz.csv + "_processed" --> edf.xyz_processed.csv
    3. /path/to/file.json + "_suf" --> /path/to/file_suf.json
    4. ds.tar.gz + "_whoops" --> ds.tar_whoops.gz (maybe unexpected)

    :param filename: input filename
    :param suffix: suffix string to be added
    """
    name, ext = os.path.splitext(filename)
    new_name = f"{name}{suffix}{ext}"
    return new_name


def create_directory_if_not_exists(directory_path):
    """
    create a directory if not exists, this function is process safe

    :param directory_path: directory path to be create
    """
    directory_path = os.path.abspath(directory_path)
    try:
        os.makedirs(directory_path, exist_ok=True)
    except FileExistsError:
        # We ignore the except from multi processes or threads.
        # Just make sure the directory exists.
        pass


def transfer_data_dir(original_dir, op_name):
    """
    Transfer the original multimodal data dir to a new dir to store the newly
    generated multimodal data. The pattern is
    `{original_dir}/__dj__produced_data__/{op_name}`
    """
    new_dir = os.path.join(original_dir, f"{Fields.multimodal_data_output_dir}/{op_name}")
    create_directory_if_not_exists(new_dir)
    return new_dir


def transfer_filename(original_filepath: Union[str, Path], op_name, save_dir: str = None, **op_kwargs):
    """
    According to the op and hashing its parameters 'op_kwargs' addition
    to the process id and current time as the 'hash_val', map the
    original_filepath to another unique file path. E.g.

    When `save_dir` is provided: '/save_dir/path/to/data/'
        /path/to/abc.jpg -->
            /save_dir/path/to/data/abc__dj_hash_#{hash_val}#.jpg
    When environment variable `DJ_PRODUCED_DATA_DIR` is provided: '/environment/path/to/data/'
        /path/to/abc.jpg -->
            /environment/path/to/data/{op_name}/abc__dj_hash_#{hash_val}#.jpg
    When neither `save_dir` nor `DJ_PRODUCED_DATA_DIR` is provided:
        1. abc.jpg -->
            __dj__produced_data__/{op_name}/
            abc__dj_hash_#{hash_val}#.jpg
        2. ./abc.jpg -->
            ./__dj__produced_data__/{op_name}/
            abc__dj_hash_#{hash_val}#.jpg
        3. /path/to/abc.jpg -->
            /path/to/__dj__produced_data__/{op_name}/
            abc__dj_hash_#{hash_val}#.jpg
        4. /path/to/__dj__produced_data__/{op_name}/
            abc__dj_hash_#{hash_val1}#.jpg -->
            /path/to/__dj__produced_data__/{op_name}/
            abc__dj_hash_#{hash_val2}#.jpg

    Priority: `save_dir` > `DJ_PRODUCED_DATA_DIR` > original data directory (default)
    """
    # check if it's valid local path, if it's not, regard it as a remote path/url and return None
    if not os.path.exists(original_filepath):
        return original_filepath

    if save_dir:
        new_dir = os.path.abspath(save_dir)
    elif produced_data_dir := os.environ.get("DJ_PRODUCED_DATA_DIR", None):
        new_dir = os.path.join(os.path.abspath(produced_data_dir), op_name)
    else:
        # produce the directory
        original_dir = os.path.dirname(original_filepath)
        dir_token = f"/{Fields.multimodal_data_output_dir}/"
        if dir_token in original_dir:
            original_dir = original_dir.split(dir_token)[0]
        new_dir = transfer_data_dir(original_dir, op_name)

    create_directory_if_not_exists(new_dir)

    # produce the unique hash code
    unique_parameters = copy.deepcopy(op_kwargs)
    unique_parameters[f"{DEFAULT_PREFIX}pid"] = os.getpid()
    unique_parameters[f"{DEFAULT_PREFIX}timestamp"] = str(datetime.now(timezone.utc))
    unique_hash = dict_to_hash(unique_parameters)

    # if the input data is produced by data-juicer, replace the hash code
    # else append hash value to filename
    def add_hash_value(text, new_hash_value):
        pattern = r"__dj_hash_#(.*?)#"

        match = re.search(pattern, text)
        # draw the string produced by data-juicer
        if match:
            text = text[: match.start()]

        return f"{text}__dj_hash_#{new_hash_value}#"

    original_filename = os.path.basename(original_filepath)
    name, ext = os.path.splitext(original_filename)
    new_name = add_hash_value(name, unique_hash)
    new_filepath = os.path.join(new_dir, f"{new_name}{ext}")

    return new_filepath


def copy_data(from_dir, to_dir, data_path):
    """
    Copy data from from_dir/data_path to to_dir/data_path.
    Return True if success.
    """
    from_path = os.path.join(from_dir, data_path)
    to_path = os.path.join(to_dir, data_path)
    if not os.path.exists(from_path):
        return False
    parent_dir = os.path.dirname(to_path)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    shutil.copy2(from_path, to_path)
    return True


def expand_outdir_and_mkdir(outdir):
    _outdir = os.path.abspath(os.path.expanduser(outdir))
    if not os.path.exists(_outdir):
        os.makedirs(_outdir)
    return _outdir


def single_partition_write_with_filename(
    df: pd.DataFrame,
    output_file_dir: str,
    keep_filename_column: bool = False,
    output_type: str = "jsonl",
) -> pd.Series:
    """
    This function processes a DataFrame and writes it to disk

    Args:
        df: A DataFrame.
        output_file_dir: The output file path.
        keep_filename_column: Whether to keep or drop the "filename" column, if it exists.
        output_type="jsonl": The type of output file to write.
    Returns:
        If the DataFrame is non-empty, return a Series containing a single element, True.
        If the DataFrame is empty, return a Series containing a single element, False.

    """  # noqa: E501
    assert "filename" in df.columns

    if len(df) > 0:
        empty_partition = False
    else:
        warnings.warn("Empty partition found")
        empty_partition = True

    # if is_cudf_type(df):
    #     import cudf
    #     success_ser = cudf.Series([empty_partition])
    # else:
    success_ser = pd.Series([empty_partition])

    if not empty_partition:
        filenames = df.filename.unique()
        filenames = list(filenames)
        num_files = len(filenames)

        for filename in filenames:
            out_df = df[df.filename == filename] if num_files > 1 else df
            if not keep_filename_column:
                out_df = out_df.drop("filename", axis=1)

            filename = Path(filename).stem
            output_file_path = os.path.join(output_file_dir, filename)

            if output_type == "jsonl":
                output_file_path = output_file_path + ".jsonl"
                out_df.to_json(
                    output_file_path,
                    orient="records",
                    lines=True,
                    force_ascii=False,
                )

            elif output_type == "parquet":
                output_file_path = output_file_path + ".parquet"
                out_df.to_parquet(output_file_path)

            else:
                raise ValueError(f"Unknown output type: {output_type}")

    return success_ser


def read_single_partition(
    files,
    filetype="jsonl",
    add_filename=False,
    input_meta: Union[str, dict] = None,
    columns: Optional[List[str]] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    This function reads a file with cuDF, sorts the columns of the DataFrame
    and adds a "filename" column.

    Args:
        files: The path to the jsonl files to read.
        add_filename: Whether to add a "filename" column to the DataFrame.
        input_meta: A dictionary or a string formatted as a dictionary, which outlines
            the field names and their respective data types within the JSONL input file.
        columns: If not None, only these columns will be read from the file.
            There is a significant performance gain when specifying columns for Parquet files.

    Returns:
        A pandas DataFrame.

    """  # noqa: E501
    if input_meta is not None and filetype != "jsonl":
        warnings.warn("input_meta is only valid for JSONL files and" "will be ignored for other file formats..")

    if filetype in ["jsonl", "json"]:
        read_kwargs = {"lines": filetype == "jsonl"}
        read_kwargs["dtype"] = False
        read_f = pd.read_json

        if input_meta is not None:
            read_kwargs["dtype"] = ast.literal_eval(input_meta) if isinstance(input_meta, str) else input_meta

    elif filetype == "parquet":
        read_kwargs = {"columns": columns}
        read_f = pd.read_parquet

    else:
        raise RuntimeError("Could not read data, please check file type")

    read_files_one_at_a_time = True
    if read_files_one_at_a_time:
        concat_f = pd.concat
        df_ls = []
        for file in files:
            df = read_f(file, **read_kwargs, **kwargs)
            if add_filename:
                df["filename"] = os.path.basename(file)
            df_ls.append(df)
        df = concat_f(df_ls, ignore_index=True)
    else:
        df = read_f(files, **read_kwargs, **kwargs)

    if filetype in ["jsonl", "json"] and columns is not None:
        if add_filename and "filename" not in columns:
            columns.append("filename")
        df = df[columns]

    df = df[sorted(df.columns)]
    return df


def get_all_files_paths_under(root, recurse_subdirectories=True, followlinks=False):
    """
    This function returns a list of all the files under a specified directory.
    Args:
        root: The path to the directory to read.
        recurse_subdirecties: Whether to recurse into subdirectories.
                              Please note that this can be slow for large
                              number of files.
        followlinks: Whether to follow symbolic links.
    """  # noqa: E501
    if recurse_subdirectories:
        file_ls = [os.path.join(r, f) for r, subdirs, files in os.walk(root, followlinks=followlinks) for f in files]
    else:
        file_ls = [entry.path for entry in os.scandir(root)]

    file_ls.sort()
    return file_ls


async def download_file(
    session: aiohttp.ClientSession, url: str, save_path: str = None, return_content=False, timeout: int = 300, **kwargs
):
    """
    Download a file from a given URL and save it to a specified directory.
    :param url: The URL of the file to download.
    :param save_path: The path where the downloaded file will be saved.
    :param return_content: Whether to return the content of the downloaded file.
    :param timeout: The timeout in seconds for each HTTP request.
    :param kwargs: The keyword arguments to pass to the HTTP request.

    :return: The response object from the HTTP request.
    """
    async with session.get(
        url, timeout=aiohttp.ClientTimeout(total=timeout), raise_for_status=True, **kwargs
    ) as response:

        assert save_path or return_content, "save_path or return_content must be set."
        content = b""

        if save_path:
            with open(save_path, "wb") as f:
                while chunk := await response.content.read():
                    f.write(chunk)
                    content += chunk
        else:
            content = await response.read()

        if return_content:
            return response, content

        return response

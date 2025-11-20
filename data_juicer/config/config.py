import argparse
import copy
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import time
import uuid
from argparse import ArgumentError
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, List, Optional, Union

import yaml
from jsonargparse import (
    ActionConfigFile,
    ArgumentParser,
    Namespace,
    dict_to_namespace,
    namespace_to_dict,
)
from jsonargparse._typehints import ActionTypeHint
from jsonargparse.typing import ClosedUnitInterval, NonNegativeInt, PositiveInt
from loguru import logger

from data_juicer.ops.base_op import OPERATORS
from data_juicer.ops.op_fusion import FUSION_STRATEGIES
from data_juicer.utils.constant import RAY_JOB_ENV_VAR
from data_juicer.utils.logger_utils import setup_logger
from data_juicer.utils.mm_utils import SpecialTokens
from data_juicer.utils.ray_utils import is_ray_mode

global_cfg = None
global_parser = None


@contextmanager
def timing_context(description):
    start_time = time.time()
    yield
    elapsed_time = time.time() - start_time
    # Use a consistent format that won't be affected by logger reconfiguration
    logger.debug(f"{description} took {elapsed_time:.2f} seconds")


def _generate_module_name(abs_path):
    """Generate a module name based on the absolute path of the file."""
    return os.path.splitext(os.path.basename(abs_path))[0]


def load_custom_operators(paths):
    """Dynamically load custom operator modules or packages in the specified path."""
    for path in paths:
        abs_path = os.path.abspath(path)
        if os.path.isfile(abs_path):
            module_name = _generate_module_name(abs_path)
            if module_name in sys.modules:
                existing_path = sys.modules[module_name].__file__
                raise RuntimeError(
                    f"Module '{module_name}' already loaded from '{existing_path}'. "
                    f"Conflict detected while loading '{abs_path}'."
                )
            try:
                spec = importlib.util.spec_from_file_location(module_name, abs_path)
                if spec is None:
                    raise RuntimeError(f"Failed to create spec for '{abs_path}'")
                module = importlib.util.module_from_spec(spec)
                # register the module first to avoid recursive import issues
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
            except Exception as e:
                raise RuntimeError(f"Error loading '{abs_path}' as '{module_name}': {e}")

        elif os.path.isdir(abs_path):
            if not os.path.isfile(os.path.join(abs_path, "__init__.py")):
                raise ValueError(f"Package directory '{abs_path}' must contain __init__.py")
            package_name = os.path.basename(abs_path)
            parent_dir = os.path.dirname(abs_path)
            if package_name in sys.modules:
                existing_path = sys.modules[package_name].__path__[0]
                raise RuntimeError(
                    f"Package '{package_name}' already loaded from '{existing_path}'. "
                    f"Conflict detected while loading '{abs_path}'."
                )
            original_sys_path = sys.path.copy()
            try:
                sys.path.insert(0, parent_dir)
                importlib.import_module(package_name)
                # record the loading path of the package (for subsequent conflict detection)
                sys.modules[package_name].__loaded_from__ = abs_path
            except Exception as e:
                raise RuntimeError(f"Error loading package '{abs_path}': {e}")
            finally:
                sys.path = original_sys_path
        else:
            raise ValueError(f"Path '{abs_path}' is neither a file nor a directory")


def init_configs(args: Optional[List[str]] = None, which_entry: object = None, load_configs_only=False):
    """
    initialize the jsonargparse parser and parse configs from one of:
        1. POSIX-style commands line args;
        2. config files in yaml (json and jsonnet supersets);
        3. environment variables
        4. hard-coded defaults

    :param args: list of params, e.g., ['--config', 'cfg.yaml'], default None.
    :param which_entry: which entry to init configs (executor/analyzer)
    :param load_configs_only: whether to load the configs only, not including backing up config files, display them, and
        setting up logger.
    :return: a global cfg object used by the DefaultExecutor or Analyzer
    """
    if args is None:
        args = sys.argv[1:]
    with timing_context("Total config initialization time"):
        with timing_context("Initializing parser"):
            parser = ArgumentParser(default_env=True, default_config_files=None, usage=argparse.SUPPRESS)

            # required but mutually exclusive args group
            required_group = parser.add_mutually_exclusive_group(required=True)
            required_group.add_argument(
                "--config", action=ActionConfigFile, help="Path to a dj basic configuration file."
            )
            required_group.add_argument(
                "--auto",
                action="store_true",
                help="Weather to use an auto analyzing "
                "strategy instead of a specific data "
                "recipe. If a specific config file is "
                "given by --config arg, this arg is "
                "disabled. Only available for Analyzer.",
            )

            parser.add_argument(
                "--auto_num",
                type=PositiveInt,
                default=1000,
                help="The number of samples to be analyzed " "automatically. It's 1000 in default.",
            )

            parser.add_argument(
                "--hpo_config", type=str, help="Path to a configuration file when using auto-HPO tool.", required=False
            )
            parser.add_argument(
                "--data_probe_algo",
                type=str,
                default="uniform",
                help='Sampling algorithm to use. Options are "uniform", '
                '"frequency_specified_field_selector", or '
                '"topk_specified_field_selector". Default is "uniform". Only '
                "used for dataset sampling",
                required=False,
            )
            parser.add_argument(
                "--data_probe_ratio",
                type=ClosedUnitInterval,
                default=1.0,
                help="The ratio of the sample size to the original dataset size. "  # noqa: E251
                "Default is 1.0 (no sampling). Only used for dataset sampling",
                required=False,
            )

            # basic global paras with extended type hints
            # e.g., files can be mode include flags
            # "fr": "path to a file that exists and is readable")
            # "fc": "path to a file that can be created if it does not exist")
            # "dw": "path to a directory that exists and is writeable")
            # "dc": "path to a directory that can be created if it does not exist")
            # "drw": "path to a directory that exists and is readable and writeable")
            parser.add_argument(
                "--project_name", type=str, default="hello_world", help="Name of your data process project."
            )
            parser.add_argument(
                "--executor_type",
                type=str,
                default="default",
                choices=["default", "ray", "ray_partitioned"],
                help='Type of executor, support "default", "ray", or "ray_partitioned".',
            )
            parser.add_argument(
                "--dataset_path",
                type=str,
                default="",
                help="Path to datasets with optional weights(0.0-1.0), 1.0 as "
                "default. Accepted format:<w1> dataset1-path <w2> dataset2-path "
                "<w3> dataset3-path ...",
            )
            parser.add_argument(
                "--dataset",
                type=Union[List[Dict], Dict],
                default=[],
                help="Dataset setting to define local/remote datasets; could be a "  # noqa: E251
                "dict or a list of dicts; refer to configs/datasets for more "
                "detailed examples",
            )
            parser.add_argument(
                "--generated_dataset_config",
                type=Dict,
                default=None,
                help="Configuration used to create a dataset. "  # noqa: E251
                "The dataset will be created from this configuration if provided. "
                "It must contain the `type` field to specify the dataset name.",
            )
            parser.add_argument(
                "--validators",
                type=List[Dict],
                default=[],
                help="List of validators to apply to the dataset. Each validator "  # noqa: E251
                "must have a `type` field specifying the validator type.",
            )
            parser.add_argument(
                "--work_dir",
                type=str,
                default=None,
                help="Path to a work directory to store outputs during Data-Juicer "  # noqa: E251
                "running. It's the directory where export_path is at in default.",
            )
            parser.add_argument(
                "--export_path",
                type=str,
                default="./outputs/hello_world/hello_world.jsonl",
                help="Path to export and save the output processed dataset. The "  # noqa: E251
                "directory to store the processed dataset will be the work "
                "directory of this process.",
            )
            parser.add_argument(
                "--export_type",
                type=str,
                default=None,
                help="The export format type. If it's not specified, Data-Juicer will parse from the export_path. The "
                "supported types can be found in Exporter._router() for standalone mode and "
                "RayExporter._SUPPORTED_FORMATS for ray mode",
            )
            parser.add_argument(
                "--export_shard_size",
                type=NonNegativeInt,
                default=0,
                help="Shard size of exported dataset in Byte. In default, it's 0, "  # noqa: E251
                "which means export the whole dataset into only one file. If "
                "it's set a positive number, the exported dataset will be split "
                "into several sub-dataset shards, and the max size of each shard "
                "won't larger than the export_shard_size",
            )
            parser.add_argument(
                "--export_in_parallel",
                type=bool,
                default=False,
                help="Whether to export the result dataset in parallel to a single "  # noqa: E251
                "file, which usually takes less time. It only works when "
                "export_shard_size is 0, and its default number of processes is "
                "the same as the argument np. **Notice**: If it's True, "
                "sometimes exporting in parallel might require much more time "
                "due to the IO blocking, especially for very large datasets. "
                "When this happens, False is a better choice, although it takes "
                "more time.",
            )
            parser.add_argument(
                "--export_extra_args",
                type=Dict,
                default={},
                help="Other optional arguments for exporting in dict. For example, the key mapping info for exporting "
                "the WebDataset format.",
            )
            parser.add_argument(
                "--keep_stats_in_res_ds",
                type=bool,
                default=False,
                help="Whether to keep the computed stats in the result dataset. If "  # noqa: E251
                "it's False, the intermediate fields to store the stats "
                "computed by Filters will be removed. Default: False.",
            )
            parser.add_argument(
                "--keep_hashes_in_res_ds",
                type=bool,
                default=False,
                help="Whether to keep the computed hashes in the result dataset. If "  # noqa: E251
                "it's False, the intermediate fields to store the hashes "
                "computed by Deduplicators will be removed. Default: False.",
            )
            parser.add_argument("--np", type=PositiveInt, default=4, help="Number of processes to process dataset.")
            parser.add_argument(
                "--text_keys",
                type=Union[str, List[str]],
                default="text",
                help="Key name of field where the sample texts to be processed, e.g., "  # noqa: E251
                "`text`, `text.instruction`, `text.output`, ... Note: currently, "
                "we support specify only ONE key for each op, for cases "
                "requiring multiple keys, users can specify the op multiple "
                "times.  We will only use the first key of `text_keys` when you "
                "set multiple keys.",
            )
            parser.add_argument(
                "--image_key",
                type=str,
                default="images",
                help="Key name of field to store the list of sample image paths.",  # noqa: E251
            )
            parser.add_argument(
                "--image_bytes_key",
                type=str,
                default="image_bytes",
                help="Key name of field to store the list of sample image bytes.",  # noqa: E251
            )
            parser.add_argument(
                "--image_special_token",
                type=str,
                default=SpecialTokens.image,
                help="The special token that represents an image in the text. In "  # noqa: E251
                'default, it\'s "<__dj__image>". You can specify your own special'
                " token according to your input dataset.",
            )
            parser.add_argument(
                "--audio_key",
                type=str,
                default="audios",
                help="Key name of field to store the list of sample audio paths.",  # noqa: E251
            )
            parser.add_argument(
                "--audio_special_token",
                type=str,
                default=SpecialTokens.audio,
                help="The special token that represents an audio in the text. In "  # noqa: E251
                'default, it\'s "<__dj__audio>". You can specify your own special'
                " token according to your input dataset.",
            )
            parser.add_argument(
                "--video_key",
                type=str,
                default="videos",
                help="Key name of field to store the list of sample video paths.",  # noqa: E251
            )
            parser.add_argument(
                "--video_special_token",
                type=str,
                default=SpecialTokens.video,
                help="The special token that represents a video in the text. In "
                'default, it\'s "<__dj__video>". You can specify your own special'
                " token according to your input dataset.",
            )
            parser.add_argument(
                "--eoc_special_token",
                type=str,
                default=SpecialTokens.eoc,
                help="The special token that represents the end of a chunk in the "  # noqa: E251
                'text. In default, it\'s "<|__dj__eoc|>". You can specify your '
                "own special token according to your input dataset.",
            )
            parser.add_argument(
                "--suffixes",
                type=Union[str, List[str]],
                default=[],
                help="Suffixes of files that will be find and loaded. If not set, we "  # noqa: E251
                "will find all suffix files, and select a suitable formatter "
                "with the most files as default.",
            )
            parser.add_argument(
                "--turbo",
                type=bool,
                default=False,
                help="Enable Turbo mode to maximize processing speed when batch size " "is 1.",  # noqa: E251
            )
            parser.add_argument(
                "--skip_op_error",
                type=bool,
                default=True,
                help="Skip errors in OPs caused by unexpected invalid samples.",  # noqa: E251
            )
            parser.add_argument(
                "--use_cache",
                type=bool,
                default=True,
                help="Whether to use the cache management of huggingface datasets. It "  # noqa: E251
                "might take up lots of disk space when using cache",
            )
            parser.add_argument(
                "--ds_cache_dir",
                type=str,
                default=None,
                help="Cache dir for HuggingFace datasets. In default it's the same "  # noqa: E251
                "as the environment variable `HF_DATASETS_CACHE`, whose default "
                'value is usually "~/.cache/huggingface/datasets". If this '
                "argument is set to a valid path by users, it will override the "
                "default cache dir. Modifying this arg might also affect the other two"
                " paths to store downloaded and extracted datasets that depend on "
                "`HF_DATASETS_CACHE`",
            )
            parser.add_argument(
                "--cache_compress",
                type=str,
                default=None,
                help="The compression method of the cache file, which can be"
                'specified in ["gzip", "zstd", "lz4"]. If this parameter is'
                "None, the cache file will not be compressed.",
            )
            parser.add_argument(
                "--open_monitor",
                type=bool,
                default=True,
                help="Whether to open the monitor to trace resource utilization for "  # noqa: E251
                "each OP during data processing. It's True in default.",
            )
            parser.add_argument(
                "--use_checkpoint",
                type=bool,
                default=False,
                help="Whether to use the checkpoint management to save the latest "  # noqa: E251
                "version of dataset to work dir when processing. Rerun the same "
                "config will reload the checkpoint and skip ops before it. Cache "
                "will be disabled when it is true . If args of ops before the "
                "checkpoint are changed, all ops will be rerun from the "
                "beginning.",
            )
            # Enhanced checkpoint configuration for PartitionedRayExecutor
            parser.add_argument(
                "--checkpoint.enabled",
                type=bool,
                default=True,
                help="Enable enhanced checkpointing for PartitionedRayExecutor",
            )
            parser.add_argument(
                "--checkpoint.strategy",
                type=str,
                default="every_op",
                choices=["every_op", "every_partition", "every_n_ops", "manual", "disabled"],
                help="Checkpoint strategy: every_op, every_partition, every_n_ops, manual, disabled",
            )
            parser.add_argument(
                "--checkpoint.n_ops",
                type=int,
                default=1,
                help="Number of operations between checkpoints for every_n_ops strategy",
            )
            parser.add_argument(
                "--checkpoint.op_names",
                type=List[str],
                default=[],
                help="List of operation names to checkpoint for manual strategy",
            )
            # Event logging configuration
            parser.add_argument(
                "--event_logging.enabled",
                type=bool,
                default=True,
                help="Enable event logging for job tracking and resumption",
            )
            # Logging configuration
            parser.add_argument(
                "--max_log_size_mb",
                type=int,
                default=100,
                help="Maximum log file size in MB before rotation",
            )
            parser.add_argument(
                "--backup_count",
                type=int,
                default=5,
                help="Number of backup log files to keep",
            )
            # Storage configuration
            parser.add_argument(
                "--event_log_dir",
                type=str,
                default=None,
                help="Separate directory for event logs (fast storage)",
            )
            parser.add_argument(
                "--checkpoint_dir",
                type=str,
                default=None,
                help="Separate directory for checkpoints (large storage)",
            )
            # Job management
            parser.add_argument(
                "--job_id",
                type=str,
                default=None,
                help="Custom job ID for resumption and tracking. If not provided, a unique ID will be auto-generated.",
            )
            parser.add_argument(
                "--temp_dir",
                type=str,
                default=None,
                help="Path to the temp directory to store intermediate caches when "  # noqa: E251
                "cache is disabled. In default it's None, so the temp dir will "
                "be specified by system. NOTICE: you should be caution when "
                "setting this argument because it might cause unexpected program "
                "behaviors when this path is set to an unsafe directory.",
            )
            parser.add_argument(
                "--open_tracer",
                type=bool,
                default=False,
                help="Whether to open the tracer to trace samples changed during "  # noqa: E251
                "process. It might take more time when opening tracer.",
            )
            parser.add_argument(
                "--op_list_to_trace",
                type=List[str],
                default=[],
                help="Which ops will be traced by tracer. If it's empty, all ops in "  # noqa: E251
                "cfg.process will be traced. Only available when open_tracer is "
                "true.",
            )
            parser.add_argument(
                "--trace_num",
                type=int,
                default=10,
                help="Number of samples extracted by tracer to show the dataset "
                "difference before and after a op. Only available when "
                "open_tracer is true.",
            )
            parser.add_argument(
                "--open_insight_mining",
                type=bool,
                default=False,
                help="Whether to open insight mining to trace the OP-wise stats/tags "  # noqa: E251
                "changes during process. It might take more time when opening "
                "insight mining.",
            )
            parser.add_argument(
                "--op_list_to_mine",
                type=List[str],
                default=[],
                help="Which OPs will be applied on the dataset to mine the insights "  # noqa: E251
                "in their stats changes. Only those OPs that produce stats or "
                "meta are valid. If it's empty, all OPs that produce stats and "
                "meta will be involved. Only available when filter_list_to_mine "
                "is true.",
            )
            parser.add_argument(
                "--op_fusion",
                type=bool,
                default=False,
                help="Whether to fuse operators that share the same intermediate "  # noqa: E251
                "variables automatically. Op fusion might reduce the memory "
                "requirements slightly but speed up the whole process.",
            )
            parser.add_argument(
                "--fusion_strategy",
                type=str,
                default="probe",
                help='OP fusion strategy. Support ["greedy", "probe"] now. "greedy" '  # noqa: E251
                "means keep the basic OP order and put the fused OP to the last "
                'of each fused OP group. "probe" means Data-Juicer will probe '
                "the running speed for each OP at the beginning and reorder the "
                "OPs and fused OPs according to their probed speed (fast to "
                'slow). It\'s "probe" in default.',
            )
            parser.add_argument(
                "--adaptive_batch_size",
                type=bool,
                default=False,
                help="Whether to use adaptive batch sizes for each OP according to "  # noqa: E251
                "the probed results. It's False in default.",
            )
            parser.add_argument(
                "--process",
                type=List[Dict],
                default=[],
                help="List of several operators with their arguments, these ops will "  # noqa: E251
                "be applied to dataset in order",
            )
            parser.add_argument(
                "--percentiles",
                type=List[float],
                default=[],
                help="Percentiles to analyze the dataset distribution. Only used in " "Analysis.",  # noqa: E251
            )
            parser.add_argument(
                "--export_original_dataset",
                type=bool,
                default=False,
                help="whether to export the original dataset with stats. If you only "  # noqa: E251
                "need the stats of the dataset, setting it to false could speed "
                "up the exporting..",
            )
            parser.add_argument(
                "--save_stats_in_one_file",
                type=bool,
                default=False,
                help="Whether to save all stats to only one file. Only used in " "Analysis.",
            )
            parser.add_argument("--ray_address", type=str, default="auto", help="The address of the Ray cluster.")

            # Partitioning configuration for PartitionedRayExecutor
            # Support both flat and nested partition configuration
            parser.add_argument(
                "--partition_size",
                type=int,
                default=10000,
                help="Number of samples per partition for PartitionedRayExecutor (legacy flat config)",
            )
            parser.add_argument(
                "--max_partition_size_mb",
                type=int,
                default=128,
                help="Maximum partition size in MB for PartitionedRayExecutor (legacy flat config)",
            )

            parser.add_argument(
                "--preserve_intermediate_data",
                type=bool,
                default=False,
                help="Preserve intermediate data for debugging (legacy flat config)",
            )

            # partition configuration
            parser.add_argument(
                "--partition.mode",
                type=str,
                default="auto",
                choices=["manual", "auto"],
                help="Partition mode: manual (specify num_of_partitions) or auto (use partition size optimizer)",
            )
            parser.add_argument(
                "--partition.num_of_partitions",
                type=int,
                default=4,
                help="Number of partitions for manual mode (ignored in auto mode)",
            )

            # Resource optimization configuration
            parser.add_argument(
                "--resource_optimization.auto_configure",
                type=bool,
                default=False,
                help="Enable automatic optimization of partition size, worker count, and other resource-dependent settings (nested resource_optimization config)",
            )

            # Intermediate storage configuration
            parser.add_argument(
                "--intermediate_storage.preserve_intermediate_data",
                type=bool,
                default=False,
                help="Preserve intermediate data for debugging (nested intermediate_storage config)",
            )
            parser.add_argument(
                "--intermediate_storage.cleanup_temp_files",
                type=bool,
                default=True,
                help="Clean up temporary files after processing (nested intermediate_storage config)",
            )
            parser.add_argument(
                "--intermediate_storage.cleanup_on_success",
                type=bool,
                default=False,
                help="Clean up intermediate files even on successful completion (nested intermediate_storage config)",
            )
            parser.add_argument(
                "--intermediate_storage.retention_policy",
                type=str,
                default="keep_all",
                choices=["keep_all", "keep_failed_only", "cleanup_all"],
                help="File retention policy (nested intermediate_storage config)",
            )
            parser.add_argument(
                "--intermediate_storage.max_retention_days",
                type=int,
                default=7,
                help="Maximum retention days for files (nested intermediate_storage config)",
            )

            # Intermediate storage format configuration
            parser.add_argument(
                "--intermediate_storage.format",
                type=str,
                default="parquet",
                choices=["parquet", "arrow", "jsonl"],
                help="Storage format for checkpoints and intermediate data (nested intermediate_storage config)",
            )
            parser.add_argument(
                "--intermediate_storage.compression",
                type=str,
                default="snappy",
                choices=["snappy", "gzip", "none"],
                help="Compression format for storage files (nested intermediate_storage config)",
            )

            parser.add_argument(
                "--intermediate_storage.write_partitions",
                type=bool,
                default=True,
                help="Whether to write intermediate partition files to disk (nested intermediate_storage config). Set to false for better performance when intermediate files aren't needed.",
            )

            parser.add_argument(
                "--partition_dir",
                type=str,
                default=None,
                help="Directory to store partition files. Supports {work_dir} placeholder. If not set, defaults to {work_dir}/partitions.",
            )

            parser.add_argument(
                "--custom-operator-paths", nargs="+", help="Paths to custom operator scripts or directories."
            )
            parser.add_argument("--debug", action="store_true", default=False, help="Whether to run in debug mode.")

            # Filter out non-essential arguments for initial parsing
            essential_args = []
            if args:
                i = 0
                while i < len(args):
                    arg = args[i]
                    # Handle --help, --config, and --auto in first pass
                    if arg == "--help":
                        essential_args.append(arg)
                    elif arg == "--config":
                        essential_args.append(arg)
                        # The next argument must be the config file path
                        if i + 1 < len(args):
                            essential_args.append(args[i + 1])
                            i += 1
                    elif arg == "--auto":
                        essential_args.append(arg)
                    i += 1

            # Parse essential arguments first
            essential_cfg = parser.parse_args(args=essential_args)

            # Now add remaining arguments based on essential config
            used_ops = None
            if essential_cfg.config:
                # Load config file to determine which operators are used
                with open(os.path.abspath(essential_cfg.config[0])) as f:
                    config_data = yaml.safe_load(f)
                    used_ops = set()
                    if "process" in config_data:
                        for op in config_data["process"]:
                            used_ops.add(list(op.keys())[0])

                # Add remaining arguments
                ops_sorted_by_types = sort_op_by_types_and_names(OPERATORS.modules.items())

                # Only add arguments for used operators
                _collect_config_info_from_class_docs(
                    [(op_name, op_class) for op_name, op_class in ops_sorted_by_types if op_name in used_ops], parser
                )

            # Parse all arguments
            with timing_context("Parsing arguments"):
                cfg = parser.parse_args(args=args)

                if cfg.executor_type == "ray":
                    os.environ[RAY_JOB_ENV_VAR] = "1"

                if cfg.custom_operator_paths:
                    load_custom_operators(cfg.custom_operator_paths)

                # check the entry
                from data_juicer.core.analyzer import Analyzer

                if not isinstance(which_entry, Analyzer) and cfg.auto:
                    err_msg = "--auto argument can only be used for analyzer!"
                    logger.error(err_msg)
                    raise NotImplementedError(err_msg)

        with timing_context("Initializing setup from config"):
            cfg = init_setup_from_cfg(cfg, load_configs_only)

        with timing_context("Updating operator process"):
            cfg = update_op_process(cfg, parser, used_ops)

        # Validate config for resumption if job_id is provided
        if not load_configs_only and hasattr(cfg, "job_id") and cfg.job_id:
            # Check if this is a resumption attempt by looking for existing job directory
            if cfg.work_dir and os.path.exists(cfg.work_dir):
                logger.info(f"🔍 Checking for job resumption: {cfg.job_id}")
                cfg._same_yaml_config = validate_config_for_resumption(cfg, cfg.work_dir, args)
            else:
                # New job, set flag to True
                cfg._same_yaml_config = True

        # copy the config file into the work directory
        if not load_configs_only:
            config_backup(cfg)

        # show the final config tables before the process started
        if not load_configs_only:
            display_config(cfg)

        global global_cfg, global_parser
        global_cfg = cfg
        global_parser = parser

        if cfg.get("debug", False):
            logger.debug("In DEBUG mode.")

        return cfg


def update_ds_cache_dir_and_related_vars(new_ds_cache_path):
    from pathlib import Path

    from datasets import config

    # update the HF_DATASETS_CACHE
    config.HF_DATASETS_CACHE = Path(new_ds_cache_path)
    # and two more PATHS that depend on HF_DATASETS_CACHE
    # - path to store downloaded datasets (e.g. remote datasets)
    config.DEFAULT_DOWNLOADED_DATASETS_PATH = os.path.join(config.HF_DATASETS_CACHE, config.DOWNLOADED_DATASETS_DIR)
    config.DOWNLOADED_DATASETS_PATH = Path(config.DEFAULT_DOWNLOADED_DATASETS_PATH)
    # - path to store extracted datasets (e.g. xxx.jsonl.zst)
    config.DEFAULT_EXTRACTED_DATASETS_PATH = os.path.join(
        config.DEFAULT_DOWNLOADED_DATASETS_PATH, config.EXTRACTED_DATASETS_DIR
    )
    config.EXTRACTED_DATASETS_PATH = Path(config.DEFAULT_EXTRACTED_DATASETS_PATH)


def init_setup_from_cfg(cfg: Namespace, load_configs_only=False):
    """
    Do some extra setup tasks after parsing config file or command line.

    1. create working directory and a log directory
    2. update cache directory
    3. update checkpoint and `temp_dir` of tempfile

    :param cfg: an original cfg
    :param cfg: an updated cfg
    """

    cfg.export_path = os.path.abspath(cfg.export_path)
    if cfg.work_dir is None:
        cfg.work_dir = os.path.dirname(cfg.export_path)

    # Call resolve_job_directories to finalize all job-related paths
    cfg = resolve_job_id(cfg)
    cfg = resolve_job_directories(cfg)

    timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
    if not load_configs_only:
        export_rel_path = os.path.relpath(cfg.export_path, start=cfg.work_dir)

        if not os.path.exists(cfg.event_log_dir):
            os.makedirs(cfg.event_log_dir, exist_ok=True)
        logfile_name = f"export_{export_rel_path}_time_{timestamp}.txt"
        setup_logger(
            save_dir=cfg.event_log_dir,
            filename=logfile_name,
            level="DEBUG" if cfg.get("debug", False) else "INFO",
            redirect=cfg.get("executor_type", "default") == "default",
        )

    # check and get dataset dir
    if cfg.get("dataset_path", None) and os.path.exists(cfg.dataset_path):
        logger.info("dataset_path config is set and a valid local path")
        cfg.dataset_path = os.path.abspath(cfg.dataset_path)
    elif cfg.get("dataset_path", "") == "" and cfg.get("dataset", None):
        logger.info("dataset_path config is empty; dataset is present")
    else:
        logger.warning(
            f"dataset_path [{cfg.get('dataset_path', '')}] is not a valid "
            f"local path, AND dataset is not present. "
            f"Please check and retry, otherwise we "
            f"will treat dataset_path as a remote dataset or a "
            f"mixture of several datasets."
        )

    # check number of processes np
    from data_juicer.utils.resource_utils import cpu_count

    sys_cpu_count = cpu_count(cfg)
    if cfg.get("np", None) and cfg.np > sys_cpu_count:
        logger.warning(
            f"Number of processes `np` is set as [{cfg.np}], which "
            f"is larger than the cpu count [{sys_cpu_count}]. Due "
            f"to the data processing of Data-Juicer is a "
            f"computation-intensive task, we recommend to set it to"
            f" a value <= cpu count. Set it to [{sys_cpu_count}] "
            f"here."
        )
        cfg.np = sys_cpu_count

    # whether or not to use cache management
    # disabling the cache or using checkpoint explicitly will turn off the
    # cache management.
    if not cfg.get("use_cache", True) or cfg.get("use_checkpoint", False):
        logger.warning("Cache management of datasets is disabled.")
        from datasets import disable_caching

        disable_caching()
        cfg.use_cache = False

        # disabled cache compression when cache is disabled
        if cfg.cache_compress:
            logger.warning("Disable cache compression due to disabled cache.")
            cfg.cache_compress = None

        # when disabling cache, enable the temp_dir argument
        logger.warning(f"Set temp directory to store temp files to " f"[{cfg.temp_dir}].")
        import tempfile

        if cfg.temp_dir is not None and not os.path.exists(cfg.temp_dir):
            os.makedirs(cfg.temp_dir, exist_ok=True)
        tempfile.tempdir = cfg.temp_dir

    # The checkpoint mode is not compatible with op fusion for now.
    if cfg.get("op_fusion", False):
        cfg.use_checkpoint = False
        cfg.fusion_strategy = cfg.fusion_strategy.lower()
        if cfg.fusion_strategy not in FUSION_STRATEGIES:
            raise NotImplementedError(
                f"Unsupported OP fusion strategy [{cfg.fusion_strategy}]. " f"Should be one of {FUSION_STRATEGIES}."
            )

    # update huggingface datasets cache directory only when ds_cache_dir is set
    from datasets import config

    if cfg.get("ds_cache_dir", None) is not None:
        logger.warning(
            f"Set dataset cache directory to {cfg.ds_cache_dir} "
            f"using the ds_cache_dir argument, which is "
            f"{config.HF_DATASETS_CACHE} before based on the env "
            f"variable HF_DATASETS_CACHE."
        )
        update_ds_cache_dir_and_related_vars(cfg.ds_cache_dir)
    else:
        cfg.ds_cache_dir = str(config.HF_DATASETS_CACHE)

    # add all filters that produce stats
    if cfg.get("auto", False):
        cfg.process = load_ops_with_stats_meta()

    # Apply text_key modification during initializing configs
    # users can freely specify text_key for different ops using `text_key`
    # otherwise, set arg text_key of each op to text_keys
    cfg.text_keys = cfg.get("text_keys", "text")
    if isinstance(cfg.text_keys, list):
        text_key = cfg.text_keys[0]
    else:
        text_key = cfg.text_keys

    SpecialTokens.image = cfg.get("image_special_token", SpecialTokens.image)
    SpecialTokens.audio = cfg.get("audio_special_token", SpecialTokens.audio)
    SpecialTokens.video = cfg.get("video_special_token", SpecialTokens.video)
    SpecialTokens.eoc = cfg.get("eoc_special_token", SpecialTokens.eoc)

    op_attrs = {
        "text_key": text_key,
        "image_key": cfg.get("image_key", "images"),
        "audio_key": cfg.get("audio_key", "audios"),
        "video_key": cfg.get("video_key", "videos"),
        "image_bytes_key": cfg.get("image_bytes_key", "image_bytes"),
        "turbo": cfg.get("turbo", False),
        "skip_op_error": cfg.get("skip_op_error", True),
        "work_dir": cfg.work_dir,
    }
    if not is_ray_mode():
        op_attrs.update({"num_proc": cfg.get("np", None)})
    cfg.process = update_op_attr(cfg.process, op_attrs)

    return cfg


def load_ops_with_stats_meta():
    import pkgutil

    import data_juicer.ops.filter as djfilter
    from data_juicer.ops import NON_STATS_FILTERS, TAGGING_OPS

    stats_filters = [
        {filter_name: {}}
        for _, filter_name, _ in pkgutil.iter_modules(djfilter.__path__)
        if filter_name not in NON_STATS_FILTERS.modules
    ]
    meta_ops = [{op_name: {}} for op_name in TAGGING_OPS.modules]
    return stats_filters + meta_ops


def update_op_attr(op_list: list, attr_dict: dict = None):
    if not attr_dict:
        return op_list
    updated_op_list = []
    for op in op_list:
        for op_name in op:
            args = op[op_name]
            if args is None:
                args = attr_dict
            else:
                for key in attr_dict:
                    if key not in args or args[key] is None:
                        args[key] = attr_dict[key]
            op[op_name] = args
        updated_op_list.append(op)
    return updated_op_list


def _collect_config_info_from_class_docs(configurable_ops, parser):
    """
    Add ops and its params to parser for command line with optimized performance.
    """
    with timing_context("Collecting operator configuration info"):
        op_params = {}

        # Add arguments for all provided operators
        for op_name, op_class in configurable_ops:
            params = parser.add_class_arguments(
                theclass=op_class, nested_key=op_name, fail_untyped=False, instantiate=False
            )
            op_params[op_name] = params

        return op_params


def sort_op_by_types_and_names(op_name_classes):
    """
    Split ops items by op type and sort them to sub-ops by name, then concat
    together.

    :param op_name_classes: a list of op modules
    :return: sorted op list , each item is a pair of op_name and
        op_class
    """
    with timing_context("Sorting operators by types and names"):
        mapper_ops = [(name, c) for (name, c) in op_name_classes if "mapper" in name]
        filter_ops = [(name, c) for (name, c) in op_name_classes if "filter" in name]
        deduplicator_ops = [(name, c) for (name, c) in op_name_classes if "deduplicator" in name]
        selector_ops = [(name, c) for (name, c) in op_name_classes if "selector" in name]
        grouper_ops = [(name, c) for (name, c) in op_name_classes if "grouper" in name]
        aggregator_ops = [(name, c) for (name, c) in op_name_classes if "aggregator" in name]
        ops_sorted_by_types = (
            sorted(mapper_ops)
            + sorted(filter_ops)
            + sorted(deduplicator_ops)
            + sorted(selector_ops)
            + sorted(grouper_ops)
            + sorted(aggregator_ops)
        )
        return ops_sorted_by_types


def update_op_process(cfg, parser, used_ops=None):
    """
    Update operator process configuration with optimized performance.

    Args:
        cfg: Configuration namespace
        parser: Argument parser
        used_ops: Set of operator names that are actually used in the config
    """
    if used_ops is None:
        used_ops = set(OPERATORS.modules.keys())

    # Get command line args for operators in one pass
    option_in_commands = set()
    full_option_in_commands = set()

    for arg in parser.args:
        if arg.startswith("--"):
            parts = arg.split("--")[1].split(".")
            op_name = parts[0]
            if op_name in used_ops:
                option_in_commands.add(op_name)
                full_option_in_commands.add(arg.split("=")[0])

    if cfg.process is None:
        cfg.process = []

    # Create direct mapping of operator names to their configs
    op_configs = {}
    for op in cfg.process:
        op_configs.setdefault(list(op.keys())[0], []).append(op[list(op.keys())[0]])

    # Process each used operator
    temp_cfg = cfg
    op_name_count = {}
    for op_name in used_ops:
        op_config = op_configs.get(op_name)
        op_config_list = []
        if op_name not in option_in_commands:
            # Update op params if set
            if op_config:
                for op_c in op_config:
                    temp_cfg = parser.merge_config(dict_to_namespace({op_name: op_c}), temp_cfg)
                    oc = namespace_to_dict(temp_cfg)[op_name]
                    op_config_list.append(oc)
                temp_cfg = parser.merge_config(dict_to_namespace({op_name: op_config_list}), temp_cfg)
        else:
            # Remove args that will be overridden by command line
            if op_config:
                for op_c in op_config:
                    for full_option in full_option_in_commands:
                        key = full_option.split(".")[1]
                        if key in op_c:
                            op_c.pop(key)
                    temp_cfg = parser.merge_config(dict_to_namespace({op_name: op_c}), temp_cfg)
                    oc = namespace_to_dict(temp_cfg)[op_name]
                    op_config_list.append(oc)
                temp_cfg = parser.merge_config(dict_to_namespace({op_name: op_config_list}), temp_cfg)

        # Update op params
        internal_op_para = temp_cfg.get(op_name)
        # Update or add the operator to process list
        if op_name in op_configs:
            # Update existing operator
            for i, op_in_process in enumerate(cfg.process):
                if isinstance(internal_op_para, list):
                    if list(op_in_process.keys())[0] == op_name:
                        if op_name not in op_name_count:
                            op_name_count[op_name] = 0
                        else:
                            op_name_count[op_name] += 1
                        cfg.process[i] = {
                            op_name: (
                                None
                                if internal_op_para is None
                                else namespace_to_dict(internal_op_para[op_name_count[op_name]])
                            )
                        }
        else:
            # Add new operator
            cfg.process.append({op_name: None if internal_op_para is None else namespace_to_dict(internal_op_para)})

    # Optimize type checking
    recognized_args = {
        action.dest for action in parser._actions if hasattr(action, "dest") and isinstance(action, ActionTypeHint)
    }

    # check the op params via type hint
    temp_parser = copy.deepcopy(parser)

    temp_args = namespace_to_arg_list(temp_cfg, includes=recognized_args, excludes=["config"])

    if temp_cfg.config:
        temp_args.extend(["--config", os.path.abspath(temp_cfg.config[0])])
    else:
        temp_args.append("--auto")

    # validate
    temp_parser.parse_args(temp_args)

    return cfg


def namespace_to_arg_list(namespace, prefix="", includes=None, excludes=None):
    arg_list = []

    for key, value in vars(namespace).items():
        if issubclass(type(value), Namespace):
            nested_args = namespace_to_arg_list(value, f"{prefix}{key}.")
            arg_list.extend(nested_args)
        elif value is not None:
            concat_key = f"{prefix}{key}"
            if includes is not None and concat_key not in includes:
                continue
            if excludes is not None and concat_key in excludes:
                continue
            if key == "process":
                arg_list.append(f"--{concat_key}={json.dumps(value, ensure_ascii=False)}")
            else:
                arg_list.append(f"--{concat_key}={value}")

    return arg_list


def save_cli_arguments(cfg: Namespace):
    """Save CLI arguments to cli.yaml in the work directory."""
    if not hasattr(cfg, "work_dir") or not cfg.work_dir:
        return

    # Get the original CLI arguments if available
    original_args = getattr(cfg, "_original_args", None)
    if not original_args:
        # Try to reconstruct from sys.argv if available
        import sys

        original_args = sys.argv[1:] if len(sys.argv) > 1 else []

    if not original_args:
        logger.warning("No CLI arguments available to save")
        return

    # Create cli.yaml in work directory
    cli_path = os.path.join(cfg.work_dir, "cli.yaml")

    # Convert args to a simple format
    cli_data = {"arguments": original_args}

    # Save as YAML
    import yaml

    with open(cli_path, "w") as f:
        yaml.dump(cli_data, f, default_flow_style=False, indent=2)

    logger.info(f"💾 Saved CLI arguments to: {cli_path}")


def validate_config_for_resumption(cfg: Namespace, work_dir: str, original_args: List[str] = None) -> bool:
    """Validate that the current config matches the job's saved config for safe resumption.

    Does verbatim comparison between:
    1. Original config.yaml + cli.yaml (saved during job creation)
    2. Current config (from current command)

    Sets cfg._same_yaml_config = True/False for the executor to use.
    """
    try:
        from pathlib import Path

        # Find the original config file in the work directory
        config_files = list(Path(work_dir).glob("*.yaml")) + list(Path(work_dir).glob("*.yml"))
        if not config_files:
            logger.warning(f"No config file found in work directory: {work_dir}")
            cfg._same_yaml_config = False
            return False

        # Find the original config.yaml (not cli.yaml)
        original_config_file = None
        for config_file in config_files:
            if config_file.name != "cli.yaml":
                original_config_file = config_file
                break

        if not original_config_file:
            logger.warning(f"No original config file found in work directory: {work_dir}")
            cfg._same_yaml_config = False
            return False

        # 1. Direct file comparison for config files
        current_config_file = cfg.config[0] if hasattr(cfg, "config") and cfg.config else None
        if not current_config_file:
            logger.error("No current config file found")
            cfg._same_yaml_config = False
            return False

        with open(original_config_file, "r") as f:
            original_config_content = f.read()
        with open(current_config_file, "r") as f:
            current_config_content = f.read()

        config_match = original_config_content.strip() == current_config_content.strip()

        # 2. Per-key comparison for CLI arguments
        cli_file = Path(work_dir) / "cli.yaml"
        cli_config = {}
        if cli_file.exists():
            with open(cli_file, "r") as f:
                cli_data = yaml.safe_load(f)
                cli_config = _parse_cli_to_config(cli_data.get("arguments", []))

        # Get current CLI arguments from the original args passed to init_configs
        current_cli_args = original_args
        if not current_cli_args:
            # Fallback: try to get from sys.argv
            import sys

            current_cli_args = sys.argv[1:] if len(sys.argv) > 1 else []

        current_cli_config = _parse_cli_to_config(current_cli_args)

        # Compare CLI arguments per key
        cli_differences = []
        all_cli_keys = set(cli_config.keys()) | set(current_cli_config.keys())
        excluded_keys = {"config", "_original_args", "backed_up_config_path", "_same_yaml_config", "job_id", "work_dir"}

        for key in all_cli_keys:
            if key in excluded_keys:
                continue

            original_value = cli_config.get(key)
            current_value = current_cli_config.get(key)

            if original_value != current_value:
                cli_differences.append({"key": key, "original": original_value, "current": current_value})

        cli_match = len(cli_differences) == 0

        if not config_match or not cli_match:
            logger.error("❌ Config validation failed - configurations don't match:")
            if not config_match:
                logger.error("   [config] Config file content differs")
            if not cli_match:
                logger.error("   [cli] CLI arguments differ:")
                for diff in cli_differences:
                    logger.error(f"      {diff['key']}: {diff['original']} → {diff['current']}")
            logger.error("💡 Use the same config file and CLI arguments for resumption")
            cfg._same_yaml_config = False
            return False

        logger.info("✅ Config validation passed - configurations match exactly")
        cfg._same_yaml_config = True
        return True

    except Exception as e:
        logger.error(f"Error validating config for resumption: {e}")
        cfg._same_yaml_config = False
        return False


def _parse_cli_to_config(cli_args: list) -> dict:
    """Parse CLI arguments into config dictionary format."""
    config = {}

    i = 0
    while i < len(cli_args):
        arg = cli_args[i]

        if arg.startswith("--"):
            key = arg[2:]  # Remove '--'

            # Check if next arg is a value (not another flag)
            if i + 1 < len(cli_args) and not cli_args[i + 1].startswith("--"):
                value = cli_args[i + 1]

                # Try to parse as different types
                if value.lower() in ["true", "false"]:
                    config[key] = value.lower() == "true"
                elif value.isdigit():
                    config[key] = int(value)
                elif value.replace(".", "").isdigit():
                    config[key] = float(value)
                else:
                    config[key] = value

                i += 2  # Skip both key and value
            else:
                # Boolean flag (no value)
                config[key] = True
                i += 1
        else:
            i += 1

    return config


def config_backup(cfg: Namespace):
    if not cfg.get("config", None):
        return
    cfg_path = os.path.abspath(cfg.config[0])

    # Use the backed_up_config_path which should be set by resolve_job_directories
    if hasattr(cfg, "backed_up_config_path"):
        target_path = cfg.backed_up_config_path
    else:
        # Fallback: use work_dir with original filename
        work_dir = cfg.work_dir
        original_config_name = os.path.basename(cfg_path)
        target_path = os.path.join(work_dir, original_config_name)

    if not os.path.exists(target_path):
        logger.info(f"Back up the input config file [{cfg_path}] to [{target_path}]")
        shutil.copyfile(cfg_path, target_path)
    else:
        logger.info(f"Config file [{cfg_path}] already exists at [{target_path}]")

    # Also save CLI arguments
    save_cli_arguments(cfg)


def display_config(cfg: Namespace):
    import pprint

    from tabulate import tabulate

    table_header = ["key", "values"]

    # remove ops outside the process list for better displaying
    shown_cfg = cfg.clone()
    for op in OPERATORS.modules.keys():
        _ = shown_cfg.pop(op)

    # construct the table as 2 columns
    config_table = [(k, pprint.pformat(v, compact=True)) for k, v in shown_cfg.items()]
    table = tabulate(config_table, headers=table_header, tablefmt="fancy_grid")

    logger.info("Configuration table: ")
    print(table)


def export_config(
    cfg: Namespace,
    path: str,
    format: str = "yaml",
    skip_none: bool = True,
    skip_check: bool = True,
    overwrite: bool = False,
    multifile: bool = True,
):
    """
    Save the config object, some params are from jsonargparse

    :param cfg: cfg object to save (Namespace type)
    :param path: the save path
    :param format: 'yaml', 'json', 'json_indented', 'parser_mode'
    :param skip_none: Whether to exclude entries whose value is None.
    :param skip_check: Whether to skip parser checking.
    :param overwrite: Whether to overwrite existing files.
    :param multifile: Whether to save multiple config files
        by using the __path__ metas.

    :return:
    """
    # remove ops outside the process list for better displaying
    cfg_to_export = cfg.clone()
    cfg_to_export = prepare_cfgs_for_export(cfg_to_export)

    global global_parser
    if not global_parser:
        init_configs()  # enable the customized type parser
    if isinstance(cfg_to_export, Namespace):
        cfg_to_export = namespace_to_dict(cfg_to_export)
    global_parser.save(
        cfg=cfg_to_export,
        path=path,
        format=format,
        skip_none=skip_none,
        skip_check=skip_check,
        overwrite=overwrite,
        multifile=multifile,
    )

    logger.info(f"Saved the configuration in {path}")


def merge_config(ori_cfg: Namespace, new_cfg: Namespace):
    """
    Merge configuration from new_cfg into ori_cfg

    :param ori_cfg: the original configuration object, whose type is
        expected as namespace from jsonargparse
    :param new_cfg: the configuration object to be merged, whose type is
        expected as dict or namespace from jsonargparse

    :return: cfg_after_merge
    """
    try:
        ori_specified_op_names = set()
        ori_specified_op_idx = {}  # {op_name: op_order}

        for op_order, op_in_process in enumerate(ori_cfg.process):
            op_name = list(op_in_process.keys())[0]
            ori_specified_op_names.add(op_name)
            ori_specified_op_idx[op_name] = op_order

        for new_k, new_v in new_cfg.items():
            # merge parameters other than `cfg.process` and DJ-OPs
            if new_k in ori_cfg and new_k != "process" and "." not in new_k:
                logger.info("=" * 15)
                logger.info(f"Before merging, the cfg item is: " f"{new_k}: {ori_cfg[new_k]}")
                ori_cfg[new_k] = new_v
                logger.info(f"After merging,  the cfg item is: " f"{new_k}: {new_v}")
                logger.info("=" * 15)
            else:
                # merge parameters of DJ-OPs into cfg.process
                # for nested style, e.g., `remove_table_text_mapper.min_col: 2`
                key_as_groups = new_k.split(".")
                if len(key_as_groups) > 1 and key_as_groups[0] in ori_specified_op_names:
                    op_name, para_name = key_as_groups[0], key_as_groups[1]
                    op_order = ori_specified_op_idx[op_name]
                    ori_cfg_val = ori_cfg.process[op_order][op_name][para_name]
                    logger.info("=" * 15)
                    logger.info(f"Before merging, the cfg item is: " f"{new_k}: {ori_cfg_val}")
                    ori_cfg.process[op_order][op_name][para_name] = new_v
                    logger.info(f"After merging,  the cfg item is: " f"{new_k}: {new_v}")
                    logger.info("=" * 15)

        ori_cfg = init_setup_from_cfg(ori_cfg)

        # copy the config file into the work directory
        config_backup(ori_cfg)

        return ori_cfg

    except ArgumentError:
        logger.error("Config merge failed")


def prepare_side_configs(ori_config: Union[str, Namespace, Dict]):
    """
    parse the config if ori_config is a string of a config file path with
        yaml, yml or json format

    :param ori_config: a config dict or a string of a config file path with
        yaml, yml or json format

    :return: a config dict
    """

    if isinstance(ori_config, str):
        # config path
        if ori_config.endswith(".yaml") or ori_config.endswith(".yml"):
            with open(ori_config) as fin:
                config = yaml.safe_load(fin)
        elif ori_config.endswith(".json"):
            with open(ori_config) as fin:
                config = json.load(fin)
        else:
            raise TypeError(
                f"Unrecognized config file type: [{ori_config}]. "
                f'Should be one of the types [".yaml", ".yml", '
                f'".json"].'
            )
    elif isinstance(ori_config, dict) or isinstance(ori_config, Namespace):
        config = ori_config
    else:
        raise TypeError(f"Unrecognized side config type: [{type(ori_config)}].")

    return config


def get_init_configs(cfg: Union[Namespace, Dict], load_configs_only: bool = True):
    """
    set init configs of data-juicer for cfg
    """
    temp_dir = tempfile.gettempdir()
    temp_file = os.path.join(temp_dir, "job_dj_config.json")
    if isinstance(cfg, Namespace):
        cfg = namespace_to_dict(cfg)

    # Remove internal attributes that are not part of the configuration schema
    # to avoid validation errors when re-initializing the config
    if isinstance(cfg, dict):
        cfg = cfg.copy()
        # Remove internal attributes that are added during config processing
        internal_attrs = [
            "_user_provided_job_id",
            "_same_yaml_config",
            "metadata_dir",
            "results_dir",
            "event_log_file",
            "job_summary_file",
            "backed_up_config_path",
        ]
        for attr in internal_attrs:
            cfg.pop(attr, None)

    # create a temp config file
    with open(temp_file, "w") as f:
        json.dump(prepare_cfgs_for_export(cfg), f)
    inited_dj_cfg = init_configs(["--config", temp_file], load_configs_only=load_configs_only)
    return inited_dj_cfg


def get_default_cfg():
    """Get default config values from config_all.yaml"""
    cfg = Namespace()

    # Get path to config_all.yaml
    config_dir = os.path.dirname(os.path.abspath(__file__))
    default_config_path = os.path.join(config_dir, "../../configs/config_min.yaml")

    # Load default values from yaml
    with open(default_config_path, "r", encoding="utf-8") as f:
        defaults = yaml.safe_load(f)

    # Convert to flat dictionary for namespace
    flat_defaults = {
        # Add other top-level keys from config_min.yaml
        **defaults
    }

    # Update cfg with defaults
    for key, value in flat_defaults.items():
        if not hasattr(cfg, key):
            setattr(cfg, key, value)

    return cfg


def prepare_cfgs_for_export(cfg):
    # 1. convert Path to str
    if "config" in cfg:
        cfg["config"] = [str(p) for p in cfg["config"]]
    # 2. remove level-1 op cfgs outside the process list
    for op in OPERATORS.modules.keys():
        if op in cfg:
            _ = cfg.pop(op)
    return cfg


def resolve_job_id(cfg):
    """Resolve or auto-generate job_id and set it on cfg."""
    job_id = getattr(cfg, "job_id", None)

    # Track whether job_id was user-provided
    if job_id is not None:
        # User explicitly provided a job_id
        setattr(cfg, "_user_provided_job_id", True)
    else:
        # No job_id provided by user
        setattr(cfg, "_user_provided_job_id", False)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        short_hash = uuid.uuid4().hex[:6]
        job_id = f"{timestamp}_{short_hash}"
        setattr(cfg, "job_id", job_id)
    return cfg


def validate_work_dir_config(work_dir: str) -> None:
    """
    Validate work_dir configuration to ensure {job_id} placement rules are followed.

    Args:
        work_dir: The work_dir string to validate

    Raises:
        ValueError: If {job_id} is not at the end of the path
    """
    if "{job_id}" in work_dir:
        # Check if {job_id} is at the end of the path
        if not work_dir.rstrip("/").endswith("{job_id}"):
            raise ValueError(
                f"Invalid work_dir configuration: '{{job_id}}' must be the last part of the path. "
                f"Current: '{work_dir}'. "
                f"Expected format: 'path/to/directory/{{job_id}}'"
            )


def resolve_job_directories(cfg):
    """
    Centralize directory resolution and placeholder substitution. Assumes job_id is already set.

    Job Directory Rules:
    - If work_dir contains '{job_id}' placeholder, it MUST be the last part of the path
    - Examples:
      ✅ work_dir: "./outputs/my_project/{job_id}"     # Valid
      ✅ work_dir: "/data/experiments/{job_id}"        # Valid
      ❌ work_dir: "./outputs/{job_id}/results"        # Invalid - {job_id} not at end
      ❌ work_dir: "./{job_id}/outputs/data"           # Invalid - {job_id} not at end

    - If work_dir does NOT contain '{job_id}', job_id will be appended automatically
    - Examples:
      work_dir: "./outputs/my_project" → work_dir: "./outputs/my_project/20250804_143022_abc123"

    After resolution, work_dir will always include job_id at the end.
    """
    # 1. placeholder map
    placeholder_map = {"work_dir": cfg.work_dir, "job_id": getattr(cfg, "job_id", "")}

    # 2. Validate {job_id} placement in work_dir before substitution
    original_work_dir = cfg.work_dir
    validate_work_dir_config(original_work_dir)

    # 3. substitute placeholders in all relevant paths (change-detection loop)
    max_passes = 10
    for _ in range(max_passes):
        changed = False
        for key in ["work_dir", "event_log_dir", "checkpoint_dir", "export_path", "dataset_path", "partition_dir"]:
            val = getattr(cfg, key, None)
            if isinstance(val, str):
                new_val = val.format(**placeholder_map)
                if new_val != val:
                    setattr(cfg, key, new_val)
                    changed = True
        # update placeholder_map in case work_dir or job_id changed
        placeholder_map = {"work_dir": cfg.work_dir, "job_id": getattr(cfg, "job_id", "")}
        if not changed:
            break
    else:
        raise RuntimeError("Too many placeholder substitution passes (possible recursive placeholders?)")

    # 4. directory resolution
    job_id = getattr(cfg, "job_id", None)
    if not job_id:
        raise ValueError("job_id must be set before resolving job directories.")

    # Ensure work_dir always includes job_id at the end
    # If work_dir already ends with job_id (from placeholder substitution), keep it as-is
    # Otherwise, append job_id automatically
    if not (cfg.work_dir.endswith(job_id) or os.path.basename(cfg.work_dir) == job_id):
        cfg.work_dir = os.path.join(cfg.work_dir, job_id)

    # All job-specific directories are under work_dir
    cfg.event_log_dir = os.path.join(cfg.work_dir, "logs")
    cfg.checkpoint_dir = os.path.join(cfg.work_dir, "checkpoints")
    cfg.partition_dir = os.path.join(cfg.work_dir, "partitions")
    cfg.metadata_dir = os.path.join(cfg.work_dir, "metadata")
    cfg.results_dir = os.path.join(cfg.work_dir, "results")
    cfg.event_log_file = os.path.join(cfg.work_dir, "events.jsonl")
    cfg.job_summary_file = os.path.join(cfg.work_dir, "job_summary.json")
    # Set backed_up_config_path using original config filename
    if hasattr(cfg, "config") and cfg.config:
        original_config_name = os.path.basename(cfg.config[0])
        cfg.backed_up_config_path = os.path.join(cfg.work_dir, original_config_name)
    else:
        cfg.backed_up_config_path = os.path.join(cfg.work_dir, "config.yaml")

    return cfg

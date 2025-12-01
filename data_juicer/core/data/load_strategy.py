import fnmatch
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Type

import datasets
from jsonargparse import Namespace
from loguru import logger

from data_juicer.core.data import DJDataset
from data_juicer.core.data.config_validator import ConfigValidator
from data_juicer.download.downloader import validate_snapshot_format
from data_juicer.format.formatter import unify_format
from data_juicer.format.load import load_formatter
from data_juicer.utils.s3_utils import create_pyarrow_s3_filesystem, validate_s3_path

# based on executor type and data source type, use different
# data load strategy to product corresponding datasets
# DJDataset, RayDataset, DaskDataset, etc


@dataclass(frozen=True)
class StrategyKey:
    """
    Immutable key for strategy registration with wildcard support
    """

    executor_type: str
    data_type: str
    data_source: str

    def matches(self, other: "StrategyKey") -> bool:
        """
        Check if this key matches another key with wildcard support

        Supports Unix-style wildcards:
        - '*' matches any string
        - '?' matches any single character
        - '[seq]' matches any character in seq
        - '[!seq]' matches any character not in seq
        """
        return (
            fnmatch.fnmatch(other.executor_type, self.executor_type)
            and fnmatch.fnmatch(other.data_type, self.data_type)
            and fnmatch.fnmatch(other.data_source, self.data_source)
        )


class DataLoadStrategy(ABC, ConfigValidator):
    """
    abstract class for data load strategy
    """

    def __init__(self, ds_config: Dict, cfg: Namespace):
        self.validate_config(ds_config)
        self.ds_config = ds_config
        self.cfg = cfg
        self.weight = ds_config.get("weight", 1.0)  # default weight is 1.0

    @abstractmethod
    def load_data(self, **kwargs) -> DJDataset:
        """Need to be implemented in the"""


class DataLoadStrategyRegistry:
    """
    Flexible strategy registry with wildcard matching
    """

    _strategies: Dict[StrategyKey, Type[DataLoadStrategy]] = {}

    @classmethod
    def get_strategy_class(
        cls, executor_type: str, data_type: str, data_source: str
    ) -> Optional[Type[DataLoadStrategy]]:
        """
        Retrieve the most specific matching strategy

        Matching priority:
        1. Exact match
        2. Wildcard matches from most specific to most general
        """
        logger.info(
            f"Getting strategy class for "
            f"exec: {executor_type}, "
            f"data_type: {data_type}, "
            f"data_source: {data_source}"
        )

        # default to wildcard if not provided
        executor_type = executor_type or "*"
        data_type = data_type or "*"
        data_source = data_source or "*"

        # Create the lookup key
        lookup_key = StrategyKey(executor_type, data_type, data_source)

        # First, check for exact match
        exact_match = cls._strategies.get(lookup_key)
        if exact_match:
            return exact_match

        # Find all matching wildcard strategies
        matching_strategies = []
        for registered_key, strategy in cls._strategies.items():
            if registered_key.matches(lookup_key):
                matching_strategies.append((registered_key, strategy))

        # Sort matching strategies by specificity (fewer wildcards first)
        if matching_strategies:

            def specificity_score(key: StrategyKey) -> int:
                """
                Calculate specificity score (lower is more specific)
                Exact match: 0
                One wildcard: 1
                Two wildcards: 2
                All wildcards: 3
                """
                return sum(1 for part in [key.executor_type, key.data_type, key.data_source] if part == "*")

            matching_strategies.sort(key=lambda x: specificity_score(x[0]))
            found = matching_strategies[0][1]
            logger.info(f"Found matching strategies: {found}")
            return found

        # No matching strategy found
        logger.warning(
            f"No matching strategy found for combination "
            f"exec: {executor_type}, "
            f"data_type: {data_type}, "
            f"data_source: {data_source}"
        )
        return None

    @classmethod
    def register(cls, executor_type: str, data_type: str, data_source: str):
        """
        Decorator for registering data load strategies with wildcard support

        :param executor_type: Type of executor (e.g., 'default', 'ray')
        :param data_type: Type of data (e.g., 'local', 'remote')
        :param data_source: Specific data source (e.g., 'arxiv', 's3')
        :return: Decorator function
        """

        def decorator(strategy_class: Type[DataLoadStrategy]):
            """
            Register the strategy class for the given key

            :param strategy_class: Strategy class to register
            :return: Original strategy class
            """
            key = StrategyKey(executor_type, data_type, data_source)
            cls._strategies[key] = strategy_class
            return strategy_class

        return decorator


class RayDataLoadStrategy(DataLoadStrategy):
    """
    abstract class for data load strategy for RayExecutor
    """

    @abstractmethod
    def load_data(self, **kwargs) -> DJDataset:
        """Need to be implemented in the"""


class DefaultDataLoadStrategy(DataLoadStrategy):
    """
    abstract class for data load strategy for LocalExecutor
    """

    @abstractmethod
    def load_data(self, **kwargs) -> DJDataset:
        """Need to be implemented in the"""


# TODO dask support
# class DaskDataLoadStrategy(DataLoadStrategy):
#     @abstractmethod
#     def load_data(self) -> Union[DaskDataset]:
#         pass

# TODO nemo support
# class NemoDataLoadStrategy(DataLoadStrategy):
#     @abstractmethod
#     def load_data(self) -> Union[NemoDataset]:
#         pass


@DataLoadStrategyRegistry.register("ray", "local", "*")
class RayLocalJsonDataLoadStrategy(RayDataLoadStrategy):
    # TODO ray defaults to json

    CONFIG_VALIDATION_RULES = {"required_fields": ["path"], "field_types": {"path": str}, "custom_validators": {}}

    def load_data(self, **kwargs):
        from data_juicer.core.data.ray_dataset import RayDataset

        path = self.ds_config["path"]

        # Convert to absolute path if relative
        if not os.path.isabs(path):
            # Try multiple base paths
            possible_paths = [
                # Current working directory
                os.path.abspath(path),
                # Original DJ root directory relative to script location
                os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", path)),
                # User's home directory
                os.path.expanduser(os.path.join("~", path)),
            ]

            # Ray work directory
            ray_work_dir = getattr(self.cfg, "work_dir", None) if self.cfg else None
            if ray_work_dir:
                possible_paths.append(os.path.abspath(os.path.join(ray_work_dir, path)))

            # Try each path
            for abs_path in possible_paths:
                if os.path.exists(abs_path):
                    path = abs_path
                    break
            else:
                # No valid path found
                raise FileNotFoundError(
                    f"Could not find file '{path}' in any location. "
                    f"Tried: {possible_paths}. "
                    f"Current working directory: {os.getcwd()}"
                )

        logger.info(f"Using resolved path for loading ray dataset: {path}")

        file_extension_map = {
            ".json": "json",
            ".jsonl": "json",
            ".txt": "text",
            ".csv": "csv",
            ".tsv": "csv",
            ".parquet": "parquet",
            ".npy": "numpy",
            ".tfrecords": "tfrecords",
            ".lance": "lance",
        }
        auto_detect = False
        data_source = self.ds_config.get("source", None)
        if data_source is None:
            auto_detect = True
        else:
            suffix = os.path.splitext(data_source)[1]
            if suffix in file_extension_map:
                data_format = file_extension_map[suffix]
            elif "." + data_source in file_extension_map:
                data_format = file_extension_map["." + data_source]
            else:
                auto_detect = True
        if auto_detect:
            item_path = path
            if os.path.isdir(item_path):
                # The first file encountered in the directory
                # determines which data reader to use.
                path_list = [path]
                not_found = True
                while not_found and len(path_list) > 0:
                    cur_path = path_list.pop()
                    for item in os.listdir(cur_path):
                        item_path = os.path.join(cur_path, item)
                        if os.path.isdir(item_path):
                            path_list.append(item_path)
                        elif os.path.isfile(item_path):
                            not_found = False
                            break
            file_extension = os.path.splitext(item_path)[1]
            # by default, we use json type to load data
            data_format = file_extension_map.get(file_extension, "json")
            logger.info(f"Try to load data as {data_format}.")
        else:
            logger.info(f"Loading {data_format} data.")
        try:
            dataset = RayDataset.read(data_format, path)
            return RayDataset(dataset, dataset_path=path, cfg=self.cfg)
        except Exception as e:
            if auto_detect:
                raise RuntimeError(
                    f"Failed to load data from {path}. "
                    f"Please check data format and set the correct `dataset.configs.source`. "
                    f"Current working directory: {os.getcwd()}. "
                    f"Error: {str(e)}"
                )
            else:
                raise RuntimeError(
                    f"Failed to load {data_format} data from {path}. "
                    f"Current working directory: {os.getcwd()}. "
                    f"Error: {str(e)}"
                )


@DataLoadStrategyRegistry.register("ray", "remote", "huggingface")
class RayHuggingfaceDataLoadStrategy(RayDataLoadStrategy):
    CONFIG_VALIDATION_RULES = {"required_fields": ["path"], "field_types": {"path": str}, "custom_validators": {}}

    def load_data(self, **kwargs):
        raise NotImplementedError("Huggingface data load strategy for Ray is not implemented")


@DataLoadStrategyRegistry.register("default", "local", "*")
class DefaultLocalDataLoadStrategy(DefaultDataLoadStrategy):
    """
    data load strategy for on disk data for LocalExecutor
    rely on AutoFormatter for actual data loading
    """

    CONFIG_VALIDATION_RULES = {"required_fields": ["path"], "field_types": {"path": str}, "custom_validators": {}}

    def load_data(self, **kwargs):
        # Get config values with defaults
        text_keys = getattr(self.cfg, "text_keys", ["text"])  # Default to ['text']
        suffixes = getattr(self.cfg, "suffixes", None)  # Default to None
        # if there is suffix_filter op, turn on the add_suffix flag
        add_suffix = False
        process_list = self.cfg.process if hasattr(self.cfg, "process") else []
        for op in process_list:
            op_name, _ = list(op.items())[0]
            if op_name == "suffix_filter":
                add_suffix = True
                break
        load_data_np = kwargs.get("num_proc", 1)

        # use proper formatter to load data
        formatter = load_formatter(
            dataset_path=self.ds_config["path"], text_keys=text_keys, suffixes=suffixes, add_suffix=add_suffix, **kwargs
        )
        # TODO more sophiscated localformatter routing
        return formatter.load_dataset(load_data_np, self.cfg)


@DataLoadStrategyRegistry.register("default", "remote", "huggingface")
class DefaultHuggingfaceDataLoadStrategy(DefaultDataLoadStrategy):
    """
    data load strategy for Huggingface dataset for LocalExecutor
    """

    CONFIG_VALIDATION_RULES = {
        "required_fields": ["path"],
        "optional_fields": ["split", "limit", "name", "data_files", "data_dir"],
        "field_types": {"path": str},
        "custom_validators": {},
    }

    def load_data(self, **kwargs):
        num_proc = kwargs.pop("num_proc", 1)
        ds = datasets.load_dataset(
            self.ds_config["path"],
            split=self.ds_config.get("split", None),
            data_files=self.ds_config.get("data_files", None),
            data_dir=self.ds_config.get("data_dir", None),
            name=self.ds_config.get("name", None),
            limit=self.ds_config.get("limit", None),
            num_proc=num_proc,
            **kwargs,
        )
        return unify_format(ds, text_keys=self.cfg.text_keys, num_proc=num_proc, global_cfg=self.cfg)


@DataLoadStrategyRegistry.register("default", "remote", "modelscope")
class DefaultModelScopeDataLoadStrategy(DefaultDataLoadStrategy):
    """
    data load strategy for ModelScope dataset for LocalExecutor
    """

    def load_data(self, **kwargs):
        raise NotImplementedError("ModelScope data load strategy is not implemented")


@DataLoadStrategyRegistry.register("default", "remote", "arxiv")
class DefaultArxivDataLoadStrategy(DefaultDataLoadStrategy):
    """
    data load strategy for arxiv dataset for LocalExecutor
    """

    CONFIG_VALIDATION_RULES = {
        "required_fields": ["path"],
        "field_types": {"path": (str)},  # has to be a string
        "custom_validators": {},
    }

    def load_data(self, **kwargs):
        raise NotImplementedError("Arxiv data load strategy is not implemented")


@DataLoadStrategyRegistry.register("default", "remote", "wiki")
class DefaultWikiDataLoadStrategy(DefaultDataLoadStrategy):
    """
    data load strategy for wiki dataset for LocalExecutor
    """

    CONFIG_VALIDATION_RULES = {"required_fields": ["path"], "field_types": {"path": str}, "custom_validators": {}}

    def load_data(self, **kwargs):
        raise NotImplementedError("Wiki data load strategy is not implemented")


@DataLoadStrategyRegistry.register("default", "remote", "commoncrawl")
class DefaultCommonCrawlDataLoadStrategy(DefaultDataLoadStrategy):
    """
    data load strategy for commoncrawl dataset for LocalExecutor
    """

    CONFIG_VALIDATION_RULES = {
        "required_fields": ["start_snapshot", "end_snapshot"],
        "optional_fields": ["aws", "url_limit"],
        "field_types": {"start_snapshot": str, "end_snapshot": str},
        "custom_validators": {
            "start_snashot": validate_snapshot_format,
            "end_snapshot": validate_snapshot_format,
            "url_limit": lambda x: x > 0,
        },
    }

    def load_data(self, **kwargs):
        raise NotImplementedError("CommonCrawl data load strategy is not implemented")


@DataLoadStrategyRegistry.register("default", "remote", "s3")
class DefaultS3DataLoadStrategy(DefaultDataLoadStrategy):
    """
    data load strategy for S3 datasets for LocalExecutor
    Uses fsspec/s3fs to access S3 files
    """

    CONFIG_VALIDATION_RULES = {
        "required_fields": ["path"],
        "optional_fields": [
            "aws_access_key_id",
            "aws_secret_access_key",
            "aws_session_token",
            "aws_region",
            "endpoint_url",
        ],
        "field_types": {"path": str},
        "custom_validators": {
            "path": lambda x: x.startswith("s3://"),
        },
    }

    def load_data(self, **kwargs):
        import os

        import datasets

        from data_juicer.format.formatter import unify_format
        from data_juicer.utils.s3_utils import get_aws_credentials

        path = self.ds_config["path"]
        validate_s3_path(path)

        load_data_np = kwargs.get("num_proc", 1)

        # Get config values with defaults
        text_keys = getattr(self.cfg, "text_keys", ["text"])

        logger.info(f"Loading dataset from S3: {path}")

        # Determine file format from extension (reuse logic from RayLocalJsonDataLoadStrategy)
        file_extension = os.path.splitext(path)[1].lower()
        file_extension_map = {
            ".json": "json",
            ".jsonl": "json",
            ".txt": "text",
            ".csv": "csv",
            ".tsv": "csv",
            ".parquet": "parquet",
        }
        data_format = file_extension_map.get(file_extension, "json")  # Default to json
        logger.info(f"Detected format: {data_format} for S3 path: {path}")

        # Create S3FileSystem with credentials from config
        # Get credentials with priority order (env vars first, then config)
        aws_access_key_id, aws_secret_access_key, aws_session_token, _ = get_aws_credentials(self.ds_config)
        # Region is auto-detected from S3 path for HuggingFace datasets, don't need it from credentials

        # Build storage_options for S3FileSystem
        # Note: region should NOT be in storage_options for HuggingFace datasets
        # as it causes issues with AioSession. Region is auto-detected from S3 path.
        storage_options = {}
        if aws_access_key_id:
            storage_options["key"] = aws_access_key_id
        if aws_secret_access_key:
            storage_options["secret"] = aws_secret_access_key
        if aws_session_token:
            storage_options["token"] = aws_session_token
        # Region is auto-detected from S3 path, don't pass it in storage_options
        # If explicit region is needed, it should be set via AWS_REGION env var
        if "endpoint_url" in self.ds_config:
            storage_options["endpoint_url"] = self.ds_config["endpoint_url"]

        # HuggingFace datasets uses storage_options (not fs parameter) for filesystem configuration
        # storage_options are passed to fsspec/s3fs internally
        # For public buckets without credentials, use anonymous access
        # HuggingFace datasets uses storage_options for filesystem configuration.
        # If storage_options is empty, s3fs will use its default credential chain (e.g., IAM role, ~/.aws/credentials).
        if storage_options.get("key") or storage_options.get("secret"):
            logger.info("Using explicit AWS credentials for S3 access")
        else:
            logger.info("Using default AWS credential chain for S3 access")

        # Allow explicit anonymous access via config
        if self.ds_config.get("anon"):
            storage_options["anon"] = True
            logger.info("Anonymous access for public S3 bucket enabled via config.")

        try:
            # Pass storage_options to load_dataset (not fs parameter)
            # storage_options are used by fsspec/s3fs internally
            ds = datasets.load_dataset(
                data_format,
                data_files=path,  # Direct S3 path
                storage_options=storage_options,  # Pass storage_options for S3 filesystem configuration
                **kwargs,
            )
            # Handle DatasetDict (multiple splits) vs Dataset (single)
            if isinstance(ds, datasets.DatasetDict):
                from data_juicer.core.data import NestedDataset

                ds = NestedDataset(datasets.concatenate_datasets([d for d in ds.values()]))
            else:
                from data_juicer.core.data import NestedDataset

                ds = NestedDataset(ds)

            # Unify format
            ds = unify_format(ds, text_keys=text_keys, num_proc=load_data_np, global_cfg=self.cfg)
            return ds
        except Exception as e:
            raise RuntimeError(
                f"Failed to load dataset from S3 path {path}. "
                f"Ensure s3fs is installed and your AWS credentials are configured. "
                f"Error: {str(e)}"
            )


@DataLoadStrategyRegistry.register("ray", "remote", "s3")
class RayS3DataLoadStrategy(RayDataLoadStrategy):
    """
    data load strategy for S3 datasets for RayExecutor
    Uses PyArrow's filesystem to read from S3
    """

    CONFIG_VALIDATION_RULES = {
        "required_fields": ["path"],
        "optional_fields": [
            "aws_access_key_id",
            "aws_secret_access_key",
            "aws_session_token",
            "aws_region",
            "endpoint_url",
        ],
        "field_types": {"path": str},
        "custom_validators": {
            "path": lambda x: x.startswith("s3://"),
        },
    }

    def load_data(self, **kwargs):
        from data_juicer.core.data.ray_dataset import RayDataset

        path = self.ds_config["path"]
        validate_s3_path(path)

        # Create S3 filesystem using utility function
        s3_fs = create_pyarrow_s3_filesystem(self.ds_config)

        logger.info(f"Loading dataset from S3: {path}")

        # Determine file format from extension or config
        file_extension_map = {
            ".json": "json",
            ".jsonl": "json",
            ".txt": "text",
            ".csv": "csv",
            ".tsv": "csv",
            ".parquet": "parquet",
            ".npy": "numpy",
            ".tfrecords": "tfrecords",
            ".lance": "lance",
        }

        auto_detect = False
        data_source = self.ds_config.get("source", None)
        if data_source is None:
            auto_detect = True
        else:
            suffix = os.path.splitext(data_source)[1]
            if suffix in file_extension_map:
                data_format = file_extension_map[suffix]
            elif "." + data_source in file_extension_map:
                data_format = file_extension_map["." + data_source]
            else:
                auto_detect = True

        if auto_detect:
            # Extract extension from path
            file_extension = os.path.splitext(path)[1]
            data_format = file_extension_map.get(file_extension, "parquet")  # Default to parquet for S3
            logger.info(f"Auto-detected data format: {data_format}")
        else:
            logger.info(f"Using specified data format: {data_format}")

        try:
            import ray.data

            # Use ray.data functions directly with PyArrow filesystem support
            # Ray's read functions support filesystem parameter via PyArrow
            if data_format in {"json", "jsonl"}:
                # For JSON, we need to use read_json_stream with filesystem
                from data_juicer.core.data.ray_dataset import read_json_stream

                dataset = read_json_stream(path, filesystem=s3_fs)
            elif data_format == "parquet":
                dataset = ray.data.read_parquet(path, filesystem=s3_fs)
            elif data_format == "csv":
                dataset = ray.data.read_csv(path, filesystem=s3_fs)
            elif data_format == "text":
                dataset = ray.data.read_text(path, filesystem=s3_fs)
            elif data_format == "numpy":
                dataset = ray.data.read_numpy(path, filesystem=s3_fs)
            elif data_format == "tfrecords":
                dataset = ray.data.read_tfrecords(path, filesystem=s3_fs)
            elif data_format == "lance":
                dataset = ray.data.read_lance(path, filesystem=s3_fs)
            else:
                raise ValueError(f"Unsupported data format for S3: {data_format}")

            return RayDataset(dataset, dataset_path=path, cfg=self.cfg)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load {data_format} data from S3 path {path}. "
                f"Ensure your AWS credentials are configured. "
                f"Error: {str(e)}"
            )

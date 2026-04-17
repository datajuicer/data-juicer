import json
import os
from multiprocessing import Pool

from datasets import Dataset as HFDataset
from loguru import logger

from data_juicer.utils.constant import Fields, HashKeys
from data_juicer.utils.file_utils import Sizes, byte_size_to_size_str


class Exporter:
    """The Exporter class is used to export a dataset to files of specific
    format."""

    def __init__(
        self,
        export_path,
        export_type=None,
        export_shard_size=0,
        export_in_parallel=True,
        num_proc=1,
        export_ds=True,
        keep_stats_in_res_ds=False,
        keep_hashes_in_res_ds=False,
        export_stats=True,
        encrypt_before_export=False,
        encryption_key_path=None,
        **kwargs,
    ):
        """
        Initialization method.

        :param export_path: the path to export datasets.
        :param export_type: the format type of the exported datasets.
        :param export_shard_size: the approximate size of each shard of exported
            dataset. In default, it's 0, which means export the dataset
            to a single file.
        :param export_in_parallel: whether to export the datasets in parallel.
        :param num_proc: number of process to export the dataset.
        :param export_ds: whether to export the dataset contents.
        :param keep_stats_in_res_ds: whether to keep stats in the result
            dataset.
        :param keep_hashes_in_res_ds: whether to keep hashes in the result
            dataset.
        :param export_stats: whether to export the stats of dataset.
        :param encrypt_before_export: whether to encrypt each exported file
            in-place immediately after writing. Requires a valid Fernet key
            accessible via ``encryption_key_path`` or the environment
            variable ``DJ_ENCRYPTION_KEY``. S3 paths are skipped (use S3
            SSE instead). Default: False.
        :param encryption_key_path: path to a file containing the Fernet key.
            Falls back to the ``DJ_ENCRYPTION_KEY`` environment variable when
            ``None``. Only used when ``encrypt_before_export`` is True.
        """
        self.export_path = export_path
        self.export_shard_size = export_shard_size
        self.export_in_parallel = export_in_parallel
        self.export_ds = export_ds
        self.keep_stats_in_res_ds = keep_stats_in_res_ds
        self.keep_hashes_in_res_ds = keep_hashes_in_res_ds
        self.export_stats = export_stats
        self.suffix = self._get_suffix(export_path) if export_type is None else export_type
        support_dict = self._router()
        if self.suffix not in support_dict:
            raise NotImplementedError(
                f"Suffix of export path [{export_path}] or specified export_type [{export_type}] is not supported "
                f"for now. Only support {list(support_dict.keys())}."
            )
        self.num_proc = num_proc
        self.max_shard_size_str = ""

        # Set up encryption for local export
        self.encrypt_before_export = encrypt_before_export
        self._fernet = None
        if encrypt_before_export:
            if export_path.startswith("s3://"):
                logger.warning(
                    "encrypt_before_export is True but export_path is an S3 "
                    "path. Local-file encryption is skipped for S3 exports. "
                    "Use S3 server-side encryption (SSE) to protect data at rest."
                )
                self.encrypt_before_export = False
            else:
                from data_juicer.utils.encryption_utils import load_fernet_key

                self._fernet = load_fernet_key(encryption_key_path)

        # Check if export_path is S3 and create storage_options if needed
        self.storage_options = None
        if export_path.startswith("s3://"):
            # Extract AWS credentials from kwargs (if provided)
            s3_config = {}
            if "aws_access_key_id" in kwargs:
                s3_config["aws_access_key_id"] = kwargs.pop("aws_access_key_id")
            if "aws_secret_access_key" in kwargs:
                s3_config["aws_secret_access_key"] = kwargs.pop("aws_secret_access_key")
            if "aws_session_token" in kwargs:
                s3_config["aws_session_token"] = kwargs.pop("aws_session_token")
            if "aws_region" in kwargs:
                s3_config["aws_region"] = kwargs.pop("aws_region")
            if "endpoint_url" in kwargs:
                s3_config["endpoint_url"] = kwargs.pop("endpoint_url")

            from data_juicer.utils.s3_utils import get_aws_credentials

            # Get credentials with priority order: environment variables > explicit config
            # This matches the pattern used in load strategies
            aws_access_key_id, aws_secret_access_key, aws_session_token, _ = get_aws_credentials(s3_config)

            # Build storage_options for HuggingFace datasets
            # Note: region should NOT be in storage_options for HuggingFace datasets
            # as it causes issues with AioSession. Region is auto-detected from S3 path.
            storage_options = {}
            if aws_access_key_id:
                storage_options["key"] = aws_access_key_id
            if aws_secret_access_key:
                storage_options["secret"] = aws_secret_access_key
            if aws_session_token:
                storage_options["token"] = aws_session_token
            if "endpoint_url" in s3_config:
                storage_options["endpoint_url"] = s3_config["endpoint_url"]

            # If no credentials provided, try anonymous access for public buckets
            # If storage_options is empty, s3fs will use its default credential chain (e.g. IAM role).
            if storage_options.get("key") or storage_options.get("secret"):
                logger.info("Using explicit AWS credentials for S3 export")
            else:
                logger.info("Using default AWS credential chain for S3 export")

            # Allow explicit anonymous access via kwargs
            if kwargs.get("anon"):
                storage_options["anon"] = True
                logger.info("Anonymous access for public S3 bucket enabled via config.")

            self.storage_options = storage_options
            logger.info(f"Detected S3 export path: {export_path}. S3 storage_options configured.")

        # get the string format of shard size
        self.max_shard_size_str = byte_size_to_size_str(self.export_shard_size)

        # we recommend users to set a shard size between MiB and TiB.
        if 0 < self.export_shard_size < Sizes.MiB:
            logger.warning(
                f"The export_shard_size [{self.max_shard_size_str}]"
                f" is less than 1MiB. If the result dataset is too "
                f"large, there might be too many shard files to "
                f"generate."
            )
        if self.export_shard_size >= Sizes.TiB:
            logger.warning(
                f"The export_shard_size [{self.max_shard_size_str}]"
                f" is larger than 1TiB. It might generate large "
                f"single shard file and make loading and exporting "
                f"slower."
            )

    def _get_suffix(self, export_path):
        """
        Get the suffix of export path and check if it's supported.

        We only support ["jsonl", "json", "parquet"] for now.

        :param export_path: the path to export datasets.
        :return: the suffix of export_path.
        """
        suffix = export_path.split(".")[-1].lower()
        return suffix

    @staticmethod
    def _ensure_meta_stats_dicts_for_export(dataset):
        """
        If __dj__meta__ or __dj__stats__ are stored as JSON strings (e.g. after
        Arrow/Ray serialization with ensure_ascii=True), parse them back to dict
        so that to_json(force_ascii=False) writes proper UTF-8 instead of \\uXXXX.
        """
        meta_key = Fields.meta
        stats_key = Fields.stats
        columns_to_fix = [c for c in [meta_key, stats_key] if c in dataset.column_names]

        if not columns_to_fix:
            return dataset

        def _parse_if_string(row):
            out = dict(row)
            for col in columns_to_fix:
                val = out.get(col)
                if isinstance(val, str):
                    try:
                        out[col] = json.loads(val)
                    except (json.JSONDecodeError, TypeError):
                        pass
            return out

        return dataset.map(_parse_if_string, desc="Preparing meta/stats for UTF-8 export")

    def _encrypt_local_file(self, path):
        """Encrypt a local file in-place if encrypt_before_export is enabled."""
        if self.encrypt_before_export and self._fernet is not None:
            from data_juicer.utils.encryption_utils import encrypt_file

            encrypt_file(path, path, self._fernet)

    def _export_impl(self, dataset, export_path, suffix, export_stats=True):
        """
        Export a dataset to specific path.

        :param dataset: the dataset to export.
        :param export_path: the path to export the dataset.
        :param suffix: suffix of export path.
        :param export_stats: whether to export stats of dataset.
        :return:
        """
        if export_stats:
            # export stats of datasets into a single file.
            logger.info("Exporting computed stats into a single file...")
            export_columns = []
            if Fields.stats in dataset.features:
                export_columns.append(Fields.stats)
            if Fields.meta in dataset.features:
                export_columns.append(Fields.meta)
            if len(export_columns):
                ds_stats = dataset.select_columns(export_columns)
                # If meta/stats were serialized as JSON strings (e.g. \\uXXXX),
                # parse back to dict so to_json(force_ascii=False) writes UTF-8.
                ds_stats = Exporter._ensure_meta_stats_dicts_for_export(ds_stats)
                stats_file = export_path.replace("." + suffix, "_stats.jsonl")
                export_kwargs = {"num_proc": self.num_proc if self.export_in_parallel else 1}
                # Add storage_options if available (for S3 export)
                if self.storage_options is not None:
                    export_kwargs["storage_options"] = self.storage_options
                Exporter.to_jsonl(ds_stats, stats_file, **export_kwargs)
                self._encrypt_local_file(stats_file)

        if self.export_ds:
            # fetch the corresponding export method according to the suffix
            if not self.keep_stats_in_res_ds:
                extra_fields = {Fields.stats, Fields.meta}
                feature_fields = set(dataset.features.keys())
                removed_fields = extra_fields.intersection(feature_fields)
                dataset = dataset.remove_columns(removed_fields)
            if not self.keep_hashes_in_res_ds:
                extra_fields = {
                    HashKeys.hash,
                    HashKeys.minhash,
                    HashKeys.simhash,
                    HashKeys.imagehash,
                    HashKeys.videohash,
                }
                feature_fields = set(dataset.features.keys())
                removed_fields = extra_fields.intersection(feature_fields)
                dataset = dataset.remove_columns(removed_fields)
            export_method = Exporter._router()[suffix]
            if self.export_shard_size <= 0:
                # export the whole dataset into one single file.
                logger.info("Export dataset into a single file...")
                export_kwargs = {"num_proc": self.num_proc if self.export_in_parallel else 1}
                # Add storage_options if available (for S3 export)
                if self.storage_options is not None:
                    export_kwargs["storage_options"] = self.storage_options
                export_method(dataset, export_path, **export_kwargs)
                self._encrypt_local_file(export_path)
            else:
                # compute the dataset size and number of shards to split
                if dataset._indices is not None:
                    dataset_nbytes = dataset.data.nbytes * len(dataset._indices) / len(dataset.data)
                else:
                    dataset_nbytes = dataset.data.nbytes
                num_shards = int(dataset_nbytes / self.export_shard_size) + 1
                num_shards = min(num_shards, len(dataset))

                # split the dataset into multiple shards
                logger.info(
                    f"Split the dataset to export into {num_shards} "
                    f"shards. Size of each shard <= "
                    f"{self.max_shard_size_str}"
                )
                shards = [dataset.shard(num_shards=num_shards, index=i, contiguous=True) for i in range(num_shards)]
                len_num = len(str(num_shards)) + 1
                num_fmt = f"%0{len_num}d"

                # regard the export path as a directory and set file names for
                # each shard
                if self.export_path.startswith("s3://"):
                    # For S3 paths, construct S3 paths for each shard
                    # Extract bucket and prefix from S3 path
                    s3_path_parts = self.export_path.replace("s3://", "").split("/", 1)
                    bucket = s3_path_parts[0]
                    prefix = s3_path_parts[1] if len(s3_path_parts) > 1 else ""
                    # Remove extension from prefix
                    if "." in prefix:
                        prefix_base = ".".join(prefix.split(".")[:-1])
                    else:
                        prefix_base = prefix
                    # Construct shard filenames
                    filenames = [
                        f"s3://{bucket}/{prefix_base}-{num_fmt % index}-of-{num_fmt % num_shards}.{self.suffix}"
                        for index in range(num_shards)
                    ]
                else:
                    # For local paths, use standard directory structure
                    dirname = os.path.dirname(os.path.abspath(self.export_path))
                    basename = os.path.basename(self.export_path).split(".")[0]
                    os.makedirs(dirname, exist_ok=True)
                    filenames = [
                        os.path.join(
                            dirname, f"{basename}-{num_fmt % index}-of-" f"{num_fmt % num_shards}" f".{self.suffix}"
                        )
                        for index in range(num_shards)
                    ]

                # export dataset into multiple shards using multiprocessing
                logger.info(f"Start to exporting to {num_shards} shards.")
                pool = Pool(self.num_proc)
                for i in range(num_shards):
                    export_kwargs = {"num_proc": 1}  # Each shard export uses single process
                    # Add storage_options if available (for S3 export)
                    if self.storage_options is not None:
                        export_kwargs["storage_options"] = self.storage_options
                    pool.apply_async(
                        export_method,
                        args=(
                            shards[i],
                            filenames[i],
                        ),
                        kwds=export_kwargs,
                    )
                pool.close()
                pool.join()
                # Encrypt each local shard after all shards are written.
                for fname in filenames:
                    self._encrypt_local_file(fname)

    def export(self, dataset):
        """
        Export method for a dataset.

        :param dataset: the dataset to export.
        :return:
        """
        self._export_impl(dataset, self.export_path, self.suffix, self.export_stats)

    def export_compute_stats(self, dataset, export_path):
        """
        Export method for saving compute status in filters
        """
        keep_stats_in_res_ds = self.keep_stats_in_res_ds
        self.keep_stats_in_res_ds = True
        self._export_impl(dataset, export_path, self.suffix, export_stats=False)
        self.keep_stats_in_res_ds = keep_stats_in_res_ds

    @staticmethod
    def _row_to_json_serializable(obj):
        """Convert a row or value to JSON-serializable form; keep Unicode as-is."""
        if isinstance(obj, dict):
            return {k: Exporter._row_to_json_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [Exporter._row_to_json_serializable(v) for v in obj]
        if hasattr(obj, "item"):  # numpy scalar
            return obj.item()
        if hasattr(obj, "tolist"):
            return obj.tolist()
        if hasattr(obj, "as_py"):  # pyarrow scalar
            return Exporter._row_to_json_serializable(obj.as_py())
        return obj

    @staticmethod
    def _write_jsonl_utf8(dataset, export_path, storage_options=None):
        """
        Write dataset to JSONL with UTF-8 text (no \\uXXXX escape).
        HuggingFace's to_json(force_ascii=False) can still escape in some paths;
        we iterate and use json.dumps(ensure_ascii=False) per row.
        Use HFDataset.__getitem__ to get raw batch (avoid NestedQueryDict wrapping
        which fails on None in list columns e.g. response_usage).
        """
        batch_size = 1000
        total = len(dataset)
        os.makedirs(os.path.dirname(os.path.abspath(export_path)), exist_ok=True)
        with open(export_path, "w", encoding="utf-8") as f:
            for start in range(0, total, batch_size):
                end = min(start + batch_size, total)
                batch = HFDataset.__getitem__(dataset, slice(start, end))
                if isinstance(batch, dict):
                    keys = list(batch.keys())
                    for i in range(len(batch[keys[0]])):
                        row = {k: batch[k][i] for k in keys}
                        row = Exporter._row_to_json_serializable(row)
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                else:
                    for row in batch:
                        row = Exporter._row_to_json_serializable(row)
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")

    @staticmethod
    def to_jsonl(dataset, export_path, num_proc=1, **kwargs):
        """
        Export method for jsonl target files.

        :param dataset: the dataset to export.
        :param export_path: the path to store the exported dataset.
        :param num_proc: the number of processes used to export the dataset.
        :param kwargs: extra arguments.
        :return:
        """
        # Use custom UTF-8 writer so all text (e.g. text, meta, dialog_history)
        # is written as UTF-8, not \\uXXXX. HF to_json(force_ascii=False) can still
        # escape in batch/multiproc paths.
        storage_options = kwargs.get("storage_options")
        if storage_options is not None:
            # S3 or custom storage: fall back to HF to_json
            dataset.to_json(export_path, force_ascii=False, num_proc=num_proc, storage_options=storage_options)
        else:
            Exporter._write_jsonl_utf8(dataset, export_path)

    @staticmethod
    def to_json(dataset, export_path, num_proc=1, **kwargs):
        """
        Export method for json target files.

        :param dataset: the dataset to export.
        :param export_path: the path to store the exported dataset.
        :param num_proc: the number of processes used to export the dataset.
        :param kwargs: extra arguments.
        :return:
        """
        # Add storage_options if provided (for S3 export)
        storage_options = kwargs.get("storage_options")
        if storage_options is not None:
            dataset.to_json(
                export_path, force_ascii=False, num_proc=num_proc, lines=False, storage_options=storage_options
            )
        else:
            dataset.to_json(export_path, force_ascii=False, num_proc=num_proc, lines=False)

    @staticmethod
    def to_parquet(dataset, export_path, **kwargs):
        """
        Export method for parquet target files.

        :param dataset: the dataset to export.
        :param export_path: the path to store the exported dataset.
        :param kwargs: extra arguments.
        :return:
        """
        # Add storage_options if provided (for S3 export)
        storage_options = kwargs.get("storage_options")
        if storage_options is not None:
            dataset.to_parquet(export_path, storage_options=storage_options)
        else:
            dataset.to_parquet(export_path)

    # suffix to export method
    @staticmethod
    def _router():
        """
        A router from different suffixes to corresponding export methods.

        :return: A dict router.
        """
        return {
            "jsonl": Exporter.to_jsonl,
            "json": Exporter.to_json,
            "parquet": Exporter.to_parquet,
        }

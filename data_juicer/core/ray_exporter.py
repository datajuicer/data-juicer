import os
from functools import partial
from pathlib import Path

from loguru import logger

from data_juicer.utils.constant import Fields, HashKeys
from data_juicer.utils.file_utils import Sizes, byte_size_to_size_str
from data_juicer.utils.model_utils import filter_arguments
from data_juicer.utils.webdataset_utils import reconstruct_custom_webdataset_format


class RayExporter:
    """The Exporter class is used to export a ray dataset to files of specific
    format."""

    # TODO: support config for export, some export methods require additional args
    _SUPPORTED_FORMATS = {
        "json",
        "jsonl",
        "parquet",
        "csv",
        "tfrecords",
        "webdataset",
        "lance",
        "iceberg",
        "paimon",
        # 'images',
        # 'numpy',
    }

    def __init__(
        self,
        export_path,
        export_type=None,
        export_shard_size=0,
        keep_stats_in_res_ds=True,
        keep_hashes_in_res_ds=False,
        encrypt_before_export=False,
        encryption_key_path=None,
        **kwargs,
    ):
        """
        Initialization method.

        :param export_path: the path to export datasets.
        :param export_type: the format type of the exported datasets.
        :param export_shard_size: the approximate size of each shard of exported
            dataset. In default, it's 0, which means export the dataset in the default setting of ray.
        :param keep_stats_in_res_ds: whether to keep stats in the result
            dataset.
        :param keep_hashes_in_res_ds: whether to keep hashes in the result
            dataset.
        :param encrypt_before_export: whether to encrypt each exported file
            in-place after Ray has finished writing. All files inside the
            export directory will be encrypted. S3 paths are skipped.
            Default: False.
        :param encryption_key_path: path to a file containing the Fernet key.
            Falls back to the ``DJ_ENCRYPTION_KEY`` environment variable when
            ``None``. Only used when ``encrypt_before_export`` is True.
        """
        self.export_path = export_path
        self.export_shard_size = export_shard_size
        self.keep_stats_in_res_ds = keep_stats_in_res_ds
        self.keep_hashes_in_res_ds = keep_hashes_in_res_ds

        if export_type:
            self.export_format = export_type
        elif export_path:
            self.export_format = self._get_export_format(export_path)
        else:
            raise ValueError("Either export_path or export_type should be provided.")
        if self.export_format not in self._SUPPORTED_FORMATS:
            raise NotImplementedError(
                f'export data format "{self.export_format}" is not supported '
                f"for now. Only support {self._SUPPORTED_FORMATS}. Please check export_type or export_path."
            )
        self.export_extra_args = kwargs if kwargs is not None else {}

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

        # Check if export_path is S3 and create filesystem if needed
        self.fs = None
        if export_path.startswith("s3://"):
            # Extract AWS credentials from export_extra_args (if provided)
            s3_config = {}
            if "aws_access_key_id" in self.export_extra_args:
                s3_config["aws_access_key_id"] = self.export_extra_args.pop("aws_access_key_id")
            if "aws_secret_access_key" in self.export_extra_args:
                s3_config["aws_secret_access_key"] = self.export_extra_args.pop("aws_secret_access_key")
            if "aws_session_token" in self.export_extra_args:
                s3_config["aws_session_token"] = self.export_extra_args.pop("aws_session_token")
            if "aws_region" in self.export_extra_args:
                s3_config["aws_region"] = self.export_extra_args.pop("aws_region")
            if "endpoint_url" in self.export_extra_args:
                s3_config["endpoint_url"] = self.export_extra_args.pop("endpoint_url")

            # Create PyArrow S3FileSystem with credentials
            # This matches the pattern used in RayS3DataLoadStrategy
            from data_juicer.utils.s3_utils import create_pyarrow_s3_filesystem

            self.fs = create_pyarrow_s3_filesystem(s3_config)
            logger.info(f"Detected S3 export path: {export_path}. S3 filesystem configured.")

        self.max_shard_size_str = ""

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

    def _get_export_format(self, export_path):
        """
        Get the suffix of export path and check if it's supported.
        We only support ["jsonl", "json", "parquet"] for now.

        :param export_path: the path to export datasets.
        :return: the export data format.
        """
        suffix = os.path.splitext(export_path)[-1].strip(".")
        if not suffix:
            logger.warning(
                f'export_path "{export_path}" does not have a suffix. '
                f'We will use "jsonl" as the default export type.'
            )
            suffix = "jsonl"

        export_format = suffix
        return export_format

    def _export_impl(self, dataset, export_path, columns=None):
        """
        Export a dataset to specific path.

        :param dataset: the dataset to export.
        :param export_path: the path to export the dataset.
        :param columns: the columns to export.
        :return:
        """
        feature_fields = dataset.columns() if not columns else columns
        removed_fields = []
        if not self.keep_stats_in_res_ds:
            extra_fields = {Fields.stats, Fields.meta}
            removed_fields.extend(list(extra_fields.intersection(feature_fields)))
        if not self.keep_hashes_in_res_ds:
            extra_fields = {
                HashKeys.hash,
                HashKeys.minhash,
                HashKeys.simhash,
                HashKeys.imagehash,
                HashKeys.videohash,
            }
            removed_fields.extend(list(extra_fields.intersection(feature_fields)))

        if len(removed_fields):
            dataset = dataset.drop_columns(removed_fields)

        router = self._router()
        if self.export_format in router:
            export_method = router[self.export_format]
        else:
            export_method = RayExporter.write_others

        export_kwargs = {
            "export_extra_args": self.export_extra_args,
            "export_format": self.export_format,
        }
        # Add filesystem if available
        if self.fs is not None:
            export_kwargs["export_extra_args"]["filesystem"] = self.fs

        if self.export_shard_size > 0:
            dataset_nbytes = dataset.size_bytes()
            dataset_num_rows = dataset.count()

            if dataset_num_rows > 0:
                num_shards = int(dataset_nbytes / self.export_shard_size) + 1
                num_shards = min(num_shards, dataset_num_rows)
                rows_per_file = max(1, int(dataset_num_rows / num_shards))
                export_kwargs["export_extra_args"]["min_rows_per_file"] = rows_per_file

        # Ensure export directory exists (Ray's write_json treats export_path as a directory)
        if not export_path.startswith("s3://"):
            os.makedirs(export_path, exist_ok=True)

        result = export_method(dataset, export_path, **export_kwargs)

        # Encrypt all exported files in-place after Ray has finished writing
        if self.encrypt_before_export and self._fernet is not None and not export_path.startswith("s3://"):
            from data_juicer.utils.encryption_utils import encrypt_file

            export_dir = Path(export_path)
            if export_dir.is_dir():
                for fpath in export_dir.iterdir():
                    if fpath.is_file():
                        encrypt_file(str(fpath), str(fpath), self._fernet)
                        logger.debug(f"Encrypted exported file: {fpath}")

        return result

    def export(self, dataset, columns=None):
        """
        Export method for a dataset.

        :param dataset: the dataset to export.
        :param columns: the columns to export.
        :return:
        """
        self._export_impl(dataset, self.export_path, columns)

    @staticmethod
    def write_json(dataset, export_path, **kwargs):
        """
        Export method for json/jsonl target files.

        :param dataset: the dataset to export.
        :param export_path: the path to store the exported dataset.
        :param kwargs: extra arguments.
        :return:
        """
        export_extra_args = kwargs.get("export_extra_args", {})
        filtered_kwargs = filter_arguments(dataset.write_json, export_extra_args)
        # Add S3 filesystem if available
        if "filesystem" in export_extra_args:
            filtered_kwargs["filesystem"] = export_extra_args["filesystem"]
        return dataset.write_json(export_path, force_ascii=False, **filtered_kwargs)

    @staticmethod
    def write_webdataset(dataset, export_path, **kwargs):
        """
        Export method for webdataset target files.

        :param dataset: the dataset to export.
        :param export_path: the path to store the exported dataset.
        :param kwargs: extra arguments.
        :return:
        """
        from data_juicer.utils.webdataset_utils import _custom_default_encoder

        # check if we need to reconstruct the customized WebDataset format
        export_extra_args = kwargs.get("export_extra_args", {})
        field_mapping = export_extra_args.get("field_mapping", {})
        if len(field_mapping) > 0:
            reconstruct_func = partial(reconstruct_custom_webdataset_format, field_mapping=field_mapping)
            dataset = dataset.map(reconstruct_func)
        filtered_kwargs = filter_arguments(dataset.write_webdataset, export_extra_args)
        # Add S3 filesystem if available
        if "filesystem" in export_extra_args:
            filtered_kwargs["filesystem"] = export_extra_args["filesystem"]

        return dataset.write_webdataset(export_path, encoder=_custom_default_encoder, **filtered_kwargs)

    @staticmethod
    def write_others(dataset, export_path, **kwargs):
        """
        Export method for other target files.

        :param dataset: the dataset to export.
        :param export_path: the path to store the exported dataset.
        :param kwargs: extra arguments.
        :return:
        """
        export_format = kwargs.get("export_format", "parquet")
        if export_format == "lance":
            # use lazy loader to check pylance installation
            from data_juicer.utils.lazy_loader import LazyLoader

            LazyLoader.check_packages(["pylance"])
        write_method = getattr(dataset, f"write_{export_format}")
        export_extra_args = kwargs.get("export_extra_args", {})
        filtered_kwargs = filter_arguments(write_method, export_extra_args)
        # Add S3 filesystem if available
        if "filesystem" in export_extra_args:
            filtered_kwargs["filesystem"] = export_extra_args["filesystem"]
        if export_path:
            return write_method(export_path, **filtered_kwargs)
        else:
            return write_method(**filtered_kwargs)

    @staticmethod
    def write_iceberg(dataset, export_path, **kwargs):
        """
        Export method for iceberg target tables.
        Checks for table existence/connectivity. If check fails, safe fall-back to JSON.
        """
        export_extra_args = kwargs.get("export_extra_args", {})
        catalog_kwargs = export_extra_args.get("catalog_kwargs", {})
        table_identifier = export_extra_args.get("table_identifier", export_path)

        use_iceberg = False

        try:
            from pyiceberg.catalog import load_catalog
            from pyiceberg.exceptions import NoSuchTableError

            try:
                catalog = load_catalog(**catalog_kwargs)
                catalog.load_table(table_identifier)
                logger.info(f"Iceberg table {table_identifier} exists. Writing to Iceberg.")
                use_iceberg = True

            except NoSuchTableError as e:
                logger.warning(
                    f"Iceberg target unavailable ({e.__class__.__name__}). Fallback to exporting to {export_path}..."
                )
                import pyarrow as pa

                schema = pa.Schema.from_pandas(dataset.limit(1).to_pandas())
                logger.info(f"Creating new Iceberg table {table_identifier} with schema: {schema}")
                try:
                    catalog.create_table(table_identifier, schema)
                    use_iceberg = True
                except Exception as e:
                    logger.error(f"Failed to create Iceberg table: {e}. Fallback to exporting to {export_path}...")
            except Exception as e:
                logger.error(f"Unexpected error checking Iceberg: {e}. Fallback to exporting to {export_path}...")
        except Exception as e:
            logger.error(f"Iceberg export is unavailable ({e.__class__.__name__}: {e}). Fallback to file export...")

        if use_iceberg:
            try:
                filtered_kwargs = filter_arguments(dataset.write_iceberg, export_extra_args)
                return dataset.write_iceberg(**filtered_kwargs)
            except Exception as e:
                logger.error(f"Write to Iceberg failed during execution: {e}. Fallback to json...")

        suffix = os.path.splitext(export_path)[-1].strip(".").lower()
        if not suffix:
            suffix = "jsonl"
            logger.warning(f"No suffix found in {export_path}, using default fallback: {suffix}")

        logger.info(f"Falling back to file export. Format: [{suffix}], Path: [{export_path}]")

        fallback_kwargs = {}
        if "filesystem" in export_extra_args:
            fallback_kwargs["filesystem"] = export_extra_args["filesystem"]
        if suffix in ["json", "jsonl"]:
            return RayExporter.write_json(dataset, export_path, **fallback_kwargs)
        else:
            fallback_kwargs["export_format"] = suffix
            return RayExporter.write_others(dataset, export_path, **fallback_kwargs)

    @staticmethod
    def write_paimon(dataset, export_path, **kwargs):
        """
        Export method for paimon target tables.
        Prefers distributed Ray writes when supported by pypaimon and falls
        back to arrow-based commit for older versions.
        Only missing pypaimon support falls back to file export; catalog and
        table write errors are surfaced to callers.
        """
        export_extra_args = kwargs.get("export_extra_args", {})
        catalog_options = export_extra_args.get("catalog_options", {})
        table_identifier = export_extra_args.get("table_identifier", export_path)
        schema_kwargs = export_extra_args.get("schema_kwargs", {})
        write_options = {
            key: value
            for key, value in export_extra_args.items()
            if key
            not in {
                "catalog_options",
                "table_identifier",
                "schema_kwargs",
                "overwrite",
                "overwrite_partition",
                "filesystem",
            }
        }

        try:
            import pyarrow as pa
            from pypaimon import Schema
            from pypaimon.catalog.catalog_exception import TableNotExistException
            from pypaimon.catalog.catalog_factory import CatalogFactory
        except ImportError as e:
            logger.error(f"Paimon export is unavailable ({e.__class__.__name__}: {e}). Fallback to file export...")
        else:
            table_write = None
            table_commit = None

            try:
                catalog = CatalogFactory.create(catalog_options)
                try:
                    table = catalog.get_table(table_identifier)
                    logger.info(f"Paimon table {table_identifier} exists. Writing to Paimon.")
                except TableNotExistException:
                    logger.info(f"Paimon table {table_identifier} does not exist. Creating it before export.")
                    pa_schema = pa.Schema.from_pandas(dataset.limit(1).to_pandas())
                    if hasattr(Schema, "from_pyarrow_schema"):
                        schema = Schema.from_pyarrow_schema(pa_schema=pa_schema, **schema_kwargs)
                    else:
                        schema = Schema(pa_schema=pa_schema, **schema_kwargs)
                    catalog.create_table(table_identifier, schema=schema, ignore_if_exists=False)
                    table = catalog.get_table(table_identifier)

                write_builder = table.new_batch_write_builder()
                overwrite = export_extra_args.get("overwrite", False)
                overwrite_partition = export_extra_args.get("overwrite_partition")
                if overwrite:
                    if overwrite_partition is None:
                        write_builder = write_builder.overwrite()
                    else:
                        write_builder = write_builder.overwrite(overwrite_partition)

                table_write = write_builder.new_write()

                if hasattr(table_write, "write_ray"):
                    filtered_kwargs = filter_arguments(table_write.write_ray, write_options)
                    return table_write.write_ray(dataset, **filtered_kwargs)

                table_commit = write_builder.new_commit()
                if hasattr(dataset, "to_arrow"):
                    arrow_table = dataset.to_arrow()
                else:
                    import ray

                    arrow_tables = ray.get(dataset.to_arrow_refs())
                    if len(arrow_tables) == 0:
                        arrow_table = pa.table({})
                    elif len(arrow_tables) == 1:
                        arrow_table = arrow_tables[0]
                    else:
                        arrow_table = pa.concat_tables(arrow_tables)
                table_write.write_arrow(arrow_table)
                commit_messages = table_write.prepare_commit()
                table_commit.commit(commit_messages)
                return
            finally:
                if table_write is not None and hasattr(table_write, "close"):
                    table_write.close()
                if table_commit is not None and hasattr(table_commit, "close"):
                    table_commit.close()

        suffix = os.path.splitext(export_path)[-1].strip(".").lower()
        if not suffix:
            suffix = "jsonl"
            logger.warning(f"No suffix found in {export_path}, using default fallback: {suffix}")

        logger.info(f"Falling back to file export. Format: [{suffix}], Path: [{export_path}]")

        fallback_kwargs = {}
        if "filesystem" in export_extra_args:
            fallback_kwargs["filesystem"] = export_extra_args["filesystem"]
        if suffix in ["json", "jsonl"]:
            return RayExporter.write_json(dataset, export_path, **fallback_kwargs)
        else:
            fallback_kwargs["export_format"] = suffix
            return RayExporter.write_others(dataset, export_path, **fallback_kwargs)

    # suffix to export method
    @staticmethod
    def _router():
        """
        A router from different suffixes to corresponding export methods.

        :return: A dict router.
        """
        return {
            "jsonl": RayExporter.write_json,
            "json": RayExporter.write_json,
            "webdataset": RayExporter.write_webdataset,
            "iceberg": RayExporter.write_iceberg,
            "paimon": RayExporter.write_paimon,
        }

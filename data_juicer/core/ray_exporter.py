import copy
import os
from functools import partial

from loguru import logger

from data_juicer.utils.constant import Fields, HashKeys
from data_juicer.utils.file_utils import Sizes, byte_size_to_size_str
from data_juicer.utils.model_utils import filter_arguments
from data_juicer.utils.s3_utils import create_filesystem_from_args
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

        fs_args = copy.deepcopy(self.export_extra_args)
        self.fs = create_filesystem_from_args(export_path, fs_args)
        self._check_shard_size()

    def _check_shard_size(self):
        if self.export_shard_size == 0:
            return
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

        return export_method(dataset, export_path, **export_kwargs)

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
        from pyiceberg.catalog import load_catalog
        from pyiceberg.exceptions import NoSuchTableError

        export_extra_args = kwargs.get("export_extra_args", {})
        catalog_kwargs = export_extra_args.get("catalog_kwargs", {})
        table_identifier = export_extra_args.get("table_identifier", export_path)

        use_iceberg = False

        try:
            catalog = load_catalog(**catalog_kwargs)
            catalog.load_table(table_identifier)
            logger.info(f"Iceberg table {table_identifier} exists. Writing to Iceberg.")
            use_iceberg = True

        except NoSuchTableError as e:
            logger.warning(
                f"Iceberg target unavailable ({e.__class__.__name__}). Fallback to exporting to {export_path}..."
            )
        except Exception as e:
            logger.error(f"Unexpected error checking Iceberg: {e}. Fallback to exporting to {export_path}...")

        if use_iceberg:
            try:
                filtered_kwargs = filter_arguments(dataset.write_iceberg, export_extra_args)
                return dataset.write_iceberg(table_identifier, **filtered_kwargs)
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
        }

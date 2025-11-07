import os
from functools import partial

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
        self.export_format = self._get_export_format(export_path) if export_type is None else export_type
        if self.export_format not in self._SUPPORTED_FORMATS:
            raise NotImplementedError(
                f'export data format "{self.export_format}" is not supported '
                f"for now. Only support {self._SUPPORTED_FORMATS}. Please check export_type or export_path."
            )
        self.export_extra_args = kwargs if kwargs is not None else {}

        # Check if export_path is S3 and create filesystem if needed
        self.s3_filesystem = None
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

            self.s3_filesystem = create_pyarrow_s3_filesystem(s3_config)
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

        export_method = RayExporter._router()[self.export_format]
        export_kwargs = {
            "export_extra_args": self.export_extra_args,
            "export_format": self.export_format,
        }
        # Add S3 filesystem if available
        if self.s3_filesystem is not None:
            export_kwargs["export_extra_args"]["filesystem"] = self.s3_filesystem
        if self.export_shard_size > 0:
            # compute the min_rows_per_file for export methods
            dataset_nbytes = dataset.data.size_bytes()
            dataset_num_rows = dataset.data.count()
            num_shards = int(dataset_nbytes / self.export_shard_size) + 1
            num_shards = min(num_shards, dataset_num_rows)
            rows_per_file = int(dataset_num_rows / num_shards)
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
        return write_method(export_path, **filtered_kwargs)

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
            "parquet": RayExporter.write_others,
            "csv": RayExporter.write_others,
            "tfrecords": RayExporter.write_others,
            "lance": RayExporter.write_others,
        }

import os
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Tuple
from urllib.parse import urlparse

from loguru import logger

from data_juicer.utils.constant import Fields, HashKeys
from data_juicer.utils.file_utils import Sizes, byte_size_to_size_str, is_remote_path
from data_juicer.utils.fs_utils import create_filesystem_for_path
from data_juicer.utils.model_utils import filter_arguments
from data_juicer.utils.webdataset_utils import reconstruct_custom_webdataset_format


@dataclass(frozen=True)
class ExportAttempt:
    """One write of a job's export; ``earlier`` lists the prefixes of its earlier writes, oldest first."""

    prefix: str
    earlier: Tuple[str, ...] = ()


def _drop_export_columns(batch, columns):
    present = [column for column in columns if column in batch.column_names]
    return batch.drop_columns(present) if present else batch


def _prefixed_filenames(prefix, file_format):
    from ray.data.datasource import FilenameProvider

    class PrefixedFilenames(FilenameProvider):
        def __init__(self):
            super().__init__()
            self.prefix = prefix
            self.file_format = file_format

        def get_filename_for_task(self, write_uuid, task_index):
            return f"{self.prefix}_{write_uuid}_{task_index:06}.{self.file_format}"

        def get_filename_for_block(self, block, write_uuid, task_index, block_index):
            return f"{self.prefix}_{write_uuid}_{task_index:06}_{block_index:06}.{self.file_format}"

    return PrefixedFilenames()


class RayExporter:
    """The Exporter class is used to export a ray dataset to files of specific
    format."""

    # Formats Ray writes as files named by a FilenameProvider, with Ray's default suffix.
    _FILE_SUFFIXES = {
        "json": "json",
        "jsonl": "json",
        "parquet": "parquet",
        "csv": "csv",
        "tfrecords": "tfrecords",
        "webdataset": "tar",
    }

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
        self.export_format = self._get_export_format(export_path) if export_type is None else export_type
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
            if urlparse(export_path).scheme.lower() in ("s3", "hdfs"):
                logger.warning(
                    "encrypt_before_export is True but export_path is a remote "
                    f"path ({export_path}). Local-file encryption is skipped. "
                    "Use server-side encryption to protect data at rest."
                )
                self.encrypt_before_export = False
            else:
                from data_juicer.utils.encryption_utils import load_fernet_key

                self._fernet = load_fernet_key(encryption_key_path)

        # Create a PyArrow filesystem when export_path points to remote
        # storage (hdfs:// or s3://), consuming the backend-specific keys
        # from export_extra_args so they won't be forwarded to write methods.
        # PyArrow's HadoopFileSystem expects paths WITHOUT the hdfs:// scheme,
        # so we also strip the scheme and use the bare path for writing.
        self.hdfs_filesystem = None
        self.s3_filesystem = None
        fs, self.export_extra_args = create_filesystem_for_path(export_path, self.export_extra_args)
        export_scheme = urlparse(export_path).scheme.lower()
        if export_scheme == "hdfs":
            self.hdfs_filesystem = fs
            logger.info(f"Detected HDFS export path: {export_path}. HDFS filesystem configured.")
        elif export_scheme == "s3":
            self.s3_filesystem = fs
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

    def _export_impl(self, dataset, export_path, columns=None, attempt=None):
        """
        Export a dataset to specific path.

        :param dataset: the dataset to export.
        :param export_path: the path to export the dataset.
        :param columns: the columns to export.
        :return:
        """
        # Schema inference can execute a lazy plan. Resolve optional fields in
        # each output batch instead, and check unknown-schema emptiness after writing.
        cols = dataset.columns(fetch_if_missing=False)
        probe_after_write = cols is None
        removed_fields = []
        if not self.keep_stats_in_res_ds:
            removed_fields.extend([Fields.stats, Fields.meta])
        if not self.keep_hashes_in_res_ds:
            extra_fields = {
                HashKeys.hash,
                HashKeys.minhash,
                HashKeys.simhash,
                HashKeys.imagehash,
                HashKeys.videohash,
            }
            removed_fields.extend(sorted(extra_fields))

        if removed_fields:
            if cols is None:
                dataset = dataset.map_batches(
                    _drop_export_columns,
                    fn_kwargs={"columns": removed_fields},
                    batch_format="pyarrow",
                    batch_size=None,
                    zero_copy_batch=True,
                )
            else:
                present = [column for column in removed_fields if column in cols]
                if present:
                    dataset = dataset.drop_columns(present)

        export_method = RayExporter._router()[self.export_format]
        export_kwargs = {
            "export_extra_args": dict(self.export_extra_args),
            "export_format": self.export_format,
        }
        if attempt is not None:
            if self.export_format not in self._FILE_SUFFIXES:
                if attempt.earlier:
                    raise NotImplementedError(
                        f'Resuming a checkpointed job cannot replace a partial "{self.export_format}" export; '
                        f"remove {export_path} before resuming"
                    )
            else:
                self._prepare_attempt(attempt, export_path)
                export_kwargs["export_extra_args"]["filename_provider"] = _prefixed_filenames(
                    attempt.prefix, self._FILE_SUFFIXES[self.export_format]
                )
        # Add S3 filesystem if available
        if self.s3_filesystem is not None:
            export_kwargs["export_extra_args"]["filesystem"] = self.s3_filesystem
        # Add HDFS filesystem if available; PyArrow needs the bare path
        # (without the hdfs:// scheme) when a filesystem is provided.
        # Keep the original path for remote-path checks below;
        # only strip the scheme when passing it to export_method.
        export_path_for_checks = export_path
        if self.hdfs_filesystem is not None:
            from data_juicer.utils.hdfs_utils import strip_hdfs_scheme

            export_kwargs["export_extra_args"]["filesystem"] = self.hdfs_filesystem
            export_path = strip_hdfs_scheme(export_path)
        if self.export_shard_size > 0:
            # Exact sizing evaluates the plan. Retain those rows so the write
            # does not execute operators (or mutate their state) a second time.
            dataset = dataset.materialize()
            dataset_nbytes = dataset.size_bytes()
            dataset_num_rows = dataset.count()
            if dataset_num_rows:
                num_shards = int(dataset_nbytes / self.export_shard_size) + 1
                num_shards = min(num_shards, dataset_num_rows)
                rows_per_file = int(dataset_num_rows / num_shards)
                export_kwargs["export_extra_args"]["min_rows_per_file"] = rows_per_file

        # Ensure export directory exists (Ray's write_json treats export_path as a directory).
        # Skip for any remote path (s3://, hdfs://, ...); the remote filesystem
        # creates directories automatically during write.
        if not is_remote_path(export_path_for_checks):
            os.makedirs(export_path_for_checks, exist_ok=True)

        result = export_method(dataset, export_path, **export_kwargs)

        # Encrypt the exported files in-place after Ray has finished writing
        if self.encrypt_before_export and self._fernet is not None and not is_remote_path(export_path_for_checks):
            from data_juicer.utils.encryption_utils import encrypt_file

            export_dir = Path(export_path)
            if export_dir.is_dir():
                for fpath in export_dir.iterdir():
                    if fpath.is_file() and (attempt is None or fpath.name.startswith(f"{attempt.prefix}_")):
                        encrypt_file(str(fpath), str(fpath), self._fernet)
                        logger.debug(f"Encrypted exported file: {fpath}")

        # Checkpoint exports are directories, including empty ones: a later
        # resume must be able to enumerate and write a new export attempt.
        if attempt is None and probe_after_write and not is_remote_path(export_path_for_checks):
            if os.path.isdir(export_path_for_checks) and not os.listdir(export_path_for_checks):
                logger.warning(f"Dataset is empty, creating empty export file at {export_path_for_checks}")
                os.rmdir(export_path_for_checks)
                with open(export_path_for_checks, "w"):
                    pass

        return result

    def _export_files(self, export_path):
        """Return the filesystem and every file path below ``export_path``."""
        import pyarrow.fs as pafs

        scheme = urlparse(export_path).scheme.lower()
        if scheme == "s3":
            filesystem = self.s3_filesystem
            parsed = urlparse(export_path)
            base = f"{parsed.netloc}{parsed.path}".rstrip("/")
        elif scheme == "hdfs":
            from data_juicer.utils.hdfs_utils import strip_hdfs_scheme

            filesystem = self.hdfs_filesystem
            base = strip_hdfs_scheme(export_path).rstrip("/")
        else:
            filesystem = pafs.LocalFileSystem()
            base = os.path.abspath(export_path)
        infos = filesystem.get_file_info(pafs.FileSelector(base, allow_not_found=True, recursive=True))
        return filesystem, [info.path for info in infos if info.type == pafs.FileType.File]

    def _remove_attempts(self, export_path, prefixes):
        if not prefixes:
            return
        filesystem, paths = self._export_files(export_path)
        starts = tuple(f"{prefix}_" for prefix in prefixes)
        stale = [path for path in paths if os.path.basename(path).startswith(starts)]
        for path in stale:
            filesystem.delete_file(path)
        if stale:
            logger.info(f"Removed {len(stale)} file(s) of earlier export attempts from {export_path}")

    def _prepare_attempt(self, attempt, export_path):
        # The latest earlier attempt stays until this one is complete, so a failed resume never
        # leaves less exported output than it found; anything older is already superseded.
        self._remove_attempts(export_path, attempt.earlier[:-1])
        _, paths = self._export_files(export_path)
        known = tuple(f"{prefix}_" for prefix in attempt.earlier)
        foreign = [path for path in paths if not os.path.basename(path).startswith(known)]
        if foreign:
            logger.warning(
                f"{len(foreign)} file(s) in {export_path} were not written by this job and are kept; "
                f"readers of the directory will see them next to the export, e.g. {foreign[0]}"
            )

    def complete_attempt(self, attempt):
        """Remove older exports after the caller verifies this attempt's output."""
        if self.export_format in self._FILE_SUFFIXES:
            self._remove_attempts(self.export_path, attempt.earlier)

    def export(self, dataset, columns=None, attempt=None):
        """
        Export method for a dataset.

        :param dataset: the dataset to export.
        :param columns: the columns to export.
        :param attempt: an ``ExportAttempt`` naming this write's files. After a successful
            write and checkpoint verification, call ``complete_attempt`` to remove earlier exports.
        :return:
        """
        self._export_impl(dataset, self.export_path, columns, attempt)

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

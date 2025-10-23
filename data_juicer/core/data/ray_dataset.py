from __future__ import annotations

import os
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Union

import pyarrow
from jsonargparse import Namespace
from loguru import logger

from data_juicer.core.data import DJDataset
from data_juicer.core.data.schema import Schema
from data_juicer.ops import Deduplicator, Filter, Mapper
from data_juicer.ops.base_op import DEFAULT_BATCH_SIZE, TAGGING_OPS
from data_juicer.utils.constant import Fields
from data_juicer.utils.file_utils import is_remote_path
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.resource_utils import cuda_device_count
from data_juicer.utils.webdataset_utils import _custom_default_decoder

ray = LazyLoader("ray")


def get_abs_path(path, dataset_dir):
    if is_remote_path(path):
        return path
    full_path = os.path.abspath(os.path.join(dataset_dir, path))
    if os.path.exists(full_path):
        return full_path
    else:
        return path


def convert_to_absolute_paths(samples, dataset_dir, path_keys):
    samples = samples.to_pydict()
    for key in path_keys:
        for idx in range(len(samples[key])):
            paths = samples[key][idx]
            if isinstance(paths, str):
                samples[key][idx] = get_abs_path(paths, dataset_dir)
            elif isinstance(paths, list):
                samples[key][idx] = [get_abs_path(item, dataset_dir) for item in paths]
    return pyarrow.Table.from_pydict(samples)


# TODO: check path for nestdataset
def set_dataset_to_absolute_path(dataset, dataset_path, cfg):
    """
    Set all the path in input data to absolute path.
    Checks dataset_dir and project_dir for valid paths.
    """
    path_keys = []
    columns = dataset.columns()
    for key in [
        cfg.get("video_key", "videos"),
        cfg.get("image_key", "images"),
        cfg.get("audio_key", "audios"),
    ]:
        if key in columns:
            path_keys.append(key)
    if len(path_keys) > 0:
        dataset_dir = os.path.dirname(dataset_path)
        logger.info(f"dataset_dir: {dataset_dir}")
        dataset = dataset.map_batches(
            partial(convert_to_absolute_paths, dataset_dir=dataset_dir, path_keys=path_keys),
            batch_format="pyarrow",
            zero_copy_batch=True,
            batch_size=DEFAULT_BATCH_SIZE,
        )
    return dataset


def preprocess_dataset(dataset: ray.data.Dataset, dataset_path, cfg) -> ray.data.Dataset:
    if dataset_path:
        dataset = set_dataset_to_absolute_path(dataset, dataset_path, cfg)
    return dataset


def get_num_gpus(op, op_proc):
    if not op.use_cuda():
        return 0
    proc_per_gpu = op_proc / cuda_device_count()
    return 1.0 / proc_per_gpu


def filter_batch(batch, filter_func):
    mask = pyarrow.array(filter_func(batch.to_pydict()))
    return batch.filter(mask)


class RayDataset(DJDataset):
    def __init__(self, dataset: ray.data.Dataset, dataset_path: str = None, cfg: Optional[Namespace] = None) -> None:
        self.data = preprocess_dataset(dataset, dataset_path, cfg)

    def schema(self) -> Schema:
        """Get dataset schema.

        Returns:
            Schema: Dataset schema containing column names and types
        """
        if self.data is None or self.data.columns() is None:
            raise ValueError("Dataset is empty or not initialized")

        return Schema.from_ray_schema(self.data.schema())

    def get(self, k: int) -> List[Dict[str, Any]]:
        """Get k rows from the dataset."""
        if k < 0:
            raise ValueError(f"k must be non-negative, got {k}")

        if k == 0:
            return []

        k = min(k, self.data.count())
        return list(self.data.limit(k).take())

    def get_column(self, column: str, k: Optional[int] = None) -> List[Any]:
        """Get column values from Ray dataset.

        Args:
            column: Name of the column to retrieve
            k: Optional number of rows to return. If None, returns all rows

        Returns:
            List of values from the specified column

        Raises:
            KeyError: If column doesn't exist
            ValueError: If k is negative
        """
        if self.data is None or self.data.columns() is None or column not in self.data.columns():
            raise KeyError(f"Column '{column}' not found in dataset")

        if k is not None:
            if k < 0:
                raise ValueError(f"k must be non-negative, got {k}")
            if k == 0:
                return []
            k = min(k, self.data.count())
            return [row[column] for row in self.data.limit(k).take()]

        return [row[column] for row in self.data.take()]

    def process(self, operators, *, exporter=None, checkpointer=None, tracer=None) -> DJDataset:
        if operators is None:
            return self
        if not isinstance(operators, list):
            operators = [operators]

        from data_juicer.utils.process_utils import calculate_ray_np

        calculate_ray_np(operators)

        for op in operators:
            self._run_single_op(op)
        return self

    def _run_single_op(self, op):
        if op._name in TAGGING_OPS.modules and Fields.meta not in self.data.columns():

            def process_batch_arrow(table: pyarrow.Table):
                new_column_data = [{} for _ in range(len(table))]
                new_table = table.append_column(Fields.meta, [new_column_data])
                return new_table

            self.data = self.data.map_batches(
                process_batch_arrow, batch_format="pyarrow", batch_size=DEFAULT_BATCH_SIZE
            )

        try:
            batch_size = getattr(op, "batch_size", 1) if op.is_batched_op() else 1
            if isinstance(op, Mapper):
                if op.use_cuda():
                    op_kwargs = op._op_cfg[op._name]
                    self.data = self.data.map_batches(
                        op.__class__,
                        fn_args=None,
                        fn_kwargs=None,
                        fn_constructor_args=None,
                        fn_constructor_kwargs=op_kwargs,
                        batch_size=batch_size,
                        num_cpus=op.cpu_required,
                        num_gpus=op.gpu_required,
                        concurrency=op.num_proc,
                        batch_format="pyarrow",
                    )
                else:
                    self.data = self.data.map_batches(
                        op.process,
                        batch_size=batch_size,
                        batch_format="pyarrow",
                        num_cpus=op.cpu_required,
                        concurrency=op.num_proc,
                    )
            elif isinstance(op, Filter):
                columns = self.data.columns()
                if Fields.stats not in columns:

                    def process_batch_arrow(table: pyarrow.Table):
                        new_column_data = [{} for _ in range(len(table))]
                        new_talbe = table.append_column(Fields.stats, [new_column_data])
                        return new_talbe

                    self.data = self.data.map_batches(
                        process_batch_arrow, batch_format="pyarrow", batch_size=DEFAULT_BATCH_SIZE
                    )
                if op.use_cuda():
                    op_kwargs = op._op_cfg[op._name]
                    self.data = self.data.map_batches(
                        op.__class__,
                        fn_args=None,
                        fn_kwargs=None,
                        fn_constructor_args=None,
                        fn_constructor_kwargs=op_kwargs,
                        batch_size=batch_size,
                        num_cpus=op.cpu_required,
                        num_gpus=op.gpu_required,
                        concurrency=op.num_proc,
                        batch_format="pyarrow",
                    )
                else:
                    self.data = self.data.map_batches(
                        op.compute_stats,
                        batch_size=batch_size,
                        batch_format="pyarrow",
                        num_cpus=op.cpu_required,
                        concurrency=op.num_proc,
                    )
                if op.stats_export_path is not None:
                    self.data.write_json(op.stats_export_path, force_ascii=False)
                if op.is_batched_op():
                    # The core computation have been done in compute_stats,
                    # and the filter process only performs simple filtering.
                    # cpu and parallelism are not set here
                    self.data = self.data.map_batches(
                        partial(filter_batch, filter_func=op.process),
                        batch_format="pyarrow",
                        zero_copy_batch=True,
                        batch_size=DEFAULT_BATCH_SIZE,
                    )
                else:
                    self.data = self.data.filter(op.process)
            elif isinstance(op, Deduplicator):
                self.data = op.run(self.data)
            else:
                logger.error("Ray executor only support Filter, Mapper and Deduplicator OPs for now")
                raise NotImplementedError
        except:  # noqa: E722
            logger.error(f"An error occurred during Op [{op._name}].")
            import traceback

            traceback.print_exc()
            exit(1)

    @classmethod
    def read(cls, data_format: str, paths: Union[str, List[str]]) -> RayDataset:
        if data_format in {"json", "jsonl"}:
            return RayDataset.read_json(paths)
        elif data_format == "webdataset":
            return RayDataset.read_webdataset(paths)
        elif data_format in {
            "parquet",
            "images",
            "parquet_bulk",
            "csv",
            "text",
            "avro",
            "numpy",
            "tfrecords",
            "binary_files",
            "lance",
        }:
            return getattr(ray.data, f"read_{data_format}")(paths)

    @classmethod
    def read_json(cls, paths: Union[str, List[str]]) -> RayDataset:
        # Note: a temp solution for reading json stream
        # TODO: replace with ray.data.read_json_stream once it is available
        import pyarrow.json as js

        try:
            js.open_json
            return read_json_stream(paths)
        except AttributeError:
            return ray.data.read_json(paths)

    @classmethod
    def read_webdataset(cls, paths: Union[str, List[str]]) -> RayDataset:
        return ray.data.read_webdataset(paths, decoder=partial(_custom_default_decoder, format="PIL"))

    def to_list(self) -> list:
        return self.data.to_pandas().to_dict(orient="records")


class JSONStreamDatasource(ray.data.read_api.JSONDatasource):
    """
    A temp Datasource for reading json stream.

    Note:

        Depends on a customized `pyarrow` with `open_json` method.
    """

    def _read_stream(self, f: "pyarrow.NativeFile", path: str):
        from pyarrow.json import open_json

        try:
            reader = open_json(
                f,
                read_options=self.read_options,
                **self.arrow_json_args,
            )
            schema = None
            while True:
                try:
                    batch = reader.read_next_batch()
                    table = pyarrow.Table.from_batches([batch], schema=schema)
                    if schema is None:
                        schema = table.schema
                    yield table
                except StopIteration:
                    return
        except pyarrow.lib.ArrowInvalid as e:
            raise ValueError(f"Failed to read JSON file: {path}.") from e


def read_json_stream(
    paths: Union[str, List[str]],
    *,
    filesystem: Optional["pyarrow.fs.FileSystem"] = None,
    parallelism: int = -1,
    ray_remote_args: Dict[str, Any] = None,
    arrow_open_stream_args: Optional[Dict[str, Any]] = None,
    meta_provider=None,
    partition_filter=None,
    partitioning=ray.data.read_api.Partitioning("hive"),
    include_paths: bool = False,
    ignore_missing_paths: bool = False,
    shuffle: Union[Literal["files"], None] = None,
    file_extensions: Optional[List[str]] = ["json", "jsonl"],
    concurrency: Optional[int] = None,
    override_num_blocks: Optional[int] = None,
    **arrow_json_args,
) -> ray.data.Dataset:
    if meta_provider is None:
        meta_provider = ray.data.read_api.DefaultFileMetadataProvider()

    datasource = JSONStreamDatasource(
        paths,
        arrow_json_args=arrow_json_args,
        filesystem=filesystem,
        open_stream_args=arrow_open_stream_args,
        meta_provider=meta_provider,
        partition_filter=partition_filter,
        partitioning=partitioning,
        ignore_missing_paths=ignore_missing_paths,
        shuffle=shuffle,
        include_paths=include_paths,
        file_extensions=file_extensions,
    )
    return ray.data.read_datasource(
        datasource,
        parallelism=parallelism,
        ray_remote_args=ray_remote_args,
        concurrency=concurrency,
        override_num_blocks=override_num_blocks,
    )

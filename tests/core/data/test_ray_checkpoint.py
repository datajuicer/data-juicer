"""Streaming checkpoint contracts and recovery integration (Ray 2.51/2.52/2.56)."""

import base64
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pyarrow as pa
import pytest
import ray

from data_juicer.core.data.ray_checkpoint_adapter import (
    Checkpoint,
    StreamingCheckpointOperator,
    insert_checkpoint,
    validate_checkpoint_adapter,
)
from data_juicer.core.data.ray_dataset import RayDataset
from data_juicer.core.executor.ray_executor_partitioned import (
    _LOGICAL_PARTITION_COLUMN,
    PartitionedRayExecutor,
)
from data_juicer.ops import Deduplicator, Filter, Mapper, Pipeline
from data_juicer.utils.ckpt_utils import CheckpointStrategy, RayCheckpointManager, write_checkpoint_block
from data_juicer.utils.constant import Fields


@pytest.fixture(scope="module")
def ray_runtime():
    ray.init(num_cpus=4, num_gpus=2, include_dashboard=False, object_store_memory=256 * 1024 * 1024)
    context = ray.data.DataContext.get_current()
    context.enable_progress_bars = False
    # Ray 2.52 budgets a new task's outputs using this size before it has a
    # sample. Keep that estimate within the fixture's small object store.
    context.target_max_block_size = 8 * 1024 * 1024
    yield
    ray.shutdown()


class AddMapper(Mapper):
    _name = "checkpoint_add_mapper"
    _batched_op = True

    def __init__(self, counter=None, **kwargs):
        super().__init__(**kwargs)
        self.counter = counter

    def process_batched(self, samples):
        if self.counter is not None:
            ray.get(self.counter.add.remote(len(samples["id"])))
        samples["value"] = [item + 1 for item in samples.get("value", samples["id"])]
        return samples


class ExpandMapper(AddMapper):
    _name = "checkpoint_expand_mapper"

    def process_batched(self, samples):
        return {key: [value for value in values for _ in range(2)] for key, values in samples.items()}


class EvenFilter(Filter):
    _name = "checkpoint_even_filter"
    _batched_op = True

    def __init__(self, empty=False, counter=None, **kwargs):
        super().__init__(**kwargs)
        self.empty = empty
        self.counter = counter

    def compute_stats_batched(self, samples):
        if self.counter is not None:
            ray.get(self.counter.add.remote(len(samples["id"])))
        for index, item in enumerate(samples["id"]):
            samples[Fields.stats][index]["even"] = item % 2 == 0
        return samples

    def process_batched(self, samples):
        return [not self.empty and stats["even"] for stats in samples[Fields.stats]]


class UniqueDeduplicator(Deduplicator):
    _name = "checkpoint_unique_deduplicator"
    _supported_exec_modes = ("ray",)

    def run(self, dataset):
        return dataset.groupby("id").map_groups(lambda batch: batch.slice(0, 1), batch_format="pyarrow")


class IdentityPipeline(Pipeline):
    _name = "checkpoint_identity_pipeline"
    _supported_exec_modes = ("ray",)

    def run(self, dataset):
        return dataset.map_batches(lambda batch: batch, batch_format="pyarrow")


@ray.remote(num_cpus=0)
class Counter:
    def __init__(self):
        self.value = 0

    def add(self, value):
        self.value += value

    def get(self):
        return self.value


def _op(cls=AddMapper, **kwargs):
    return cls(num_proc=1, num_cpus=1, auto_op_parallelism=False, batch_size=2, **kwargs)


def _executor(manager):
    executor = PartitionedRayExecutor.__new__(PartitionedRayExecutor)
    executor.cfg = {"auto_op_parallelism": False}
    executor.ckpt_manager = manager
    executor.pipeline_dag = None
    executor._log_event = Mock()
    manager.event_logger = executor
    return executor


def _manifest(manager, index, pid=0):
    path = os.path.join(manager.ckpt_dir, manager.resolve_checkpoint_filename(index, pid))
    return manager._read_manifest(path)


def _manifest_schema(manifest):
    return pa.ipc.read_schema(pa.BufferReader(base64.b64decode(manifest["schema"])))


def test_sparse_checkpoint_writes_only_present_partitions(tmp_path, record_property):
    import pyarrow.parquet as pq

    manager = RayCheckpointManager(str(tmp_path))
    tx = manager.begin_checkpoint(0, "sparse", list(range(100)))
    started = time.perf_counter()
    for pid in range(100):
        table = pa.table({"id": [pid], _LOGICAL_PARTITION_COLUMN: [pid]})
        schema, entries = write_checkpoint_block(table, tx.directories, _LOGICAL_PARTITION_COLUMN)
        assert len(entries) == 1
        assert entries[0][0] == pid
        tx.add_result((schema, entries))
    assert len(tx.schemas) == 1
    tx.publish()
    record_property("checkpoint_seconds", time.perf_counter() - started)
    data_files = [path for path in tmp_path.rglob("*") if path.is_file() and path.suffix in (".parquet", ".arrow")]
    assert len(data_files) == 100
    record_property("data_files", len(data_files))
    record_property("data_bytes", sum(path.stat().st_size for path in data_files))
    for pid in range(100):
        manifest = _manifest(manager, 0, pid)
        assert manifest["rows"] == 1
        assert len(manifest["files"]) == 1
        assert manifest["files"][0]["rows"] == 1
        path = tmp_path / manager.resolve_checkpoint_filename(0, pid) / manifest["files"][0]["path"]
        assert pq.read_table(path).to_pylist() == [{"id": pid}]


def test_checkpoint_unifies_distinct_block_schemas_once(tmp_path, monkeypatch):
    manager = RayCheckpointManager(str(tmp_path))
    tx = manager.begin_checkpoint(0, "schema", [0, 1, 2])
    tables = [
        pa.table({"id": [0], "nullable": [None], _LOGICAL_PARTITION_COLUMN: [0]}),
        pa.table({"id": [1], "nullable": [3], "extra": ["value"], _LOGICAL_PARTITION_COLUMN: [1]}),
    ]
    for table in tables * 2:
        tx.add_result(write_checkpoint_block(table, tx.directories, _LOGICAL_PARTITION_COLUMN))
    assert len(tx.schemas) == 2
    assert list(tx.schemas) == [
        base64.b64encode(table.drop([_LOGICAL_PARTITION_COLUMN]).schema.serialize()).decode("ascii") for table in tables
    ]
    unify = Mock(wraps=pa.unify_schemas)
    monkeypatch.setattr(pa, "unify_schemas", unify)
    tx.publish()
    assert unify.call_count == 1
    expected = pa.schema([("id", pa.int64()), ("nullable", pa.int64()), ("extra", pa.string())])
    for pid in range(3):
        assert _manifest_schema(_manifest(manager, 0, pid)).equals(expected)
    assert _manifest(manager, 0, 2)["files"] == []


@pytest.mark.parametrize("tags", [[0, 99], [0, None]])
def test_checkpoint_rejects_invalid_partition_tags_before_writing(tmp_path, tags):
    tx = RayCheckpointManager(str(tmp_path)).begin_checkpoint(0, "invalid", [0, 1])
    with pytest.raises(ValueError, match="Unexpected logical partition tags"):
        write_checkpoint_block(
            pa.table({"id": [1, 2], _LOGICAL_PARTITION_COLUMN: tags}), tx.directories, _LOGICAL_PARTITION_COLUMN
        )
    assert not list(Path(tx.root).rglob("*.parquet"))
    tx.abort()


def test_checkpoint_rejects_removed_partition_tag(tmp_path):
    tx = RayCheckpointManager(str(tmp_path)).begin_checkpoint(0, "invalid", [0, 1])
    with pytest.raises(ValueError, match="removed the internal logical-partition tag"):
        write_checkpoint_block(pa.table({"id": [1]}), tx.directories, _LOGICAL_PARTITION_COLUMN)
    tx.abort()


@pytest.mark.parametrize("partitioned", [False, True])
@pytest.mark.parametrize("missing_tag", [False, True])
def test_zero_row_block_returns_schema_without_files(tmp_path, partitioned, missing_tag):
    tx = RayCheckpointManager(str(tmp_path)).begin_checkpoint(0, "empty", [0, 1] if partitioned else [None])
    schema = pa.schema([("id", pa.int64()), ("nullable", pa.string()), ("meta", pa.struct([]))])
    table = pa.Table.from_batches([], schema=schema)
    if not missing_tag:
        table = table.append_column(_LOGICAL_PARTITION_COLUMN, pa.array([], type=pa.int64()))
    column = _LOGICAL_PARTITION_COLUMN if partitioned else None
    result_schema, entries = write_checkpoint_block(table, tx.directories, column)
    assert entries == []
    expected = schema if partitioned else table.schema
    assert pa.ipc.read_schema(pa.BufferReader(base64.b64decode(result_schema))).equals(expected)
    tx.add_result((result_schema, entries))
    tx.publish()
    for pid in tx.partition_ids:
        directory = tmp_path / tx.manager.resolve_checkpoint_filename(0, pid)
        assert {path.name for path in directory.iterdir()} == {"_checkpoint_v2", "manifest.json"}
        manifest = _manifest(tx.manager, 0, pid)
        assert manifest["rows"] == 0
        assert manifest["files"] == []
        assert _manifest_schema(manifest).equals(expected)


@pytest.mark.parametrize("empty", [False, True])
def test_sparse_checkpoint_schema_order_and_continuation(ray_runtime, tmp_path, empty):
    manager = RayCheckpointManager(str(tmp_path))
    tx = manager.begin_checkpoint(0, "sparse", [0, 1, 2])
    table = pa.table(
        {
            "id": [4, 1, 3, 0, 2],
            "nullable": pa.array([None, 1, None, 3, 4], type=pa.int64()),
            "meta": [{}, {}, {}, {}, {}],
            _LOGICAL_PARTITION_COLUMN: [0, 2, 0, 2, 0],
        }
    )
    if empty:
        table = table.slice(0, 0)
    # Include an empty block as well as multiple interleaved partition slices.
    for block in [table.slice(0, 2), table.slice(0, 0), table.slice(2)]:
        tx.add_result(write_checkpoint_block(block, tx.directories, _LOGICAL_PARTITION_COLUMN))
    tx.publish()
    for pid in range(3):
        expected = table.filter(pa.compute.equal(table[_LOGICAL_PARTITION_COLUMN], pid)).drop(
            [_LOGICAL_PARTITION_COLUMN]
        )
        manifest = _manifest(manager, 0, pid)
        assert manager.find_latest_checkpoint(pid)[0] == 0
        if not expected.num_rows:
            assert manifest["files"] == []
            directory = tmp_path / manager.resolve_checkpoint_filename(0, pid)
            assert {path.name for path in directory.iterdir()} == {"_checkpoint_v2", "manifest.json"}
        loaded = manager.load_checkpoint(0, partition_id=pid)
        assert loaded.take_all() == expected.to_pylist()
        assert loaded.schema().base_schema.equals(expected.schema)
        direct = RayDataset(ray.data.from_arrow(expected), auto_op_parallelism=False).process([_op(IdentityPipeline)])
        resumed = RayDataset(loaded, auto_op_parallelism=False).process([_op(IdentityPipeline)])
        # Compare Arrow rows directly: pandas can turn nullable integers into NaN
        # differently depending on the physical block/batch boundaries.
        assert resumed.data.take_all() == direct.data.take_all()
        direct_schema, resumed_schema = direct.data.schema(), resumed.data.schema()
        if expected.num_rows:
            assert resumed_schema.base_schema.equals(direct_schema.base_schema)


@pytest.mark.parametrize("delivered_block", [False, True])
def test_zero_row_global_checkpoint_loads_and_continues(ray_runtime, tmp_path, delivered_block):
    manager = RayCheckpointManager(str(tmp_path))
    schema = pa.schema([("id", pa.int64()), ("nullable", pa.string()), ("meta", pa.struct([]))])
    table = pa.Table.from_batches([], schema=schema)
    if delivered_block:
        manager.save_checkpoint(ray.data.from_arrow(table), 0, partition_id=None)
    else:
        tx = manager.begin_checkpoint(0, "empty", [None])
        tx.publish(schema)
    manifest = _manifest(manager, 0, None)
    assert manifest["files"] == []
    assert manifest["rows"] == 0
    assert _manifest_schema(manifest).equals(schema)
    assert manager.find_latest_checkpoint(None)[0] == 0
    loaded = manager.load_checkpoint(0, partition_id=None)
    assert loaded.take_all() == []
    assert loaded.schema().base_schema.equals(schema)
    result = RayDataset(loaded, auto_op_parallelism=False).process([_op()], checkpointer=manager.for_partitions([None]))
    assert result.to_list() == []
    assert _manifest(manager, 0, None)["files"] == []


@pytest.mark.parametrize("file_format", ["parquet", "arrow"])
def test_existing_v2_empty_files_remain_readable(ray_runtime, tmp_path, file_format):
    import pyarrow.parquet as pq

    manager = RayCheckpointManager(str(tmp_path))
    tx = manager.begin_checkpoint(0, "old_empty", [0])
    table = pa.Table.from_batches([], schema=pa.schema([("id", pa.int64()), ("nullable", pa.string())]))
    filename = f"empty.{file_format}"
    path = os.path.join(tx.directories[0], filename)
    if file_format == "parquet":
        pq.write_table(table, path)
    else:
        with pa.OSFile(path, "wb") as sink, pa.ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)
    tx.add_result(
        (
            base64.b64encode(table.schema.serialize()).decode("ascii"),
            [(0, {"path": filename, "format": file_format, "rows": 0, "size_bytes": os.path.getsize(path)})],
        )
    )
    tx.publish()
    assert manager.find_latest_checkpoint()[0] == 0
    loaded = manager.load_checkpoint(0)
    assert loaded is not None
    assert loaded.take_all() == []
    assert _manifest_schema(_manifest(manager, 0)).equals(table.schema)


def test_empty_block_slice_does_not_write_data(tmp_path):
    manager = RayCheckpointManager(str(tmp_path))
    tx = manager.begin_checkpoint(0, "slice", [0, 1])
    table = pa.table({"id": [1, 2], _LOGICAL_PARTITION_COLUMN: [99, 99]})
    # Tags outside the admitted slice do not belong to this checkpoint block.
    schema, entries = write_checkpoint_block(table, tx.directories, _LOGICAL_PARTITION_COLUMN, (1, 1))
    assert entries == []
    tx.add_result((schema, entries))
    tx.publish()
    for pid in range(2):
        assert _manifest(manager, 0, pid)["files"] == []
        assert _manifest_schema(_manifest(manager, 0, pid)).equals(pa.schema([("id", pa.int64())]))


def test_adapter_contract(monkeypatch):
    validate_checkpoint_adapter()
    monkeypatch.setattr(ray, "__version__", "0.0.0")
    with pytest.raises(RuntimeError, match="adapter contract"):
        validate_checkpoint_adapter()


def test_transaction_ignores_unpublished_and_retry_files(tmp_path):
    manager = RayCheckpointManager(str(tmp_path))
    tx = manager.begin_checkpoint(0, "mapper", [0])
    table = pa.table({"id": [1, 2], "nullable": pa.array([None, 3], type=pa.int64()), "meta": [{}, {}]})
    write_checkpoint_block(table, tx.directories)  # abandoned attempt
    tx.add_result(write_checkpoint_block(table, tx.directories))
    assert manager.find_latest_checkpoint() is None
    tx.publish()
    manifest = _manifest(manager, 0)
    assert manifest["rows"] == 2
    assert len(manifest["files"]) == 1
    assert manifest["files"][0]["format"] == "arrow"
    assert manager.find_latest_checkpoint()[0] == 0


@pytest.mark.parametrize("damage", ["missing_file", "truncated_file", "manifest", "missing_manifest"])
@pytest.mark.parametrize("pid", [0, None])
def test_damaged_manifest_is_recomputed(tmp_path, damage, pid):
    manager = RayCheckpointManager(str(tmp_path))
    tx = manager.begin_checkpoint(2, "mapper", [pid])
    tx.add_result(write_checkpoint_block(pa.table({"id": [1]}), tx.directories))
    tx.publish()
    path = tmp_path / manager.resolve_checkpoint_filename(2, pid)
    manifest = _manifest(manager, 2, pid)
    if damage == "missing_file":
        (path / manifest["files"][0]["path"]).unlink()
    elif damage == "truncated_file":
        (path / manifest["files"][0]["path"]).write_bytes(b"bad")
    elif damage == "manifest":
        (path / "manifest.json").write_text("{broken")
    else:
        (path / "manifest.json").unlink()
    assert manager.find_latest_checkpoint(pid) is None
    assert manager.load_checkpoint(2, partition_id=pid) is None
    tx = manager.begin_checkpoint(2, "mapper", [pid])
    tx.add_result(write_checkpoint_block(pa.table({"id": [1]}), tx.directories))
    tx.publish()
    assert manager.find_latest_checkpoint(pid)[0] == 2
    assert _manifest(manager, 2, pid)["rows"] == 1
    assert not list(tmp_path.glob(".checkpoint-*"))


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("has_split", [False, True])
def test_checkpoint_path_does_not_add_hive_columns(ray_runtime, tmp_path, legacy, has_split):
    import pyarrow.parquet as pq

    manager = RayCheckpointManager(str(tmp_path / "split=train"))
    table = pa.table({"id": [7], **({"split": [42]} if has_split else {})})
    if legacy:
        path = Path(manager.ckpt_dir, manager.resolve_checkpoint_filename(0, 0))
        path.mkdir()
        pq.write_table(table, path / "part.parquet")
    else:
        manager.save_checkpoint(ray.data.from_arrow(table), 0)
    loaded = manager.load_checkpoint(0)
    assert loaded is not None
    assert loaded.take_all() == table.to_pylist()
    assert loaded.schema().base_schema.equals(table.schema)


def test_replacement_publication_failure_restores_old_checkpoint(tmp_path, monkeypatch):
    manager = RayCheckpointManager(str(tmp_path))
    original = manager.begin_checkpoint(0, "mapper", [0])
    original.add_result(write_checkpoint_block(pa.table({"id": [1]}), original.directories))
    original.publish()
    manifest = _manifest(manager, 0)
    replacement = manager.begin_checkpoint(0, "mapper", [0])
    replacement.add_result(write_checkpoint_block(pa.table({"id": [2]}), replacement.directories))
    rename = os.rename

    def fail_publish(source, destination):
        if source == replacement.directories[0]:
            raise OSError("injected publication failure")
        return rename(source, destination)

    monkeypatch.setattr(os, "rename", fail_publish)
    with pytest.raises(OSError, match="injected publication failure"):
        replacement.publish()
    replacement.abort()
    assert _manifest(manager, 0) == manifest
    assert not list(tmp_path.glob(".checkpoint-*"))


@pytest.mark.parametrize("pid", [0, None])
def test_resume_recomputes_damaged_boundary(ray_runtime, tmp_path, pid):
    manager = RayCheckpointManager(str(tmp_path))
    executor = _executor(manager)
    first, second = Counter.remote(), Counter.remote()
    ops = [_op(counter=first), _op(counter=second)]

    def source():
        return RayDataset(ray.data.range(4, override_num_blocks=2), auto_op_parallelism=False)

    expected = executor._process_with_checkpointing(source(), pid, ops).to_list()
    path = tmp_path / manager.resolve_checkpoint_filename(1, pid)
    (path / _manifest(manager, 1, pid)["files"][0]["path"]).write_bytes(b"bad")
    assert manager.find_latest_checkpoint(pid)[0] == 0
    assert executor._process_with_checkpointing(source(), pid, ops).to_list() == expected
    assert ray.get(first.get.remote()) == 4
    assert ray.get(second.get.remote()) == 8
    assert manager.find_latest_checkpoint(pid)[0] == 1
    assert manager.load_checkpoint(1, partition_id=pid).take_all() == expected


def test_exit_before_publication_and_empty_global_scope(tmp_path):
    manager = RayCheckpointManager(str(tmp_path))
    tx = manager.begin_checkpoint(4, "global", [None])
    tx.add_result(write_checkpoint_block(pa.table({"id": [1]}), tx.directories))
    tx.abort()
    assert manager.find_latest_checkpoint(None) is None
    tx = manager.begin_checkpoint(4, "global", [None])
    tx.publish(pa.schema([("id", pa.int64())]))
    assert manager.find_latest_checkpoint(None)[0] == 4
    assert manager.find_latest_checkpoint(0) is None
    assert _manifest(manager, 4, None)["rows"] == 0


@pytest.mark.parametrize("actor,gpus", [(False, 0), (True, 0), (True, 0.5)])
def test_mapper_filter_expansion_resume_and_repeat_reads(ray_runtime, tmp_path, actor, gpus):
    manager = RayCheckpointManager(str(tmp_path))
    executor = _executor(manager)
    counter = Counter.remote()
    mapper = _op(counter=counter, ray_execution_mode="actor" if actor else "task", num_gpus=gpus)
    ops = [mapper, _op(ExpandMapper), _op(EvenFilter, ray_execution_mode="actor" if actor else "task", num_gpus=gpus)]
    data = RayDataset(ray.data.range(8, override_num_blocks=4), auto_op_parallelism=False)
    result = executor._process_with_checkpointing(data, 0, ops)
    rows = result.to_list()
    assert [row["id"] for row in rows] == [0, 0, 2, 2, 4, 4, 6, 6]
    assert [row["value"] for row in rows] == [1, 1, 3, 3, 5, 5, 7, 7]
    assert all(row[Fields.stats]["even"] for row in rows)
    assert ray.get(counter.get.remote()) == 8
    assert result.count() == len(rows)
    assert result.to_list() == rows
    # Resume the complete final boundary, including a repeated operator name.
    resumed = executor._process_with_checkpointing(data, 0, ops)
    assert resumed.to_list() == rows
    assert ray.get(counter.get.remote()) == 8
    assert [_manifest(manager, i)["rows"] for i in range(3)] == [8, 16, 8]
    events = [call.kwargs for call in executor._log_event.call_args_list]
    for index in range(3):
        saved = next(
            i
            for i, event in enumerate(events)
            if event["event_type"].value == "checkpoint_save" and event["operation_idx"] == index
        )
        complete = next(
            i
            for i, event in enumerate(events)
            if event["event_type"].value == "op_complete" and event["operation_idx"] == index
        )
        assert saved < complete


@pytest.mark.parametrize("empty", [False, True])
def test_execution_group_and_zero_output(ray_runtime, tmp_path, empty):
    manager = RayCheckpointManager(str(tmp_path))
    executor = _executor(manager)
    data = ray.data.from_arrow(pa.table({"id": [0, 2, 1, 3], _LOGICAL_PARTITION_COLUMN: [0, 0, 1, 1]}))
    result = executor._process_execution_group_with_checkpointing(
        RayDataset(data, auto_op_parallelism=False), [0, 1], [_op(EvenFilter, empty=empty), _op()]
    )
    assert [row["id"] for row in result.to_list()] == ([] if empty else [0, 2])
    for index in range(2):
        for pid in range(2):
            manifest = _manifest(manager, index, pid)
            assert manifest["rows"] == (0 if empty or pid == 1 else 2)
            if not manifest["rows"]:
                assert manifest["files"] == []
            loaded = manager.load_checkpoint(index, partition_id=pid)
            assert _LOGICAL_PARTITION_COLUMN not in (loaded.columns() or [])
            assert loaded.count() == manifest["rows"]


def test_nullable_schema_and_legacy_recovery(ray_runtime, tmp_path):
    manager = RayCheckpointManager(str(tmp_path))
    table = pa.table({"id": [1, 2], "nullable": pa.array([None, 3], type=pa.int64()), "meta": [{}, {}]})
    data = RayDataset(ray.data.from_arrow(table), auto_op_parallelism=False)
    manager.save_checkpoint(data, 0)
    assert manager.load_checkpoint(0).take_all() == table.to_pylist()
    legacy = tmp_path / manager.resolve_checkpoint_filename(1, 1)
    legacy.mkdir()
    import pyarrow.parquet as pq

    pq.write_table(table.drop(["meta"]), legacy / "part.parquet")
    assert manager.load_checkpoint(1, partition_id=1).take_all() == table.drop(["meta"]).to_pylist()


def test_stats_export_does_not_execute_upstream_twice(ray_runtime, tmp_path):
    manager = RayCheckpointManager(str(tmp_path / "checkpoints"))
    counter = Counter.remote()
    stats_counter = Counter.remote()
    data = RayDataset(ray.data.range(8, override_num_blocks=4), auto_op_parallelism=False)
    ops = [_op(counter=counter), _op(EvenFilter, counter=stats_counter, stats_export_path=str(tmp_path / "stats"))]
    result = _executor(manager)._process_with_checkpointing(data, 0, ops)
    assert result.count() == 4
    assert ray.get(counter.get.remote()) == 8
    assert ray.get(stats_counter.get.remote()) == 8
    assert _manifest(manager, 0)["rows"] == 8
    assert _manifest(manager, 1)["rows"] == 4


def test_global_dedup_pipeline_checkpoint_and_resume(ray_runtime, tmp_path):
    manager = RayCheckpointManager(str(tmp_path))
    executor = _executor(manager)
    counter = Counter.remote()
    ops = [_op(counter=counter), _op(UniqueDeduplicator), _op(IdentityPipeline), _op(counter=counter)]
    source = RayDataset(ray.data.from_items([{"id": 1}, {"id": 1}, {"id": 2}]), auto_op_parallelism=False)
    executor._process_with_simple_partitioning = lambda data, operators: executor._process_with_checkpointing(
        data, 0, operators
    )
    result = executor._process_with_convergence(source, ops, [1, 2])
    assert sorted(row["id"] for row in result.to_list()) == [1, 2]
    assert manager.find_latest_checkpoint(None)[0] == 3
    assert manager.find_latest_checkpoint(0)[0] == 0
    assert ray.get(counter.get.remote()) == 5
    executor._process_with_simple_partitioning = Mock(side_effect=AssertionError("must skip partitions"))
    resumed = executor._process_with_convergence(source, ops, [1, 2])
    assert resumed.to_list() == result.to_list()
    assert ray.get(counter.get.remote()) == 5


@ray.remote(num_cpus=0)
class Gate:
    def __init__(self):
        import asyncio

        self.event = asyncio.Event()

    async def wait(self):
        import asyncio

        await asyncio.wait_for(self.event.wait(), timeout=30)

    async def release(self):
        self.event.set()


def test_two_operators_overlap_before_first_finishes(ray_runtime, tmp_path):
    manager = RayCheckpointManager(str(tmp_path))
    gate = Gate.remote()

    def first(batch):
        if batch["id"][0].as_py() > 0:
            ray.get(gate.wait.remote())
        return batch

    def second(batch):
        ray.get(gate.release.remote())
        return batch

    data = ray.data.range(8, override_num_blocks=4).map_batches(first, batch_format="pyarrow", concurrency=1)
    data = insert_checkpoint(data, manager, 0, "first", [0])
    data = data.map_batches(second, batch_format="pyarrow", concurrency=1)
    result = insert_checkpoint(data, manager, 1, "second", [0]).materialize()
    assert [row["id"] for row in result.take_all()] == list(range(8))
    assert manager.find_latest_checkpoint()[0] == 1


def test_writer_failure_aborts_and_propagates(ray_runtime, tmp_path, monkeypatch):
    import data_juicer.core.data.ray_checkpoint_adapter as adapter

    def fail_write(*args):
        raise OSError("injected checkpoint write failure")

    monkeypatch.setattr(adapter, "write_checkpoint_block", fail_write)
    manager = RayCheckpointManager(str(tmp_path))
    with pytest.raises(Exception, match="injected checkpoint write failure"):
        insert_checkpoint(ray.data.range(4), manager, 0, "first", [0]).materialize()
    assert manager.find_latest_checkpoint() is None
    assert not list(tmp_path.glob(".checkpoint-*"))


def test_published_checkpoint_survives_downstream_failure(ray_runtime, tmp_path):
    manager = RayCheckpointManager(str(tmp_path))
    gate = Gate.remote()
    data = insert_checkpoint(
        ray.data.range(8, override_num_blocks=4),
        manager,
        0,
        "first",
        [0],
        on_complete=lambda: ray.get(gate.release.remote()),
    )

    def fail(batch):
        if batch["id"][0].as_py() >= 6:
            ray.get(gate.wait.remote())
            raise ValueError("injected downstream failure")
        return batch

    data = data.map_batches(fail, batch_format="pyarrow")
    with pytest.raises(Exception, match="injected downstream failure"):
        insert_checkpoint(data, manager, 1, "second", [0]).materialize()
    assert manager.find_latest_checkpoint()[0] == 0
    assert manager.load_checkpoint(0).take_all() == [{"id": i} for i in range(8)]
    assert not list(tmp_path.glob(".checkpoint-*"))


@pytest.mark.parametrize(
    "strategy,names,expected",
    [
        (CheckpointStrategy.EVERY_N_OPS, [], [3]),
        (CheckpointStrategy.MANUAL, ["checkpoint_add_mapper"], [2, 3]),
    ],
)
def test_policy_uses_original_indices(ray_runtime, tmp_path, strategy, names, expected):
    manager = RayCheckpointManager(
        str(tmp_path), checkpoint_strategy=strategy, checkpoint_n_ops=2, checkpoint_op_names=names
    )
    data = RayDataset(ray.data.range(4), auto_op_parallelism=False)
    result = data.process([_op(), _op()], checkpointer=manager.for_partitions([None], op_offset=2))
    assert result.count() == 4
    assert [
        index for index in range(5) if (tmp_path / manager.resolve_checkpoint_filename(index, None)).exists()
    ] == expected


def test_mixed_partition_progress_resumes_only_remaining_calls(ray_runtime, tmp_path):
    manager = RayCheckpointManager(str(tmp_path))
    executor = _executor(manager)
    first, second = Counter.remote(), Counter.remote()
    ops = [_op(counter=first), _op(counter=second)]
    p0 = RayDataset(ray.data.from_items([{"id": 0}, {"id": 1}]), auto_op_parallelism=False)
    p1 = RayDataset(ray.data.from_items([{"id": 2}, {"id": 3}]), auto_op_parallelism=False)
    executor._process_with_checkpointing(p0, 0, ops)
    executor._process_with_checkpointing(p1, 1, ops[:1])
    # New physical block layouts do not change the saved logical partition data.
    p0 = RayDataset(ray.data.range(2, override_num_blocks=2), auto_op_parallelism=False)
    p1 = RayDataset(ray.data.range(2, override_num_blocks=1), auto_op_parallelism=False)
    rows = executor._process_with_checkpointing(p0, 0, ops).to_list()
    rows += executor._process_with_checkpointing(p1, 1, ops).to_list()
    assert [row["id"] for row in rows] == list(range(4))
    assert [row["value"] for row in rows] == [2, 3, 4, 5]
    assert ray.get(first.get.remote()) == 4
    assert ray.get(second.get.remote()) == 4


def test_retry_does_not_commit_abandoned_attempt(ray_runtime, tmp_path, monkeypatch):
    import data_juicer.core.data.ray_checkpoint_adapter as adapter

    marker = str(tmp_path / "worker-exited")

    def crash_once(block, *args):
        result = write_checkpoint_block(block, *args)
        try:
            fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            return result
        os.close(fd)
        os._exit(17)

    monkeypatch.setattr(adapter, "write_checkpoint_block", crash_once)
    manager = RayCheckpointManager(str(tmp_path))
    result = insert_checkpoint(ray.data.range(4, override_num_blocks=1), manager, 0, "retry", [0]).materialize()
    assert result.count() == 4
    manifest = _manifest(manager, 0)
    assert manifest["rows"] == 4
    assert len(manifest["files"]) == 1
    assert manager.load_checkpoint(0).take_all() == [{"id": i} for i in range(4)]


@pytest.mark.parametrize("oversized", [False, True])
def test_backpressure_completion_and_cancellation(ray_runtime, tmp_path, monkeypatch, oversized):
    import data_juicer.core.data.ray_checkpoint_adapter as adapter
    from ray.data._internal.execution.operators.input_data_buffer import InputDataBuffer
    from ray.data._internal.execution.interfaces import RefBundle
    from ray.data.block import BlockAccessor

    gate = Gate.remote()

    def blocked_write(block, *args):
        ray.get(gate.wait.remote())
        return write_checkpoint_block(block, *args)

    monkeypatch.setattr(adapter, "write_checkpoint_block", blocked_write)
    manager = RayCheckpointManager(str(tmp_path))
    context = ray.data.DataContext.get_current().copy()
    table = pa.table({"id": [0, 1]})
    context.target_max_block_size = 4 if oversized else 1024
    metadata = BlockAccessor.for_block(table).get_metadata()
    entry = (ray.put(table), metadata)
    if ray.__version__ == "2.56.1":
        from ray.data._internal.execution.interfaces import BlockEntry

        entry = BlockEntry(*entry)
    slice_kwargs = {}
    if ray.__version__ == "2.56.1":
        from ray.data._internal.execution.interfaces.ref_bundle import BlockSlice

        slice_kwargs["slices"] = [BlockSlice(0, 1), BlockSlice(1, 2), None]
    bundle = RefBundle([entry, entry, entry], owns_blocks=False, schema=table.schema, **slice_kwargs)
    logical = SimpleNamespace(
        name="Checkpoint",
        schema=table.schema,
        manager=manager,
        op_idx=0,
        op_name="test",
        partition_ids=[0],
        partition_column=None,
        on_complete=None,
    )
    splitter = adapter.CheckpointBlockSplitter(InputDataBuffer(context, input_data=[]), context)
    splitter.start(context.execution_options)
    physical = StreamingCheckpointOperator(logical, splitter, context)
    physical.start(context.execution_options)
    splitter.add_input(bundle, 0)

    def feed():
        while splitter.has_next() and physical.can_add_input():
            physical.add_input(splitter.get_next(), 0)
        if not splitter.has_next():
            physical.all_inputs_done()

    feed()
    assert len(physical.get_active_tasks()) == (1 if oversized else 2)
    assert not physical.can_add_input()
    assert manager.find_latest_checkpoint() is None
    if oversized:
        # Cancellation stops pending writers and never publishes success.
        physical._do_shutdown(force=True)
        assert not physical.get_active_tasks()
        assert manager.find_latest_checkpoint() is None
        assert not list(tmp_path.glob(".checkpoint-*"))
    else:
        ray.get(gate.release.remote())
        outputs = []
        while splitter.has_next() or physical.num_active_tasks() or physical.has_next():
            for task in list(physical.get_active_tasks()):
                ray.get(task.get_waitable())
                task.on_task_finished()
            while physical.has_next():
                outputs.append(physical.get_next())
            feed()
            assert len(physical.get_active_tasks()) <= 2
            assert len(physical._pending) + len(physical._waiting) <= 2
        assert len(outputs) == 3
        assert all(output.block_refs[0] == bundle.block_refs[0] for output in outputs)
        assert _manifest(manager, 0)["rows"] == (4 if slice_kwargs else 6)


def test_scope_mismatch_manifest_is_ignored(tmp_path):
    manager = RayCheckpointManager(str(tmp_path))
    transaction = manager.begin_checkpoint(2, "test", [None])
    transaction.publish()
    path = tmp_path / manager.resolve_checkpoint_filename(2, None) / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["op_idx"] = 3
    path.write_text(json.dumps(manifest))
    assert manager.find_latest_checkpoint(None) is None


def test_public_global_first_recipe_validates_input_on_resume(ray_runtime, tmp_path):
    import yaml
    from data_juicer.config import init_configs
    from data_juicer.ops.base_op import OPERATORS

    for cls in (UniqueDeduplicator, IdentityPipeline, AddMapper):
        if cls._name not in OPERATORS.modules:
            OPERATORS.register_module(cls._name)(cls)
    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps({"id": value}) + "\n" for value in [1, 1, 2, 3]))
    recipe = {
        "project_name": "streaming-global-checkpoint",
        "executor_type": "ray_partitioned",
        "dataset_path": str(source),
        "export_path": str(tmp_path / "output.jsonl"),
        "work_dir": str(tmp_path / "work"),
        "strict_preflight": False,
        "auto_op_parallelism": False,
        "op_fusion": True,
        "partition": {
            "mode": "manual",
            "num_of_partitions": 2,
            "max_concurrent_partitions": 1,
            "gpu_preflight_enabled": False,
        },
        "checkpoint": {"enabled": True, "strategy": "every_op"},
        "process": [
            {cls._name: {"num_proc": 1, "num_cpus": 1, "auto_op_parallelism": False}}
            for cls in (UniqueDeduplicator, IdentityPipeline, AddMapper)
        ],
    }
    config_path = tmp_path / "recipe.yaml"
    config_path.write_text(yaml.safe_dump(recipe))
    cfg = init_configs(["--config", str(config_path)])
    executor = PartitionedRayExecutor(cfg)
    result = executor.run()
    rows = result.to_list()
    assert sorted(row["id"] for row in rows) == [1, 2, 3]
    assert executor.ckpt_manager.find_latest_checkpoint(None)[0] == 2
    assert executor.ckpt_manager.find_latest_checkpoint(0) is None
    assert Path(executor.ckpt_manager.ckpt_dir, "partitioning_info.json").exists()
    ops = executor._prepare_operators()
    assert len(ops) == 3
    executor._is_resuming = True
    cfg._resume_requested = True
    resumed = executor._process_with_convergence(executor.datasetbuilder.load_dataset(), ops, [0, 1])
    assert resumed.to_list() == rows
    source.write_text("".join(json.dumps({"id": value}) + "\n" for value in [1, 1, 2, 99]))
    with pytest.raises(RuntimeError, match="hash|content|mismatch"):
        executor._process_with_convergence(executor.datasetbuilder.load_dataset(), ops, [0, 1])


@pytest.mark.parametrize("gpu_fraction,actors,stages", [(1, 2, 2), (0.6, 1, 3)])
def test_explicit_gpu_stage_reservations_keep_necessary_barrier(ray_runtime, tmp_path, gpu_fraction, actors, stages):
    manager = RayCheckpointManager(str(tmp_path))
    executor = _executor(manager)
    ops = [
        AddMapper(
            num_proc=actors,
            num_cpus=1,
            num_gpus=gpu_fraction,
            batch_size=2,
            auto_op_parallelism=False,
            ray_execution_mode="actor",
        )
        for _ in range(stages)
    ]
    data = RayDataset(ray.data.range(8, override_num_blocks=4), auto_op_parallelism=False)
    result = executor._process_with_checkpointing(data, 0, ops)
    assert [row["value"] for row in result.to_list()] == list(range(stages, 8 + stages))
    assert [op.num_proc for op in ops] == [actors] * stages
    events = [call.kwargs for call in executor._log_event.call_args_list]
    complete = next(
        i
        for i, event in enumerate(events)
        if event["event_type"].value == "op_complete" and event["operation_idx"] == stages - 2
    )
    started = next(
        i
        for i, event in enumerate(events)
        if event["event_type"].value == "op_start" and event["operation_idx"] == stages - 1
    )
    assert complete < started

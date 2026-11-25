"""Streaming checkpoint contracts and recovery integration (Ray 2.51/2.56)."""

import json
import os
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
    ray.data.DataContext.get_current().enable_progress_bars = False
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
def test_damaged_manifest_is_never_selected(tmp_path, damage):
    manager = RayCheckpointManager(str(tmp_path))
    tx = manager.begin_checkpoint(2, "mapper", [0])
    tx.add_result(write_checkpoint_block(pa.table({"id": [1]}), tx.directories))
    tx.publish()
    path = tmp_path / manager.resolve_checkpoint_filename(2, 0)
    manifest = _manifest(manager, 2)
    if damage == "missing_file":
        (path / manifest["files"][0]["path"]).unlink()
    elif damage == "truncated_file":
        (path / manifest["files"][0]["path"]).write_bytes(b"bad")
    elif damage == "manifest":
        (path / "manifest.json").write_text("{broken")
    else:
        (path / "manifest.json").unlink()
    assert manager.find_latest_checkpoint() is None
    assert manager.load_checkpoint(2) is None


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

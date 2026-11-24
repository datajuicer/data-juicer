"""Configuration, eligibility and public partitioned execution contracts."""

import gc
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import ray
import yaml

from data_juicer.config import init_configs
from data_juicer.config.config import build_base_parser
from data_juicer.core.data.ray_dataset import RayDataset
from data_juicer.core.elasticjuicer import (
    partition_compute_flow as compute_flow_runtime,
)
from data_juicer.core.elasticjuicer.compute_flow import (
    ComputeFlowError,
    PartitionComputeFlowConfig,
    partition_compute_flow_config,
)
from data_juicer.core.elasticjuicer.partition_compute_flow import (
    _ROW_COLUMN,
    PartitionComputeFlowSession,
    _PoolUnavailable,
    segment_rejection,
)
from data_juicer.core.elasticjuicer.stage_identity import assign_stage_identities
from data_juicer.core.executor.gpu_memory_probe import GPUMemoryProbe
from data_juicer.core.executor.ray_executor_partitioned import PartitionedRayExecutor
from data_juicer.utils.constant import Fields
from tests.core.elasticjuicer.compute_flow_test_ops import ComputeFlowTestMapper


def options(**kw):
    return PartitionComputeFlowConfig.parse(dict(enabled=True, workers=2, batch_size=4, **kw))


def init_cli(args):
    # The public config saver reads sys.argv during initialization. Emulate the
    # real CLI so pytest flags cannot leak into checkpoint compatibility checks.
    with patch.object(sys, "argv", ["dj-process", *args]):
        return init_configs(args)


def operator(**kw):
    config = dict(
        batch_size=4,
        num_proc=1,
        num_cpus=1,
        num_gpus=0.25,
        ray_execution_mode="actor",
        adaptive_batching=True,
        skip_op_error=False,
    )
    config.update(kw)
    op = ComputeFlowTestMapper(**config)
    op._gpu_memory_fraction = 0.15
    return op


@pytest.mark.parametrize(
    "value",
    [
        True,
        {"enabled": "false"},
        {"unknown": 1},
        {"workers": 0},
        {"workers": True},
        {"batch_size": -1},
        {"batch_size": 0},
        {"batch_size": True},
        {"batch_size": None},
        {"batch_size": "invalid"},
        {"memory_fraction_limit": 1.1},
        {"task_timeout_seconds": float("nan")},
        {"mode": "invalid"},
        {"interval_seconds": "fast"},
        {"starvation_seconds": True},
        {"ewma_alpha": "slow"},
    ],
)
def test_invalid_options(value):
    with pytest.raises(ComputeFlowError):
        PartitionComputeFlowConfig.parse(value)


def test_parser_default_and_cli():
    parser = build_base_parser()
    assert parser.parse_args(["--auto"]).partition.compute_flow is None
    parsed = parser.parse_args(["--auto", '--partition.compute_flow={"enabled":true,"workers":2}'])
    assert PartitionComputeFlowConfig.parse(parsed.partition.compute_flow).workers == 2
    assert PartitionComputeFlowConfig.parse(parsed.partition.compute_flow).batch_size == "auto"
    parsed = parser.parse_args(["--auto", '--partition.compute_flow={"batch_size":"auto"}'])
    assert PartitionComputeFlowConfig.parse(parsed.partition.compute_flow).batch_size == "auto"


@pytest.mark.parametrize(
    "patch",
    [
        {"executor_type": "ray"},
        {"elastic_juicer_adaptive_batching": False},
        {"op_fusion": True},
        {"elastic_juicer_profile_seed": True},
    ],
)
def test_invalid_public_combinations(patch):
    cfg = {
        "executor_type": "ray_partitioned",
        "elastic_juicer_adaptive_batching": True,
        "partition": {"compute_flow": {"enabled": True}},
    }
    cfg.update(patch)
    with pytest.raises(ComputeFlowError):
        partition_compute_flow_config(cfg)


def test_eligibility_uses_measured_joint_memory_and_strict_mapper_contract():
    first, second = operator(), operator()
    assert segment_rejection([first, second], options()) is None
    second._gpu_memory_fraction = 0.7
    assert "co-residency" in segment_rejection([first, second], options())
    del second._gpu_memory_fraction
    assert "preflight" in segment_rejection([first, second], options())
    assert "error-skipping" in segment_rejection([operator(skip_op_error=True)], options())
    assert "runtime" in segment_rejection([operator(runtime_env={"env_vars": {"A": "B"}})], options())
    assert "contract" in segment_rejection([operator(adaptive_batching=False)], options())


@pytest.fixture(scope="module")
def cluster():
    assert not ray.is_initialized()
    ray.init(
        address="local",
        num_cpus=6,
        num_gpus=2,
        include_dashboard=False,
        object_store_memory=128 * 1024**2,
        log_to_driver=False,
    )
    yield
    ray.shutdown()


@pytest.fixture
def measured_probe(monkeypatch):
    # Logical GPU tests inject only preflight estimates; real Ray scheduling,
    # actors, transport, checkpointing and public executor code run normally.
    def resolve(self, dataset, ops):
        for op in ops:
            if op.accelerator == "cuda":
                op._gpu_memory_fraction = 0.15

    monkeypatch.setattr(GPUMemoryProbe, "resolve", resolve)


def recipe(tmp_path, *, enabled=True, checkpoint=True, every_op=False, failure=None, failure_file=None):
    source = tmp_path / "input.jsonl"
    if not source.exists():
        source.write_text(
            "".join(
                json.dumps({"id": i, "value": i + 1, "text": str(i), Fields.meta: {"calls": 0}}) + "\n"
                for i in range(36)
            )
        )
    base = dict(
        batch_size=4,
        num_proc=1,
        num_cpus=1,
        num_gpus=0.25,
        ray_execution_mode="actor",
        adaptive_batching=True,
        skip_op_error=False,
        threshold=2,
    )
    cfg = dict(
        project_name="partition-compute-flow",
        executor_type="ray_partitioned",
        dataset_path=str(source),
        export_path=str(tmp_path / "result.jsonl"),
        work_dir=str(tmp_path / "work"),
        strict_preflight=False,
        auto_op_parallelism=False,
        elastic_juicer_adaptive_batching=True,
        partition=dict(
            mode="manual",
            num_of_partitions=4,
            execution_group_size=2,
            max_concurrent_partitions=1,
            gpu_preflight_enabled=True,
            compute_flow=dict(enabled=enabled, workers=2, batch_size=4, max_inflight_batches=8, interval_seconds=0.01),
        ),
        checkpoint=dict(enabled=checkpoint, strategy="every_op" if every_op else "every_n_ops", n_ops=2),
        process=[
            {"compute_flow_test_mapper": dict(base, scale=2, delay=0.005)},
            {"compute_flow_test_mapper": dict(base, scale=3, delay=0.02, failure=failure, failure_file=failure_file)},
        ],
    )
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return init_cli(["--config", str(path)])


def assert_rows(result):
    rows = result.data.take_all()
    assert [r["id"] for r in rows] == list(range(36))
    assert [r["value"] for r in rows] == [(i + 1) * 6 for i in range(36)]
    assert all(r[Fields.meta]["calls"] == 2 for r in rows)
    assert all(_ROW_COLUMN not in r and "__data_juicer_logical_partition_id__" not in r for r in rows)
    return rows


def wait_for_gpus():
    # Ray Data's fallback path may retain actor handles in reference cycles.
    gc.collect()
    deadline = time.monotonic() + 10
    while ray.available_resources().get("GPU", 0) < 2 and time.monotonic() < deadline:
        time.sleep(0.05)
    assert ray.available_resources().get("GPU", 0) == 2


def test_public_run_reuses_workers_and_preserves_checkpoints(cluster, measured_probe, tmp_path, monkeypatch):
    cfg = recipe(tmp_path)
    executor = PartitionedRayExecutor(cfg)
    result = executor.run()
    actual = assert_rows(result)
    summary = cfg._resolved_compute_flow_summary
    assert summary["executed_intervals"] == 2
    assert len(summary["pools"]) == 1
    assert summary["pools"][0]["controller"]["reassignments"] > 0
    events = [json.loads(line) for line in (Path(summary["artifact_dir"]) / "events.jsonl").read_text().splitlines()]
    instances = {}
    for event in events:
        if event["kind"] == "complete":
            instances.setdefault((event["slot"], event["stage"]), set()).add(event["operator_instance"])
    assert len(instances) == 4 and all(len(v) == 1 for v in instances.values())
    assert any(e.get("oom_retries", 0) > 0 for e in events)
    assert {event["call"] for event in events if event["kind"] == "complete"} == {1, 2}
    wait_for_gpus()

    def forbidden(*args, **kwargs):
        raise AssertionError("Completed checkpoint rows must never be reprocessed")

    monkeypatch.setattr(RayDataset, "process", forbidden)
    monkeypatch.setattr(PartitionComputeFlowSession, "execute", forbidden)
    cfg._resume_requested = True
    executor._is_resuming = True
    ops = executor._prepare_operators()
    resumed = executor._process_with_simple_partitioning(executor.datasetbuilder.load_dataset(), ops)
    assert resumed.data.take_all() == actual


@pytest.mark.parametrize("pool_starts", [True, False])
def test_shared_pool_can_run_when_ordinary_actor_minimums_do_not_fit(
    cluster, measured_probe, tmp_path, monkeypatch, pool_starts
):
    cfg = recipe(tmp_path, checkpoint=False)
    for item in cfg.process:
        next(iter(item.values())).update(num_gpus=1, num_proc=2)

    def forbid_fallback(*args, **kwargs):
        pytest.fail("Ray Data cannot fit the four whole-GPU actors")

    monkeypatch.setattr(RayDataset, "process", forbid_fallback)
    executor = PartitionedRayExecutor(cfg)
    if pool_starts:
        assert_rows(executor.run())
        assert cfg._resolved_compute_flow_summary["executed_intervals"] == 1
        wait_for_gpus()
    else:

        def fail_start(*args):
            raise _PoolUnavailable("injected startup failure")

        monkeypatch.setattr(PartitionComputeFlowSession, "_ensure_pool", fail_start)
        with pytest.raises(ComputeFlowError, match="Resident execution is required.*injected startup failure"):
            executor.run()
        assert cfg._resolved_compute_flow_summary["failure"]["type"] == "ComputeFlowError"


def test_every_op_barriers_and_disabled_path_match(cluster, measured_probe, tmp_path):
    enabled = tmp_path / "enabled"
    disabled = tmp_path / "disabled"
    enabled.mkdir()
    disabled.mkdir()
    cfg = recipe(enabled, every_op=True)
    enabled_rows = assert_rows(PartitionedRayExecutor(cfg).run())
    assert cfg._resolved_compute_flow_summary["executed_intervals"] == 4
    assert all(len(p["stages"]) == 1 for p in cfg._resolved_compute_flow_summary["pools"])
    wait_for_gpus()
    baseline = recipe(disabled, enabled=False, every_op=True)
    baseline_rows = assert_rows(PartitionedRayExecutor(baseline).run())
    assert enabled_rows == baseline_rows
    assert not hasattr(baseline, "_resolved_compute_flow_summary")
    wait_for_gpus()


def test_missing_joint_memory_falls_back_before_resident_execution(cluster, tmp_path, monkeypatch):
    monkeypatch.setattr(GPUMemoryProbe, "resolve", lambda *args: None)
    cfg = recipe(tmp_path, checkpoint=False)
    assert_rows(PartitionedRayExecutor(cfg).run())
    assert cfg._resolved_compute_flow_summary["executed_intervals"] == 0
    assert cfg._resolved_compute_flow_summary["pools"] == []
    assert "preflight" in cfg._resolved_compute_flow_summary["fallbacks"][0]
    wait_for_gpus()


@pytest.mark.parametrize("failure", ["ordinary", "row_order", "partition"])
def test_failure_releases_workers_and_does_not_retry_on_legacy_path(
    cluster, measured_probe, tmp_path, monkeypatch, failure
):
    cfg = recipe(tmp_path, failure=failure)

    def forbidden(*args, **kwargs):
        raise AssertionError("No fallback after partial execution")

    monkeypatch.setattr(RayDataset, "process", forbidden)
    with pytest.raises(Exception, match="injected late failure|row identity/order|logical partition"):
        PartitionedRayExecutor(cfg).run()
    assert cfg._resolved_compute_flow_summary["failure"] is not None
    wait_for_gpus()


@pytest.mark.parametrize("mixed_progress", [False, True])
def test_public_resume_after_partial_checkpoint_commit(cluster, measured_probe, tmp_path, monkeypatch, mixed_progress):
    failure_file = tmp_path / "transient_failure"
    audit = tmp_path / "mapper_calls"
    recipe(tmp_path)
    path = tmp_path / "recipe.yaml"
    spec = yaml.safe_load(path.read_text())
    spec["auto_op_parallelism"] = True
    spec["checkpoint"]["n_ops"] = 1 if mixed_progress else 3
    base = dict(spec["process"][0]["compute_flow_test_mapper"], num_proc=-1, num_gpus=1, audit_path=str(audit))
    spec["process"] = [
        {
            "compute_flow_test_mapper": dict(
                base, scale=scale, failure="transient" if scale == 5 else None, failure_file=str(failure_file)
            )
        }
        for scale in (2, 3, 5)
    ]
    path.write_text(yaml.safe_dump(spec))
    # The Ray node advertises logical GPUs. Only CUDA memory discovery and probe measurements are synthetic.
    monkeypatch.setattr("data_juicer.utils.ray_utils.ray_available_gpu_memories", lambda: [16000, 16000])
    monkeypatch.setattr("data_juicer.utils.resource_utils.is_cuda_available", lambda: True)
    monkeypatch.setattr("data_juicer.ops.base_op.is_cuda_available", lambda: True)
    args = ["--config", str(path)]
    reference = init_cli(args)
    reference.export_path = str(tmp_path / "reference.jsonl")
    expected = PartitionedRayExecutor(reference).run().data.take_all()
    assert [r["id"] for r in expected] == list(range(36))
    assert all(r["value"] == (r["id"] + 1) * 30 and r[Fields.meta]["calls"] == 3 for r in expected)
    failure_file.touch()
    cfg = init_cli(args)
    executor = PartitionedRayExecutor(cfg)
    if mixed_progress:
        save = executor.ckpt_manager.save_checkpoint

        def interrupted_save(dataset, op_idx, op_name, partition_id, **kwargs):
            if partition_id == 3 and op_idx == 0:
                raise RuntimeError("injected transient failure")
            return save(dataset, op_idx, op_name, partition_id, **kwargs)

        monkeypatch.setattr(executor.ckpt_manager, "save_checkpoint", interrupted_save)
    with pytest.raises(Exception, match="injected transient failure"):
        executor.run()
    assert [executor.ckpt_manager.find_latest_checkpoint(i)[0] for i in (0, 1)] == [2, 2]
    assert executor.ckpt_manager.find_latest_checkpoint(3) is None
    partial = executor.ckpt_manager.find_latest_checkpoint(2)
    assert (partial[0] if partial else None) == (0 if mixed_progress else None)

    failure_file.unlink()
    offset = audit.stat().st_size
    planned = []
    probe = GPUMemoryProbe.resolve

    def record_preflight(self, dataset, ops):
        planned.extend((row["id"], op.scale) for row in dataset.data.take_all() for op in ops)
        return probe(self, dataset, ops)

    monkeypatch.setattr(GPUMemoryProbe, "resolve", record_preflight)
    resumed_cfg = init_cli([*args, "--resume", cfg.job_id])
    assert PartitionedRayExecutor(resumed_cfg).run().data.take_all() == expected
    assert exported_rows(resumed_cfg.export_path) == exported_rows(reference.export_path)
    with audit.open() as stream:
        stream.seek(offset)
        calls = sorted(tuple(map(int, line.split())) for line in stream)
    assert calls == [
        (row, scale) for row in range(18, 36) for scale in ((3, 5) if mixed_progress and row < 27 else (2, 3, 5))
    ]
    assert resumed_cfg._resolved_compute_flow_summary["executed_intervals"] >= 2
    assert sorted(planned) == calls
    wait_for_gpus()


def test_public_resume_skips_empty_filtered_partition(cluster, measured_probe, tmp_path):
    (tmp_path / "input.jsonl").write_text(
        "".join(
            json.dumps({"id": i, "value": i + 1, "text": "x" if i < 4 else "keep", Fields.meta: {"calls": 0}}) + "\n"
            for i in range(8)
        )
    )
    recipe(tmp_path)
    path = tmp_path / "recipe.yaml"
    spec = yaml.safe_load(path.read_text())
    spec["partition"]["num_of_partitions"] = 2
    spec["process"] = [{"text_length_filter": {"min_len": 2}}, spec["process"][0]]
    spec["checkpoint"] = {"enabled": True, "strategy": "manual", "op_names": ["text_length_filter"]}
    path.write_text(yaml.safe_dump(spec))
    args = ["--config", str(path)]

    reference = init_cli(args)
    reference.export_path = str(tmp_path / "reference.jsonl")
    PartitionedRayExecutor(reference).run()
    expected = exported_rows(reference.export_path)
    assert [(r["id"], r["value"]) for r in expected] == [(i, (i + 1) * 2) for i in range(4, 8)]

    cfg = init_cli(args)
    executor = PartitionedRayExecutor(cfg)
    with patch.object(executor.exporter, "export", side_effect=RuntimeError("injected export failure")):
        with pytest.raises(RuntimeError, match="injected export failure"):
            executor.run()

    resumed = PartitionedRayExecutor(init_cli([*args, "--resume", cfg.job_id]))
    with (
        patch.object(GPUMemoryProbe, "resolve", autospec=True, side_effect=GPUMemoryProbe.resolve) as probe,
        patch.object(resumed, "_process_execution_interval", wraps=resumed._process_execution_interval) as execute,
    ):
        resumed.run()
    # Ray may omit an empty Parquet shard and replay the CPU filter. Count every
    # GPU Mapper invocation, including attempts with no sample rows.
    for calls in (probe.call_args_list, execute.call_args_list):
        mapper_inputs = [call.args[-2] for call in calls if call.args[-1][0]._name == "compute_flow_test_mapper"]
        assert len(mapper_inputs) == 1
        assert [r["id"] for r in mapper_inputs[0].data.take_all()] == list(range(4, 8))
    assert exported_rows(cfg.export_path) == expected


def test_restored_empty_checkpoint_finishes_next_boundary(cluster, tmp_path):
    executor = PartitionedRayExecutor(recipe(tmp_path, every_op=True))
    executor._plan_on_resume = True
    executor._compute_flow_session = Mock(spec=PartitionComputeFlowSession)
    ops = executor._prepare_operators()
    executor._initialize_dag_execution(executor.cfg, ops=ops)
    manager = executor.ckpt_manager
    # Seed a readable zero-row checkpoint even on Ray versions that omit empty shards.
    checkpoint = Path(manager.ckpt_dir) / manager.resolve_checkpoint_filename(0, 0)
    checkpoint.mkdir()
    pq.write_table(pa.table({"id": pa.array([], type=pa.int64())}), checkpoint / "empty.parquet")
    with (
        patch.object(executor, "_configure_pre_partition_resources") as plan,
        patch.object(executor, "_process_execution_interval") as execute,
        patch.object(executor, "_post_execute_operations_with_dag_monitoring") as complete,
        patch.object(manager, "save_checkpoint", wraps=manager.save_checkpoint) as save,
    ):
        result = executor._process_with_checkpointing(executor.datasetbuilder.load_dataset(), 0, ops)
    assert result.data.count() == 0
    executor._compute_flow_session.release.assert_called_once_with()
    plan.assert_not_called()
    execute.assert_not_called()
    complete.assert_called_once()
    metrics = complete.call_args.kwargs["metrics"]
    assert (metrics["input_rows"], metrics["output_rows"]) == (0, 0)
    save.assert_called_once_with(result, 1, ops[1]._name, 0, cfg=executor.cfg)


def test_empty_and_reserved_input_do_not_create_pool(cluster, tmp_path):
    session = PartitionComputeFlowSession(options(), tmp_path)
    try:
        empty = ray.data.from_arrow(pa.table({"id": [], "value": []}))
        assert session.execute(empty, [operator()]).count() == 0
        assert session.pools == []
        with pytest.raises(ComputeFlowError, match="reserved column"):
            session.execute(ray.data.from_items([{_ROW_COLUMN: 1}]), [operator()])
        assert session.pools == []
    finally:
        session.finish()


@pytest.mark.parametrize("limit", ["bytes", "timeout"])
def test_runtime_limits_close_owned_pool(cluster, tmp_path, limit):
    ops = [operator(threshold=8, delay=0.1)]
    assign_stage_identities(ops)
    kw = {"max_batch_bytes": 1} if limit == "bytes" else {"task_timeout_seconds": 0.01}
    session = PartitionComputeFlowSession(options(**kw), tmp_path)
    data = ray.data.from_items([{"id": i, "value": i, Fields.meta: {"calls": 0}} for i in range(4)])
    try:
        with pytest.raises((ComputeFlowError, TimeoutError), match="max_batch_bytes|time"):
            session.execute(data, ops)
        assert session.actors == {}
    finally:
        session.finish()
    wait_for_gpus()


def test_tagging_mapper_metadata_setup_matches_ray_data(cluster, measured_probe, tmp_path):
    source = tmp_path / "input.jsonl"
    source.write_text("".join(json.dumps({"id": i, "value": i + 1, "text": str(i)}) + "\n" for i in range(36)))
    cfg = recipe(tmp_path, checkpoint=False)
    assert_rows(PartitionedRayExecutor(cfg).run())
    assert cfg._resolved_compute_flow_summary["executed_intervals"] == 1
    wait_for_gpus()


def exported_rows(path):
    return [json.loads(line) for shard in sorted(Path(path).glob("*.json")) for line in shard.read_text().splitlines()]


@pytest.mark.parametrize("enabled", [False, True])
def test_no_checkpoint_executes_one_ordered_dataset(cluster, measured_probe, tmp_path, monkeypatch, enabled):
    cfg = recipe(tmp_path, checkpoint=False, enabled=enabled)
    executor = PartitionedRayExecutor(cfg)
    for method in ("_split_dataset_deterministic", "_tag_execution_partition", "_process_partition_execution_groups"):
        monkeypatch.setattr(executor, method, lambda *args: pytest.fail("unexpected logical partition processing"))
    load = executor.datasetbuilder.load_dataset

    def multi_block_input(**kwargs):
        dataset = load(**kwargs)
        dataset.data = dataset.data.repartition(6, shuffle=False).materialize()
        return dataset

    monkeypatch.setattr(executor.datasetbuilder, "load_dataset", multi_block_input)
    with patch.object(executor, "_process_execution_interval", wraps=executor._process_execution_interval) as execute:
        rows = assert_rows(executor.run())
    assert [(row["id"], row["value"]) for row in exported_rows(cfg.export_path)] == [
        (row["id"], row["value"]) for row in rows
    ]
    assert execute.call_count == 1
    if enabled:
        assert cfg._resolved_compute_flow_summary["executed_intervals"] == 1
    wait_for_gpus()


@pytest.mark.parametrize("enabled", [False, True])
def test_ordinary_execution_validates_minimums_before_mapper_work(
    cluster, measured_probe, tmp_path, monkeypatch, enabled
):
    cfg = recipe(tmp_path, checkpoint=False, enabled=enabled)
    for entry in cfg.process:
        entry["compute_flow_test_mapper"].update(num_proc=2, num_gpus=1, adaptive_batching=not enabled)
    monkeypatch.setattr(RayDataset, "process", lambda *args, **kwargs: pytest.fail("unschedulable actor pools"))
    with pytest.raises(RuntimeError, match="Ray Data actor minimums"):
        PartitionedRayExecutor(cfg).run()


def test_mixed_ray_data_and_resident_intervals_collect_handles(cluster, measured_probe, tmp_path):
    cfg = recipe(tmp_path)
    base = dict(cfg.process[0]["compute_flow_test_mapper"], threshold=64, delay=0)
    cfg.process = [
        {"compute_flow_test_mapper": dict(base, scale=scale, adaptive_batching=i != 0)}
        for i, scale in enumerate((2, 3, 5, 7))
    ]
    # Deterministically defer cyclic GC: each execution group must hand off
    # from two ordinary Ray Data actors to a two-GPU resident pool itself.
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        rows = PartitionedRayExecutor(cfg).run().data.take_all()
    finally:
        if was_enabled:
            gc.enable()
    assert [r["id"] for r in rows] == list(range(36))
    assert [r["value"] for r in rows] == [(i + 1) * 210 for i in range(36)]
    assert all(r[Fields.meta]["calls"] == 4 for r in rows)
    summary = cfg._resolved_compute_flow_summary
    assert summary["executed_intervals"] == 2
    assert summary["failure"] is None
    assert any("contract" in reason for reason in summary["fallbacks"])
    wait_for_gpus()


def test_resource_wait_respects_configured_timeout_beyond_five_seconds(tmp_path, monkeypatch):
    clock = [0.0]
    collections = []
    ops = [operator()]
    assign_stage_identities(ops)

    def available():
        # A background executor drops its last live references at t=6;
        # a subsequent collection is needed to release its cyclic handles.
        return {"CPU": 6, "GPU": 2 if collections and collections[-1] >= 6 else 0}

    def sleep(seconds):
        clock[0] += seconds

    def reached_construction(**kwargs):
        raise RuntimeError("resource admission succeeded")

    monkeypatch.setattr(compute_flow_runtime, "time", SimpleNamespace(monotonic=lambda: clock[0], sleep=sleep))
    monkeypatch.setattr(compute_flow_runtime, "gc", SimpleNamespace(collect=lambda: collections.append(clock[0])))
    monkeypatch.setattr(
        compute_flow_runtime,
        "ray",
        SimpleNamespace(
            nodes=lambda: [{"Alive": True, "Resources": {"CPU": 6, "GPU": 2}}],
            available_resources=available,
            remote=reached_construction,
            exceptions=ray.exceptions,
        ),
    )
    session = PartitionComputeFlowSession(options(startup_timeout_seconds=8), tmp_path)
    try:
        with pytest.raises(RuntimeError, match="resource admission succeeded"):
            session._ensure_pool(ops)
        assert 6 <= clock[0] < 8
    finally:
        session.finish()


@pytest.mark.parametrize("reason", ["capacity", "busy", "readiness"])
def test_pool_admission_failure_uses_ray_data(cluster, measured_probe, tmp_path, monkeypatch, reason):
    cfg = recipe(tmp_path)
    if reason == "capacity":
        cfg.partition.compute_flow["workers"] = 3
    elif reason == "busy":
        cfg.partition.compute_flow["startup_timeout_seconds"] = 0.2
    held_actor = None
    if reason == "busy":

        @ray.remote(num_cpus=0, num_gpus=0.25)
        class OtherJob:
            def ping(self):
                return "alive"

        held_actor = OtherJob.remote()
        assert ray.get(held_actor.ping.remote()) == "alive"
    elif reason == "readiness":

        def not_ready(*args, **kwargs):
            raise ray.exceptions.GetTimeoutError("injected readiness timeout")

        monkeypatch.setattr(
            compute_flow_runtime,
            "ray",
            SimpleNamespace(
                nodes=ray.nodes,
                available_resources=ray.available_resources,
                remote=ray.remote,
                get=not_ready,
                kill=ray.kill,
                exceptions=ray.exceptions,
            ),
        )
    try:
        assert_rows(PartitionedRayExecutor(cfg).run())
        summary = cfg._resolved_compute_flow_summary
        assert summary["executed_intervals"] == 0
        assert summary["failure"] is None and summary["fallbacks"]
        if reason != "readiness":
            assert summary["pools"] == []
        if held_actor is not None:
            assert ray.get(held_actor.ping.remote()) == "alive"
    finally:
        if held_actor is not None:
            ray.kill(held_actor, no_restart=True)
    wait_for_gpus()


@pytest.mark.parametrize("failure", [RuntimeError("injected initializer failure"), KeyboardInterrupt()])
def test_pool_creation_errors_are_not_silently_replayed(cluster, tmp_path, monkeypatch, failure):
    def failed_get(*args, **kwargs):
        raise failure

    monkeypatch.setattr(
        compute_flow_runtime,
        "ray",
        SimpleNamespace(
            nodes=ray.nodes,
            available_resources=ray.available_resources,
            remote=ray.remote,
            get=failed_get,
            kill=ray.kill,
            exceptions=ray.exceptions,
        ),
    )
    ops = [operator()]
    assign_stage_identities(ops)
    session = PartitionComputeFlowSession(options(), tmp_path)
    try:
        with pytest.raises(type(failure)):
            session.execute(ray.data.from_items([{"id": 0, "value": 1}]), ops)
        assert session.fallbacks == [] and session.failure is not None
        assert session.actors == {}
    finally:
        session.finish()
    wait_for_gpus()


@pytest.mark.parametrize("configured_batch", [None, 32])
def test_transport_batch_preserves_operator_caps_by_default(cluster, tmp_path, monkeypatch, configured_batch):
    settings = {"enabled": True, "workers": 2}
    if configured_batch is not None:
        settings["batch_size"] = configured_batch
    # Different caps ensure auto chooses the largest, while the first stage
    # still microbatches independently and absorbs an injected OOM locally.
    ops = [operator(batch_size=250, threshold=200), operator(batch_size=1000, threshold=2000, scale=3)]
    assign_stage_identities(ops)
    warnings = []
    monkeypatch.setattr(compute_flow_runtime.logger, "warning", warnings.append)
    session = PartitionComputeFlowSession(PartitionComputeFlowConfig.parse(settings), tmp_path)
    data = ray.data.from_arrow(pa.table({"id": list(range(2000)), "value": list(range(1, 2001))}))
    try:
        result = session.execute(data, ops)
        rows = result.take_all()
        assert [r["value"] for r in rows] == [i * 6 for i in range(1, 2001)]
        assert all(r[Fields.meta]["calls"] == 2 for r in rows)
        events = [json.loads(line) for line in (session.root / "events.jsonl").read_text().splitlines()]
        completed = [event for event in events if event["kind"] == "complete"]
        if configured_batch is None:
            assert len(completed) == 4 and result.num_blocks() == 2
            assert {e["rows"] for e in completed} == {1000}
            assert any(e["batch"]["success_lower_bound"] == 1000 for e in completed)
            assert any(e["oom_retries"] > 0 for e in completed)
            assert not warnings
        else:
            assert len(completed) == 126 and result.num_blocks() == 63
            assert {e["rows"] for e in completed} == {16, 32}
            assert not any(e["oom_retries"] for e in completed)
            assert len(warnings) == 1 and "caps actual mapper batches" in warnings[0]
    finally:
        session.finish()
    wait_for_gpus()

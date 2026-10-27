"""Configuration, eligibility and public partitioned execution contracts."""

import gc
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pytest
import ray
import yaml

from data_juicer.config import init_configs
from data_juicer.config.config import build_base_parser
from data_juicer.core.data.ray_dataset import RayDataset
from data_juicer.core.elasticjuicer.compute_flow import (
    ComputeFlowError,
    PartitionComputeFlowConfig,
    partition_compute_flow_config,
)
from data_juicer.core.elasticjuicer.partition_compute_flow import (
    _ROW_COLUMN,
    PartitionComputeFlowSession,
    segment_rejection,
)
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


def test_public_resume_after_partial_checkpoint_commit(cluster, measured_probe, tmp_path, monkeypatch):
    failure_file = tmp_path / "transient_failure"
    failure_file.touch()
    cfg = recipe(tmp_path, failure="transient", failure_file=str(failure_file))
    executor = PartitionedRayExecutor(cfg)
    with pytest.raises(Exception, match="injected transient failure"):
        executor.run()
    assert executor.ckpt_manager.find_latest_checkpoint(0)[0] == 1
    assert executor.ckpt_manager.find_latest_checkpoint(1)[0] == 1
    assert executor.ckpt_manager.find_latest_checkpoint(2) is None
    assert cfg._resolved_compute_flow_summary["executed_intervals"] == 1
    failure_audit = Path(cfg._resolved_compute_flow_summary["artifact_dir"]) / "events.jsonl"
    failure_events = failure_audit.read_bytes()
    wait_for_gpus()
    failure_file.unlink()
    processed_ids = []
    original = RayDataset.process

    def record_missing_partitions(self, ops, **kwargs):
        ids = [row["id"] for row in self.data.take_all()]
        assert not set(ids) & set(range(18)), "checkpointed rows were reprocessed"
        processed_ids.extend(ids)
        return original(self, ops, **kwargs)

    monkeypatch.setattr(RayDataset, "process", record_missing_partitions)
    resumed_cfg = init_cli(["--config", str(tmp_path / "recipe.yaml"), "--resume", cfg.job_id])
    resumed = PartitionedRayExecutor(resumed_cfg)
    assert_rows(resumed.run())
    assert set(processed_ids) == set(range(18, 36))
    assert failure_audit.read_bytes() == failure_events
    assert resumed_cfg._resolved_compute_flow_summary["pools"] == []
    assert "resume" in resumed_cfg._resolved_compute_flow_summary["fallbacks"][0]
    wait_for_gpus()


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
    from data_juicer.core.elasticjuicer.stage_identity import assign_stage_identities

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
    assert cfg._resolved_compute_flow_summary["executed_intervals"] == 2
    wait_for_gpus()

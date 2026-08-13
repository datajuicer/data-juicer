import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from jsonargparse import Namespace

from data_juicer.config.config import resolve_job_id
from data_juicer.core.executor.elastic_sharding.context import (
    ELASTIC_SHARD_CHILD_ENV,
    LaunchContext,
    detect_launch_context,
    should_wrap_executor,
)
from data_juicer.core.executor.elastic_sharding.executor import ElasticShardingExecutor
from data_juicer.core.executor.elastic_sharding.job import _refresh_lock
from data_juicer.core.executor.elastic_sharding.rendezvous import (
    RendezvousResult,
    SharedRendezvous,
)
from data_juicer.core.executor.elastic_sharding.safety import analyze_shardability
from data_juicer.core.executor.factory import ExecutorFactory


def _cfg(tmp_path: Path, process):
    dataset_path = tmp_path / "input.jsonl"
    dataset_path.write_text('{"text":"hello"}\n', encoding="utf-8")
    recipe_path = tmp_path / "recipe.yaml"
    recipe_path.write_text("process: []\n", encoding="utf-8")
    return SimpleNamespace(
        config=[recipe_path],
        executor_type="default",
        ray_address="auto",
        dataset_path=str(dataset_path),
        dataset=[],
        generated_dataset_config=None,
        export_path=str(tmp_path / "output.jsonl"),
        export_type=None,
        export_shard_size=0,
        export_in_parallel=False,
        data_probe_ratio=1.0,
        decrypt_after_reading=False,
        encrypt_before_export=False,
        process=process,
        elastic_sharding=SimpleNamespace(mode="auto", run_id=None),
    )


def test_launch_context_requires_complete_rank_metadata_and_stable_identity():
    context = detect_launch_context(
        {"WORLD_SIZE": "4", "RANK": "2", "LOCAL_RANK": "0", "PAI_JOB_ID": "job-7"},
        hostname="worker-b",
    )
    assert context == LaunchContext(4, 2, 0, "job-7", "worker-b", "torch")
    assert detect_launch_context({"WORLD_SIZE": "1", "RANK": "0"}) is None

    try:
        detect_launch_context({"WORLD_SIZE": "4"})
    except ValueError as exc:
        assert "RANK" in str(exc)
    else:
        raise AssertionError("incomplete distributed metadata must be rejected")


def test_factory_wraps_only_outer_stable_distributed_process(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, [{"whitespace_normalization_mapper": {}}])
    env = {"WORLD_SIZE": "2", "RANK": "0", "PAI_JOB_ID": "submission-1"}
    assert should_wrap_executor(cfg, env)
    assert not should_wrap_executor(cfg, {**env, ELASTIC_SHARD_CHILD_ENV: "1"})

    for name, value in env.items():
        monkeypatch.setenv(name, value)
    executor = ExecutorFactory.create_executor_from_config(cfg)
    assert isinstance(executor, ElasticShardingExecutor)


def test_resolve_job_id_is_shared_across_distributed_ranks(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("PAI_JOB_ID", "pai-submission-99")
    first = resolve_job_id(Namespace(job_id=None, resume=None, elastic_sharding=Namespace(mode="auto", run_id=None)))
    second = resolve_job_id(Namespace(job_id=None, resume=None, elastic_sharding=Namespace(mode="auto", run_id=None)))
    assert first.job_id == second.job_id
    assert first.job_id.startswith("elastic_")
    assert first._elastic_auto_job_id


def test_shared_rendezvous_deduplicates_hosts_and_elects_one_leader(tmp_path):
    contexts = (
        LaunchContext(3, 0, 0, "run-1", "node-a", "torch"),
        LaunchContext(3, 1, 1, "run-1", "node-a", "torch"),
        LaunchContext(3, 2, 0, "run-1", "node-b", "torch"),
    )

    def wait(context):
        return SharedRendezvous(
            tmp_path / "coord",
            context,
            fingerprint="same-config",
            timeout_secs=5,
            poll_interval_secs=0.01,
        ).wait()

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(wait, contexts))
    assert all(result.hostnames == ("node-a", "node-b") for result in results)
    assert all(result.host_leader_ranks == (0, 2) for result in results)


def test_shardability_accepts_record_local_ops_and_rejects_global_ops(tmp_path):
    local = analyze_shardability(_cfg(tmp_path, [{"whitespace_normalization_mapper": {}}]))
    assert local.eligible
    assert local.operator_names == ("whitespace_normalization_mapper",)

    json_cfg = _cfg(tmp_path, [{"whitespace_normalization_mapper": {}}])
    json_cfg.export_type = "json"
    json_report = analyze_shardability(json_cfg)
    assert not json_report.eligible
    assert any("supports JSONL export" in reason for reason in json_report.reasons)

    global_cfg = _cfg(tmp_path, [{"document_deduplicator": {}}])
    global_report = analyze_shardability(global_cfg)
    assert not global_report.eligible
    assert any("global operation" in reason for reason in global_report.reasons)


def test_heartbeat_refreshes_only_the_current_claim(tmp_path):
    lock_path = tmp_path / "shard.lock"
    lock_path.write_text(json.dumps({"token": "owner"}), encoding="utf-8")
    os.utime(lock_path, (1, 1))
    assert not _refresh_lock(lock_path, "replaced-owner")
    assert lock_path.stat().st_mtime == 1
    assert _refresh_lock(lock_path, "owner")
    assert lock_path.stat().st_mtime > time.time() - 5


def test_coordinator_only_fallback_executes_once(tmp_path):
    cfg = _cfg(tmp_path, [{"document_deduplicator": {}}])
    root = tmp_path / "coord"
    rendezvous = RendezvousResult(
        members=(
            {"rank": 0, "hostname": "node-a"},
            {"rank": 1, "hostname": "node-b"},
        ),
        coordinator_rank=0,
        host_leader_ranks=(0, 1),
    )
    calls = []
    coordinator = ElasticShardingExecutor(cfg)
    coordinator._context = LaunchContext(2, 0, 0, "run-1", "node-a", "torch")
    coordinator._run_base = lambda *_args: calls.append("run") or "value"
    assert coordinator._run_coordinator_only(rendezvous, root, "fingerprint", "global op", None, False) == "value"

    follower = ElasticShardingExecutor(cfg)
    follower._context = LaunchContext(2, 1, 0, "run-1", "node-b", "torch")
    follower._run_base = lambda *_args: calls.append("unexpected")
    assert follower._run_coordinator_only(rendezvous, root, "fingerprint", "global op", None, False) is None
    assert calls == ["run"]

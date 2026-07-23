import hashlib
import importlib.util
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "demos" / "elastic_sharding" / "shard_job.py"
SPEC = importlib.util.spec_from_file_location("elastic_shard_job", SCRIPT_PATH)
elastic_shard_job = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = elastic_shard_job
SPEC.loader.exec_module(elastic_shard_job)

TWO_NODE_SCRIPT_PATH = REPO_ROOT / "demos" / "elastic_sharding" / "two_node_test.py"
TWO_NODE_SPEC = importlib.util.spec_from_file_location(
    "elastic_two_node_test",
    TWO_NODE_SCRIPT_PATH,
)
elastic_two_node_test = importlib.util.module_from_spec(TWO_NODE_SPEC)
sys.modules[TWO_NODE_SPEC.name] = elastic_two_node_test
TWO_NODE_SPEC.loader.exec_module(elastic_two_node_test)


def _write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _prepare_job(tmp_path, monkeypatch, *, num_shards=3):
    data_dir = tmp_path / "input"
    _write_jsonl(
        data_dir / "a.jsonl",
        [
            {"id": 0, "text": "a", "images": ["media/a.jpg"]},
            {"id": 1, "text": "b" * 80},
            {"id": 2, "text": "c"},
        ],
    )
    _write_jsonl(
        data_dir / "nested" / "b.jsonl",
        [
            {"id": 3, "text": "d" * 40},
            {"id": 4, "text": "e", "images": ["https://example.test/e.jpg"]},
            {"id": 5, "text": "f"},
        ],
    )
    recipe_path = tmp_path / "recipe.yaml"
    recipe_path.write_text(
        yaml.safe_dump(
            {
                "dataset_path": str(data_dir),
                "executor_type": "ray",
                "ray_address": "local",
                "process": [
                    {
                        "whitespace_normalization_mapper": {
                            "text_key": "text",
                            "index_key": "global_index",
                        }
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        elastic_shard_job,
        "_git_info",
        lambda: {"commit": "test-commit", "dirty": False},
    )
    job_dir = tmp_path / "job"
    return_code = elastic_shard_job.main(
        [
            "prepare",
            "--config",
            str(recipe_path),
            "--job-dir",
            str(job_dir),
            "--num-shards",
            str(num_shards),
        ]
    )
    assert return_code == 0
    return job_dir, data_dir, recipe_path


def _load_shard_records(job_dir):
    manifest = elastic_shard_job._load_manifest(job_dir)
    records = []
    for shard in manifest["shards"]:
        shard_path = job_dir / shard["path"]
        with shard_path.open("r", encoding="utf-8") as handle:
            records.extend(json.loads(line) for line in handle)
    return manifest, records


def _publish_done_results(job_dir, owners=None):
    manifest = elastic_shard_job._load_manifest(job_dir)
    expected = bytearray()
    for shard_index, shard in enumerate(manifest["shards"]):
        source_path = job_dir / shard["path"]
        result_path = job_dir / "attempts" / shard["id"] / "manual" / "processed.jsonl"
        result_path.parent.mkdir(parents=True)
        payload = source_path.read_bytes()
        result_path.write_bytes(payload)
        expected.extend(payload)
        elastic_shard_job._atomic_write_json(
            elastic_shard_job._done_path(job_dir, shard["id"]),
            {
                "shard_id": shard["id"],
                "status": "done",
                "output_path": result_path.relative_to(job_dir).as_posix(),
                "rows": shard["rows"],
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "hostname": (owners[shard_index % len(owners)] if owners else "test-node"),
            },
        )
    return bytes(expected)


def test_prepare_preserves_order_and_normalizes_metadata(tmp_path, monkeypatch):
    job_dir, data_dir, recipe_path = _prepare_job(tmp_path, monkeypatch)

    manifest, records = _load_shard_records(job_dir)
    assert manifest["schema_version"] == 2
    assert manifest["execution"] == {
        "executor_type": "ray",
        "ray_address": "local",
        "recipe_executor_type": "ray",
    }
    assert manifest["num_shards"] == 3
    assert len(manifest["shards"]) == 3
    assert all(shard["rows"] > 0 for shard in manifest["shards"])
    assert [record["id"] for record in records] == list(range(6))
    assert [record["global_index"] for record in records] == list(range(6))
    assert records[0]["images"] == [str((data_dir / "media" / "a.jpg").resolve())]
    assert records[4]["images"] == ["https://example.test/e.jpg"]
    assert sum(shard["rows"] for shard in manifest["shards"]) == 6

    # Repeating the exact prepare request is an idempotent no-op.
    assert (
        elastic_shard_job.main(
            [
                "prepare",
                "--config",
                str(recipe_path),
                "--job-dir",
                str(job_dir),
                "--num-shards",
                "3",
            ]
        )
        == 0
    )

    # Size and mtime are only fast checks; content hash still detects a change.
    source_path = data_dir / "a.jsonl"
    old_mtime_ns = source_path.stat().st_mtime_ns
    source_path.write_text(
        source_path.read_text(encoding="utf-8").replace('"text": "a"', '"text": "z"', 1),
        encoding="utf-8",
    )
    os.utime(source_path, ns=(old_mtime_ns, old_mtime_ns))
    assert (
        elastic_shard_job.main(
            [
                "prepare",
                "--config",
                str(recipe_path),
                "--job-dir",
                str(job_dir),
                "--num-shards",
                "3",
            ]
        )
        == 2
    )


def test_prepare_rejects_whole_dataset_operator(tmp_path, monkeypatch):
    dataset_path = tmp_path / "input.jsonl"
    _write_jsonl(dataset_path, [{"text": "one"}, {"text": "two"}])
    recipe_path = tmp_path / "recipe.yaml"
    recipe_path.write_text(
        yaml.safe_dump(
            {
                "dataset_path": str(dataset_path),
                "executor_type": "ray",
                "ray_address": "local",
                "process": [{"document_deduplicator": {}}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        elastic_shard_job,
        "_git_info",
        lambda: {"commit": "test-commit", "dirty": False},
    )
    job_dir = tmp_path / "job"

    assert (
        elastic_shard_job.main(
            [
                "prepare",
                "--config",
                str(recipe_path),
                "--job-dir",
                str(job_dir),
                "--num-shards",
                "1",
            ]
        )
        == 2
    )
    assert not job_dir.exists()


def test_default_recipe_is_accepted_and_overridden_with_ray():
    validation = elastic_shard_job._validate_recipe(
        {
            "executor_type": "default",
            "process": [{"whitespace_normalization_mapper": {"text_key": "text"}}],
        }
    )
    assert any("overridden with executor_type=ray" in warning for warning in validation["warnings"])


def test_claim_is_exclusive_and_stale_lock_is_reclaimed(tmp_path, monkeypatch):
    job_dir, _, _ = _prepare_job(tmp_path, monkeypatch, num_shards=1)
    shard = elastic_shard_job._load_manifest(job_dir)["shards"][0]

    def claim_once(_):
        return elastic_shard_job._create_claim(
            job_dir,
            shard,
            timeout_secs=3600,
            max_retries=3,
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        claims = list(pool.map(claim_once, range(16)))
    winners = [claim for claim in claims if claim is not None]
    assert len(winners) == 1
    first_claim = winners[0]
    lock_path = elastic_shard_job._lock_path(job_dir, shard["id"])
    assert lock_path.exists()
    assert len(elastic_shard_job._attempt_directories(job_dir, shard["id"])) == 1

    os.utime(lock_path, (1, 1))
    second_claim = elastic_shard_job._create_claim(
        job_dir,
        shard,
        timeout_secs=1,
        max_retries=3,
    )
    assert second_claim is not None
    assert second_claim["token"] != first_claim["token"]
    assert not elastic_shard_job._release_lock(lock_path, first_claim["token"])
    assert elastic_shard_job._read_json(lock_path)["token"] == second_claim["token"]
    first_attempt = job_dir / first_claim["attempt_dir"] / "attempt.json"
    assert elastic_shard_job._read_json(first_attempt)["status"] == "stale"
    assert len(list((job_dir / "state" / "stale_locks").glob("*.lock"))) == 1


def test_process_claim_uses_isolated_paths_and_publishes_done(tmp_path, monkeypatch):
    job_dir, _, _ = _prepare_job(tmp_path, monkeypatch, num_shards=1)
    manifest = elastic_shard_job._load_manifest(job_dir)
    shard = manifest["shards"][0]
    claim = elastic_shard_job._create_claim(
        job_dir,
        shard,
        timeout_secs=3600,
        max_retries=3,
    )
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs["env"]
        ray_export_path = Path(command[command.index("--export_path") + 1])
        input_path = Path(command[command.index("--dataset_path") + 1])
        ray_export_path.mkdir()
        lines = input_path.read_bytes().splitlines(keepends=True)
        midpoint = len(lines) // 2
        (ray_export_path / "part-00000.json").write_bytes(b"".join(lines[:midpoint]))
        (ray_export_path / "part-00001.json").write_bytes(b"".join(lines[midpoint:]))
        (ray_export_path / "_SUCCESS").write_text("not json", encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setattr(elastic_shard_job.subprocess, "run", fake_run)
    assert elastic_shard_job._process_claim(job_dir, manifest, shard, claim) == "done"

    done = elastic_shard_job._read_json(elastic_shard_job._done_path(job_dir, shard["id"]))
    assert done["rows"] == shard["rows"]
    assert done["executor_type"] == "ray"
    assert done["ray_address"] == "local"
    assert len(done["ray_output_files"]) == 2
    assert (job_dir / done["output_path"]).read_bytes() == (job_dir / shard["path"]).read_bytes()
    assert not elastic_shard_job._lock_path(job_dir, shard["id"]).exists()
    assert Path(captured["env"]["HF_HOME"]).is_relative_to(job_dir / "cache")
    assert Path(captured["env"]["XDG_CACHE_HOME"]).is_relative_to(job_dir / "cache")
    assert captured["env"]["PYTHONPATH"].split(os.pathsep)[0] == str(REPO_ROOT)
    assert captured["command"][captured["command"].index("--executor_type") + 1] == "ray"
    assert captured["command"][captured["command"].index("--ray_address") + 1] == "local"
    assert elastic_shard_job._effective_execution(
        manifest,
        SimpleNamespace(ray_address="auto"),
    ) == {"executor_type": "ray", "ray_address": "auto"}


def test_max_retries_are_in_addition_to_initial_attempt(tmp_path, monkeypatch):
    job_dir, _, _ = _prepare_job(tmp_path, monkeypatch, num_shards=1)
    shard = elastic_shard_job._load_manifest(job_dir)["shards"][0]

    last_claim = None
    for attempt_index in range(4):
        claim = elastic_shard_job._create_claim(
            job_dir,
            shard,
            timeout_secs=3600,
            max_retries=3,
        )
        assert claim is not None
        metadata_path = job_dir / claim["attempt_dir"] / "attempt.json"
        metadata = elastic_shard_job._read_json(metadata_path)
        metadata["status"] = "failed"
        elastic_shard_job._atomic_write_json(metadata_path, metadata)
        if attempt_index < 3:
            elastic_shard_job._release_lock(
                elastic_shard_job._lock_path(job_dir, shard["id"]),
                claim["token"],
            )
        else:
            last_claim = claim

    assert last_claim is not None
    assert elastic_shard_job._publish_failure_and_release(
        job_dir,
        shard["id"],
        last_claim["token"],
        {
            "shard_id": shard["id"],
            "status": "failed",
            "failures": 4,
        },
    )

    assert (
        elastic_shard_job._create_claim(
            job_dir,
            shard,
            timeout_secs=3600,
            max_retries=3,
        )
        is None
    )
    failed = elastic_shard_job._read_json(elastic_shard_job._failed_path(job_dir, shard["id"]))
    assert failed["failures"] == 4


def test_retry_and_ordered_merge(tmp_path, monkeypatch):
    job_dir, _, _ = _prepare_job(tmp_path, monkeypatch, num_shards=2)
    manifest = elastic_shard_job._load_manifest(job_dir)
    failed_shard = manifest["shards"][0]["id"]
    failed_path = elastic_shard_job._failed_path(job_dir, failed_shard)
    elastic_shard_job._atomic_write_json(
        failed_path,
        {"shard_id": failed_shard, "status": "failed"},
    )
    attempt_root = job_dir / "attempts" / failed_shard
    attempt_root.mkdir(parents=True)
    (attempt_root / "old-attempt").mkdir()

    assert elastic_shard_job.main(["retry", "--job-dir", str(job_dir), "--shard-id", failed_shard]) == 0
    assert not failed_path.exists()
    assert not attempt_root.exists()
    assert list((job_dir / "state" / "history" / "failed").glob("*.json"))
    assert list((job_dir / "state" / "history" / "attempts").iterdir())

    # Merge refuses partial state, then validates and joins results in manifest order.
    output_path = tmp_path / "merged.jsonl"
    assert elastic_shard_job.main(["merge", "--job-dir", str(job_dir), "--output", str(output_path)]) == 2
    expected = _publish_done_results(job_dir)
    assert elastic_shard_job.main(["merge", "--job-dir", str(job_dir), "--output", str(output_path)]) == 0
    assert output_path.read_bytes() == expected
    merge_metadata = elastic_shard_job._read_json(job_dir / "merge.json")
    assert merge_metadata["rows"] == manifest["total_rows"]
    assert merge_metadata["sha256"] == hashlib.sha256(expected).hexdigest()


def test_two_node_wrapper_verifies_distinct_owners_and_merges(tmp_path, monkeypatch):
    job_dir, _, _ = _prepare_job(tmp_path, monkeypatch, num_shards=4)
    expected = _publish_done_results(job_dir, owners=["node-a", "node-b"])
    output_path = tmp_path / "two-node-output.jsonl"

    assert (
        elastic_two_node_test.main(
            [
                "verify",
                "--job-dir",
                str(job_dir),
                "--output",
                str(output_path),
                "--expect-nodes",
                "2",
            ]
        )
        == 0
    )
    assert output_path.read_bytes() == expected


def test_dlc_entrypoint_coordinates_two_instances_with_one_command(tmp_path, monkeypatch):
    job_dir = tmp_path / "dlc-job"
    counters = {"prepare": 0, "worker": 0, "verify": 0, "next_shard": 0}
    counter_lock = threading.Lock()

    def fake_prepare(args):
        with counter_lock:
            counters["prepare"] += 1
        time.sleep(0.02)
        (job_dir / "state" / "done").mkdir(parents=True)
        elastic_two_node_test._atomic_write_json(
            job_dir / "manifest.json",
            {
                "num_shards": 4,
                "shards": [{"id": f"shard-{index:05d}"} for index in range(4)],
            },
        )
        return 0

    def fake_worker(args):
        owner = threading.current_thread().name
        with counter_lock:
            counters["worker"] += 1
            first_shard = counters["next_shard"]
            counters["next_shard"] += args.max_shards
        for shard_index in range(first_shard, first_shard + args.max_shards):
            elastic_two_node_test._atomic_write_json(
                job_dir / "state" / "done" / f"shard-{shard_index:05d}.json",
                {"hostname": owner, "status": "done"},
            )
        return 0

    def fake_verify(args):
        with counter_lock:
            counters["verify"] += 1
        owners = {
            elastic_two_node_test._read_json(path)["hostname"] for path in (job_dir / "state" / "done").glob("*.json")
        }
        assert len(owners) == 2
        assert args.expect_nodes == 2
        return 0

    monkeypatch.setattr(elastic_two_node_test, "prepare", fake_prepare)
    monkeypatch.setattr(elastic_two_node_test, "worker", fake_worker)
    monkeypatch.setattr(elastic_two_node_test, "verify", fake_verify)
    command = [
        "dlc",
        "--job-dir",
        str(job_dir),
        "--nodes",
        "2",
        "--num-shards",
        "4",
        "--wait-timeout-secs",
        "2",
        "--poll-interval-secs",
        "0.001",
    ]

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="dlc-node") as pool:
        return_codes = list(pool.map(lambda _: elastic_two_node_test.main(command), range(2)))

    assert return_codes == [0, 0]
    assert counters == {
        "prepare": 1,
        "worker": 2,
        "verify": 1,
        "next_shard": 4,
    }
    coordination_dir = elastic_two_node_test._coordination_dir(job_dir)
    assert elastic_two_node_test._result_code(coordination_dir / "prepare-result.json") == 0
    assert elastic_two_node_test._result_code(coordination_dir / "finalize-result.json") == 0


def test_dlc_entrypoint_propagates_prepare_failure(tmp_path, monkeypatch):
    job_dir = tmp_path / "failed-dlc-job"
    calls = {"prepare": 0}
    counter_lock = threading.Lock()

    def fake_prepare(args):
        with counter_lock:
            calls["prepare"] += 1
        time.sleep(0.02)
        return 2

    def unexpected_worker(args):
        raise AssertionError("worker must not run after preparation fails")

    monkeypatch.setattr(elastic_two_node_test, "prepare", fake_prepare)
    monkeypatch.setattr(elastic_two_node_test, "worker", unexpected_worker)
    command = [
        "dlc",
        "--job-dir",
        str(job_dir),
        "--nodes",
        "2",
        "--num-shards",
        "4",
        "--wait-timeout-secs",
        "2",
        "--poll-interval-secs",
        "0.001",
    ]

    with ThreadPoolExecutor(max_workers=2) as pool:
        return_codes = list(pool.map(lambda _: elastic_two_node_test.main(command), range(2)))

    assert return_codes == [2, 2]
    assert calls["prepare"] == 1


def test_dlc_entrypoint_propagates_worker_failure(tmp_path, monkeypatch):
    job_dir = tmp_path / "worker-failed-dlc-job"
    worker_calls = 0
    counter_lock = threading.Lock()

    def fake_prepare(args):
        (job_dir / "state" / "done").mkdir(parents=True)
        elastic_two_node_test._atomic_write_json(
            job_dir / "manifest.json",
            {
                "num_shards": 4,
                "shards": [{"id": f"shard-{index:05d}"} for index in range(4)],
            },
        )
        return 0

    def fake_worker(args):
        nonlocal worker_calls
        with counter_lock:
            call_index = worker_calls
            worker_calls += 1
        if call_index == 0:
            time.sleep(0.02)
            return 7
        return 0

    def unexpected_verify(args):
        raise AssertionError("verify must not run after a worker fails")

    monkeypatch.setattr(elastic_two_node_test, "prepare", fake_prepare)
    monkeypatch.setattr(elastic_two_node_test, "worker", fake_worker)
    monkeypatch.setattr(elastic_two_node_test, "verify", unexpected_verify)
    command = [
        "dlc",
        "--job-dir",
        str(job_dir),
        "--nodes",
        "2",
        "--num-shards",
        "4",
        "--wait-timeout-secs",
        "2",
        "--poll-interval-secs",
        "0.001",
    ]

    with ThreadPoolExecutor(max_workers=2) as pool:
        return_codes = list(pool.map(lambda _: elastic_two_node_test.main(command), range(2)))

    assert return_codes == [7, 7]
    assert worker_calls == 2
    abort = elastic_two_node_test._read_json(elastic_two_node_test._coordination_dir(job_dir) / "abort.json")
    assert abort["return_code"] == 7

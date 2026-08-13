"""Core executor that turns worker-broadcast launches into one shard job."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Optional

from loguru import logger
from pydantic import PositiveInt

from data_juicer.config import init_configs
from data_juicer.core.executor.base import ExecutorBase

from . import job
from .context import LaunchContext, elastic_mode, launch_context_for_config
from .rendezvous import RendezvousResult, SharedRendezvous
from .safety import ShardabilityReport, analyze_shardability


class ElasticShardingError(RuntimeError):
    pass


def _nested_value(cfg: Any, name: str, default: Any) -> Any:
    elastic_cfg = getattr(cfg, "elastic_sharding", None)
    if elastic_cfg is None:
        return default
    if hasattr(elastic_cfg, "get"):
        return elastic_cfg.get(name, default)
    return getattr(elastic_cfg, name, default)


def _read_json_if_matching(path: Path, fingerprint: str) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict) or value.get("fingerprint") != fingerprint:
        return None
    return value


class ElasticShardingExecutor(ExecutorBase):
    """Coordinate one process per launch rank using a shared POSIX directory.

    Auto mode activates only after a complete rendezvous proves at least two
    distinct hostnames and the pipeline passes conservative record-local
    checks. One rank per hostname runs the durable shard worker. Other ranks
    wait for the single coordinator to atomically publish the merged result.
    """

    def __init__(self, cfg=None):
        self.cfg = init_configs() if cfg is None else cfg
        self.executor_type = "elastic_sharding"

    def _base_executor(self):
        from data_juicer.core.executor.factory import ExecutorFactory

        executor_class = ExecutorFactory.create_executor(self.cfg.executor_type)
        return executor_class(self.cfg)

    def _run_base(self, load_data_np: Optional[PositiveInt], skip_return: bool):
        return self._base_executor().run(load_data_np=load_data_np, skip_return=skip_return)

    def _config_fingerprint(self, context: LaunchContext) -> str:
        digest = hashlib.sha256()
        digest.update(str(context.run_id).encode("utf-8"))
        effective = {
            key: getattr(self.cfg, key, None)
            for key in (
                "executor_type",
                "ray_address",
                "dataset_path",
                "export_path",
                "export_type",
                "export_shard_size",
                "export_in_parallel",
                "data_probe_ratio",
                "custom_operator_paths",
                "process",
            )
        }
        digest.update(json.dumps(effective, sort_keys=True, default=str).encode("utf-8"))
        config_paths = getattr(self.cfg, "config", None) or []
        if config_paths:
            path = Path(str(config_paths[0])).expanduser().resolve()
            try:
                digest.update(path.read_bytes())
            except OSError:
                digest.update(str(path).encode("utf-8"))
        return digest.hexdigest()

    def _coordination_root(self, fingerprint: str) -> Path:
        configured = _nested_value(self.cfg, "coordination_dir", None)
        if configured:
            parent = Path(str(configured).format(work_dir=self.cfg.work_dir, job_id=self.cfg.job_id)).expanduser()
            return (parent / str(self.cfg.job_id) / fingerprint[:12]).resolve()
        return (Path(self.cfg.work_dir) / "elastic_sharding" / fingerprint[:12]).resolve()

    def _wait_for_result(self, path: Path, fingerprint: str) -> dict[str, Any]:
        timeout_secs = int(_nested_value(self.cfg, "result_timeout_secs", 7 * 24 * 60 * 60))
        poll_secs = float(_nested_value(self.cfg, "poll_interval_secs", 5))
        deadline = time.monotonic() + timeout_secs
        while time.monotonic() < deadline:
            result = _read_json_if_matching(path, fingerprint)
            if result is not None:
                return result
            time.sleep(poll_secs)
        raise ElasticShardingError(f"Timed out waiting for elastic-sharding result: {path}")

    @staticmethod
    def _raise_failed_result(result: dict[str, Any]) -> None:
        if int(result.get("return_code", 2)) != 0:
            raise ElasticShardingError(str(result.get("error", "elastic-sharding execution failed")))

    def _run_coordinator_only(
        self,
        rendezvous: RendezvousResult,
        root: Path,
        fingerprint: str,
        reason: str,
        load_data_np: Optional[PositiveInt],
        skip_return: bool,
    ):
        result_path = root / "coordinator-only-result.json"
        previous = _read_json_if_matching(result_path, fingerprint)
        if previous is not None:
            self._raise_failed_result(previous)
            return None

        if self._context.rank == rendezvous.coordinator_rank:
            logger.warning(f"Elastic sharding is not applicable; running once on rank 0: {reason}")
            try:
                value = self._run_base(load_data_np, skip_return)
            except Exception as exc:
                job._atomic_write_json(
                    result_path,
                    {
                        "fingerprint": fingerprint,
                        "return_code": 2,
                        "error": f"Coordinator-only execution failed: {exc}",
                    },
                )
                raise
            job._atomic_write_json(
                result_path,
                {"fingerprint": fingerprint, "return_code": 0, "reason": reason},
            )
            return value

        result = self._wait_for_result(result_path, fingerprint)
        self._raise_failed_result(result)
        return None

    def _recipe_path(self) -> Path:
        paths = getattr(self.cfg, "config", None) or []
        if not paths:
            raise ElasticShardingError("Elastic sharding requires a file-backed --config recipe")
        return Path(str(paths[0])).expanduser().resolve()

    def _select_num_shards(self, report: ShardabilityReport, host_count: int) -> int:
        del report  # reserved for future operator cost-aware sizing
        recipe_path = self._recipe_path()
        recipe = job._load_yaml(recipe_path)
        validation = job._validate_recipe(recipe)
        dataset_path = job._resolve_dataset_path(recipe, str(self.cfg.dataset_path))
        files = job._discover_jsonl_files(dataset_path)
        scan = job._scan_dataset(
            files,
            dataset_path=dataset_path,
            media_keys=validation["media_keys"],
            index_keys=validation["index_keys"],
        )
        shards_per_node = int(_nested_value(self.cfg, "shards_per_node", 2))
        target_size_mb = int(_nested_value(self.cfg, "target_shard_size_mb", 1024))
        max_shards = int(_nested_value(self.cfg, "max_shards", 4096))
        size_shards = math.ceil(scan["total_bytes"] / (target_size_mb * 1024 * 1024))
        desired = max(host_count * shards_per_node, size_shards, 1)
        return min(scan["total_rows"], max_shards, desired)

    def _prepare(self, root: Path, fingerprint: str, report: ShardabilityReport, host_count: int) -> Path:
        prepared_path = root / "prepared.json"
        previous = _read_json_if_matching(prepared_path, fingerprint)
        if previous is not None:
            self._raise_failed_result(previous)
            return Path(previous["job_dir"])

        job_dir = root / "shard-job"
        try:
            num_shards = self._select_num_shards(report, host_count)
            prepare_args = argparse.Namespace(
                config=str(self._recipe_path()),
                dataset_path=str(self.cfg.dataset_path),
                job_dir=str(job_dir),
                num_shards=num_shards,
                lock_timeout_secs=int(_nested_value(self.cfg, "lock_timeout_secs", job.DEFAULT_LOCK_TIMEOUT_SECS)),
                heartbeat_interval_secs=int(
                    _nested_value(self.cfg, "heartbeat_interval_secs", job.DEFAULT_HEARTBEAT_INTERVAL_SECS)
                ),
                max_retries=int(_nested_value(self.cfg, "max_retries", job.DEFAULT_MAX_RETRIES)),
                poll_interval_secs=int(_nested_value(self.cfg, "poll_interval_secs", job.DEFAULT_POLL_INTERVAL_SECS)),
                ray_address="local",
            )
            return_code = job.prepare_job(prepare_args)
            if return_code != 0:
                raise ElasticShardingError(f"Shard preparation returned {return_code}")
            result = {
                "fingerprint": fingerprint,
                "return_code": 0,
                "job_dir": str(job_dir),
                "num_shards": num_shards,
            }
        except Exception as exc:
            result = {"fingerprint": fingerprint, "return_code": 2, "error": f"Shard preparation failed: {exc}"}
            job._atomic_write_json(prepared_path, result)
            raise
        job._atomic_write_json(prepared_path, result)
        return job_dir

    def _wait_for_prepared(self, root: Path, fingerprint: str) -> Path:
        result = self._wait_for_result(root / "prepared.json", fingerprint)
        self._raise_failed_result(result)
        return Path(result["job_dir"])

    def _run_worker(self, job_dir: Path) -> int:
        return job.worker_job(
            argparse.Namespace(
                job_dir=str(job_dir),
                max_shards=None,
                lock_timeout_secs=None,
                heartbeat_interval_secs=None,
                max_retries=None,
                poll_interval_secs=None,
                ray_address=None,
                allow_version_mismatch=False,
            )
        )

    def _wait_for_terminal_job(self, job_dir: Path) -> dict[str, Any]:
        manifest = job._load_manifest(job_dir)
        poll_secs = int(manifest["policy"]["poll_interval_secs"])
        timeout_secs = int(_nested_value(self.cfg, "result_timeout_secs", 7 * 24 * 60 * 60))
        deadline = time.monotonic() + timeout_secs
        while time.monotonic() < deadline:
            status = job._collect_status(
                job_dir,
                manifest,
                timeout_secs=int(manifest["policy"]["lock_timeout_secs"]),
            )
            if status["terminal"]:
                return status
            time.sleep(poll_secs)
        raise ElasticShardingError(f"Timed out waiting for terminal shard job: {job_dir}")

    def _publish_merge(self, root: Path, job_dir: Path, fingerprint: str) -> dict[str, Any]:
        final_path = root / "final-result.json"
        previous = _read_json_if_matching(final_path, fingerprint)
        if previous is not None:
            return previous
        merge_path = job_dir / "merge.json"
        try:
            merge = job._read_json(merge_path)
            output_path = Path(self.cfg.export_path).expanduser().resolve()
            validation = job._validate_jsonl(output_path)
            if (
                Path(merge.get("output_path", "")).resolve() == output_path
                and validation["rows"] == merge.get("rows")
                and validation["sha256"] == merge.get("sha256")
            ):
                recovered = {
                    "fingerprint": fingerprint,
                    "return_code": 0,
                    "output": str(output_path),
                    "num_shards": merge.get("num_shards"),
                    "recovered_merge": True,
                }
                job._atomic_write_json(final_path, recovered)
                return recovered
        except (job.ShardJobError, OSError):
            pass
        try:
            status = self._wait_for_terminal_job(job_dir)
            if not status["complete"]:
                raise ElasticShardingError(
                    f"Shard job failed: done={status['counts']['done']} failed={status['counts']['failed']}"
                )
            return_code = job.merge_job(
                argparse.Namespace(
                    job_dir=str(job_dir),
                    output=str(self.cfg.export_path),
                    lock_timeout_secs=None,
                    heartbeat_interval_secs=None,
                    max_retries=None,
                    poll_interval_secs=None,
                    overwrite=False,
                )
            )
            result = {
                "fingerprint": fingerprint,
                "return_code": return_code,
                "output": str(self.cfg.export_path),
                "num_shards": status["total"],
            }
        except Exception as exc:
            result = {"fingerprint": fingerprint, "return_code": 2, "error": f"Shard finalization failed: {exc}"}
            job._atomic_write_json(final_path, result)
            raise
        job._atomic_write_json(final_path, result)
        return result

    def _run_sharded(
        self,
        rendezvous: RendezvousResult,
        root: Path,
        fingerprint: str,
        report: ShardabilityReport,
    ):
        if self._context.rank == rendezvous.coordinator_rank:
            job_dir = self._prepare(root, fingerprint, report, len(rendezvous.hostnames))
        else:
            job_dir = self._wait_for_prepared(root, fingerprint)

        if rendezvous.is_host_leader(self._context.rank):
            worker_return_code = self._run_worker(job_dir)
            if worker_return_code != 0:
                logger.error(f"Elastic shard worker returned {worker_return_code}; waiting for terminal job state")

        if self._context.rank == rendezvous.coordinator_rank:
            result = self._publish_merge(root, job_dir, fingerprint)
        else:
            result = self._wait_for_result(root / "final-result.json", fingerprint)
        self._raise_failed_result(result)
        logger.info(
            f"Elastic sharding completed {result.get('num_shards')} shards across "
            f"{len(rendezvous.hostnames)} nodes: {result.get('output')}"
        )
        return None

    def run(self, load_data_np: Optional[PositiveInt] = None, skip_return: bool = False):
        mode = elastic_mode(self.cfg)
        try:
            context = launch_context_for_config(self.cfg)
        except ValueError as exc:
            if mode == "on":
                raise ElasticShardingError(str(exc)) from exc
            logger.warning(f"Ignoring invalid distributed launch metadata in auto mode: {exc}")
            return self._run_base(load_data_np, skip_return)
        if context is None or not context.has_stable_run_id:
            if mode == "on":
                raise ElasticShardingError(
                    "elastic_sharding.mode=on requires distributed rank metadata and a stable run_id/job_id"
                )
            return self._run_base(load_data_np, skip_return)

        self._context = context
        fingerprint = self._config_fingerprint(context)
        root = self._coordination_root(fingerprint)
        rendezvous_timeout = int(_nested_value(self.cfg, "rendezvous_timeout_secs", 120))
        poll_interval = float(_nested_value(self.cfg, "rendezvous_poll_interval_secs", 1))
        try:
            rendezvous = SharedRendezvous(
                root,
                context,
                fingerprint=fingerprint,
                timeout_secs=rendezvous_timeout,
                poll_interval_secs=poll_interval,
            ).wait()
        except job.ShardJobError as exc:
            if mode == "on":
                raise
            logger.warning(f"Elastic rendezvous was not completed; retaining ordinary executor behavior: {exc}")
            return self._run_base(load_data_np, skip_return)

        if len(rendezvous.hostnames) < 2:
            return self._run_coordinator_only(
                rendezvous,
                root,
                fingerprint,
                "distributed launch contains only one distinct hostname",
                load_data_np,
                skip_return,
            )

        report = analyze_shardability(self.cfg)
        if not report.eligible:
            explanation = "; ".join(report.reasons)
            if mode == "on":
                raise ElasticShardingError(f"Pipeline is not eligible for elastic sharding: {explanation}")
            return self._run_coordinator_only(
                rendezvous,
                root,
                fingerprint,
                explanation,
                load_data_np,
                skip_return,
            )
        for warning in report.warnings:
            logger.warning(f"Elastic sharding: {warning}")
        logger.info(
            f"Detected {len(rendezvous.hostnames)} nodes and a record-local pipeline; "
            f"starting one shard worker on ranks {rendezvous.host_leader_ranks}"
        )
        return self._run_sharded(rendezvous, root, fingerprint, report)

#!/usr/bin/env python3
"""Shared-filesystem elastic sharding for Data-Juicer processing jobs.

The module provides the installed ``dj-process-sharded`` command as well as
the lower-level prepare/worker/status/retry/merge state machine.  Nodes do not
form a cross-node Ray cluster: shared storage coordinates shard ownership and
each claiming node runs an isolated Data-Juicer Ray executor.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml

SCHEMA_VERSION = 3
DEFAULT_LOCK_TIMEOUT_SECS = 35 * 60 * 60
DEFAULT_MAX_RETRIES = 3
DEFAULT_POLL_INTERVAL_SECS = 20
DEFAULT_WAIT_TIMEOUT_SECS = 35 * 60 * 60
DEFAULT_WAIT_POLL_INTERVAL_SECS = 2.0
RUN_ID_ENV_NAMES = ("PAI_JOB_ID", "DLC_JOB_ID", "JOB_ID")
REMOTE_PATH_PREFIXES = ("http://", "https://", "s3://", "gs://", "hdfs://")

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
PROCESS_DATA_MODULE = "data_juicer.tools.process_data"


class ShardJobError(RuntimeError):
    """An expected, user-facing job error."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ShardJobError(f"Unable to read JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ShardJobError(f"Expected a JSON object in {path}")
    return value


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    """Atomically replace a JSON metadata file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _exclusive_write_json(path: Path, value: dict[str, Any]) -> bool:
    """Publish JSON only when ``path`` does not exist.

    The unique temporary file and hard-link publication keep readers from
    observing partially written metadata and avoid replacing another winner.
    Both files are always on the same shared filesystem.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(tmp_path, path)
            return True
        except FileExistsError:
            return False
    finally:
        tmp_path.unlink(missing_ok=True)


def _write_claim(path: Path, value: dict[str, Any]) -> bool:
    """Create a claim with POSIX O_EXCL semantics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False
    with os.fdopen(fd, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return True


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        raise ShardJobError(f"Unable to read recipe {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise ShardJobError(f"Recipe must contain a YAML mapping: {path}")
    return config


def _git_info() -> dict[str, Any]:
    try:
        commit_result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        status_result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return {"commit": "unknown", "dirty": None}
    commit = commit_result.stdout.strip() if commit_result.returncode == 0 else "unknown"
    dirty = bool(status_result.stdout.strip()) if status_result.returncode == 0 else None
    return {"commit": commit, "dirty": dirty}


def _resolve_config_path(value: str, *, base: Path | None = None) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else ((base or Path.cwd()) / path).resolve()


def _resolve_dataset_path(config: dict[str, Any], override: str | None) -> Path:
    if override:
        dataset_path = Path(override).expanduser().resolve()
    else:
        configured = config.get("dataset_path")
        if not isinstance(configured, str) or not configured.strip():
            raise ShardJobError("Recipe must set a local dataset_path, or prepare must receive --dataset-path")
        dataset_path = _resolve_config_path(configured)
    if not dataset_path.exists():
        raise ShardJobError(f"Dataset path does not exist: {dataset_path}")
    return dataset_path


def _discover_jsonl_files(dataset_path: Path) -> list[Path]:
    if dataset_path.is_file():
        files = [dataset_path] if dataset_path.suffix.lower() == ".jsonl" else []
    else:
        files = sorted(
            (path for path in dataset_path.rglob("*") if path.is_file() and path.suffix.lower() == ".jsonl"),
            key=lambda path: path.relative_to(dataset_path).as_posix(),
        )
    if not files:
        raise ShardJobError(f"No .jsonl files found under {dataset_path}")
    return files


def _validate_recipe(config: dict[str, Any]) -> dict[str, Any]:
    executor_type = config.get("executor_type", "default")
    if executor_type not in {"default", "ray"}:
        raise ShardJobError(
            "Elastic sharding supports recipes for executor_type=default or ray; "
            f"got executor_type={executor_type!r}"
        )

    process = config.get("process")
    if not isinstance(process, list):
        raise ShardJobError("Recipe must define an explicit process list")

    custom_paths = config.get("custom_operator_paths") or config.get("custom-operator-paths") or []
    if isinstance(custom_paths, str):
        custom_paths = [custom_paths]
    if custom_paths:
        from data_juicer.config.config import load_custom_operators

        load_custom_operators([str(_resolve_config_path(path)) for path in custom_paths])

    from data_juicer.ops import OPERATORS, Filter, Mapper

    index_keys: set[str] = set()
    warnings: list[str] = []
    if executor_type == "default":
        warnings.append("recipe executor_type=default will be overridden with executor_type=ray for shard workers")
    for index, op_config in enumerate(process):
        if not isinstance(op_config, dict) or len(op_config) != 1:
            raise ShardJobError(f"process[{index}] must contain exactly one operator")
        op_name, op_args = next(iter(op_config.items()))
        op_args = op_args or {}
        if not isinstance(op_args, dict):
            raise ShardJobError(f"Arguments for operator {op_name!r} must be a mapping")

        op_class = OPERATORS.modules.get(op_name)
        if op_class is None:
            raise ShardJobError(f"Unknown operator {op_name!r}; cannot prove it is shard-safe")
        shard_safe_type = issubclass(op_class, (Mapper, Filter))
        explicitly_global = bool(getattr(op_class, "is_global_operation", False))
        global_name = any(token in op_name.lower() for token in ("deduplicator", "global_", "full_dataset_"))
        if not shard_safe_type or explicitly_global or global_name:
            raise ShardJobError(
                f"Operator {op_name!r} requires or may require whole-dataset semantics "
                "and is not supported by elastic sharding"
            )

        stats_export_path = op_args.get("stats_export_path")
        if stats_export_path:
            raise ShardJobError(
                f"Operator {op_name!r} sets stats_export_path, which would cause cross-shard output collisions"
            )

        index_key = op_args.get("index_key")
        if index_key is not None:
            if not isinstance(index_key, str) or not index_key:
                raise ShardJobError(f"Operator {op_name!r} has an invalid index_key")
            index_keys.add(index_key)

        if op_args.get("save_dir"):
            raise ShardJobError(f"Operator {op_name!r} sets save_dir, which may cause cross-shard output collisions")

    media_keys: list[str] = []
    for config_key, default in (
        ("image_key", "images"),
        ("audio_key", "audios"),
        ("video_key", "videos"),
    ):
        value = config.get(config_key, default)
        if not isinstance(value, str) or not value:
            raise ShardJobError(f"{config_key} must be a non-empty string")
        if value not in media_keys:
            media_keys.append(value)

    return {
        "index_keys": sorted(index_keys),
        "media_keys": media_keys,
        "warnings": warnings,
    }


def _normalize_record(
    record: dict[str, Any],
    *,
    global_index: int,
    media_root: Path,
    media_keys: Iterable[str],
    index_keys: Iterable[str],
) -> bytes:
    for key in media_keys:
        value = record.get(key)
        if value is None:
            continue
        if not isinstance(value, list):
            raise ShardJobError(f"Media field {key!r} must be a list or null")
        normalized_paths: list[Any] = []
        for item in value:
            if not isinstance(item, str):
                raise ShardJobError(f"Every item in media field {key!r} must be a string")
            if not item or item.startswith(REMOTE_PATH_PREFIXES) or Path(item).is_absolute():
                normalized_paths.append(item)
            else:
                normalized_paths.append(str((media_root / item).resolve()))
        record[key] = normalized_paths

    for key in index_keys:
        record.setdefault(key, global_index)

    return (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def _parse_record(raw_line: bytes, path: Path, line_number: int) -> dict[str, Any]:
    if not raw_line.strip():
        raise ShardJobError(f"Blank JSONL record at {path}:{line_number}")
    try:
        text = raw_line.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ShardJobError(f"Invalid UTF-8 at {path}:{line_number}: {exc}") from exc
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ShardJobError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    if not isinstance(value, dict):
        raise ShardJobError(f"JSONL record must be an object at {path}:{line_number}")
    return value


def _scan_dataset(
    files: list[Path],
    *,
    dataset_path: Path,
    media_keys: Iterable[str],
    index_keys: Iterable[str],
) -> dict[str, Any]:
    media_root = dataset_path if dataset_path.is_dir() else dataset_path.parent
    total_rows = 0
    total_bytes = 0
    normalized_digest = hashlib.sha256()
    input_files: list[dict[str, Any]] = []

    for path in files:
        raw_digest = hashlib.sha256()
        file_rows = 0
        with path.open("rb") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                raw_digest.update(raw_line)
                record = _parse_record(raw_line, path, line_number)
                try:
                    normalized = _normalize_record(
                        record,
                        global_index=total_rows,
                        media_root=media_root,
                        media_keys=media_keys,
                        index_keys=index_keys,
                    )
                except ShardJobError as exc:
                    raise ShardJobError(f"{exc} at {path}:{line_number}") from exc
                normalized_digest.update(normalized)
                total_bytes += len(normalized)
                total_rows += 1
                file_rows += 1

        relpath = path.name if dataset_path.is_file() else path.relative_to(dataset_path).as_posix()
        stat = path.stat()
        input_files.append(
            {
                "path": str(path),
                "relative_path": relpath,
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "rows": file_rows,
                "sha256": raw_digest.hexdigest(),
            }
        )

    if total_rows == 0:
        raise ShardJobError("Input dataset contains no records")
    return {
        "total_rows": total_rows,
        "total_bytes": total_bytes,
        "normalized_sha256": normalized_digest.hexdigest(),
        "input_files": input_files,
    }


def _iter_normalized_records(
    files: list[Path],
    *,
    dataset_path: Path,
    media_keys: Iterable[str],
    index_keys: Iterable[str],
) -> Iterator[bytes]:
    media_root = dataset_path if dataset_path.is_dir() else dataset_path.parent
    global_index = 0
    for path in files:
        with path.open("rb") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                record = _parse_record(raw_line, path, line_number)
                try:
                    normalized = _normalize_record(
                        record,
                        global_index=global_index,
                        media_root=media_root,
                        media_keys=media_keys,
                        index_keys=index_keys,
                    )
                except ShardJobError as exc:
                    raise ShardJobError(f"{exc} at {path}:{line_number}") from exc
                yield normalized
                global_index += 1


def _write_shards(
    stage_dir: Path,
    files: list[Path],
    *,
    dataset_path: Path,
    media_keys: Iterable[str],
    index_keys: Iterable[str],
    num_shards: int,
    total_rows: int,
    total_bytes: int,
    expected_sha256: str,
) -> list[dict[str, Any]]:
    shards_dir = stage_dir / "shards"
    shards_dir.mkdir(parents=True)
    width = max(5, len(str(num_shards)))

    shard_metadata: list[dict[str, Any]] = []
    shard_index = 0
    shard_rows = 0
    shard_bytes = 0
    shard_digest = hashlib.sha256()
    complete_digest = hashlib.sha256()
    written_rows = 0
    written_bytes = 0

    def shard_name(index: int) -> str:
        return f"part-{index:0{width}d}-of-{num_shards:0{width}d}"

    current_name = shard_name(shard_index)
    current_path = shards_dir / f"{current_name}.jsonl"
    current_handle = current_path.open("wb")

    def close_current() -> None:
        nonlocal shard_rows, shard_bytes, shard_digest
        current_handle.flush()
        os.fsync(current_handle.fileno())
        current_handle.close()
        shard_metadata.append(
            {
                "id": current_name,
                "index": shard_index,
                "path": current_path.relative_to(stage_dir).as_posix(),
                "rows": shard_rows,
                "size_bytes": shard_bytes,
                "sha256": shard_digest.hexdigest(),
            }
        )
        shard_rows = 0
        shard_bytes = 0
        shard_digest = hashlib.sha256()

    try:
        for normalized in _iter_normalized_records(
            files,
            dataset_path=dataset_path,
            media_keys=media_keys,
            index_keys=index_keys,
        ):
            current_handle.write(normalized)
            shard_digest.update(normalized)
            complete_digest.update(normalized)
            shard_rows += 1
            shard_bytes += len(normalized)
            written_rows += 1
            written_bytes += len(normalized)

            if shard_index >= num_shards - 1:
                continue
            remaining_rows = total_rows - written_rows
            remaining_shards = num_shards - shard_index - 1
            crossed_target = written_bytes * num_shards >= total_bytes * (shard_index + 1)
            must_split = remaining_rows == remaining_shards
            if crossed_target or must_split:
                close_current()
                shard_index += 1
                current_name = shard_name(shard_index)
                current_path = shards_dir / f"{current_name}.jsonl"
                current_handle = current_path.open("wb")
    except Exception:
        if not current_handle.closed:
            current_handle.close()
        raise

    close_current()

    if len(shard_metadata) != num_shards or any(shard["rows"] <= 0 for shard in shard_metadata):
        raise ShardJobError(
            f"Internal partitioning error: expected {num_shards} non-empty shards, got {len(shard_metadata)}"
        )
    if written_rows != total_rows or written_bytes != total_bytes:
        raise ShardJobError("Input changed between the scan and write passes")
    if complete_digest.hexdigest() != expected_sha256:
        raise ShardJobError("Input content changed between the scan and write passes")
    return shard_metadata


def _manifest_path(job_dir: Path) -> Path:
    return job_dir / "manifest.json"


def _load_manifest(job_dir: Path) -> dict[str, Any]:
    manifest = _read_json(_manifest_path(job_dir))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ShardJobError(
            f"Unsupported manifest schema {manifest.get('schema_version')!r}; expected {SCHEMA_VERSION}"
        )
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ShardJobError("Manifest does not contain any shards")
    execution = manifest.get("execution")
    if not isinstance(execution, dict) or execution.get("executor_type") != "ray" or not execution.get("ray_address"):
        raise ShardJobError("Manifest does not contain valid Ray execution settings")
    return manifest


def _existing_job_matches(
    job_dir: Path,
    *,
    dataset_path: Path,
    config_sha256: str,
    num_shards: int,
    ray_address: str,
    files: list[Path],
) -> bool:
    manifest_path = _manifest_path(job_dir)
    if not manifest_path.exists():
        return False
    manifest = _load_manifest(job_dir)
    if (
        manifest.get("dataset_path") != str(dataset_path)
        or manifest.get("recipe", {}).get("sha256") != config_sha256
        or manifest.get("num_shards") != num_shards
        or manifest.get("execution", {}).get("ray_address") != ray_address
        or manifest.get("submission_working_dir", str(REPO_ROOT)) != str(Path.cwd())
    ):
        return False
    saved_files = manifest.get("input_files", [])
    if len(saved_files) != len(files):
        return False
    for saved, current in zip(saved_files, files):
        stat = current.stat()
        if (
            saved.get("path") != str(current)
            or saved.get("size_bytes") != stat.st_size
            or saved.get("mtime_ns") != stat.st_mtime_ns
            or saved.get("sha256") != _sha256_file(current)
        ):
            return False
    return True


def prepare_job(args: argparse.Namespace) -> int:
    config_path = Path(args.config).expanduser().resolve()
    if not config_path.is_file():
        raise ShardJobError(f"Recipe does not exist: {config_path}")
    if args.num_shards <= 0:
        raise ShardJobError("--num-shards must be positive")
    ray_address = args.ray_address.strip()
    if not ray_address:
        raise ShardJobError("--ray-address must not be empty")

    config = _load_yaml(config_path)
    validation = _validate_recipe(config)
    recipe_ray_address = config.get("ray_address")
    if recipe_ray_address is not None and str(recipe_ray_address) != ray_address:
        validation["warnings"].append(
            f"recipe ray_address={recipe_ray_address!r} will be overridden with {ray_address!r}"
        )
    validation["warnings"].append(
        f"elastic sharding is configured with {args.num_shards} shard(s). For best throughput, keep "
        "--num-shards no greater than the number of worker nodes (ideally equal: one shard per node). "
        "A worker processes claimed shards sequentially, and each additional shard starts another "
        "Data-Juicer process; use extra shards only when load balancing or finer retry granularity "
        "justifies the startup overhead"
    )
    dataset_path = _resolve_dataset_path(config, args.dataset_path)
    files = _discover_jsonl_files(dataset_path)
    job_dir = Path(args.job_dir).expanduser().resolve()
    config_sha256 = _sha256_file(config_path)

    if job_dir.exists():
        if _existing_job_matches(
            job_dir,
            dataset_path=dataset_path,
            config_sha256=config_sha256,
            num_shards=args.num_shards,
            ray_address=ray_address,
            files=files,
        ):
            print(f"Job is already prepared and unchanged: {job_dir}")
            return 0
        raise ShardJobError(
            f"Job directory already exists and does not match this request: {job_dir}. " "Choose a new --job-dir."
        )

    scan = _scan_dataset(
        files,
        dataset_path=dataset_path,
        media_keys=validation["media_keys"],
        index_keys=validation["index_keys"],
    )
    if args.num_shards > scan["total_rows"]:
        raise ShardJobError(f"--num-shards ({args.num_shards}) exceeds input records ({scan['total_rows']})")

    job_dir.parent.mkdir(parents=True, exist_ok=True)
    stage_dir = job_dir.parent / f".{job_dir.name}.preparing.{uuid.uuid4().hex}"
    stage_dir.mkdir(mode=0o755)
    try:
        recipe_snapshot = stage_dir / "recipe.yaml"
        shutil.copy2(config_path, recipe_snapshot)
        shards = _write_shards(
            stage_dir,
            files,
            dataset_path=dataset_path,
            media_keys=validation["media_keys"],
            index_keys=validation["index_keys"],
            num_shards=args.num_shards,
            total_rows=scan["total_rows"],
            total_bytes=scan["total_bytes"],
            expected_sha256=scan["normalized_sha256"],
        )

        for relative_dir in (
            "state/locks",
            "state/done",
            "state/failed",
            "state/stale_locks",
            "state/history/failed",
            "state/history/attempts",
            "state/history/claims",
            "attempts",
        ):
            (stage_dir / relative_dir).mkdir(parents=True, exist_ok=True)

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "job_id": job_dir.name,
            "created_at": _utc_now(),
            "submission_working_dir": str(Path.cwd()),
            "dataset_path": str(dataset_path),
            "media_root": str(dataset_path if dataset_path.is_dir() else dataset_path.parent),
            "media_keys": validation["media_keys"],
            "index_keys": validation["index_keys"],
            "input_files": scan["input_files"],
            "total_rows": scan["total_rows"],
            "total_size_bytes": scan["total_bytes"],
            "normalized_sha256": scan["normalized_sha256"],
            "num_shards": args.num_shards,
            "shards": shards,
            "recipe": {
                "source_path": str(config_path),
                "snapshot_path": "recipe.yaml",
                "sha256": config_sha256,
            },
            "data_juicer": _git_info(),
            "execution": {
                "executor_type": "ray",
                "ray_address": ray_address,
                "recipe_executor_type": config.get("executor_type", "default"),
            },
            "policy": {
                "lock_timeout_secs": args.lock_timeout_secs,
                "max_retries": args.max_retries,
                "poll_interval_secs": args.poll_interval_secs,
            },
            "warnings": validation["warnings"],
        }
        _atomic_write_json(stage_dir / "manifest.json", manifest)
        os.rename(stage_dir, job_dir)
    except Exception:
        shutil.rmtree(stage_dir, ignore_errors=True)
        raise

    print(
        f"Prepared {args.num_shards} shards from {scan['total_rows']} records "
        f"({scan['total_bytes']} normalized bytes) in {job_dir}"
    )
    for warning in validation["warnings"]:
        print(f"WARNING: {warning}", file=sys.stderr)
    return 0


def _job_path(job_dir: Path, relative_path: str) -> Path:
    resolved = (job_dir / relative_path).resolve()
    try:
        resolved.relative_to(job_dir.resolve())
    except ValueError as exc:
        raise ShardJobError(f"Manifest path escapes job directory: {relative_path}") from exc
    return resolved


def _lock_path(job_dir: Path, shard_id: str) -> Path:
    return job_dir / "state" / "locks" / f"{shard_id}.lock"


def _done_path(job_dir: Path, shard_id: str) -> Path:
    return job_dir / "state" / "done" / f"{shard_id}.json"


def _failed_path(job_dir: Path, shard_id: str) -> Path:
    return job_dir / "state" / "failed" / f"{shard_id}.json"


def _attempt_directories(job_dir: Path, shard_id: str) -> list[Path]:
    attempt_root = job_dir / "attempts" / shard_id
    return sorted((path for path in attempt_root.glob("*") if path.is_dir()), key=lambda path: path.name)


def _attempt_failures(job_dir: Path, shard_id: str) -> int:
    failures = 0
    for directory in _attempt_directories(job_dir, shard_id):
        metadata_path = directory / "attempt.json"
        if not metadata_path.exists():
            continue
        try:
            status = _read_json(metadata_path).get("status")
        except ShardJobError:
            continue
        if status in {"failed", "stale"}:
            failures += 1
    return failures


def _read_lock_fd(fd: int) -> dict[str, Any]:
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        chunks = []
        while True:
            chunk = os.read(fd, 64 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        value = json.loads(b"".join(chunks).decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_lock_fd(fd: int, value: dict[str, Any]) -> None:
    """Rewrite a claim in place while preserving its inode and path.

    A successful claim becomes a permanent terminal fence.  Replacing the
    file with ``os.replace`` would briefly decouple the advisory lock from the
    visible inode, so terminal transitions use the already locked descriptor.
    """
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    os.lseek(fd, 0, os.SEEK_SET)
    os.ftruncate(fd, 0)
    written = 0
    while written < len(payload):
        written += os.write(fd, payload[written:])
    os.fsync(fd)


def _read_claim(lock_path: Path) -> dict[str, Any] | None:
    """Read one claim under a shared advisory lock.

    ``None`` means the path did not exist.  An empty mapping means the file was
    present but malformed; callers must treat that conservatively and must not
    reclaim it automatically.
    """
    try:
        fd = os.open(lock_path, os.O_RDONLY)
    except FileNotFoundError:
        return None
    try:
        fcntl.flock(fd, fcntl.LOCK_SH)
        return _read_lock_fd(fd)
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _lock_fd_still_at_path(fd: int, lock_path: Path) -> bool:
    try:
        fd_stat = os.fstat(fd)
        path_stat = lock_path.stat()
    except FileNotFoundError:
        return False
    return (fd_stat.st_dev, fd_stat.st_ino) == (path_stat.st_dev, path_stat.st_ino)


def _release_lock(lock_path: Path, token: str) -> bool:
    """Remove only the lock identified by ``token``.

    The short advisory lock serializes release with stale-lock takeover. The
    claim itself still uses O_EXCL and does not keep an advisory lock open
    during shard processing.
    """
    try:
        fd = os.open(lock_path, os.O_RDONLY)
    except FileNotFoundError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        if not _lock_fd_still_at_path(fd, lock_path):
            return False
        metadata = _read_lock_fd(fd)
        if metadata.get("token") != token:
            return False
        if metadata.get("status", "running") != "running":
            return False
        lock_path.unlink()
        return True
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _mark_attempt_stale(job_dir: Path, lock_metadata: dict[str, Any]) -> None:
    relative_attempt = lock_metadata.get("attempt_dir")
    if not isinstance(relative_attempt, str):
        return
    attempt_dir = _job_path(job_dir, relative_attempt)
    metadata_path = attempt_dir / "attempt.json"
    if not metadata_path.exists():
        return
    try:
        metadata = _read_json(metadata_path)
    except ShardJobError:
        return
    if metadata.get("status") == "running":
        metadata.update({"status": "stale", "stale_at": _utc_now()})
        _atomic_write_json(metadata_path, metadata)


def _reclaim_stale_lock(job_dir: Path, lock_path: Path, timeout_secs: int) -> bool:
    try:
        fd = os.open(lock_path, os.O_RDONLY)
    except FileNotFoundError:
        return True
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        if not _lock_fd_still_at_path(fd, lock_path):
            return True
        lock_metadata = _read_lock_fd(fd)
        # Invalid claims and terminal fences are never reclaimed implicitly.
        # Keeping the path present is what prevents a client with a stale
        # negative cache for state/done from creating a duplicate claim.
        if not lock_metadata.get("token"):
            return False
        if lock_metadata.get("status", "running") != "running":
            return False
        if time.time() - os.fstat(fd).st_mtime <= timeout_secs:
            return False
        stale_path = job_dir / "state" / "stale_locks" / f"{lock_path.stem}.{int(time.time())}.{uuid.uuid4().hex}.lock"
        # Update the retry counter before removing the visible lock, so the
        # next claimant cannot miss this expired attempt.
        _mark_attempt_stale(job_dir, lock_metadata)
        os.rename(lock_path, stale_path)
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    return True


def _publish_failure_and_seal_claim(
    job_dir: Path,
    shard_id: str,
    token: str,
    failure_metadata: dict[str, Any],
) -> bool:
    """Publish terminal failure and retain the claim as a durable fence."""
    lock_path = _lock_path(job_dir, shard_id)
    try:
        fd = os.open(lock_path, os.O_RDWR)
    except FileNotFoundError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        if not _lock_fd_still_at_path(fd, lock_path):
            return False
        claim = _read_lock_fd(fd)
        if claim.get("token") != token:
            return False
        published = False
        done_exists = _done_path(job_dir, shard_id).exists()
        if not done_exists:
            published = _exclusive_write_json(
                _failed_path(job_dir, shard_id),
                {**failure_metadata, "token": token},
            )
        _write_lock_fd(
            fd,
            {
                **claim,
                "status": "failed" if published else "superseded",
                "terminal_at": _utc_now(),
                "terminal_path": (
                    f"state/failed/{shard_id}.json" if published or not done_exists else f"state/done/{shard_id}.json"
                ),
            },
        )
        return True
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _publish_done_and_seal_claim(
    job_dir: Path,
    shard_id: str,
    token: str,
    done_metadata: dict[str, Any],
) -> tuple[bool, bool]:
    """Publish success and retain the claim as a durable terminal fence.

    Returns ``(owned, published)``. The ownership check, done publication,
    and terminal transition share one advisory-lock critical section so an
    expired attempt cannot publish after a replacement has claimed the shard.
    The claim path is deliberately not removed: its continued existence makes
    a later ``O_CREAT|O_EXCL`` fail at the shared-filesystem server even when a
    client temporarily has a stale negative lookup for the done file.
    """
    lock_path = _lock_path(job_dir, shard_id)
    try:
        fd = os.open(lock_path, os.O_RDWR)
    except FileNotFoundError:
        return False, False
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        if not _lock_fd_still_at_path(fd, lock_path):
            return False, False
        claim = _read_lock_fd(fd)
        if claim.get("token") != token:
            return False, False
        published = False
        failed_exists = _failed_path(job_dir, shard_id).exists()
        if not failed_exists:
            published = _exclusive_write_json(
                _done_path(job_dir, shard_id),
                done_metadata,
            )
        _write_lock_fd(
            fd,
            {
                **claim,
                "status": "done" if published else "superseded",
                "terminal_at": _utc_now(),
                "terminal_path": (
                    f"state/done/{shard_id}.json" if published or not failed_exists else f"state/failed/{shard_id}.json"
                ),
            },
        )
        return True, published
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _create_claim(
    job_dir: Path,
    shard: dict[str, Any],
    *,
    timeout_secs: int,
    max_retries: int,
) -> dict[str, Any] | None:
    shard_id = shard["id"]
    if _done_path(job_dir, shard_id).exists() or _failed_path(job_dir, shard_id).exists():
        return None

    lock_path = _lock_path(job_dir, shard_id)
    if lock_path.exists() and not _reclaim_stale_lock(job_dir, lock_path, timeout_secs):
        return None
    if _done_path(job_dir, shard_id).exists() or _failed_path(job_dir, shard_id).exists():
        return None

    attempt_directories = _attempt_directories(job_dir, shard_id)
    attempt_statuses: list[str | None] = []
    for directory in attempt_directories:
        metadata_path = directory / "attempt.json"
        if not metadata_path.exists():
            attempt_statuses.append(None)
            continue
        try:
            attempt_statuses.append(_read_json(metadata_path).get("status"))
        except ShardJobError:
            attempt_statuses.append(None)

    # Defense in depth for shared filesystems whose directory-entry caches can
    # temporarily hide both terminal markers and a recently changed claim.
    # A new attempt is valid only after every prior attempt is explicitly
    # failed/stale.  In particular, running/done/superseded/unknown histories
    # must never trigger another expensive Data-Juicer process.
    if attempt_statuses and any(status not in {"failed", "stale"} for status in attempt_statuses):
        return None

    failure_count = sum(status in {"failed", "stale"} for status in attempt_statuses)
    # ``max_retries`` counts retries after the initial attempt. For example,
    # max_retries=3 permits four total failed attempts before terminal state.
    if failure_count > max_retries:
        # A Worker may have died after recording the last failed attempt but
        # before sealing its running claim. Stale takeover removes that old
        # claim, so rebuild a terminal fence before publishing state/failed.
        # The order is intentional: a temporarily invisible failed marker is
        # safe while this O_EXCL path still prevents any new owner.
        token = uuid.uuid4().hex
        terminal_at = _utc_now()
        terminal_claim = {
            "shard_id": shard_id,
            "token": token,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "claimed_at": terminal_at,
            "claimed_at_epoch": time.time(),
            "attempt_number": len(attempt_directories),
            "status": "failed",
            "terminal_at": terminal_at,
            "terminal_path": f"state/failed/{shard_id}.json",
            "reason": "retry limit reached before claim",
        }
        if _write_claim(lock_path, terminal_claim):
            _exclusive_write_json(
                _failed_path(job_dir, shard_id),
                {
                    "shard_id": shard_id,
                    "status": "failed",
                    "failed_at": terminal_at,
                    "failures": failure_count,
                    "reason": "retry limit reached before claim",
                    "token": token,
                },
            )
        return None

    attempt_number = len(attempt_directories) + 1
    token = uuid.uuid4().hex
    attempt_name = f"{attempt_number:04d}-{token}"
    attempt_dir = job_dir / "attempts" / shard_id / attempt_name
    attempt_dir.mkdir(parents=True, exist_ok=False)
    claim = {
        "shard_id": shard_id,
        "token": token,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "claimed_at": _utc_now(),
        "claimed_at_epoch": time.time(),
        "attempt_number": attempt_number,
        "attempt_dir": attempt_dir.relative_to(job_dir).as_posix(),
        "status": "running",
    }
    if not _write_claim(lock_path, claim):
        attempt_dir.rmdir()
        return None
    _atomic_write_json(
        attempt_dir / "attempt.json",
        {
            **claim,
            "status": "running",
            "input_path": shard["path"],
        },
    )
    return claim


def _validate_jsonl(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    rows = 0
    size_bytes = 0
    try:
        with path.open("rb") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                digest.update(raw_line)
                size_bytes += len(raw_line)
                _parse_record(raw_line, path, line_number)
                rows += 1
    except OSError as exc:
        raise ShardJobError(f"Unable to read result {path}: {exc}") from exc
    return {"rows": rows, "size_bytes": size_bytes, "sha256": digest.hexdigest()}


def _materialize_ray_output(
    ray_export_path: Path,
    output_path: Path,
    job_dir: Path,
) -> dict[str, Any]:
    """Combine Ray's output directory into one deterministic JSONL result."""
    if ray_export_path.is_file():
        source_files = [ray_export_path]
    elif ray_export_path.is_dir():
        source_files = sorted(
            (path for path in ray_export_path.rglob("*") if path.is_file() and not path.name.startswith((".", "_"))),
            key=lambda path: path.relative_to(ray_export_path).as_posix(),
        )
    else:
        raise ShardJobError(f"Ray did not create its export path: {ray_export_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f".{output_path.name}.{uuid.uuid4().hex}.tmp")
    source_metadata: list[dict[str, Any]] = []
    try:
        with tmp_path.open("wb") as output_handle:
            for source_path in source_files:
                source_digest = hashlib.sha256()
                source_rows = 0
                source_bytes = 0
                with source_path.open("rb") as input_handle:
                    for line_number, raw_line in enumerate(input_handle, start=1):
                        _parse_record(raw_line, source_path, line_number)
                        source_digest.update(raw_line)
                        source_rows += 1
                        source_bytes += len(raw_line)
                        output_handle.write(raw_line)
                        if not raw_line.endswith(b"\n"):
                            output_handle.write(b"\n")
                source_metadata.append(
                    {
                        "path": source_path.relative_to(job_dir).as_posix(),
                        "rows": source_rows,
                        "size_bytes": source_bytes,
                        "sha256": source_digest.hexdigest(),
                    }
                )
            output_handle.flush()
            os.fsync(output_handle.fileno())
        os.replace(tmp_path, output_path)
    except OSError as exc:
        raise ShardJobError(f"Unable to materialize Ray output {ray_export_path}: {exc}") from exc
    finally:
        tmp_path.unlink(missing_ok=True)
    return {"ray_output_files": source_metadata}


def _current_commit_matches(manifest: dict[str, Any], allow_mismatch: bool) -> None:
    expected = manifest.get("data_juicer", {}).get("commit", "unknown")
    current = _git_info().get("commit", "unknown")
    if expected == "unknown" or current == "unknown" or expected == current:
        return
    message = f"Data-Juicer commit mismatch: job={expected}, worker={current}"
    if allow_mismatch:
        print(f"WARNING: {message}", file=sys.stderr)
    else:
        raise ShardJobError(message + "; use --allow-version-mismatch only if this is intentional")


def _build_process_command(
    job_dir: Path,
    manifest: dict[str, Any],
    shard: dict[str, Any],
    claim: dict[str, Any],
) -> tuple[list[str], Path, Path, Path, Path]:
    attempt_dir = _job_path(job_dir, claim["attempt_dir"])
    input_path = _job_path(job_dir, shard["path"])
    recipe_path = _job_path(job_dir, manifest["recipe"]["snapshot_path"])
    output_path = attempt_dir / "processed.jsonl"
    ray_export_path = attempt_dir / "ray-output.jsonl"
    attempt_job_id = f"{manifest['job_id']}_{shard['id']}_{claim['token'][:12]}"
    work_base = attempt_dir / "work"
    resolved_work_dir = work_base / attempt_job_id
    command = [
        sys.executable,
        "-m",
        PROCESS_DATA_MODULE,
        "--config",
        str(recipe_path),
        "--dataset_path",
        str(input_path),
        "--export_path",
        str(ray_export_path),
        "--executor_type",
        "ray",
        "--ray_address",
        manifest["execution"]["ray_address"],
        "--work_dir",
        str(work_base),
        "--job_id",
        attempt_job_id,
        "--event_log_dir",
        str(attempt_dir / "logs"),
        "--checkpoint_dir",
        str(attempt_dir / "checkpoints"),
        "--partition_dir",
        str(attempt_dir / "partitions"),
    ]
    return command, ray_export_path, output_path, attempt_dir, resolved_work_dir


def _process_claim(
    job_dir: Path,
    manifest: dict[str, Any],
    shard: dict[str, Any],
    claim: dict[str, Any],
) -> str:
    shard_id = shard["id"]
    token = claim["token"]
    lock_path = _lock_path(job_dir, shard_id)
    command, ray_export_path, output_path, attempt_dir, resolved_work_dir = _build_process_command(
        job_dir, manifest, shard, claim
    )
    log_path = attempt_dir / "process.log"
    metadata_path = attempt_dir / "attempt.json"
    metadata = _read_json(metadata_path)
    metadata["command"] = command
    metadata["ray_export_path"] = ray_export_path.relative_to(job_dir).as_posix()
    _atomic_write_json(metadata_path, metadata)

    env = os.environ.copy()
    env["DJ_PRODUCED_DATA_DIR"] = str(attempt_dir / "produced")
    # Worker hosts commonly mount the home directory read-only. Reuse a
    # job-local cache unless the caller explicitly selected cache locations.
    cache_dir = Path(env.get("XDG_CACHE_HOME", job_dir / "cache"))
    env.setdefault("XDG_CACHE_HOME", str(cache_dir))
    env.setdefault("HF_HOME", str(cache_dir / "huggingface"))
    env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    # Data-Juicer backs up the recipe into the resolved work directory before
    # all configurations guarantee that directory exists.
    resolved_work_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        with log_path.open("w", encoding="utf-8") as log_handle:
            result = subprocess.run(
                command,
                cwd=Path(manifest.get("submission_working_dir", REPO_ROOT)),
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        return_code = result.returncode
    except OSError as exc:
        return_code = -1
        with log_path.open("a", encoding="utf-8") as log_handle:
            log_handle.write(f"\nUnable to start Data-Juicer: {exc}\n")

    validation: dict[str, Any] | None = None
    error: str | None = None
    if return_code == 0 and ray_export_path.exists():
        try:
            ray_metadata = _materialize_ray_output(ray_export_path, output_path, job_dir)
            validation = _validate_jsonl(output_path)
            validation.update(ray_metadata)
        except ShardJobError as exc:
            error = str(exc)
    elif return_code == 0:
        error = f"Data-Juicer returned 0 but did not create {ray_export_path}"
    else:
        error = f"Data-Juicer returned exit code {return_code}"

    metadata = _read_json(metadata_path)
    metadata.update(
        {
            "finished_at": _utc_now(),
            "duration_secs": time.time() - started,
            "return_code": return_code,
            "output_path": output_path.relative_to(job_dir).as_posix(),
        }
    )

    if validation is not None:
        done_metadata = {
            "shard_id": shard_id,
            "status": "done",
            "completed_at": _utc_now(),
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "token": token,
            "attempt_number": claim["attempt_number"],
            "output_path": output_path.relative_to(job_dir).as_posix(),
            "executor_type": "ray",
            "ray_address": manifest["execution"]["ray_address"],
            **validation,
            "data_juicer_commit": _git_info().get("commit", "unknown"),
        }
        owned, published = _publish_done_and_seal_claim(
            job_dir,
            shard_id,
            token,
            done_metadata,
        )
        if published:
            attempt_status = "done"
        elif owned:
            attempt_status = "superseded"
        else:
            latest_metadata = _read_json(metadata_path)
            attempt_status = "stale" if latest_metadata.get("status") == "stale" else "lost"
            if latest_metadata.get("stale_at") is not None:
                metadata["stale_at"] = latest_metadata["stale_at"]
        metadata.update(
            {
                "status": attempt_status,
                "result": validation,
            }
        )
        _atomic_write_json(metadata_path, metadata)
        if published:
            return "done"
        return "superseded" if owned else "lost"

    metadata.update({"status": "failed", "error": error})
    _atomic_write_json(metadata_path, metadata)

    failures = _attempt_failures(job_dir, shard_id)
    max_retries = int(manifest["policy"]["max_retries"])
    if failures > max_retries:
        owned = _publish_failure_and_seal_claim(
            job_dir,
            shard_id,
            token,
            {
                "shard_id": shard_id,
                "status": "failed",
                "failed_at": _utc_now(),
                "failures": failures,
                "last_error": error,
                "last_attempt": claim["attempt_number"],
                "last_log": log_path.relative_to(job_dir).as_posix(),
                "token": token,
            },
        )
    else:
        owned = _release_lock(lock_path, token)
    return "failed" if owned else "lost"


def _effective_policy(manifest: dict[str, Any], args: argparse.Namespace) -> dict[str, int]:
    saved = manifest.get("policy", {})
    return {
        "lock_timeout_secs": int(
            args.lock_timeout_secs
            if getattr(args, "lock_timeout_secs", None) is not None
            else saved.get("lock_timeout_secs", DEFAULT_LOCK_TIMEOUT_SECS)
        ),
        "max_retries": int(
            args.max_retries
            if getattr(args, "max_retries", None) is not None
            else saved.get("max_retries", DEFAULT_MAX_RETRIES)
        ),
        "poll_interval_secs": int(
            args.poll_interval_secs
            if getattr(args, "poll_interval_secs", None) is not None
            else saved.get("poll_interval_secs", DEFAULT_POLL_INTERVAL_SECS)
        ),
    }


def _effective_execution(manifest: dict[str, Any], args: argparse.Namespace) -> dict[str, str]:
    saved = manifest.get("execution", {})
    ray_address = (
        args.ray_address if getattr(args, "ray_address", None) is not None else saved.get("ray_address", "local")
    )
    ray_address = str(ray_address).strip()
    if not ray_address:
        raise ShardJobError("--ray-address must not be empty")
    return {"executor_type": "ray", "ray_address": ray_address}


def _collect_status(
    job_dir: Path,
    manifest: dict[str, Any],
    *,
    timeout_secs: int,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    counts = {
        "pending": 0,
        "running": 0,
        "stale": 0,
        "committing": 0,
        "conflict": 0,
        "done": 0,
        "failed": 0,
    }
    now = time.time()
    for shard in manifest["shards"]:
        shard_id = shard["id"]
        done_path = _done_path(job_dir, shard_id)
        failed_path = _failed_path(job_dir, shard_id)
        lock_path = _lock_path(job_dir, shard_id)
        done_exists = done_path.exists()
        failed_exists = failed_path.exists()
        record: dict[str, Any] = {"shard_id": shard_id, "rows": shard["rows"]}
        claim = _read_claim(lock_path)
        claim_status = None
        if claim is not None:
            claim_status = claim.get("status", "running")
            record.update(
                {
                    "claim_status": claim_status,
                    "claim_token": claim.get("token"),
                    "owner": claim.get("hostname"),
                    "pid": claim.get("pid"),
                }
            )

        if done_exists and failed_exists:
            status = "conflict"
            record["conflict_reason"] = "done and failed markers both exist"
        elif done_exists:
            try:
                done = _read_json(done_path)
                record.update({"owner": done.get("hostname"), "done_token": done.get("token")})
            except ShardJobError:
                done = {}
                record["metadata_error"] = True
            if claim is None:
                status = "conflict"
                record["conflict_reason"] = "done marker has no durable terminal claim"
            elif claim_status not in {"done", "superseded"}:
                status = "conflict"
                record["conflict_reason"] = f"done marker coexists with claim_status={claim_status!r}"
            elif claim_status == "done" and done.get("token") != claim.get("token"):
                status = "conflict"
                record["conflict_reason"] = "done token differs from terminal claim token"
            elif claim_status == "superseded" and claim.get("terminal_path") != f"state/done/{shard_id}.json":
                status = "conflict"
                record["conflict_reason"] = "superseded claim points to a different terminal marker"
            else:
                status = "done"
        elif failed_exists:
            try:
                failed_metadata = _read_json(failed_path)
                record.update({"error": failed_metadata.get("last_error")})
            except ShardJobError:
                failed_metadata = {}
                record["metadata_error"] = True
            if claim is None:
                status = "conflict"
                record["conflict_reason"] = "failed marker has no durable terminal claim"
            elif claim_status not in {"failed", "superseded"}:
                status = "conflict"
                record["conflict_reason"] = f"failed marker coexists with claim_status={claim_status!r}"
            elif claim_status == "failed" and failed_metadata.get("token") != claim.get("token"):
                status = "conflict"
                record["conflict_reason"] = "failed token differs from terminal claim token"
            elif claim_status == "superseded" and claim.get("terminal_path") != f"state/failed/{shard_id}.json":
                status = "conflict"
                record["conflict_reason"] = "superseded claim points to a different terminal marker"
            else:
                status = "failed"
        elif claim is not None:
            if not claim:
                status = "conflict"
                record["conflict_reason"] = "claim file is malformed"
            elif claim_status in {"done", "failed", "superseded"}:
                # The claim transition is durable, but this client can still
                # have a short-lived negative cache for the terminal marker.
                status = "committing"
            elif claim_status != "running":
                status = "conflict"
                record["conflict_reason"] = f"unknown claim_status={claim_status!r}"
            else:
                try:
                    age = max(0.0, now - lock_path.stat().st_mtime)
                except FileNotFoundError:
                    age = 0.0
                status = "stale" if age > timeout_secs else "running"
                record["lock_age_secs"] = age
        elif lock_path.exists():
            # The path appeared after _read_claim. Treat the inconsistent view
            # as a conflict instead of calling it pending and racing a claim.
            status = "conflict"
            record["conflict_reason"] = "claim appeared during status collection"
        else:
            status = "pending"

        # Preserve metadata diagnostics for a claim that disappeared between
        # open/stat calls without ever reporting the shard as safely complete.
        if claim is not None and claim and claim.get("token") is None:
            status = "conflict"
            record["conflict_reason"] = "claim has no token"
        record["status"] = status
        counts[status] += 1
        records.append(record)

    return {
        "job_id": manifest["job_id"],
        "job_dir": str(job_dir),
        "execution": manifest["execution"],
        "total": len(records),
        "counts": counts,
        "complete": counts["done"] == len(records),
        "terminal": counts["done"] + counts["failed"] == len(records),
        "shards": records,
    }


def worker_job(args: argparse.Namespace) -> int:
    job_dir = Path(args.job_dir).expanduser().resolve()
    manifest = _load_manifest(job_dir)
    policy = _effective_policy(manifest, args)
    execution = _effective_execution(manifest, args)
    manifest["policy"].update(policy)
    manifest["execution"].update(execution)
    _current_commit_matches(manifest, args.allow_version_mismatch)

    expected_recipe_sha = manifest["recipe"]["sha256"]
    recipe_path = _job_path(job_dir, manifest["recipe"]["snapshot_path"])
    if _sha256_file(recipe_path) != expected_recipe_sha:
        raise ShardJobError("Recipe snapshot hash does not match manifest")

    processed = 0
    failed = 0
    hostname = socket.gethostname()
    print(f"Worker {hostname}:{os.getpid()} started for {manifest['job_id']}")

    while True:
        if args.max_shards is not None and processed >= args.max_shards:
            print(f"Reached --max-shards={args.max_shards}")
            return 2 if failed else 0

        claim: dict[str, Any] | None = None
        selected_shard: dict[str, Any] | None = None
        for shard in manifest["shards"]:
            claim = _create_claim(
                job_dir,
                shard,
                timeout_secs=policy["lock_timeout_secs"],
                max_retries=policy["max_retries"],
            )
            if claim is not None:
                selected_shard = shard
                break

        if claim is not None and selected_shard is not None:
            print(f"Claimed {selected_shard['id']} (attempt {claim['attempt_number']})")
            result = _process_claim(job_dir, manifest, selected_shard, claim)
            processed += 1
            if result == "failed":
                failed += 1
            print(f"Finished {selected_shard['id']}: {result}")
            continue

        status = _collect_status(
            job_dir,
            manifest,
            timeout_secs=policy["lock_timeout_secs"],
        )
        counts = status["counts"]
        if status["complete"]:
            print(f"All {status['total']} shards completed")
            return 0
        if counts["conflict"]:
            print(
                f"Job state conflict: conflict={counts['conflict']} done={counts['done']} "
                f"failed={counts['failed']}",
                file=sys.stderr,
            )
            return 2
        if status["terminal"]:
            print(
                f"Job finished with failures: done={counts['done']} failed={counts['failed']}",
                file=sys.stderr,
            )
            return 2
        print(
            "No shard claim available; "
            f"pending={counts['pending']} running={counts['running']} stale={counts['stale']} "
            f"committing={counts['committing']} conflict={counts['conflict']}. "
            f"Waiting {policy['poll_interval_secs']}s..."
        )
        time.sleep(policy["poll_interval_secs"])


def status_job(args: argparse.Namespace) -> int:
    job_dir = Path(args.job_dir).expanduser().resolve()
    manifest = _load_manifest(job_dir)
    policy = _effective_policy(manifest, args)
    status = _collect_status(job_dir, manifest, timeout_secs=policy["lock_timeout_secs"])
    if args.json:
        print(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        counts = status["counts"]
        print(
            f"Job {status['job_id']}: executor={status['execution']['executor_type']} "
            f"ray_address={status['execution']['ray_address']} total={status['total']} "
            f"pending={counts['pending']} running={counts['running']} stale={counts['stale']} "
            f"committing={counts['committing']} conflict={counts['conflict']} "
            f"done={counts['done']} failed={counts['failed']}"
        )
        if args.all:
            for shard in status["shards"]:
                owner = f" owner={shard.get('owner')}" if shard.get("owner") else ""
                print(f"  {shard['shard_id']}: {shard['status']}{owner}")
    return 0 if status["complete"] else (2 if status["counts"]["failed"] or status["counts"]["conflict"] else 1)


def retry_job(args: argparse.Namespace) -> int:
    job_dir = Path(args.job_dir).expanduser().resolve()
    manifest = _load_manifest(job_dir)
    valid_ids = {shard["id"] for shard in manifest["shards"]}
    selected_ids = set(args.shard_id or [])
    targets = sorted(
        shard_id
        for shard_id in valid_ids
        if _failed_path(job_dir, shard_id).exists() and (args.all_failed or shard_id in selected_ids)
    )
    if not args.all_failed:
        unknown = selected_ids - valid_ids
        if unknown:
            raise ShardJobError(f"Unknown shard IDs: {', '.join(sorted(unknown))}")
    if not targets:
        print("No failed shards selected")
        return 0

    history_failed = job_dir / "state" / "history" / "failed"
    history_attempts = job_dir / "state" / "history" / "attempts"
    history_claims = job_dir / "state" / "history" / "claims"
    history_failed.mkdir(parents=True, exist_ok=True)
    history_attempts.mkdir(parents=True, exist_ok=True)
    history_claims.mkdir(parents=True, exist_ok=True)

    # Validate the full selection before moving anything, so a bad target
    # cannot leave a multi-shard retry request only partially applied.
    for shard_id in targets:
        if _done_path(job_dir, shard_id).exists():
            raise ShardJobError(f"Cannot retry {shard_id}: it is already complete")
        claim = _read_claim(_lock_path(job_dir, shard_id))
        if claim is None:
            raise ShardJobError(f"Cannot retry {shard_id}: its terminal claim is missing")
        if claim.get("status") not in {"failed", "superseded"}:
            raise ShardJobError(f"Cannot retry {shard_id}: claim status is {claim.get('status', 'unknown')!r}")

    for shard_id in targets:
        suffix = f"{int(time.time())}.{uuid.uuid4().hex}"
        failed_path = _failed_path(job_dir, shard_id)
        lock_path = _lock_path(job_dir, shard_id)
        try:
            fd = os.open(lock_path, os.O_RDONLY)
        except FileNotFoundError as exc:
            raise ShardJobError(f"Cannot retry {shard_id}: its terminal claim disappeared") from exc
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            if not _lock_fd_still_at_path(fd, lock_path):
                raise ShardJobError(f"Cannot retry {shard_id}: its terminal claim changed")
            claim = _read_lock_fd(fd)
            if claim.get("status") not in {"failed", "superseded"}:
                raise ShardJobError(
                    f"Cannot retry {shard_id}: claim status changed to {claim.get('status', 'unknown')!r}"
                )
            if _done_path(job_dir, shard_id).exists():
                raise ShardJobError(f"Cannot retry {shard_id}: it is already complete")

            os.rename(failed_path, history_failed / f"{shard_id}.{suffix}.json")
            attempt_root = job_dir / "attempts" / shard_id
            if attempt_root.exists():
                os.rename(attempt_root, history_attempts / f"{shard_id}.{suffix}")
            # Removing the terminal fence is the final requeue step. Holding
            # its advisory lock prevents a worker from observing a half-
            # archived retry state and claiming the shard too early.
            os.rename(lock_path, history_claims / f"{shard_id}.{suffix}.json")
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
        print(f"Requeued {shard_id}")
    return 0


def merge_job(args: argparse.Namespace) -> int:
    job_dir = Path(args.job_dir).expanduser().resolve()
    manifest = _load_manifest(job_dir)
    policy = _effective_policy(manifest, args)
    status = _collect_status(job_dir, manifest, timeout_secs=policy["lock_timeout_secs"])
    if not status["complete"]:
        counts = status["counts"]
        raise ShardJobError(
            "Cannot merge an incomplete job: "
            f"pending={counts['pending']} running={counts['running']} stale={counts['stale']} "
            f"committing={counts['committing']} conflict={counts['conflict']} "
            f"failed={counts['failed']} done={counts['done']}"
        )

    output_path = Path(args.output).expanduser().resolve()
    if output_path.exists() and not args.overwrite:
        raise ShardJobError(f"Output already exists: {output_path}; pass --overwrite to replace it")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f".{output_path.name}.{uuid.uuid4().hex}.tmp")
    total_rows = 0
    output_digest = hashlib.sha256()

    try:
        with tmp_path.open("wb") as output_handle:
            for shard in manifest["shards"]:
                done = _read_json(_done_path(job_dir, shard["id"]))
                result_path = _job_path(job_dir, done["output_path"])
                result_digest = hashlib.sha256()
                result_rows = 0
                last_had_newline = True
                with result_path.open("rb") as input_handle:
                    for line_number, raw_line in enumerate(input_handle, start=1):
                        _parse_record(raw_line, result_path, line_number)
                        result_digest.update(raw_line)
                        result_rows += 1
                        output_handle.write(raw_line)
                        output_digest.update(raw_line)
                        last_had_newline = raw_line.endswith(b"\n")
                    if result_rows and not last_had_newline:
                        output_handle.write(b"\n")
                        output_digest.update(b"\n")
                if result_rows != done.get("rows") or result_digest.hexdigest() != done.get("sha256"):
                    raise ShardJobError(f"Result checksum or row count mismatch for {shard['id']}")
                total_rows += result_rows
            output_handle.flush()
            os.fsync(output_handle.fileno())
        os.replace(tmp_path, output_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    merge_metadata = {
        "job_id": manifest["job_id"],
        "merged_at": _utc_now(),
        "output_path": str(output_path),
        "rows": total_rows,
        "sha256": output_digest.hexdigest(),
        "num_shards": manifest["num_shards"],
    }
    _atomic_write_json(job_dir / "merge.json", merge_metadata)
    print(f"Merged {manifest['num_shards']} shards and {total_rows} rows into {output_path}")
    return 0


def _resolve_output_path(config: dict[str, Any], override: str | None) -> Path:
    configured = override if override is not None else config.get("export_path")
    if not isinstance(configured, str) or not configured.strip():
        raise ShardJobError("Recipe must set export_path, or run must receive --output")
    if configured.startswith(REMOTE_PATH_PREFIXES):
        raise ShardJobError("Elastic sharding only supports a local export_path")
    export_type = config.get("export_type")
    if export_type not in (None, "jsonl"):
        raise ShardJobError(f"Elastic sharding only supports export_type=jsonl; got {export_type!r}")
    output_path = _resolve_config_path(configured)
    if output_path.suffix.lower() != ".jsonl":
        raise ShardJobError(f"Elastic sharding requires a .jsonl export_path; got {output_path}")
    return output_path


def _preflight_sharded_run(args: argparse.Namespace) -> dict[str, Any]:
    """Validate that a run can preserve semantics when processed per shard."""
    config_path = _resolve_config_path(args.config)
    if not config_path.is_file():
        raise ShardJobError(f"Recipe does not exist: {config_path}")
    config = _load_yaml(config_path)
    validation = _validate_recipe(config)

    unsupported_flags = {
        "export_original_dataset": "per-run original-dataset export",
        "encrypt_before_export": "encrypted intermediate output",
    }
    for key, description in unsupported_flags.items():
        if config.get(key):
            raise ShardJobError(f"Recipe enables {description} ({key})")
    if config.get("export_shard_size", 0):
        raise ShardJobError("Recipe sets export_shard_size; elastic sharding owns final output partitioning")

    dataset_path = _resolve_dataset_path(config, args.dataset_path)
    files = _discover_jsonl_files(dataset_path)
    output_path = _resolve_output_path(config, args.output)
    return {
        "config_path": config_path,
        "config": config,
        "validation": validation,
        "dataset_path": dataset_path,
        "files": files,
        "output_path": output_path,
    }


def _best_effort_output_path(args: argparse.Namespace) -> Path | None:
    """Resolve a local output for overwrite protection without blocking fallback."""
    try:
        config = _load_yaml(_resolve_config_path(args.config))
        configured = args.output if args.output is not None else config.get("export_path")
        if not isinstance(configured, str) or not configured.strip() or configured.startswith(REMOTE_PATH_PREFIXES):
            return None
        return _resolve_config_path(configured)
    except (OSError, ShardJobError):
        return None


def _resolve_run_id(args: argparse.Namespace) -> str:
    if args.run_id is not None:
        run_id = args.run_id.strip()
        if not run_id:
            raise ShardJobError("--run-id must not be empty")
        return run_id
    for name in RUN_ID_ENV_NAMES:
        run_id = os.environ.get(name, "").strip()
        if run_id:
            return run_id
    return "default"


def _coordination_dir(job_dir: Path, run_id: str) -> Path:
    run_key = hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:20]
    return job_dir.parent / f".{job_dir.name}.sharded-coordination" / run_key


def _resolved_or_remote_path(value: str) -> str:
    if value.startswith(REMOTE_PATH_PREFIXES):
        return value
    return str(Path(value).expanduser().resolve())


def _run_request(args: argparse.Namespace, run_id: str, mode: str, output_path: Path | None) -> dict[str, Any]:
    config_path = _resolve_config_path(args.config)
    config_sha256 = _sha256_file(config_path) if config_path.is_file() else None
    return {
        "run_id": run_id,
        "mode": mode,
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "dataset_path": (_resolved_or_remote_path(args.dataset_path) if args.dataset_path else None),
        "output_path": (
            _resolved_or_remote_path(args.output)
            if args.output
            else (str(output_path) if output_path is not None else None)
        ),
        "num_shards": args.num_shards,
        "ray_address": args.ray_address,
    }


def _ensure_run_request(coordination_dir: Path, request: dict[str, Any]) -> None:
    request_path = coordination_dir / "request.json"
    if _exclusive_write_json(request_path, request):
        return
    saved = _read_json(request_path)
    if saved != request:
        raise ShardJobError(
            "This job-dir/run-id was already used with different arguments; " "choose a new --run-id or --job-dir"
        )


def _try_acquire_phase(coordination_dir: Path, phase: str) -> bool:
    return _write_claim(
        coordination_dir / f"{phase}.lock",
        {
            "phase": phase,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "created_at": _utc_now(),
        },
    )


def _phase_result_code(result_path: Path) -> int:
    result = _read_json(result_path)
    return_code = result.get("return_code")
    if not isinstance(return_code, int):
        raise ShardJobError(f"Coordination result has no integer return_code: {result_path}")
    return return_code


def _wait_for_phase_result(args: argparse.Namespace, result_path: Path, phase: str) -> int:
    deadline = time.monotonic() + args.wait_timeout_secs
    while True:
        if result_path.exists():
            return _phase_result_code(result_path)
        if time.monotonic() >= deadline:
            raise ShardJobError(f"Timed out after {args.wait_timeout_secs:g}s waiting for {phase}")
        time.sleep(args.wait_poll_interval_secs)


def _write_phase_result(result_path: Path, phase: str, return_code: int, error: str | None = None) -> None:
    result = {
        "phase": phase,
        "return_code": return_code,
        "hostname": socket.gethostname(),
        "finished_at": _utc_now(),
    }
    if error is not None:
        result["error"] = error
    _atomic_write_json(result_path, result)


def _original_process_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "-m",
        PROCESS_DATA_MODULE,
        "--config",
        str(_resolve_config_path(args.config)),
    ]
    if args.dataset_path:
        command.extend(["--dataset_path", _resolved_or_remote_path(args.dataset_path)])
    if args.output:
        command.extend(["--export_path", _resolved_or_remote_path(args.output)])
    return command


def _coordinated_fallback(
    args: argparse.Namespace,
    coordination_dir: Path,
    reason: str,
    output_path: Path | None,
) -> int:
    result_path = coordination_dir / "fallback-result.json"
    if result_path.exists():
        return _phase_result_code(result_path)
    if not _try_acquire_phase(coordination_dir, "fallback"):
        return _wait_for_phase_result(args, result_path, "unsharded Data-Juicer process")

    print(f"WARNING: elastic sharding disabled: {reason}", file=sys.stderr, flush=True)
    print("Falling back to one original dj-process invocation.", file=sys.stderr, flush=True)
    command = _original_process_command(args)
    error = None
    if output_path is not None and output_path.exists() and not args.overwrite:
        return_code = 2
        error = f"Output already exists: {output_path}; pass --overwrite to replace it"
        print(f"ERROR: {error}", file=sys.stderr)
    else:
        try:
            return_code = subprocess.run(command, check=False).returncode
        except OSError as exc:
            return_code = 2
            error = f"Unable to start Data-Juicer: {exc}"
            print(f"ERROR: {error}", file=sys.stderr)
    _write_phase_result(result_path, "fallback", return_code, error)
    return return_code


def _coordinated_prepare(args: argparse.Namespace, coordination_dir: Path, output_path: Path) -> int:
    result_path = coordination_dir / "prepare-result.json"
    if result_path.exists():
        return _phase_result_code(result_path)
    if not _try_acquire_phase(coordination_dir, "prepare"):
        return _wait_for_phase_result(args, result_path, "shard preparation")

    print(f"hostname={socket.gethostname()} elected as prepare coordinator.", flush=True)
    error = None
    if output_path.exists() and not args.overwrite:
        return_code = 2
        error = f"Output already exists: {output_path}; pass --overwrite to replace it"
        print(f"ERROR: {error}", file=sys.stderr)
    else:
        try:
            return_code = prepare_job(
                argparse.Namespace(
                    config=args.config,
                    dataset_path=args.dataset_path,
                    job_dir=args.job_dir,
                    num_shards=args.num_shards,
                    ray_address=args.ray_address,
                    lock_timeout_secs=args.lock_timeout_secs,
                    max_retries=args.max_retries,
                    poll_interval_secs=args.poll_interval_secs,
                )
            )
        except (OSError, ShardJobError) as exc:
            return_code = 2
            error = str(exc)
            print(f"ERROR: shard preparation failed: {error}", file=sys.stderr)
    _write_phase_result(result_path, "prepare", return_code, error)
    return return_code


def _publish_run_failure(coordination_dir: Path, return_code: int, reason: str) -> int:
    result_path = coordination_dir / "run-failure.json"
    _exclusive_write_json(
        result_path,
        {
            "phase": "worker",
            "return_code": return_code,
            "reason": reason,
            "hostname": socket.gethostname(),
            "finished_at": _utc_now(),
        },
    )
    return _phase_result_code(result_path)


def _coordinated_finalize(args: argparse.Namespace, coordination_dir: Path, output_path: Path) -> int:
    result_path = coordination_dir / "finalize-result.json"
    if result_path.exists():
        return _phase_result_code(result_path)
    if not _try_acquire_phase(coordination_dir, "finalize"):
        return _wait_for_phase_result(args, result_path, "final output merge")

    print(f"hostname={socket.gethostname()} elected as finalize coordinator.", flush=True)
    error = None
    try:
        return_code = merge_job(
            argparse.Namespace(
                job_dir=args.job_dir,
                output=str(output_path),
                lock_timeout_secs=args.lock_timeout_secs,
                overwrite=args.overwrite,
            )
        )
    except (OSError, ShardJobError) as exc:
        return_code = 2
        error = str(exc)
        print(f"ERROR: final merge failed: {error}", file=sys.stderr)
    _write_phase_result(result_path, "finalize", return_code, error)
    return return_code


def run_job(args: argparse.Namespace) -> int:
    """Run one command on one or more nodes, sharding only when it is safe."""
    job_dir = Path(args.job_dir).expanduser().resolve()
    args.job_dir = str(job_dir)
    args.config = str(_resolve_config_path(args.config))
    run_id = _resolve_run_id(args)

    try:
        preflight = _preflight_sharded_run(args)
        mode = "sharded"
        reason = None
        output_path = preflight["output_path"]
    except Exception as exc:  # Let the original dj-process own unsupported/invalid recipes.
        mode = "fallback"
        reason = f"{type(exc).__name__}: {exc}"
        output_path = _best_effort_output_path(args)

    coordination_dir = _coordination_dir(job_dir, run_id)
    coordination_dir.mkdir(parents=True, exist_ok=True)
    _ensure_run_request(coordination_dir, _run_request(args, run_id, mode, output_path))

    if mode == "fallback":
        return _coordinated_fallback(args, coordination_dir, reason or "recipe is not shard-safe", output_path)

    final_result_path = coordination_dir / "finalize-result.json"
    if final_result_path.exists():
        return _phase_result_code(final_result_path)
    failure_path = coordination_dir / "run-failure.json"
    if failure_path.exists():
        return _phase_result_code(failure_path)
    prepare_code = _coordinated_prepare(args, coordination_dir, output_path)
    if prepare_code != 0:
        return prepare_code

    worker_code = worker_job(
        argparse.Namespace(
            job_dir=str(job_dir),
            max_shards=None,
            lock_timeout_secs=args.lock_timeout_secs,
            max_retries=args.max_retries,
            poll_interval_secs=args.poll_interval_secs,
            ray_address=args.ray_address,
            allow_version_mismatch=args.allow_version_mismatch,
        )
    )
    if worker_code != 0:
        return _publish_run_failure(
            coordination_dir,
            worker_code,
            f"worker on hostname={socket.gethostname()} exited with code {worker_code}",
        )
    if failure_path.exists():
        return _phase_result_code(failure_path)
    return _coordinated_finalize(args, coordination_dir, output_path)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive number")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser(
        "run",
        help="Run a complete job, automatically falling back when the recipe is not shard-safe",
    )
    run.add_argument("--config", required=True, help="Data-Juicer YAML recipe")
    run.add_argument("--dataset-path", help="Override recipe dataset_path")
    run.add_argument("--job-dir", required=True, help="Shared POSIX directory for shards and job state")
    run.add_argument(
        "--num-shards",
        required=True,
        type=_positive_int,
        help="Exact shard count; for throughput keep it <= worker nodes, ideally one shard per node",
    )
    run.add_argument("--output", help="Override recipe export_path; sharded mode requires local JSONL")
    run.add_argument("--ray-address", default="local", help="Node-local Ray address used by shard workers")
    run.add_argument("--run-id", help="Submission ID shared by all nodes; defaults to scheduler ID or 'default'")
    run.add_argument("--lock-timeout-secs", type=_positive_int, default=DEFAULT_LOCK_TIMEOUT_SECS)
    run.add_argument("--max-retries", type=_non_negative_int, default=DEFAULT_MAX_RETRIES)
    run.add_argument("--poll-interval-secs", type=_positive_int, default=DEFAULT_POLL_INTERVAL_SECS)
    run.add_argument("--wait-timeout-secs", type=_positive_float, default=DEFAULT_WAIT_TIMEOUT_SECS)
    run.add_argument(
        "--wait-poll-interval-secs",
        type=_positive_float,
        default=DEFAULT_WAIT_POLL_INTERVAL_SECS,
        help="Polling interval while another node coordinates prepare/fallback/finalize",
    )
    run.add_argument("--allow-version-mismatch", action="store_true")
    run.add_argument("--overwrite", action="store_true", help="Allow replacing an existing final output")
    run.set_defaults(func=run_job)

    prepare = subparsers.add_parser("prepare", help="Validate a recipe and pre-split its JSONL input")
    prepare.add_argument("--config", required=True, help="Data-Juicer YAML recipe")
    prepare.add_argument("--dataset-path", help="Override recipe dataset_path with a local JSONL file/directory")
    prepare.add_argument("--job-dir", required=True, help="New job directory on a shared POSIX filesystem")
    prepare.add_argument(
        "--num-shards",
        required=True,
        type=_positive_int,
        help="Exact shard count; for throughput keep it <= worker nodes, ideally one shard per node",
    )
    prepare.add_argument("--lock-timeout-secs", type=_positive_int, default=DEFAULT_LOCK_TIMEOUT_SECS)
    prepare.add_argument("--max-retries", type=_non_negative_int, default=DEFAULT_MAX_RETRIES)
    prepare.add_argument("--poll-interval-secs", type=_positive_int, default=DEFAULT_POLL_INTERVAL_SECS)
    prepare.add_argument(
        "--ray-address",
        default="local",
        help="Ray address saved for workers; 'local' starts an isolated Ray instance per attempt",
    )
    prepare.set_defaults(func=prepare_job)

    worker = subparsers.add_parser("worker", help="Claim and process shards until the job reaches a terminal state")
    worker.add_argument("--job-dir", required=True)
    worker.add_argument("--max-shards", type=_positive_int, help="Maximum claims handled by this worker invocation")
    worker.add_argument("--lock-timeout-secs", type=_positive_int)
    worker.add_argument("--max-retries", type=_non_negative_int)
    worker.add_argument("--poll-interval-secs", type=_positive_int)
    worker.add_argument("--ray-address", help="Override the prepared Ray address for this worker")
    worker.add_argument("--allow-version-mismatch", action="store_true")
    worker.set_defaults(func=worker_job)

    status = subparsers.add_parser("status", help="Show shard state without modifying it")
    status.add_argument("--job-dir", required=True)
    status.add_argument("--lock-timeout-secs", type=_positive_int)
    status.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    status.add_argument("--all", action="store_true", help="Print every shard")
    status.set_defaults(func=status_job)

    retry = subparsers.add_parser("retry", help="Archive failed attempts and requeue failed shards")
    retry.add_argument("--job-dir", required=True)
    retry_group = retry.add_mutually_exclusive_group(required=True)
    retry_group.add_argument("--all-failed", action="store_true")
    retry_group.add_argument("--shard-id", action="append", help="Shard ID to requeue; may be repeated")
    retry.set_defaults(func=retry_job)

    merge = subparsers.add_parser("merge", help="Validate and merge all successful shard outputs")
    merge.add_argument("--job-dir", required=True)
    merge.add_argument("--output", required=True)
    merge.add_argument("--lock-timeout-secs", type=_positive_int)
    merge.add_argument("--overwrite", action="store_true")
    merge.set_defaults(func=merge_job)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (OSError, ShardJobError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

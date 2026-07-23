#!/usr/bin/env python3
"""Shared-filesystem sharding demo for elastic multi-node Data-Juicer jobs.

The script deliberately lives in ``demos`` instead of the executor package:
each shard is processed by an isolated ``ray`` executor on the claiming node,
while this file coordinates preparation, claiming, retries, status, and merge.
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

SCHEMA_VERSION = 2
DEFAULT_LOCK_TIMEOUT_SECS = 35 * 60 * 60
DEFAULT_MAX_RETRIES = 3
DEFAULT_POLL_INTERVAL_SECS = 20
REMOTE_PATH_PREFIXES = ("http://", "https://", "s3://", "gs://", "hdfs://")

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
PROCESS_DATA_SCRIPT = REPO_ROOT / "tools" / "process_data.py"


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


def _resolve_config_path(value: str, *, base: Path = REPO_ROOT) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


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
                "and is not supported by this demo"
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
            warnings.append(
                f"operator {op_name!r} uses an explicit save_dir; the caller must ensure generated filenames "
                "do not collide across shards"
            )

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
            "attempts",
        ):
            (stage_dir / relative_dir).mkdir(parents=True, exist_ok=True)

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "job_id": job_dir.name,
            "created_at": _utc_now(),
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
        if _read_lock_fd(fd).get("token") != token:
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
        if time.time() - os.fstat(fd).st_mtime <= timeout_secs:
            return False
        lock_metadata = _read_lock_fd(fd)
        stale_path = job_dir / "state" / "stale_locks" / f"{lock_path.stem}.{int(time.time())}.{uuid.uuid4().hex}.lock"
        # Update the retry counter before removing the visible lock, so the
        # next claimant cannot miss this expired attempt.
        _mark_attempt_stale(job_dir, lock_metadata)
        os.rename(lock_path, stale_path)
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    return True


def _publish_failure_and_release(
    job_dir: Path,
    shard_id: str,
    token: str,
    failure_metadata: dict[str, Any],
) -> bool:
    """Publish terminal failure only while still owning the current lock."""
    lock_path = _lock_path(job_dir, shard_id)
    try:
        fd = os.open(lock_path, os.O_RDONLY)
    except FileNotFoundError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        if not _lock_fd_still_at_path(fd, lock_path):
            return False
        if _read_lock_fd(fd).get("token") != token:
            return False
        if not _done_path(job_dir, shard_id).exists():
            _exclusive_write_json(_failed_path(job_dir, shard_id), failure_metadata)
        lock_path.unlink()
        return True
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

    failure_count = _attempt_failures(job_dir, shard_id)
    # ``max_retries`` counts retries after the initial attempt. For example,
    # max_retries=3 permits four total failed attempts before terminal state.
    if failure_count > max_retries:
        _exclusive_write_json(
            _failed_path(job_dir, shard_id),
            {
                "shard_id": shard_id,
                "status": "failed",
                "failed_at": _utc_now(),
                "failures": failure_count,
                "reason": "retry limit reached before claim",
            },
        )
        return None

    attempt_number = len(_attempt_directories(job_dir, shard_id)) + 1
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
        str(PROCESS_DATA_SCRIPT),
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
                cwd=REPO_ROOT,
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
        published = _exclusive_write_json(_done_path(job_dir, shard_id), done_metadata)
        metadata.update(
            {
                "status": "done" if published else "superseded",
                "result": validation,
            }
        )
        _atomic_write_json(metadata_path, metadata)
        _release_lock(lock_path, token)
        return "done" if published else "superseded"

    metadata.update({"status": "failed", "error": error})
    _atomic_write_json(metadata_path, metadata)

    failures = _attempt_failures(job_dir, shard_id)
    max_retries = int(manifest["policy"]["max_retries"])
    if failures > max_retries:
        owned = _publish_failure_and_release(
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
    counts = {"pending": 0, "running": 0, "stale": 0, "done": 0, "failed": 0}
    now = time.time()
    for shard in manifest["shards"]:
        shard_id = shard["id"]
        done_path = _done_path(job_dir, shard_id)
        failed_path = _failed_path(job_dir, shard_id)
        lock_path = _lock_path(job_dir, shard_id)
        record: dict[str, Any] = {"shard_id": shard_id, "rows": shard["rows"]}
        if done_path.exists():
            status = "done"
            try:
                record.update({"owner": _read_json(done_path).get("hostname")})
            except ShardJobError:
                record["metadata_error"] = True
        elif failed_path.exists():
            status = "failed"
            try:
                record.update({"error": _read_json(failed_path).get("last_error")})
            except ShardJobError:
                record["metadata_error"] = True
        elif lock_path.exists():
            try:
                age = max(0.0, now - lock_path.stat().st_mtime)
            except FileNotFoundError:
                age = 0.0
            status = "stale" if age > timeout_secs else "running"
            record["lock_age_secs"] = age
            try:
                lock = _read_json(lock_path)
                record.update({"owner": lock.get("hostname"), "pid": lock.get("pid")})
            except ShardJobError:
                record["metadata_error"] = True
        else:
            status = "pending"
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
    hostname = socket.gethostname()
    print(f"Worker {hostname}:{os.getpid()} started for {manifest['job_id']}")

    while True:
        if args.max_shards is not None and processed >= args.max_shards:
            print(f"Reached --max-shards={args.max_shards}")
            return 0

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
        if status["terminal"]:
            print(
                f"Job finished with failures: done={counts['done']} failed={counts['failed']}",
                file=sys.stderr,
            )
            return 2
        print(
            "No shard claim available; "
            f"pending={counts['pending']} running={counts['running']} stale={counts['stale']}. "
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
            f"done={counts['done']} failed={counts['failed']}"
        )
        if args.all:
            for shard in status["shards"]:
                owner = f" owner={shard.get('owner')}" if shard.get("owner") else ""
                print(f"  {shard['shard_id']}: {shard['status']}{owner}")
    return 0 if status["complete"] else (2 if status["counts"]["failed"] else 1)


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
    history_failed.mkdir(parents=True, exist_ok=True)
    history_attempts.mkdir(parents=True, exist_ok=True)

    # Validate the full selection before moving anything, so a bad target
    # cannot leave a multi-shard retry request only partially applied.
    for shard_id in targets:
        if _lock_path(job_dir, shard_id).exists():
            raise ShardJobError(f"Cannot retry {shard_id}: a lock still exists")
        if _done_path(job_dir, shard_id).exists():
            raise ShardJobError(f"Cannot retry {shard_id}: it is already complete")

    for shard_id in targets:
        suffix = f"{int(time.time())}.{uuid.uuid4().hex}"
        failed_path = _failed_path(job_dir, shard_id)
        os.rename(failed_path, history_failed / f"{shard_id}.{suffix}.json")
        attempt_root = job_dir / "attempts" / shard_id
        if attempt_root.exists():
            os.rename(attempt_root, history_attempts / f"{shard_id}.{suffix}")
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Validate a recipe and pre-split its JSONL input")
    prepare.add_argument("--config", required=True, help="Data-Juicer YAML recipe")
    prepare.add_argument("--dataset-path", help="Override recipe dataset_path with a local JSONL file/directory")
    prepare.add_argument("--job-dir", required=True, help="New job directory on a shared POSIX filesystem")
    prepare.add_argument("--num-shards", required=True, type=_positive_int, help="Exact number of non-empty shards")
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

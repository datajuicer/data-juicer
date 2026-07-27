#!/usr/bin/env python3
"""Worker-broadcast DLC launcher for the elastic sharding demo.

Use this entry point only when the DLC job type starts the configured command
on every Worker instance. Shared-storage leader election then coordinates
preparation and finalization without platform-specific rank variables.

For an MPIJob, the configured command runs on its Launcher instead. The
Launcher must use ``mpirun`` (normally with DLC's generated hostfile) to start
one ``worker`` process on every GPU Worker; do not invoke the ``dlc`` command
only on the MPI Launcher.

The explicit ``prepare``, ``worker``, and ``verify`` commands are retained for
manual or scheduler-driven testing.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import socket
import subprocess
import sys
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
SHARD_JOB_SCRIPT = SCRIPT_DIR / "shard_job.py"
DEFAULT_CONFIG = SCRIPT_DIR / "configs" / "demo.yaml"
DEFAULT_WAIT_TIMEOUT_SECS = 35 * 60 * 60
DEFAULT_POLL_INTERVAL_SECS = 2.0


def _run_shard_job(arguments: list[str]) -> int:
    command = [sys.executable, str(SHARD_JOB_SCRIPT), *arguments]
    print(f"$ {shlex.join(command)}", flush=True)
    return subprocess.run(command, check=False).returncode


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _coordination_dir(job_dir: Path) -> Path:
    return job_dir.parent / f".{job_dir.name}.dlc-coordination"


def _try_acquire_phase(lock_path: Path, phase: str) -> bool:
    """Atomically elect one DLC instance for a coordination phase."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            lock_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
    except FileExistsError:
        return False

    metadata = {
        "phase": phase,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "created_at": _utc_now(),
    }
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return True


def _result_code(path: Path) -> int:
    result = _read_json(path)
    return_code = result.get("return_code")
    if not isinstance(return_code, int):
        raise ValueError(f"Coordination result has no integer return_code: {path}")
    return return_code


def _wait_for_phase(
    *,
    success_path: Path | None,
    result_path: Path,
    timeout_secs: float,
    poll_interval_secs: float,
    phase: str,
) -> int:
    deadline = time.monotonic() + timeout_secs
    while True:
        if success_path is not None and success_path.exists():
            return 0
        if result_path.exists():
            return _result_code(result_path)
        if time.monotonic() >= deadline:
            print(
                f"ERROR: timed out after {timeout_secs:g}s waiting for DLC "
                f"{phase}. Check whether another worker exited unexpectedly.",
                file=sys.stderr,
            )
            return 2
        time.sleep(poll_interval_secs)


def _coordinated_prepare(
    args: argparse.Namespace,
    job_dir: Path,
    coordination_dir: Path,
) -> int:
    manifest_path = job_dir / "manifest.json"
    result_path = coordination_dir / "prepare-result.json"

    # Re-run the idempotency check when a prepared job already exists. This
    # catches accidental reuse with a different recipe, input, or shard count.
    if manifest_path.exists():
        return prepare(args)
    if result_path.exists():
        return _result_code(result_path)

    lock_path = coordination_dir / "prepare.lock"
    if not _try_acquire_phase(lock_path, "prepare"):
        return _wait_for_phase(
            success_path=manifest_path,
            result_path=result_path,
            timeout_secs=args.wait_timeout_secs,
            poll_interval_secs=args.poll_interval_secs,
            phase="preparation",
        )

    print(
        f"hostname={socket.gethostname()} elected as DLC prepare coordinator.",
        flush=True,
    )
    try:
        return_code = prepare(args)
    except Exception as exc:
        _atomic_write_json(
            result_path,
            {
                "phase": "prepare",
                "return_code": 2,
                "hostname": socket.gethostname(),
                "finished_at": _utc_now(),
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise

    _atomic_write_json(
        result_path,
        {
            "phase": "prepare",
            "return_code": return_code,
            "hostname": socket.gethostname(),
            "finished_at": _utc_now(),
        },
    )
    return return_code


def _publish_abort(
    coordination_dir: Path,
    *,
    return_code: int,
    reason: str,
) -> None:
    abort_path = coordination_dir / "abort.json"
    if abort_path.exists():
        return
    if not _try_acquire_phase(coordination_dir / "abort.lock", "abort"):
        return
    _atomic_write_json(
        abort_path,
        {
            "phase": "worker",
            "return_code": return_code,
            "reason": reason,
            "hostname": socket.gethostname(),
            "finished_at": _utc_now(),
        },
    )


def _wait_for_terminal_state(
    *,
    job_dir: Path,
    coordination_dir: Path,
    timeout_secs: float,
    poll_interval_secs: float,
) -> int:
    manifest = _read_json(job_dir / "manifest.json")
    shards = manifest.get("shards")
    if not isinstance(shards, list):
        raise ValueError("manifest.json has no shards list")

    deadline = time.monotonic() + timeout_secs
    while True:
        abort_path = coordination_dir / "abort.json"
        if abort_path.exists():
            return _result_code(abort_path)

        done_count = sum((job_dir / "state" / "done" / f"{shard['id']}.json").exists() for shard in shards)
        failed_count = sum((job_dir / "state" / "failed" / f"{shard['id']}.json").exists() for shard in shards)
        if failed_count:
            _publish_abort(
                coordination_dir,
                return_code=2,
                reason=f"{failed_count} shard(s) reached terminal failure",
            )
            return _wait_for_phase(
                success_path=None,
                result_path=abort_path,
                timeout_secs=timeout_secs,
                poll_interval_secs=poll_interval_secs,
                phase="failure propagation",
            )
        if done_count == len(shards):
            return 0
        if time.monotonic() >= deadline:
            _publish_abort(
                coordination_dir,
                return_code=2,
                reason=(
                    f"timed out after {timeout_secs:g}s waiting for all shards " f"({done_count}/{len(shards)} done)"
                ),
            )
            return _wait_for_phase(
                success_path=None,
                result_path=abort_path,
                timeout_secs=timeout_secs,
                poll_interval_secs=poll_interval_secs,
                phase="timeout propagation",
            )
        time.sleep(poll_interval_secs)


def _coordinated_finalize(
    args: argparse.Namespace,
    coordination_dir: Path,
) -> int:
    result_path = coordination_dir / "finalize-result.json"
    if result_path.exists():
        return _result_code(result_path)

    lock_path = coordination_dir / "finalize.lock"
    if not _try_acquire_phase(lock_path, "finalize"):
        return _wait_for_phase(
            success_path=None,
            result_path=result_path,
            timeout_secs=args.wait_timeout_secs,
            poll_interval_secs=args.poll_interval_secs,
            phase="final verification",
        )

    print(
        f"hostname={socket.gethostname()} elected as DLC finalize coordinator.",
        flush=True,
    )
    try:
        return_code = verify(
            argparse.Namespace(
                job_dir=args.job_dir,
                output=args.output,
                expect_nodes=(args.nodes if args.require_all_nodes else 1),
            )
        )
        error = None
    except Exception as exc:
        return_code = 2
        error = f"{type(exc).__name__}: {exc}"
        print(f"ERROR: final verification failed: {error}", file=sys.stderr)

    result = {
        "phase": "finalize",
        "return_code": return_code,
        "hostname": socket.gethostname(),
        "finished_at": _utc_now(),
    }
    if error is not None:
        result["error"] = error
    _atomic_write_json(result_path, result)
    return return_code


def prepare(args: argparse.Namespace) -> int:
    command = [
        "prepare",
        "--config",
        str(Path(args.config).expanduser().resolve()),
        "--job-dir",
        str(Path(args.job_dir).expanduser().resolve()),
        "--num-shards",
        str(args.num_shards),
        "--ray-address",
        args.ray_address,
    ]
    if args.dataset_path:
        command.extend(
            [
                "--dataset-path",
                str(Path(args.dataset_path).expanduser().resolve()),
            ]
        )
    return _run_shard_job(command)


def worker(args: argparse.Namespace) -> int:
    shard_limit = (
        f"at most {args.max_shards} shard claim(s)"
        if args.max_shards is not None
        else "an unlimited number of shard claims"
    )
    print(
        f"Starting DLC worker on hostname={socket.gethostname()}; " f"this invocation will handle {shard_limit}.",
        flush=True,
    )
    command = [
        "worker",
        "--job-dir",
        str(Path(args.job_dir).expanduser().resolve()),
    ]
    if args.max_shards is not None:
        command.extend(["--max-shards", str(args.max_shards)])
    if args.ray_address:
        command.extend(["--ray-address", args.ray_address])
    return _run_shard_job(command)


def dlc(args: argparse.Namespace) -> int:
    """Run the complete elastic job from every DLC Worker instance."""
    job_dir = Path(args.job_dir).expanduser().resolve()
    coordination_dir = _coordination_dir(job_dir)
    coordination_dir.mkdir(parents=True, exist_ok=True)

    max_shards: int | None = None
    if args.require_all_nodes:
        if args.nodes is None:
            raise ValueError("--require-all-nodes requires --nodes")
        max_shards = math.ceil(args.num_shards / args.nodes)
        if args.num_shards < args.nodes:
            raise ValueError("--num-shards must be at least --nodes")
        if (args.nodes - 1) * max_shards >= args.num_shards:
            raise ValueError(
                "This shard/node combination cannot guarantee that every DLC "
                "Worker participates. Increase --num-shards or use a multiple "
                "of --nodes."
            )

    node_description = args.nodes if args.nodes is not None else "not-enforced"
    shard_limit = max_shards if max_shards is not None else "unlimited"
    print(
        f"DLC instance hostname={socket.gethostname()} starting; "
        f"nodes={node_description}, shards={args.num_shards}, "
        f"require_all_nodes={args.require_all_nodes}, "
        f"max_shards_per_worker={shard_limit}.",
        flush=True,
    )
    prepare_code = _coordinated_prepare(args, job_dir, coordination_dir)
    if prepare_code != 0:
        return prepare_code

    final_result_path = coordination_dir / "finalize-result.json"
    if final_result_path.exists():
        return _result_code(final_result_path)
    abort_path = coordination_dir / "abort.json"
    if abort_path.exists():
        return _result_code(abort_path)

    worker_code = worker(
        argparse.Namespace(
            job_dir=str(job_dir),
            max_shards=max_shards,
            ray_address=args.ray_address,
        )
    )
    if worker_code != 0:
        _publish_abort(
            coordination_dir,
            return_code=worker_code,
            reason=(f"worker on hostname={socket.gethostname()} exited with " f"code {worker_code}"),
        )
        return _wait_for_phase(
            success_path=None,
            result_path=abort_path,
            timeout_secs=args.wait_timeout_secs,
            poll_interval_secs=args.poll_interval_secs,
            phase="worker failure propagation",
        )

    terminal_code = _wait_for_terminal_state(
        job_dir=job_dir,
        coordination_dir=coordination_dir,
        timeout_secs=args.wait_timeout_secs,
        poll_interval_secs=args.poll_interval_secs,
    )
    if terminal_code != 0:
        return terminal_code
    return _coordinated_finalize(args, coordination_dir)


def status(args: argparse.Namespace) -> int:
    return _run_shard_job(
        [
            "status",
            "--job-dir",
            str(Path(args.job_dir).expanduser().resolve()),
            "--all",
        ]
    )


def _validate_merged_jsonl(path: Path) -> int:
    rows = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"Blank line in merged output at line {line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Non-object JSON value at line {line_number}")
            rows += 1
    return rows


def verify(args: argparse.Namespace) -> int:
    job_dir = Path(args.job_dir).expanduser().resolve()
    status_code = _run_shard_job(["status", "--job-dir", str(job_dir), "--all"])
    if status_code != 0:
        print("Verification stopped: not all shards are complete.", file=sys.stderr)
        return status_code

    manifest = _read_json(job_dir / "manifest.json")
    owner_counts: Counter[str] = Counter()
    for shard in manifest["shards"]:
        done_path = job_dir / "state" / "done" / f"{shard['id']}.json"
        done = _read_json(done_path)
        hostname = done.get("hostname")
        if not isinstance(hostname, str) or not hostname:
            raise ValueError(f"Done metadata has no hostname: {done_path}")
        owner_counts[hostname] += 1

    print("Shard owners:")
    for owner, count in sorted(owner_counts.items()):
        print(f"  {owner}: {count} shard(s)")
    if len(owner_counts) < args.expect_nodes:
        print(
            f"Verification failed: expected at least {args.expect_nodes} distinct "
            f"hostnames, found {len(owner_counts)}.",
            file=sys.stderr,
        )
        return 2

    output_path = Path(args.output).expanduser().resolve() if args.output else job_dir / "merged.jsonl"
    merge_code = _run_shard_job(
        [
            "merge",
            "--job-dir",
            str(job_dir),
            "--output",
            str(output_path),
            "--overwrite",
        ]
    )
    if merge_code != 0:
        return merge_code

    rows = _validate_merged_jsonl(output_path)
    print(
        f"PASS: {manifest['num_shards']} shards were completed by "
        f"{len(owner_counts)} node(s); merged {rows} rows into {output_path}"
    )
    return 0


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive number")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare_parser = commands.add_parser("prepare", help="Prepare a sharded test job")
    prepare_parser.add_argument("--job-dir", required=True, help="Shared job directory")
    prepare_parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Existing shard-safe Data-Juicer recipe",
    )
    prepare_parser.add_argument(
        "--dataset-path",
        help="Optional JSONL file/directory overriding the recipe dataset_path",
    )
    prepare_parser.add_argument("--num-shards", type=_positive_int, default=4)
    prepare_parser.add_argument("--ray-address", default="local")
    prepare_parser.set_defaults(func=prepare)

    dlc_parser = commands.add_parser(
        "dlc",
        help="Run a complete job when DLC broadcasts this command to every Worker",
    )
    dlc_parser.add_argument("--job-dir", required=True, help="Shared job directory")
    dlc_parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Existing shard-safe Data-Juicer recipe",
    )
    dlc_parser.add_argument(
        "--dataset-path",
        help="Optional JSONL file/directory overriding the recipe dataset_path",
    )
    dlc_parser.add_argument(
        "--nodes",
        type=_positive_int,
        help=("Expected DLC Worker count for logging; required and enforced only " "with --require-all-nodes"),
    )
    dlc_parser.add_argument("--num-shards", type=_positive_int, default=4)
    dlc_parser.add_argument(
        "--require-all-nodes",
        action="store_true",
        help=("Strict smoke-test mode: cap claims per Worker and verify that all " "--nodes hostnames participated"),
    )
    dlc_parser.add_argument(
        "--ray-address",
        default="local",
        help="Ray address used independently inside each node (default: local)",
    )
    dlc_parser.add_argument(
        "--output",
        help="Merged output; defaults to merged.jsonl in the job directory",
    )
    dlc_parser.add_argument(
        "--wait-timeout-secs",
        type=_positive_float,
        default=DEFAULT_WAIT_TIMEOUT_SECS,
        help="Maximum wait for another DLC instance or all shards (default: 35h)",
    )
    dlc_parser.add_argument(
        "--poll-interval-secs",
        type=_positive_float,
        default=DEFAULT_POLL_INTERVAL_SECS,
    )
    dlc_parser.set_defaults(func=dlc)

    worker_parser = commands.add_parser("worker", help="Run on any Worker node")
    worker_parser.add_argument("--job-dir", required=True, help="Shared job directory")
    worker_parser.add_argument(
        "--max-shards",
        type=_positive_int,
        help="Optional maximum number of claims handled by this invocation",
    )
    worker_parser.add_argument(
        "--ray-address",
        help="Optional worker override, for example auto for a persistent local Ray head",
    )
    worker_parser.set_defaults(func=worker)

    status_parser = commands.add_parser("status", help="Show current shard ownership state")
    status_parser.add_argument("--job-dir", required=True)
    status_parser.set_defaults(func=status)

    verify_parser = commands.add_parser("verify", help="Check ownership and merge")
    verify_parser.add_argument("--job-dir", required=True)
    verify_parser.add_argument("--output", help="Merged output; defaults inside the job directory")
    verify_parser.add_argument("--expect-nodes", type=_positive_int, default=1)
    verify_parser.set_defaults(func=verify)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

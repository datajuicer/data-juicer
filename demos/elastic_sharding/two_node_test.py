#!/usr/bin/env python3
"""Backward-compatible strict two-node wrapper for ``dlc_job.py``.

New jobs should call ``dlc_job.py`` directly. This wrapper keeps the original
two-node defaults: two expected Workers, strict participation, two claims per
Worker for four shards, and ``two-node-merged.jsonl`` as the default output.
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dlc_job import main as _generic_main  # noqa: E402


def _has_option(arguments: list[str], option: str) -> bool:
    return option in arguments or any(value.startswith(f"{option}=") for value in arguments)


def _option_value(arguments: list[str], option: str) -> str | None:
    for index, value in enumerate(arguments):
        if value == option and index + 1 < len(arguments):
            return arguments[index + 1]
        if value.startswith(f"{option}="):
            return value.split("=", 1)[1]
    return None


def _compat_arguments(arguments: list[str]) -> list[str]:
    converted = list(arguments)
    if not converted:
        return converted

    command = converted[0]
    if command == "dlc":
        if not _has_option(converted, "--nodes"):
            converted.extend(["--nodes", "2"])
        if not _has_option(converted, "--require-all-nodes"):
            converted.append("--require-all-nodes")
        if not _has_option(converted, "--output"):
            job_dir = _option_value(converted, "--job-dir")
            if job_dir:
                converted.extend(["--output", str(Path(job_dir) / "two-node-merged.jsonl")])
    elif command == "worker" and not _has_option(converted, "--max-shards"):
        converted.extend(["--max-shards", "2"])
    elif command == "verify":
        if not _has_option(converted, "--expect-nodes"):
            converted.extend(["--expect-nodes", "2"])
        if not _has_option(converted, "--output"):
            job_dir = _option_value(converted, "--job-dir")
            if job_dir:
                converted.extend(["--output", str(Path(job_dir) / "two-node-merged.jsonl")])
    return converted


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    return _generic_main(_compat_arguments(arguments))


if __name__ == "__main__":
    raise SystemExit(main())

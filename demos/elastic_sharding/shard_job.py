#!/usr/bin/env python3
"""Compatibility CLI for the core elastic-sharding state machine."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_juicer.core.executor.elastic_sharding import job as _job

# This demo historically exposed helper functions that its tests and some
# external smoke harnesses import directly. Re-export them while keeping the
# implementation in core.
_EXPORTED_NAMES = tuple(name for name in dir(_job) if not name.startswith("__") and name != "main")
globals().update({name: getattr(_job, name) for name in _EXPORTED_NAMES})


def main(argv=None):
    # Preserve the old module's monkeypatch behavior for downstream smoke
    # harnesses that replace a helper before invoking the CLI.
    for name in _EXPORTED_NAMES:
        setattr(_job, name, globals()[name])
    return _job.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Compatibility wrapper for the installed elastic sharding command.

New integrations should invoke ``dj-process-sharded``.  The demo keeps this
script so existing prepare/worker/status/retry/merge commands continue to
work from a source checkout.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from data_juicer.tools.elastic_sharding import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Filter processed jsonl by meta.agent_bad_case_tier (and optional model).

Example:
  python demos/agent/scripts/slice_export_by_tier.py \\
    --input ./outputs/agent_quality/processed.jsonl \\
    --tier high_precision \\
    --output ./outputs/agent_quality/hp.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from dj_export_row import get_dj_meta, iter_merged_export_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--tier",
        required=True,
        help="high_precision | watchlist | none (exact match)",
    )
    ap.add_argument("--model", default="", help="Optional filter agent_request_model")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    n_in = n_out = 0
    with open(args.output, "w", encoding="utf-8") as fout:
        for _, row in iter_merged_export_rows(args.input):
            if args.limit is not None and n_out >= args.limit:
                break
            n_in += 1
            meta = get_dj_meta(row)
            if str(meta.get("agent_bad_case_tier", "")) != args.tier:
                continue
            if args.model and str(meta.get("agent_request_model", "")) != args.model:
                continue
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_out += 1
    print(f"Read ~{n_in} lines, wrote {n_out} to {args.output}")


if __name__ == "__main__":
    main()

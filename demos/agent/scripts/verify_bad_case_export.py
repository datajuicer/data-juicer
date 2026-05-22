#!/usr/bin/env python3
"""Sanity-check a dj-process jsonl export for bad-case workflow fields.

Reads ``processed.jsonl`` and, if ``meta`` is absent (default export), merges
``processed_stats.jsonl`` line-by-line (Data-Juicer ``keep_stats_in_res_ds: false``).

Exit 0 if checks pass; non-zero if file missing or required keys absent.

Example:
  python demos/agent/scripts/verify_bad_case_export.py \\
    --input ./outputs/agent_bad_case_smoke/processed.jsonl

  python demos/agent/scripts/verify_bad_case_export.py \\
    --input ./batch1/processed.jsonl --input ./batch2/processed.jsonl
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from dj_export_row import get_dj_meta, infer_stats_jsonl_path, iter_merged_export_rows


def _check_row(row: dict, line_no: int) -> List[str]:
    errs: List[str] = []
    meta = get_dj_meta(row)
    if not meta:
        errs.append(
            f"line {line_no}: missing __dj__meta__/meta "
            "(use sibling *_stats.jsonl next to processed.jsonl, or set keep_stats_in_res_ds: true)"
        )
        return errs
    if "agent_bad_case_tier" not in meta:
        errs.append(f"line {line_no}: meta.agent_bad_case_tier missing")
    sigs = meta.get("agent_bad_case_signals")
    if not isinstance(sigs, list):
        errs.append(f"line {line_no}: meta.agent_bad_case_signals should be a list")
    return errs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input",
        action="append",
        dest="inputs",
        required=True,
        metavar="PATH",
        help="processed.jsonl from dj-process (repeatable; checked in order)",
    )
    ap.add_argument(
        "--min-rows",
        type=int,
        default=1,
        help="Minimum non-empty JSON lines to require (default 1)",
    )
    ap.add_argument(
        "--require-insight",
        action="store_true",
        help="Require meta.agent_insight_llm (full recipe with step 10)",
    )
    args = ap.parse_args()

    for inp in args.inputs:
        if not os.path.isfile(inp):
            print(f"ERROR: not found: {inp}", file=sys.stderr)
            return 2

    all_errs: List[str] = []
    n_json_total = 0
    for inp in args.inputs:
        n_json = 0
        for line_no, row in iter_merged_export_rows(inp):
            if not isinstance(row, dict):
                all_errs.append(f"{inp}: line {line_no}: root must be object")
                continue
            n_json += 1
            for msg in _check_row(row, line_no):
                all_errs.append(f"{inp}: {msg}")
            if args.require_insight:
                meta = get_dj_meta(row)
                if not isinstance(meta.get("agent_insight_llm"), dict):
                    all_errs.append(
                        f"{inp}: line {line_no}: agent_insight_llm missing (need full pipeline)"
                    )
        n_json_total += n_json
        stats_side = infer_stats_jsonl_path(inp)
        merged_note = ""
        if os.path.isfile(stats_side):
            merged_note = f" (merged with {os.path.basename(stats_side)} where needed)"
        print(
            f"OK: {inp} — checked {n_json} row(s), bad-case fields present.{merged_note}"
        )

    if n_json_total < args.min_rows:
        print(
            f"ERROR: expected at least {args.min_rows} JSON rows in total, got {n_json_total}",
            file=sys.stderr,
        )
        return 3

    if all_errs:
        for e in all_errs[:50]:
            print(e, file=sys.stderr)
        if len(all_errs) > 50:
            print(f"... and {len(all_errs) - 50} more errors", file=sys.stderr)
        return 4

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

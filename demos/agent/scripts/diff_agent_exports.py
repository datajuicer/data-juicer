#!/usr/bin/env python3
# Copyright 2025 The Data-Juicer Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare two agent JSONL exports by ``id`` and print meta/stats field changes.

Usage (repo root)::

    python demos/agent/scripts/diff_agent_exports.py \\
        --before ./outputs/run_a/processed.jsonl \\
        --after ./outputs/run_b/train_data.jsonl \\
        --meta-keys agent_training_dataset_tier agent_bad_case_tier \\
        --stats-keys llm_analysis_score llm_difficulty_score

Requires ``jsonlines`` (same as Data-Juicer env).
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Iterable, List, Optional, Set

META_CANDIDATES = ("__dj__meta__", "meta")
STATS_CANDIDATES = ("__dj__stats__", "stats")


def _pick(d: dict, keys: Iterable[str]) -> Optional[dict]:
    for k in keys:
        v = d.get(k)
        if isinstance(v, dict):
            return v
    return None


def _json_sig(x: Any) -> str:
    return json.dumps(x, sort_keys=True, default=str)


def _load_index(path: str) -> Dict[str, dict]:
    import jsonlines

    out: Dict[str, dict] = {}
    with jsonlines.open(path) as reader:
        for row in reader:
            rid = row.get("id")
            if rid is None:
                continue
            out[str(rid)] = row
    return out


def main() -> None:
    desc = "Diff agent JSONL rows by id on selected meta/stats keys."
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--before", required=True, help="Baseline JSONL path")
    p.add_argument("--after", required=True, help="New JSONL path")
    p.add_argument(
        "--meta-keys",
        nargs="*",
        default=[],
        help="Top-level keys under meta (__dj__meta__)",
    )
    p.add_argument(
        "--stats-keys",
        nargs="*",
        default=[],
        help="Top-level keys under stats (__dj__stats__)",
    )
    p.add_argument("--limit", type=int, default=50, help="Max changed ids to print")
    args = p.parse_args()

    before = _load_index(args.before)
    after = _load_index(args.after)
    ids: Set[str] = set(before) & set(after)

    changed: List[str] = []
    for rid in sorted(ids):
        a, b = before[rid], after[rid]
        ma = _pick(a, META_CANDIDATES)
        mb = _pick(b, META_CANDIDATES)
        sa = _pick(a, STATS_CANDIDATES)
        sb = _pick(b, STATS_CANDIDATES)
        diffs: List[str] = []
        for k in args.meta_keys:
            va = ma.get(k) if ma else None
            vb = mb.get(k) if mb else None
            if _json_sig(va) != _json_sig(vb):
                diffs.append(f"meta.{k}: {va!r} -> {vb!r}")
        for k in args.stats_keys:
            va = sa.get(k) if sa else None
            vb = sb.get(k) if sb else None
            if _json_sig(va) != _json_sig(vb):
                diffs.append(f"stats.{k}: {va!r} -> {vb!r}")
        if diffs:
            changed.append(rid)

    only_after = set(after) - set(before)
    only_before = set(before) - set(after)
    msg = (
        f"shared_ids={len(ids)} only_before={len(only_before)} "
        f"only_after={len(only_after)} changed={len(changed)}"
    )
    print(msg)
    for rid in changed[: args.limit]:
        a, b = before[rid], after[rid]
        ma = _pick(a, META_CANDIDATES)
        mb = _pick(b, META_CANDIDATES)
        sa = _pick(a, STATS_CANDIDATES)
        sb = _pick(b, STATS_CANDIDATES)
        print(f"\n=== id={rid} ===")
        for k in args.meta_keys:
            va = ma.get(k) if ma else None
            vb = mb.get(k) if mb else None
            if _json_sig(va) != _json_sig(vb):
                print(f"  meta.{k}: {va!r} -> {vb!r}")
        for k in args.stats_keys:
            va = sa.get(k) if sa else None
            vb = sb.get(k) if sb else None
            if _json_sig(va) != _json_sig(vb):
                print(f"  stats.{k}: {va!r} -> {vb!r}")
    if len(changed) > args.limit:
        rest = len(changed) - args.limit
        print(f"\n... ({rest} more changed ids truncated)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Summarize bad-case tiers and signal codes by model / pt (optional pandas).

Example:
  python demos/agent/scripts/analyze_bad_case_cohorts.py \\
    --input ./outputs/agent_quality/processed.jsonl \\
    --out-csv ./outputs/agent_quality/cohort_summary.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Sequence, Tuple, Union

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from dj_export_row import get_dj_meta, iter_merged_export_rows


def load_merged_rows(
    path: Union[str, Sequence[str]],
    limit: Optional[int] = None,
) -> List[dict]:
    """Load jsonl with optional ``*_stats.jsonl`` merge (same as CLI).

    If ``path`` is a sequence of paths, read each file in order (no on-disk merge).
    ``limit`` caps the total row count across all files.
    """
    paths: List[str] = [path] if isinstance(path, str) else list(path)
    rows: List[dict] = []
    for p in paths:
        for _, row in iter_merged_export_rows(p):
            if limit is not None and len(rows) >= limit:
                return rows
            rows.append(row)
    return rows


def aggregate_cohort_stdlib(rows: List[dict]) -> List[dict]:
    """Return list of dict rows for CSV: model, pt, tier, count, top_signals."""
    tier_counts: DefaultDict[Tuple[str, str, str], int] = defaultdict(int)
    signal_counts: DefaultDict[Tuple[str, str, str], Counter] = defaultdict(Counter)

    for row in rows:
        meta = get_dj_meta(row)
        model = str(meta.get("agent_request_model", "_unknown"))
        pt = str(meta.get("agent_pt", "_unknown"))
        tier = str(meta.get("agent_bad_case_tier", "none"))
        tier_counts[(model, pt, tier)] += 1
        sigs = meta.get("agent_bad_case_signals") or []
        if isinstance(sigs, list):
            for s in sigs:
                if isinstance(s, dict) and s.get("code"):
                    signal_counts[(model, pt, tier)][str(s["code"])] += 1

    models_pts = {(m, p) for (m, p, _) in tier_counts}
    out = []
    for model, pt in sorted(models_pts):
        for tier in ("high_precision", "watchlist", "none"):
            c = tier_counts.get((model, pt, tier), 0)
            top_ct = signal_counts.get((model, pt, tier), Counter())
            top = ", ".join(f"{k}:{v}" for k, v in top_ct.most_common(5))
            out.append(
                {
                    "agent_request_model": model,
                    "agent_pt": pt,
                    "tier": tier,
                    "count": c,
                    "top_signal_codes": top,
                }
            )
    return out


def _aggregate_pandas(rows: List[dict]) -> "Any":
    import pandas as pd

    records = []
    for row in rows:
        meta = get_dj_meta(row)
        insight = meta.get("agent_insight_llm") or {}
        records.append(
            {
                "model": meta.get("agent_request_model"),
                "pt": meta.get("agent_pt"),
                "tier": meta.get("agent_bad_case_tier"),
                "priority": insight.get("human_review_priority"),
                "headline": (insight.get("headline") or "")[:120],
            }
        )
    df = pd.DataFrame(records)
    g = df.groupby(["model", "pt", "tier"], dropna=False).size().reset_index(name="count")
    return g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-csv", default="", help="Write cohort summary CSV")
    ap.add_argument("--limit", type=int, default=None, help="Max lines to read")
    ap.add_argument(
        "--pandas-priority",
        action="store_true",
        help="Use pandas groupby (needs pandas); adds human_review_priority slice",
    )
    args = ap.parse_args()

    rows = load_merged_rows(args.input, args.limit)
    print(f"Loaded {len(rows)} records from {args.input}")

    if args.pandas_priority:
        try:
            g = _aggregate_pandas(rows)
            print(g.to_string(index=False))
            if args.out_csv:
                g.to_csv(args.out_csv, index=False)
                print(f"Wrote {args.out_csv}")
        except ImportError:
            print("pandas not installed; falling back to stdlib aggregation")
            args.pandas_priority = False

    if not args.pandas_priority:
        flat = aggregate_cohort_stdlib(rows)
        if args.out_csv:
            import csv

            with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(
                    f,
                    fieldnames=[
                        "agent_request_model",
                        "agent_pt",
                        "tier",
                        "count",
                        "top_signal_codes",
                    ],
                )
                w.writeheader()
                w.writerows(flat)
            print(f"Wrote {args.out_csv}")
        else:
            for r in flat:
                if r["count"] or r["top_signal_codes"]:
                    print(r)


if __name__ == "__main__":
    main()

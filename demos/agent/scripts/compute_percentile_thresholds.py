#!/usr/bin/env python3
"""Compute percentiles for token / latency / perplexity; optional calibration JSON.

Example (report only):
  python demos/agent/scripts/compute_percentile_thresholds.py \\
    --input ./outputs/agent_quality/processed.jsonl

Example (write P95 thresholds for agent_bad_case_signal_mapper):
  python demos/agent/scripts/compute_percentile_thresholds.py \\
    --input ./outputs/agent_quality/processed.jsonl \\
    --write-calibration ./outputs/agent_quality/bad_case_calibration.json \\
    --calibration-percentile 95
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from dj_export_row import get_dj_meta, get_dj_stats, iter_merged_export_rows


def _percentile(sorted_vals: List[float], p: float) -> Optional[float]:
    if not sorted_vals:
        return None
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


def _gather(
    path: str,
    by_pt: bool,
) -> Tuple[DefaultDict[str, Dict[str, List[float]]], int]:
    """group_key -> {'tokens': [...], 'latency': [...]}"""
    buckets: DefaultDict[str, Dict[str, List[float]]] = defaultdict(
        lambda: {"tokens": [], "latency": []}
    )
    n = 0
    for _, row in iter_merged_export_rows(path):
        meta = get_dj_meta(row)
        model = meta.get("agent_request_model", "_unknown_model")
        pt = meta.get("agent_pt", "_unknown_pt")
        key = f"{model}"
        if by_pt:
            key = f"{model} | pt={pt}"
        t = meta.get("total_tokens")
        if t is not None:
            try:
                buckets[key]["tokens"].append(float(t))
            except (TypeError, ValueError):
                pass
        lat = meta.get("agent_total_cost_time_ms")
        if lat is not None:
            try:
                buckets[key]["latency"].append(float(lat))
            except (TypeError, ValueError):
                pass
        n += 1
    return buckets, n


def _gather_by_request_model_only(path: str) -> Tuple[Dict[str, Dict[str, List[float]]], int]:
    """Single key: agent_request_model (for calibration JSON). Includes perplexity."""
    buckets: DefaultDict[str, Dict[str, List[float]]] = defaultdict(
        lambda: {"tokens": [], "latency": [], "perplexity": []}
    )
    n = 0
    for _, row in iter_merged_export_rows(path):
        meta = get_dj_meta(row)
        stats = get_dj_stats(row)
        model = str(meta.get("agent_request_model") or "_unknown_model")
        t = meta.get("total_tokens")
        if t is not None:
            try:
                buckets[model]["tokens"].append(float(t))
            except (TypeError, ValueError):
                pass
        lat = meta.get("agent_total_cost_time_ms")
        if lat is not None:
            try:
                buckets[model]["latency"].append(float(lat))
            except (TypeError, ValueError):
                pass
        ppl = stats.get("perplexity")
        if ppl is not None:
            try:
                buckets[model]["perplexity"].append(float(ppl))
            except (TypeError, ValueError):
                pass
        n += 1
    return dict(buckets), n


def _pool_all(buckets: Dict[str, Dict[str, List[float]]]) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {"tokens": [], "latency": [], "perplexity": []}
    for b in buckets.values():
        out["tokens"].extend(b.get("tokens") or [])
        out["latency"].extend(b.get("latency") or [])
        out["perplexity"].extend(b.get("perplexity") or [])
    return out


def _build_calibration_doc(
    buckets: Dict[str, Dict[str, List[float]]],
    percentile: float,
) -> Dict[str, Any]:
    """Thresholds: flag when value *exceeds* P95 (tail / high cost / high ppl)."""
    pooled = _pool_all(buckets)

    def row_from_lists(lists: Dict[str, List[float]]) -> Dict[str, Any]:
        r: Dict[str, Any] = {}
        tv = sorted(lists.get("tokens") or [])
        lv = sorted(lists.get("latency") or [])
        pv = sorted(lists.get("perplexity") or [])
        pt = _percentile(tv, percentile)
        lm = _percentile(lv, percentile)
        pp = _percentile(pv, percentile)
        if pt is not None:
            r["max_total_tokens"] = int(round(pt))
        if lm is not None:
            r["max_latency_ms"] = int(round(lm))
        if pp is not None:
            r["perplexity_high_threshold"] = float(pp)
        return r

    default_row = row_from_lists(pooled)
    by_model: Dict[str, Any] = {}
    for model, lists in buckets.items():
        row = row_from_lists(lists)
        if row:
            by_model[model] = row

    return {
        "version": 1,
        "percentile": percentile,
        "description": (
            "Thresholds are inclusive P-percentile values from a prior export; "
            "agent_bad_case_signal_mapper flags samples *strictly above* "
            "max_total_tokens / max_latency_ms, and perplexity *>=* threshold."
        ),
        "default": default_row,
        "by_request_model": by_model,
    }


def _report(buckets: Dict[str, Dict[str, List[float]]], percentiles: List[float]) -> None:
    for key in sorted(buckets.keys()):
        print(f"\n=== {key} ===")
        for field, label in (("tokens", "total_tokens"), ("latency", "latency_ms")):
            vals = sorted(buckets[key][field])
            if not vals:
                print(f"  {label}: (no values)")
                continue
            parts = [f"n={len(vals)}"]
            for p in percentiles:
                pv = _percentile(vals, p)
                if pv is not None:
                    parts.append(f"p{p:g}={pv:.1f}")
            print(f"  {label}: " + ", ".join(parts))


def main() -> None:
    ap = argparse.ArgumentParser(description="Percentiles for agent export jsonl")
    ap.add_argument("--input", required=True, help="Path to processed.jsonl")
    ap.add_argument(
        "--by-pt",
        action="store_true",
        help="Split report buckets by meta.agent_pt as well as agent_request_model",
    )
    ap.add_argument(
        "--percentiles",
        default="50,90,95,99",
        help="Comma-separated percentiles for console report",
    )
    ap.add_argument(
        "--write-calibration",
        default="",
        help="Write JSON for agent_bad_case_signal_mapper (by request_model + default)",
    )
    ap.add_argument(
        "--calibration-percentile",
        type=float,
        default=95.0,
        help="Percentile used as threshold in calibration file (default 95)",
    )
    args = ap.parse_args()
    pct = [float(x) for x in args.percentiles.split(",") if x.strip()]

    if args.write_calibration:
        buckets_m, n = _gather_by_request_model_only(args.input)
        print(f"Read {n} lines from {args.input} (for calibration by request_model)")
        if not buckets_m:
            print("No data; check meta.agent_request_model and usage_counter.")
            return
        doc = _build_calibration_doc(buckets_m, args.calibration_percentile)
        with open(args.write_calibration, "w", encoding="utf-8") as out:
            json.dump(doc, out, ensure_ascii=False, indent=2)
        print(f"Wrote {args.write_calibration} (percentile={args.calibration_percentile:g})")
        print(
            "Recipe: set agent_bad_case_signal_mapper "
            "auto_calibrate_thresholds: true and "
            f"calibration_json_path: {args.write_calibration!r}"
        )
        return

    buckets, n = _gather(args.input, by_pt=bool(args.by_pt))
    print(f"Read {n} lines from {args.input}")
    if not buckets:
        print("No meta buckets; check copy_lineage_fields and usage_counter.")
        return
    _report(buckets, pct)
    print(
        "\nTip: use --write-calibration PATH --calibration-percentile 95 "
        "to emit JSON for auto thresholds in the recipe."
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Pre-training transition report for agent dataset quality.

This script compares two exports by ``id`` and reports:
- Red / Yellow / Green transition matrix
- Net green gain and risky promotion ratios
- Signal burden deltas (high / medium bad-case signals)
- Ranking sanity checks (Spearman + top-k coverage / lift)
- Optional stage retention and drop-profile analysis
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dj_export_row import get_dj_meta, get_dj_stats, iter_merged_export_rows

VALID_COLORS = ("red", "yellow", "green")


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def _load_index(path: str, limit: Optional[int] = None) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for _, row in iter_merged_export_rows(path):
        rid = row.get("id")
        if rid is None:
            continue
        out[str(rid)] = row
        if limit is not None and len(out) >= limit:
            break
    return out


def _normalize_01(v: Optional[float]) -> Optional[float]:
    if v is None:
        return None
    return max(0.0, min(1.0, float(v)))


def _normalize_difficulty(v: Optional[float]) -> Optional[float]:
    if v is None:
        return None
    if v > 1.0:
        return _normalize_01(v / 5.0)
    return _normalize_01(v)


def _max_taxonomy_severity(meta: Dict[str, Any]) -> Optional[float]:
    tax = meta.get("agent_error_taxonomy")
    if not isinstance(tax, dict):
        return None
    buckets = tax.get("buckets")
    if not isinstance(buckets, dict):
        return None
    vals: List[float] = []
    for item in buckets.values():
        if not isinstance(item, dict):
            continue
        sev = _safe_float(item.get("severity"))
        if sev is not None:
            vals.append(_normalize_01(sev) or 0.0)
    return max(vals) if vals else None


def _cross_model_gap(meta: Dict[str, Any]) -> Optional[float]:
    pair = meta.get("agent_cross_model_pair")
    if not isinstance(pair, dict):
        return None
    if not bool(pair.get("has_pairwise_contrast")):
        return None
    delta = _safe_float(pair.get("delta_to_best"))
    if delta is None:
        return None
    return _normalize_01(delta * 2.0)


def _hardness_proxy(row: dict) -> Optional[float]:
    """Proxy for ``model is likely wrong and sample is hard``."""
    meta = get_dj_meta(row)
    stats = get_dj_stats(row)
    parts: List[float] = []

    d = _normalize_difficulty(_safe_float(stats.get("llm_difficulty_score")))
    if d is not None:
        parts.append(d)

    ratio = _safe_float(meta.get("tool_success_ratio"))
    if ratio is not None and ratio >= 0.0:
        parts.append(_normalize_01(1.0 - ratio) or 0.0)

    tmax = _max_taxonomy_severity(meta)
    if tmax is not None:
        parts.append(tmax)

    gap = _cross_model_gap(meta)
    if gap is not None:
        parts.append(gap)

    if not parts:
        return None
    return sum(parts) / len(parts)


def _quality_proxy(row: dict) -> Optional[float]:
    """Independent quality proxy, distinct from ``agent_learnable_value``."""
    meta = get_dj_meta(row)
    stats = get_dj_stats(row)
    parts: List[float] = []

    a = _normalize_01(_safe_float(stats.get("llm_analysis_score")))
    if a is not None:
        parts.append(a)

    tsr = _safe_float(meta.get("tool_success_ratio"))
    if tsr is not None and tsr >= 0.0:
        parts.append(_normalize_01(tsr) or 0.0)

    mx = _max_taxonomy_severity(meta)
    if mx is not None:
        parts.append(_normalize_01(1.0 - mx) or 0.0)

    if not parts:
        return None
    return sum(parts) / len(parts)


def _color_of_row(row: dict, require_safety_ok: bool) -> str:
    meta = get_dj_meta(row)
    tier = str(
        meta.get("agent_training_dataset_tier")
        or meta.get("agent_learnable_value_tier")
        or "none"
    ).lower()
    bad_tier = str(meta.get("agent_bad_case_tier") or "none").lower()

    tax = meta.get("agent_error_taxonomy") or {}
    hard_drop = (
        bool(tax.get("hard_drop_recommended"))
        if isinstance(tax, dict)
        else False
    )

    gate = meta.get("agent_training_safety_gate") or {}
    gate_ok = (
        isinstance(gate, dict)
        and gate.get("ok") is True
        and not bool(gate.get("skipped"))
    )

    if hard_drop or bad_tier == "high_precision":
        return "red"
    if bad_tier == "watchlist" or tier == "bronze":
        return "yellow"
    if tier in ("gold", "silver") and bad_tier == "none" and not hard_drop:
        if require_safety_ok and not gate_ok:
            return "yellow"
        return "green"
    return "yellow"


def _signal_burden(row: dict) -> Tuple[int, int]:
    meta = get_dj_meta(row)
    sigs = meta.get("agent_bad_case_signals")
    if not isinstance(sigs, list):
        return 0, 0
    high = 0
    medium = 0
    for item in sigs:
        if not isinstance(item, dict):
            continue
        w = str(item.get("weight") or "").lower()
        if w == "high":
            high += 1
        elif w == "medium":
            medium += 1
    return high, medium


def _avg_rank(values: Sequence[float]) -> List[float]:
    pairs = sorted((v, i) for i, v in enumerate(values))
    ranks = [0.0] * len(values)
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[pairs[k][1]] = avg
        i = j
    return ranks


def _pearson(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    if len(x) != len(y) or len(x) < 2:
        return None
    mx = statistics.fmean(x)
    my = statistics.fmean(y)
    vx = sum((a - mx) ** 2 for a in x)
    vy = sum((b - my) ** 2 for b in y)
    if vx <= 0.0 or vy <= 0.0:
        return None
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    return cov / math.sqrt(vx * vy)


def _spearman(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    if len(x) != len(y) or len(x) < 2:
        return None
    return _pearson(_avg_rank(x), _avg_rank(y))


def _coverage_at_k(rows: List[Tuple[float, str]], ratio: float) -> Optional[float]:
    if not rows:
        return None
    n = max(1, int(math.ceil(len(rows) * ratio)))
    top = sorted(rows, key=lambda t: t[0], reverse=True)[:n]
    green = sum(1 for _, c in top if c == "green")
    return green / len(top) if top else None


def _rate_at_k(rows: List[Tuple[float, bool]], ratio: float) -> Optional[float]:
    if not rows:
        return None
    n = max(1, int(math.ceil(len(rows) * ratio)))
    top = sorted(rows, key=lambda t: t[0], reverse=True)[:n]
    hits = sum(1 for _, flag in top if flag)
    return hits / len(top) if top else None


def _mean_at_k(rows: List[Tuple[float, float]], ratio: float) -> Optional[float]:
    if not rows:
        return None
    n = max(1, int(math.ceil(len(rows) * ratio)))
    top = sorted(rows, key=lambda t: t[0], reverse=True)[:n]
    return statistics.fmean(v for _, v in top)


def _quantile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    pos = (len(s) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return s[int(pos)]
    return s[lo] * (hi - pos) + s[hi] * (pos - lo)


def _fmt(v: Optional[float]) -> str:
    if v is None:
        return "n/a"
    return f"{v:.4f}"


def _parse_stage_arg(items: Sequence[str]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"--stage expects NAME=PATH, got: {item!r}")
        name, path = item.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError(f"--stage expects NAME=PATH, got: {item!r}")
        out.append((name, path))
    return out


def _drop_profile(
    prev_rows: Dict[str, dict],
    dropped_ids: Sequence[str],
    require_safety_ok: bool,
) -> Dict[str, Any]:
    profile = {
        "count": len(dropped_ids),
        "color_counts": {c: 0 for c in VALID_COLORS},
        "hard_drop_recommended": 0,
        "sys_or_harness_noise": 0,
        "bad_case_high_precision": 0,
        "bad_case_watchlist": 0,
    }
    for rid in dropped_ids:
        row = prev_rows.get(rid)
        if row is None:
            continue
        color = _color_of_row(row, require_safety_ok)
        profile["color_counts"][color] += 1
        meta = get_dj_meta(row)
        tax = meta.get("agent_error_taxonomy") or {}
        if isinstance(tax, dict) and bool(tax.get("hard_drop_recommended")):
            profile["hard_drop_recommended"] += 1
        sysn = meta.get("agent_sys_log_noise") or {}
        harn = meta.get("agent_harness_noise") or {}
        if (isinstance(sysn, dict) and sysn.get("is_likely_noise")) or (
            isinstance(harn, dict) and harn.get("is_likely_noise")
        ):
            profile["sys_or_harness_noise"] += 1
        bad_tier = str(meta.get("agent_bad_case_tier") or "none").lower()
        if bad_tier == "high_precision":
            profile["bad_case_high_precision"] += 1
        elif bad_tier == "watchlist":
            profile["bad_case_watchlist"] += 1
    return profile


def _stage_retention_report(
    stage_defs: List[Tuple[str, str]],
    limit: Optional[int],
    require_safety_ok: bool,
) -> Dict[str, Any]:
    loaded = [(name, _load_index(path, limit)) for name, path in stage_defs]
    rows = []
    for i in range(len(loaded) - 1):
        prev_name, prev_map = loaded[i]
        next_name, next_map = loaded[i + 1]
        prev_ids = set(prev_map)
        next_ids = set(next_map)
        kept_ids = sorted(prev_ids & next_ids)
        drop_ids = sorted(prev_ids - next_ids)
        added_ids = sorted(next_ids - prev_ids)
        retention = len(kept_ids) / len(prev_ids) if prev_ids else 0.0
        rows.append(
            {
                "from": prev_name,
                "to": next_name,
                "from_count": len(prev_ids),
                "to_count": len(next_ids),
                "kept_count": len(kept_ids),
                "dropped_count": len(drop_ids),
                "added_count": len(added_ids),
                "retention": retention,
                "drop_profile": _drop_profile(prev_map, drop_ids, require_safety_ok),
            }
        )
    return {"stages": [name for name, _ in stage_defs], "edges": rows}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Transition matrix + ranking sanity report "
            "(training-free offline evaluation)."
        )
    )
    ap.add_argument("--before", required=True, help="Baseline export jsonl")
    ap.add_argument("--after", required=True, help="Candidate export jsonl")
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max ids to read per side",
    )
    ap.add_argument(
        "--require-safety-ok",
        action="store_true",
        help="Treat non-ok safety gate as not-green in strict mapping.",
    )
    ap.add_argument(
        "--k-ratios",
        default="0.1,0.2,0.3",
        help="Comma-separated top-k coverage ratios.",
    )
    ap.add_argument(
        "--hardness-quantile",
        type=float,
        default=0.75,
        help="Quantile to define hard label from hardness proxy.",
    )
    ap.add_argument(
        "--stage",
        action="append",
        default=[],
        help=(
            "Optional stage retention input (repeatable): "
            "--stage raw=path --stage r1=path --stage r2=path"
        ),
    )
    ap.add_argument(
        "--out-json",
        default="",
        help="Optional output path for machine-readable summary JSON.",
    )
    args = ap.parse_args()

    before = _load_index(args.before, args.limit)
    after = _load_index(args.after, args.limit)
    shared_ids = sorted(set(before) & set(after))

    matrix = {b: {a: 0 for a in VALID_COLORS} for b in VALID_COLORS}
    risky_red_to_green = 0
    red_to_green_total = 0

    before_h: List[int] = []
    before_m: List[int] = []
    after_h: List[int] = []
    after_m: List[int] = []

    lv_before: List[float] = []
    q_before: List[float] = []
    h_before: List[float] = []
    lv_after: List[float] = []
    q_after: List[float] = []
    h_after: List[float] = []

    rank_green_before: List[Tuple[float, str]] = []
    rank_green_after: List[Tuple[float, str]] = []
    rank_hard_before: List[Tuple[float, bool]] = []
    rank_hard_after: List[Tuple[float, bool]] = []
    rank_hard_value_before: List[Tuple[float, float]] = []
    rank_hard_value_after: List[Tuple[float, float]] = []

    for rid in shared_ids:
        rb = before[rid]
        ra = after[rid]
        cb = _color_of_row(rb, args.require_safety_ok)
        ca = _color_of_row(ra, args.require_safety_ok)
        matrix[cb][ca] += 1

        hb, mb = _signal_burden(rb)
        ha, ma = _signal_burden(ra)
        before_h.append(hb)
        before_m.append(mb)
        after_h.append(ha)
        after_m.append(ma)

        mb_meta = get_dj_meta(rb)
        ma_meta = get_dj_meta(ra)
        lvb = _safe_float(mb_meta.get("agent_learnable_value"))
        lva = _safe_float(ma_meta.get("agent_learnable_value"))
        qb = _quality_proxy(rb)
        qa = _quality_proxy(ra)
        hbv = _hardness_proxy(rb)
        hav = _hardness_proxy(ra)

        if lvb is not None and qb is not None:
            lv_before.append(lvb)
            q_before.append(qb)
        if lva is not None and qa is not None:
            lv_after.append(lva)
            q_after.append(qa)
        if lvb is not None and hbv is not None:
            h_before.append(hbv)
        if lva is not None and hav is not None:
            h_after.append(hav)

        if lvb is not None:
            rank_green_before.append((lvb, cb))
        if lva is not None:
            rank_green_after.append((lva, ca))
        if lvb is not None and hbv is not None:
            rank_hard_value_before.append((lvb, hbv))
        if lva is not None and hav is not None:
            rank_hard_value_after.append((lva, hav))

        if cb == "red" and ca == "green":
            red_to_green_total += 1
            meta_after = get_dj_meta(ra)
            bad_tier = str(meta_after.get("agent_bad_case_tier") or "none").lower()
            tax = meta_after.get("agent_error_taxonomy") or {}
            hard_drop = (
                bool(tax.get("hard_drop_recommended"))
                if isinstance(tax, dict)
                else False
            )
            if bad_tier != "none" or hard_drop:
                risky_red_to_green += 1

    hq_before = _quantile(h_before, args.hardness_quantile)
    hq_after = _quantile(h_after, args.hardness_quantile)
    if hq_before is not None:
        rank_hard_before = [
            (lv, hv >= hq_before) for lv, hv in rank_hard_value_before
        ]
    if hq_after is not None:
        rank_hard_after = [
            (lv, hv >= hq_after) for lv, hv in rank_hard_value_after
        ]

    n = len(shared_ids)
    red_before = sum(matrix["red"].values())
    yellow_before = sum(matrix["yellow"].values())
    green_before = sum(matrix["green"].values())
    green_after = (
        matrix["red"]["green"]
        + matrix["yellow"]["green"]
        + matrix["green"]["green"]
    )
    red_after = (
        matrix["red"]["red"]
        + matrix["yellow"]["red"]
        + matrix["green"]["red"]
    )

    up_prob = (
        (matrix["red"]["green"] + matrix["yellow"]["green"])
        / max(1, red_before + yellow_before)
    )
    down_prob = (
        (matrix["green"]["yellow"] + matrix["green"]["red"])
        / max(1, green_before)
        if green_before > 0
        else 0.0
    )
    net_green_gain = up_prob - down_prob

    sp_q_before = _spearman(lv_before, q_before)
    sp_q_after = _spearman(lv_after, q_after)

    sp_h_before = _spearman(
        [x[0] for x in rank_hard_value_before],
        [x[1] for x in rank_hard_value_before],
    )
    sp_h_after = _spearman(
        [x[0] for x in rank_hard_value_after],
        [x[1] for x in rank_hard_value_after],
    )

    k_ratios = [float(x) for x in args.k_ratios.split(",") if x.strip()]
    green_cov_before = {k: _coverage_at_k(rank_green_before, k) for k in k_ratios}
    green_cov_after = {k: _coverage_at_k(rank_green_after, k) for k in k_ratios}
    hard_prec_before = {k: _rate_at_k(rank_hard_before, k) for k in k_ratios}
    hard_prec_after = {k: _rate_at_k(rank_hard_after, k) for k in k_ratios}
    hard_mean_before = {k: _mean_at_k(rank_hard_value_before, k) for k in k_ratios}
    hard_mean_after = {k: _mean_at_k(rank_hard_value_after, k) for k in k_ratios}

    global_h_before = (
        statistics.fmean(v for _, v in rank_hard_value_before)
        if rank_hard_value_before
        else None
    )
    global_h_after = (
        statistics.fmean(v for _, v in rank_hard_value_after)
        if rank_hard_value_after
        else None
    )
    hard_lift_before = {
        k: (
            (hard_mean_before[k] / global_h_before)
            if hard_mean_before[k] is not None and global_h_before
            else None
        )
        for k in k_ratios
    }
    hard_lift_after = {
        k: (
            (hard_mean_after[k] / global_h_after)
            if hard_mean_after[k] is not None and global_h_after
            else None
        )
        for k in k_ratios
    }

    stage_report = None
    if args.stage:
        stage_defs = _parse_stage_arg(args.stage)
        if len(stage_defs) >= 2:
            stage_report = _stage_retention_report(
                stage_defs,
                args.limit,
                args.require_safety_ok,
            )

    print(
        f"shared_ids={n} "
        f"only_before={len(before) - n} "
        f"only_after={len(after) - n}"
    )
    print("\nTransition Matrix (before -> after):")
    print("            red    yellow   green")
    for src in VALID_COLORS:
        print(
            f"{src:>8}  "
            f"{matrix[src]['red']:>6}  "
            f"{matrix[src]['yellow']:>7}  "
            f"{matrix[src]['green']:>6}"
        )

    print("\nHeadline Metrics:")
    print(f"- green_coverage_before={green_before / max(1, n):.4f}")
    print(f"- green_coverage_after ={green_after / max(1, n):.4f}")
    print(f"- red_rate_before      ={red_before / max(1, n):.4f}")
    print(f"- red_rate_after       ={red_after / max(1, n):.4f}")
    print(f"- net_green_gain       ={net_green_gain:.4f}")
    print(
        "- harmful_promotion_rate(red->green)="
        f"{risky_red_to_green / max(1, red_to_green_total):.4f}"
    )

    print("\nSignal Burden (mean per shared id):")
    print(f"- high_before={statistics.fmean(before_h) if before_h else 0.0:.4f}")
    print(f"- high_after ={statistics.fmean(after_h) if after_h else 0.0:.4f}")
    print(f"- medium_before={statistics.fmean(before_m) if before_m else 0.0:.4f}")
    print(f"- medium_after ={statistics.fmean(after_m) if after_m else 0.0:.4f}")

    print("\nRanking Sanity:")
    print(f"- spearman(lv vs quality_proxy) before={_fmt(sp_q_before)}")
    print(f"- spearman(lv vs quality_proxy) after ={_fmt(sp_q_after)}")
    print(f"- spearman(lv vs hardness_proxy) before={_fmt(sp_h_before)}")
    print(f"- spearman(lv vs hardness_proxy) after ={_fmt(sp_h_after)}")
    for k in k_ratios:
        pct = int(k * 100)
        print(
            f"- top_{pct:02d}pct_green before={_fmt(green_cov_before[k])} "
            f"after={_fmt(green_cov_after[k])}"
        )
        print(
            f"  top_{pct:02d}pct_hard_precision before={_fmt(hard_prec_before[k])} "
            f"after={_fmt(hard_prec_after[k])}"
        )
        print(
            f"  top_{pct:02d}pct_hard_lift before={_fmt(hard_lift_before[k])} "
            f"after={_fmt(hard_lift_after[k])}"
        )

    if stage_report:
        print("\nStage Retention:")
        for edge in stage_report["edges"]:
            print(
                f"- {edge['from']} -> {edge['to']}: "
                f"from={edge['from_count']} to={edge['to_count']} "
                f"kept={edge['kept_count']} dropped={edge['dropped_count']} "
                f"retention={edge['retention']:.4f}"
            )
            dp = edge["drop_profile"]
            cc = dp["color_counts"]
            print(
                "  drop_profile: "
                f"red={cc['red']} yellow={cc['yellow']} green={cc['green']} "
                f"hard_drop={dp['hard_drop_recommended']} "
                f"noise={dp['sys_or_harness_noise']} "
                f"high_precision={dp['bad_case_high_precision']} "
                f"watchlist={dp['bad_case_watchlist']}"
            )

    if args.out_json:
        out = {
            "shared_ids": n,
            "only_before": len(before) - n,
            "only_after": len(after) - n,
            "matrix": matrix,
            "metrics": {
                "green_coverage_before": green_before / max(1, n),
                "green_coverage_after": green_after / max(1, n),
                "red_rate_before": red_before / max(1, n),
                "red_rate_after": red_after / max(1, n),
                "net_green_gain": net_green_gain,
                "harmful_promotion_rate": (
                    risky_red_to_green / max(1, red_to_green_total)
                ),
            },
            "signal_burden": {
                "high_before_mean": (
                    statistics.fmean(before_h) if before_h else 0.0
                ),
                "high_after_mean": (
                    statistics.fmean(after_h) if after_h else 0.0
                ),
                "medium_before_mean": (
                    statistics.fmean(before_m) if before_m else 0.0
                ),
                "medium_after_mean": (
                    statistics.fmean(after_m) if after_m else 0.0
                ),
            },
            "ranking_sanity": {
                "spearman_quality_before": sp_q_before,
                "spearman_quality_after": sp_q_after,
                "spearman_hardness_before": sp_h_before,
                "spearman_hardness_after": sp_h_after,
                "hard_threshold_before": hq_before,
                "hard_threshold_after": hq_after,
                "topk_green_before": {
                    str(k): green_cov_before[k] for k in k_ratios
                },
                "topk_green_after": {
                    str(k): green_cov_after[k] for k in k_ratios
                },
                "topk_hard_precision_before": {
                    str(k): hard_prec_before[k] for k in k_ratios
                },
                "topk_hard_precision_after": {
                    str(k): hard_prec_after[k] for k in k_ratios
                },
                "topk_hard_lift_before": {
                    str(k): hard_lift_before[k] for k in k_ratios
                },
                "topk_hard_lift_after": {
                    str(k): hard_lift_after[k] for k in k_ratios
                },
            },
            "stage_retention": stage_report,
        }
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\nWrote JSON summary: {args.out_json}")


if __name__ == "__main__":
    main()

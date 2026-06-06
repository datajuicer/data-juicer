# Copyright 2025 The Data-Juicer Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Collapse existing LLM / dialog meta into a small training-oriented taxonomy (no LLM).

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from data_juicer.ops.base_op import OPERATORS, TAGGING_OPS, Mapper
from data_juicer.utils.constant import Fields, MetaKeys, StatsKeys

OP_NAME = "agent_error_taxonomy_mapper"

_DIALOG_KEYS = (
    MetaKeys.dialog_memory_consistency,
    MetaKeys.dialog_coreference,
    MetaKeys.agent_trace_coherence,
)


def _norm_1_to_5(score: Any) -> Optional[float]:
    if score is None:
        return None
    try:
        v = float(score)
    except (TypeError, ValueError):
        return None
    if v < 0 or v > 5:
        return None
    return (v - 1.0) / 4.0


def _parse_llm_record(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return {}


def _evidence_leaf(x: Any) -> str:
    """Serialize taxonomy evidence leaves as strings for stable HF Arrow ``meta`` structs."""
    if x is None:
        return ""
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, (int, float)):
        return str(float(x))
    if isinstance(x, list):
        return json.dumps(x, ensure_ascii=False)
    return str(x)


def _dim_scores(rec: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    ds = rec.get("dimension_scores") or {}
    if not isinstance(ds, dict):
        return out
    for k, v in ds.items():
        try:
            out[str(k)] = float(v)
        except (TypeError, ValueError):
            continue
    return out


@TAGGING_OPS.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
class AgentErrorTaxonomyMapper(Mapper):
    """Populate ``meta.agent_error_taxonomy`` from existing stats/meta (rules only)."""

    def __init__(
        self,
        hard_drop_safety_severity: float = 0.75,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hard_drop_safety_severity = float(hard_drop_safety_severity)

    def _axis_low_evidence(self, meta: dict, keys: Tuple[str, ...]) -> List[str]:
        lows: List[str] = []
        for k in keys:
            rec = meta.get(k)
            if not isinstance(rec, dict) or rec.get("skipped") or rec.get("error"):
                continue
            sc = rec.get("score")
            nv = _norm_1_to_5(sc)
            if nv is None:
                continue
            if nv < 0.35:
                lows.append(k)
        return lows

    def process_single(self, sample):
        meta = sample.setdefault(Fields.meta, {})
        stats = sample.get(Fields.stats) or {}
        if not isinstance(stats, dict):
            stats = {}

        analysis_rec = _parse_llm_record(stats.get(StatsKeys.llm_analysis_record))
        quality_rec = _parse_llm_record(stats.get(StatsKeys.llm_quality_record))
        a_dims = _dim_scores(analysis_rec)
        q_dims = _dim_scores(quality_rec)

        def _a(name: str) -> Optional[float]:
            v = a_dims.get(name)
            return float(v) if v is not None else None

        def _q(name: str) -> Optional[float]:
            v = q_dims.get(name)
            return float(v) if v is not None else None

        reasoning_norms: List[float] = []
        for k in _DIALOG_KEYS:
            rec = meta.get(k)
            if isinstance(rec, dict) and rec.get("score") is not None:
                nv = _norm_1_to_5(rec.get("score"))
                if nv is not None:
                    reasoning_norms.append(nv)
        rq = _a("response_quality")
        if rq is not None:
            reasoning_norms.append(_norm_1_to_5(rq) or 0.0)
        reasoning_sev = 1.0 - (sum(reasoning_norms) / len(reasoning_norms)) if reasoning_norms else 0.0

        tool_rel = meta.get(MetaKeys.agent_tool_relevance)
        tr_norm = None
        if isinstance(tool_rel, dict) and tool_rel.get("score") is not None:
            tr_norm = _norm_1_to_5(tool_rel.get("score"))
        ratio = meta.get(MetaKeys.tool_success_ratio)
        ratio_pen = 0.0
        if ratio is not None:
            try:
                rf = float(ratio)
                if rf >= 0.0:
                    ratio_pen = max(0.0, 1.0 - rf)
            except (TypeError, ValueError):
                ratio_pen = 0.0
        chain_ok = meta.get(MetaKeys.agent_tool_chain_complete)
        chain_pen = 0.35 if chain_ok is False else 0.0
        tr_pen = (1.0 - tr_norm) if tr_norm is not None else 0.0
        tool_use_sev = min(1.0, max(tr_pen, ratio_pen) + chain_pen)

        instr = _a("instruction_following")
        if instr is None:
            instr_sev = 0.0
        else:
            n = _norm_1_to_5(instr)
            instr_sev = max(0.0, min(1.0, 1.0 - (n if n is not None else 0.0)))

        safety = _a("safety")
        grammar = _q("grammar")
        s1 = 1.0 - (_norm_1_to_5(safety) or 0.0) if safety is not None else 0.0
        s2 = 1.0 - (_norm_1_to_5(grammar) or 0.0) if grammar is not None else 0.0
        safety_style_sev = max(0.0, min(1.0, max(s1, s2)))

        buckets = {
            "reasoning": {
                "severity": float(reasoning_sev),
                "evidence": {
                    "dialog_axes": _evidence_leaf(self._axis_low_evidence(meta, _DIALOG_KEYS)),
                    "llm_analysis.response_quality": _evidence_leaf(rq),
                },
            },
            "tool_use": {
                "severity": float(tool_use_sev),
                "evidence": {
                    "meta.tool_success_ratio": _evidence_leaf(ratio),
                    "meta.agent_tool_relevance.score": _evidence_leaf(
                        tool_rel.get("score") if isinstance(tool_rel, dict) else None
                    ),
                    "meta.agent_tool_chain_complete": _evidence_leaf(chain_ok),
                },
            },
            "instruction_following": {
                "severity": float(instr_sev),
                "evidence": {"llm_analysis.instruction_following": _evidence_leaf(instr)},
            },
            "safety_style": {
                "severity": float(safety_style_sev),
                "evidence": {
                    "llm_analysis.safety": _evidence_leaf(safety),
                    "llm_quality.grammar": _evidence_leaf(grammar),
                },
            },
        }
        meta[MetaKeys.agent_error_taxonomy] = {
            "buckets": buckets,
            "hard_drop_recommended": bool(safety_style_sev >= self.hard_drop_safety_severity),
        }
        return sample

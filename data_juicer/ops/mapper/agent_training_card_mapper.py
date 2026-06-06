# Copyright 2025 The Data-Juicer Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Build a compact ``meta.agent_training_card`` for dataset cards / hand-off (rules only).

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from data_juicer.ops.base_op import OPERATORS, TAGGING_OPS, Mapper
from data_juicer.utils.constant import Fields, MetaKeys, StatsKeys

OP_NAME = "agent_training_card_mapper"


def _ensure_meta_dict(sample: dict) -> dict:
    mk = Fields.meta
    m = sample.get(mk)
    if isinstance(m, dict):
        return m
    import json

    if isinstance(m, str) and m.strip():
        try:
            parsed = json.loads(m)
            if isinstance(parsed, dict):
                sample[mk] = parsed
                return parsed
        except json.JSONDecodeError:
            pass
    sample[mk] = {}
    return sample[mk]


def _top_taxonomy_axes(tax: Any, n: int = 2) -> List[str]:
    if not isinstance(tax, dict):
        return []
    buckets = tax.get("buckets")
    if not isinstance(buckets, dict):
        return []
    scored: List[Tuple[str, float]] = []
    for name, b in buckets.items():
        if isinstance(b, dict) and isinstance(b.get("severity"), (int, float)):
            scored.append((str(name), float(b["severity"])))
    scored.sort(key=lambda x: -x[1])
    return [x[0] for x in scored[:n]]


def _float_metric(v: Any) -> float:
    """Always a finite float so Arrow never pins a metric leaf as ``null`` only."""
    if v is None:
        return -1.0
    try:
        return float(v)
    except (TypeError, ValueError):
        return -1.0


def _safety_gate_label(gate: Any) -> str:
    if isinstance(gate, dict) and "ok" in gate:
        return "true" if gate.get("ok") else "false"
    return "unknown"


def _to_jsonable(x: Any) -> Any:
    """Recursively coerce numpy scalars / odd types so HF Arrow can infer a stable meta schema."""
    if x is None:
        return None
    if isinstance(x, (str, bool, int)):
        return x
    if isinstance(x, float):
        return x
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if hasattr(x, "item"):
        try:
            return _to_jsonable(x.item())
        except Exception:
            return str(x)
    return str(x)


@TAGGING_OPS.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
class AgentTrainingCardMapper(Mapper):
    """Aggregate training-dataset fields into ``meta.agent_training_card`` (JSON string for Arrow)."""

    def __init__(
        self,
        include_distilled: bool = True,
        include_rewrite_hints: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Global recipe config may set ``turbo: false``; force non-batched HF ``map`` so
        # Arrow can widen ``__dj__meta__`` when appending this op’s output.
        self.turbo = True
        self.include_distilled = bool(include_distilled)
        self.include_rewrite_hints = bool(include_rewrite_hints)

    def process_single(self, sample: dict) -> dict:
        meta = _ensure_meta_dict(sample)
        stats = sample.get(Fields.stats) or {}
        if not isinstance(stats, dict):
            stats = {}

        tax = meta.get(MetaKeys.agent_error_taxonomy) or {}
        tier = str(
            meta.get(MetaKeys.agent_training_dataset_tier) or meta.get(MetaKeys.agent_learnable_value_tier) or "none"
        )
        lv = meta.get(MetaKeys.agent_learnable_value)
        hard_drop = bool(tax.get("hard_drop_recommended")) if isinstance(tax, dict) else False
        chain_ok = meta.get(MetaKeys.agent_tool_chain_complete)

        sft_ready = tier in ("gold", "silver", "bronze") and not hard_drop and chain_ok is not False

        pair = meta.get(MetaKeys.agent_cross_model_pair) or {}
        delta = pair.get("delta_to_best")
        preference_ready = bool(pair.get("has_pairwise_contrast")) and delta is not None and float(delta) > 1e-6

        gate = meta.get(MetaKeys.agent_training_safety_gate) or {}
        safety_label = _safety_gate_label(gate if isinstance(gate, dict) else None)

        distill = meta.get(MetaKeys.agent_distilled_trajectory) if self.include_distilled else None
        has_distill = False
        if isinstance(distill, dict):
            has_distill = not distill.get("skipped") and bool(distill.get("distilled_final_reply"))

        rh = meta.get(MetaKeys.agent_rewrite_hints) if self.include_rewrite_hints else None

        # Avoid nested dict/list variance across rows in HF Arrow ``meta`` structs.
        learnable_value_json = None if lv is None else json.dumps(_to_jsonable(lv), ensure_ascii=False)

        card: Dict[str, Any] = {
            "training_dataset_tier": tier,
            "learnable_value_json": learnable_value_json,
            "target_capabilities": _top_taxonomy_axes(tax, 2),
            "training_ready": {
                "sft": bool(sft_ready),
                "preference": bool(preference_ready),
                "rft": False,
            },
            "safety_gate_ok": safety_label,
            "hard_drop_recommended": hard_drop,
            "llm_difficulty_score": _float_metric(stats.get(StatsKeys.llm_difficulty_score)),
            "llm_analysis_score": _float_metric(stats.get(StatsKeys.llm_analysis_score)),
            "recommended_usage": self._usage_line(
                tier=tier,
                caps=_top_taxonomy_axes(tax, 2),
                sft_ready=sft_ready,
                pref_ready=preference_ready,
                has_distill=bool(has_distill),
            ),
            # Always set these keys so Arrow/HF infers a stable struct for every row.
            "distilled_present": bool(self.include_distilled and isinstance(distill, dict) and has_distill),
            "rewrite_hints_present": bool(
                self.include_rewrite_hints and isinstance(rh, dict) and bool(rh.get("hints") or rh.get("parse_ok"))
            ),
        }

        meta[MetaKeys.agent_training_card] = json.dumps(_to_jsonable(card), ensure_ascii=False)
        return sample

    @staticmethod
    def _usage_line(
        *,
        tier: str,
        caps: List[str],
        sft_ready: bool,
        pref_ready: bool,
        has_distill: bool,
    ) -> str:
        parts = [f"tier={tier}"]
        if caps:
            parts.append("focus=" + ",".join(caps))
        if sft_ready:
            parts.append("SFT candidate")
        if pref_ready:
            parts.append("preference-pair candidate")
        if has_distill:
            parts.append("has teacher distilled reply")
        return "; ".join(parts)

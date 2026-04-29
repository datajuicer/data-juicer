# Copyright 2025 The Data-Juicer Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# LLM-based synthesis of numeric stats + qualitative LLM eval records into a
# single auditable insight object (attribution for viz / human quality audit).

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from loguru import logger
from pydantic import PositiveInt

from data_juicer.ops.base_op import OPERATORS, TAGGING_OPS, Mapper
from data_juicer.utils.agent_output_locale import (
    agent_insight_system_prompt,
    normalize_preferred_output_lang,
)
from data_juicer.utils.constant import Fields, MetaKeys, StatsKeys
from data_juicer.utils.model_utils import get_model, prepare_model

OP_NAME = "agent_insight_llm_mapper"

_DIALOG_QUALITY_LLM_META_KEYS = (
    MetaKeys.dialog_memory_consistency,
    MetaKeys.dialog_coreference,
    MetaKeys.dialog_topic_shift,
    MetaKeys.dialog_error_recovery,
    MetaKeys.dialog_clarification_quality,
    MetaKeys.dialog_proactivity,
    MetaKeys.dialog_non_repetition,
    MetaKeys.agent_trace_coherence,
    MetaKeys.agent_tool_relevance,
)


def _dialog_quality_llm_pack(meta: dict) -> Optional[dict]:
    """Compact 1–5 axis scores from lightweight turn/trace LLM mappers."""
    out: Dict[str, Any] = {}
    for k in _DIALOG_QUALITY_LLM_META_KEYS:
        rec = meta.get(k)
        if not isinstance(rec, dict):
            continue
        if rec.get("skipped"):
            out[k] = {"skipped": True}
            continue
        if rec.get("error"):
            out[k] = {"error": str(rec.get("error"))}
            continue
        piece: Dict[str, Any] = {}
        if rec.get("score") is not None:
            piece["score"] = rec["score"]
        reason = rec.get("reason")
        if isinstance(reason, str) and reason.strip():
            piece["reason"] = reason[:400]
        if piece:
            out[k] = piece
    return out if out else None


def _json_safe(x: Any, depth: int = 0) -> Any:
    if depth > 6:
        return "..."
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, dict):
        return {str(k): _json_safe(v, depth + 1) for k, v in list(x.items())[:40]}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v, depth + 1) for v in x[:30]]
    if hasattr(x, "tolist"):
        try:
            return x.tolist()
        except Exception:
            pass
    if hasattr(x, "item"):
        try:
            return x.item()
        except Exception:
            pass
    return str(x)[:500]


def _truncate_record(rec: Any, max_chars: int = 1200) -> Any:
    if rec is None:
        return None
    # record may be a JSON string (serialized by _normalize_record); parse it first
    if isinstance(rec, str):
        if not rec:
            return None
        try:
            rec = json.loads(rec)
        except Exception:
            return rec[:max_chars] + "…" if len(rec) > max_chars else rec
    s = json.dumps(_json_safe(rec), ensure_ascii=False)
    if len(s) <= max_chars:
        return json.loads(s)
    return s[:max_chars] + "…"


def _parse_llm_json(raw: str) -> Optional[dict]:
    if not raw or not isinstance(raw, str):
        return None
    text = raw.strip()
    m = re.search(r"\{[\s\S]*\}\s*$", text)
    if m:
        text = m.group(0)
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    return None


def _build_evidence_pack(
    sample: dict,
    query_key: str,
    response_key: str,
    query_max: int,
    response_max: int,
) -> dict:
    meta = sample.get(Fields.meta) or {}
    stats = sample.get(Fields.stats) or {}
    q = (sample.get(query_key) or "").strip()
    r = (sample.get(response_key) or "").strip()

    def _head(labels: Any, n: int = 8) -> Any:
        if not isinstance(labels, list):
            return labels
        return labels[:n]

    dq_pack = _dialog_quality_llm_pack(meta)

    return {
        "lineage": {
            "agent_request_model": meta.get(MetaKeys.agent_request_model),
            "agent_pt": meta.get(MetaKeys.agent_pt),
            "agent_total_cost_time_ms": meta.get(MetaKeys.agent_total_cost_time_ms),
        },
        "usage_tokens": {
            "prompt_tokens": meta.get(MetaKeys.prompt_tokens),
            "completion_tokens": meta.get(MetaKeys.completion_tokens),
            "total_tokens": meta.get(MetaKeys.total_tokens),
        },
        "tools": {
            "tool_success_count": meta.get(MetaKeys.tool_success_count),
            "tool_fail_count": meta.get(MetaKeys.tool_fail_count),
            "tool_success_ratio": meta.get(MetaKeys.tool_success_ratio),
            "primary_tool_type": meta.get(MetaKeys.primary_tool_type),
            "dominant_tool_types": _head(meta.get(MetaKeys.dominant_tool_types)),
            "agent_skill_insights": _head(meta.get(MetaKeys.agent_skill_insights)),
        },
        "dialog_tags": {
            "dialog_intent_labels": _head(meta.get(MetaKeys.dialog_intent_labels)),
            "dialog_topic_labels": _head(meta.get(MetaKeys.dialog_topic_labels)),
            "dialog_sentiment_labels": _head(meta.get(MetaKeys.dialog_sentiment_labels)),
            "agent_turn_count": meta.get(MetaKeys.agent_turn_count),
        },
        "scores": {
            "llm_analysis_score": stats.get(StatsKeys.llm_analysis_score),
            "llm_quality_score": stats.get(StatsKeys.llm_quality_score),
            "llm_difficulty_score": stats.get(StatsKeys.llm_difficulty_score),
        },
        "llm_eval_support": {
            "llm_analysis_record": _truncate_record(stats.get(StatsKeys.llm_analysis_record)),
            "llm_quality_record": _truncate_record(stats.get(StatsKeys.llm_quality_record)),
            "llm_difficulty_record": _truncate_record(stats.get(StatsKeys.llm_difficulty_record)),
        },
        "text_stats": {
            "text_len": stats.get(StatsKeys.text_len),
            "num_words": stats.get(StatsKeys.num_words),
            "perplexity": stats.get(StatsKeys.perplexity),
            "lang_score": stats.get(StatsKeys.lang_score),
        },
        "deterministic_bad_case": {
            "signals": meta.get(MetaKeys.agent_bad_case_signals),
            "tier": meta.get(MetaKeys.agent_bad_case_tier),
        },
        "dialog_quality_llm": _truncate_record(dq_pack, max_chars=2800) if dq_pack else None,
        "query_preview": q[:query_max],
        "response_preview": r[:response_max],
    }


@TAGGING_OPS.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
class AgentInsightLLMMapper(Mapper):
    """Synthesize stats + LLM eval text into ``meta.agent_insight_llm`` (JSON).

    Intended to run **after** filters/mappers that populate ``stats`` and
    ``agent_bad_case_signal_mapper``. Use ``run_for_tiers`` to limit API cost.

    Output is best-effort JSON; raw model text is stored in
    ``meta.agent_insight_llm_raw`` if parsing fails.
    """

    def __init__(
        self,
        api_model: str = "gpt-4o",
        *,
        api_endpoint: Optional[str] = None,
        response_path: Optional[str] = None,
        system_prompt: Optional[str] = None,
        query_key: str = "query",
        response_key: str = "response",
        query_preview_max_chars: int = 500,
        response_preview_max_chars: int = 500,
        run_for_tiers: Optional[List[str]] = None,
        try_num: PositiveInt = 2,
        model_params: Dict = {},
        sampling_params: Dict = {},
        preferred_output_lang: str = "en",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.preferred_output_lang = normalize_preferred_output_lang(preferred_output_lang)
        self.system_prompt = system_prompt or agent_insight_system_prompt(self.preferred_output_lang)
        self.query_key = query_key
        self.response_key = response_key
        self.query_preview_max_chars = query_preview_max_chars
        self.response_preview_max_chars = response_preview_max_chars
        # None = all samples; else require meta.agent_bad_case_tier in this list
        self.run_for_tiers = run_for_tiers
        self.try_num = try_num
        self.sampling_params = sampling_params or {}
        self.model_key = prepare_model(
            model_type="api",
            model=api_model,
            endpoint=api_endpoint,
            response_path=response_path,
            **model_params,
        )

    def process_single(self, sample, rank=None):
        meta = sample.setdefault(Fields.meta, {})
        if MetaKeys.agent_insight_llm in meta:
            return sample

        tier = meta.get(MetaKeys.agent_bad_case_tier)
        if self.run_for_tiers is not None and tier not in self.run_for_tiers:
            return sample

        meta[MetaKeys.agent_pipeline_output_lang] = self.preferred_output_lang

        pack = _build_evidence_pack(
            sample,
            self.query_key,
            self.response_key,
            self.query_preview_max_chars,
            self.response_preview_max_chars,
        )
        user_content = json.dumps(pack, ensure_ascii=False)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

        raw = ""
        for _ in range(self.try_num):
            try:
                client = get_model(self.model_key, rank=rank)
                raw = client(messages, **self.sampling_params)
                if raw and isinstance(raw, str) and raw.strip():
                    break
            except Exception as e:
                logger.warning("agent_insight_llm_mapper: %s", e)

        meta[MetaKeys.agent_insight_llm_raw] = raw if isinstance(raw, str) else ""
        parsed = _parse_llm_json(raw) if raw else None
        if parsed is not None:
            meta[MetaKeys.agent_insight_llm] = parsed
        else:
            if self.preferred_output_lang == "zh":
                headline = "解析失败，见 meta.agent_insight_llm_raw"
                audit = "JSON parse failed"
            else:
                headline = "Parse failed; see meta.agent_insight_llm_raw"
                audit = "JSON parse failed"
            meta[MetaKeys.agent_insight_llm] = {
                "headline": headline,
                "root_causes": [],
                "narrative_alignment": "mixed",
                "human_review_priority": "P2",
                "viz_facets": [],
                "audit_notes": audit,
            }
        return sample

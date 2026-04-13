# Copyright 2025 The Data-Juicer Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Static attribution hints for agent_bad_case_signal.

Used by the HTML report to map signal codes to upstream evidence fields.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

# 与 agent_bad_case_signal_mapper._DIALOG_QUALITY_SCORE_META_KEYS / MetaKeys 对齐（报告侧不 import data_juicer）
DIALOG_QUALITY_SCORE_META_KEYS: Tuple[str, ...] = (
    "dialog_memory_consistency",
    "dialog_coreference",
    "dialog_topic_shift",
    "dialog_error_recovery",
    "dialog_clarification_quality",
    "dialog_proactivity",
    "dialog_non_repetition",
    "agent_trace_coherence",
    "agent_tool_relevance",
)

# role:
#   primary — 常驱动 high_precision 或 high 权重主证据
#   structured — §5b 轴分等结构化质检摘要（多为 medium，但建议与主证据同读）
#   appendix — 启发式 / 弱证据
SIGNAL_SUPPORT_ROWS: List[Dict[str, Any]] = [
    {
        "code": "tool_message_error_pattern",
        "role": "primary",
        "weight_hint": "high",
        "upstream": (
            "meta.tool_fail_count、tool_unknown_count；"
            "tool_success_tagger_mapper（messages role=tool）；"
            "第 9 步需 fail≥min_tool_fail_count_for_signal 才发本信号（减轻单条试错 bias）"
        ),
    },
    {
        "code": "llm_agent_analysis_eval_low",
        "role": "primary",
        "weight_hint": "high 或 medium",
        "upstream": (
            "stats.llm_analysis_score、"
            "stats.llm_analysis_record.recommendation；"
            "来自 llm_analysis_filter（discard + 低分 → high）"
        ),
    },
    {
        "code": "llm_reply_quality_eval_low",
        "role": "primary",
        "weight_hint": "high 或 medium",
        "upstream": (
            "stats.llm_quality_score、stats.llm_quality_record；"
            "来自 llm_quality_score_filter"
        ),
    },
    {
        "code": "dialog_turn_quality_meta_low",
        "role": "structured",
        "weight_hint": "medium（无额外 LLM；常与 watchlist 共现）",
        "upstream": (
            "meta.dialog_*、meta.agent_trace_coherence、"
            "meta.agent_tool_relevance（1–5）；"
            "agent_bad_case_signal_mapper 汇总上游 dialog_* / agent_* 质检算子"
        ),
    },
    {
        "code": "low_tool_success_ratio",
        "role": "appendix",
        "weight_hint": "medium",
        "upstream": (
            "meta.tool_success_ratio、success/fail 轮次；"
            "tool_success_tagger_mapper"
        ),
    },
    {
        "code": "suspect_empty_or_trivial_final_response",
        "role": "appendix",
        "weight_hint": "medium",
        "upstream": "样本 query / response 长度启发式",
    },
    {
        "code": "high_token_usage",
        "role": "appendix",
        "weight_hint": "medium",
        "upstream": (
            "meta.total_tokens；usage_counter_mapper；"
            "阈值可配或由 calibration JSON（P95）决定"
        ),
    },
    {
        "code": "high_latency_ms",
        "role": "appendix",
        "weight_hint": "medium",
        "upstream": (
            "meta.agent_total_cost_time_ms；"
            "agent_dialog_normalize copy_lineage_fields；阈值同可校准"
        ),
    },
    {
        "code": "negative_sentiment_label_hint",
        "role": "appendix",
        "weight_hint": "medium",
        "upstream": (
            "meta.dialog_sentiment_labels；"
            "dialog_sentiment_detection_mapper（易噪）"
        ),
    },
    {
        "code": "high_perplexity",
        "role": "appendix",
        "weight_hint": "medium",
        "upstream": (
            "stats.perplexity；perplexity_filter（KenLM；"
            "默认菜谱可关）"
        ),
    },
    {
        "code": "hard_query_low_reply_quality_conjunction",
        "role": "appendix",
        "weight_hint": "medium",
        "upstream": (
            "stats.llm_difficulty_score ∩ stats.llm_quality_score；"
            "难度与质量联合启发式"
        ),
    },
]

APPENDIX_CODES = {
    r["code"] for r in SIGNAL_SUPPORT_ROWS if r["role"] == "appendix"
}
PRIMARY_CODES = {
    r["code"] for r in SIGNAL_SUPPORT_ROWS if r["role"] == "primary"
}
STRUCTURED_CODES = {
    r["code"] for r in SIGNAL_SUPPORT_ROWS if r["role"] == "structured"
}

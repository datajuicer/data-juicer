# Copyright 2025 The Data-Juicer Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Preferred output language helpers for agent / dialog LLM operators.

YAML / op kwargs use ``preferred_output_lang`` (e.g. ``zh``, ``en``, ``zh-CN``).
Normalized to ``zh`` or ``en`` for prompt selection. JSON **keys** stay English
where required for parsing; **free-text** fields follow this locale.
"""

from __future__ import annotations

from typing import Optional

# Keep in sync with dialog_quality_llm_utils.JSON_SCORE_REASON_EN intent.
_JSON_SCHEMA_EN = (
    "Output exactly one JSON object, no markdown. Schema:\n"
    '{"score": <integer 1-5>, "reason": "<brief justification>"}\n'
    "Higher score = better for this criterion."
)

_JSON_SCHEMA_ZH = (
    "请只输出一个 JSON 对象，不要使用 markdown。模式：\n"
    '{"score": <1-5 的整数>, "reason": "<简短理由>"}\n'
    "分数越高表示该维度表现越好。"
)


def normalize_preferred_output_lang(value: Optional[str]) -> str:
    """Return ``zh`` or ``en`` (default ``en`` if missing/unknown)."""
    if value is None:
        return "en"
    s = str(value).strip().lower().replace("_", "-")
    if not s:
        return "en"
    if s.startswith("zh"):
        return "zh"
    if s in ("en", "english", "eng"):
        return "en"
    return "en"


def dialog_score_json_instruction(lang: str) -> str:
    """Instruction block for 1–5 + reason JSON (dialog / trace quality mappers)."""
    return _JSON_SCHEMA_ZH if normalize_preferred_output_lang(lang) == "zh" else _JSON_SCHEMA_EN


def rubric_reason_language_clause(lang: str) -> str:
    """Append to system prompt: rubric may be English; ``reason`` follows locale."""
    if normalize_preferred_output_lang(lang) == "zh":
        return (
            "Language: The scoring rubric above may be in English; apply it faithfully. "
            "The JSON field `reason` must be written in concise **Simplified Chinese (简体中文)**."
        )
    return (
        "Language: Write the JSON field `reason` in concise **English**, "
        "even if the user dialog is in another language."
    )


def llm_filter_free_text_language_appendix(lang: Optional[str]) -> str:
    """Append to LLMAnalysisFilter ``system_prompt`` for rationale / tags language."""
    if lang is None or str(lang).strip() == "":
        return ""
    if normalize_preferred_output_lang(lang) == "zh":
        return (
            "\n\n【输出语言】保持 JSON 键名与上述 schema 一致（英文）；"
            "`rationale`、tags 中自然语言描述、以及 flags 的可读语义请使用**简体中文**。"
        )
    return (
        "\n\n【Output language】Keep JSON keys as in the schema (English); "
        "write `rationale`, human-readable tag strings, and flag semantics in **English**."
    )


def agent_insight_system_prompt(lang: str) -> str:
    """System prompt for ``agent_insight_llm_mapper``."""
    if normalize_preferred_output_lang(lang) == "zh":
        return (
            "你是智能体交互日志的谨慎分析员（自动数据科学风格）。"
            "你会收到**单条**样本的 JSON：数值 stats、query/response 短预览、"
            "可选的 LLM 评估摘录、dialog_quality_llm（各轴 1–5 分）、"
            "对话标签、工具使用、以及确定性的 bad-case 信号。\n\n"
            "任务：\n"
            "1）综合数字与文字依据；指出一致或冲突。\n"
            "2）只使用输入 JSON 里出现的键作归因，不要编造事实。\n"
            "3）不确定时标为低置信度（偏 precision，便于人工质检）。\n\n"
            "请只输出一个 JSON 对象，键名必须完全一致：\n"
            "- headline: 一句话卡片标题（中文）\n"
            "- root_causes: 数组，元素为 "
            "{factor, confidence (high|medium|low), cited_fields (来自输入的字段路径字符串), "
            "rationale_one_line}；其中说明用中文\n"
            '- narrative_alignment: "aligned"|"mixed"|"conflict"\n'
            '- human_review_priority: "P0"|"P1"|"P2"|"P3"（P0 少用）\n'
            "- viz_facets: 字符串数组，用于图表维度（如 agent_request_model, agent_pt）\n"
            "- audit_notes: 可选字符串，中文备注\n\n"
            "只输出合法 JSON，不要用 markdown 代码围栏。"
        )
    return (
        "You are a careful analyst for agent interaction logs "
        "(auto data-scientist style).\n"
        "You receive ONE sample as a JSON object: numeric stats, short "
        "previews of query/response, optional LLM evaluation rationales, "
        "dialog_quality_llm (1–5 per-axis scores from lightweight turn/trace "
        "judges), dialog tags, tool usage, and deterministic bad-case signals.\n\n"
        "Tasks:\n"
        "1) Integrate numbers and qualitative rationales; note agreement "
        "or conflict.\n"
        "2) Attribute only with keys from the input JSON (no invented "
        "facts).\n"
        '3) Prefer "uncertain" / low confidence (precision for human QA).\n\n'
        "Output a single JSON object with exactly these keys:\n"
        "- headline: one short sentence for dashboard cards (English)\n"
        "- root_causes: array of {factor, confidence (high|medium|low), "
        "cited_fields (strings from input), rationale_one_line} — in English\n"
        '- narrative_alignment: "aligned"|"mixed"|"conflict"\n'
        '- human_review_priority: "P0"|"P1"|"P2"|"P3" (P0 sparingly)\n'
        "- viz_facets: strings for chart dimensions "
        "(e.g. agent_request_model, agent_pt)\n"
        "- audit_notes: optional caveats in English\n\n"
        "Respond with valid JSON only, no markdown fences."
    )


def dialog_detection_output_language_note(lang: str, mode: str) -> str:
    """Append to dialog intent/topic/sentiment/intensity system prompts.

    Chinese prefixes (意图分析：, 话题类别：, …) stay for regex parsers; body follows locale.
    """
    m = (mode or "").strip().lower()
    zh = normalize_preferred_output_lang(lang) == "zh"
    if zh:
        notes = {
            "intent": "\n\n【语言】「意图分析」「意图类别」后的说明请使用简体中文。",
            "topic": "\n\n【语言】「话题分析」「话题类别」后的说明请使用简体中文。",
            "sentiment": "\n\n【语言】「情感分析」「情感类别」后的说明请使用简体中文。",
            "intensity": "\n\n【语言】「情绪分析」后的说明与情绪值整数请按规定格式；分析句用简体中文。",
        }
    else:
        notes = {
            "intent": (
                "\n\n【Language】Keep the exact prefixes "
                "「意图分析：」「意图类别：」 as in the examples; "
                "write the text after each colon in **English**."
            ),
            "topic": (
                "\n\n【Language】Keep the exact prefixes "
                "「话题分析：」「话题类别：」 as in the examples; "
                "write the text after each colon in **English**."
            ),
            "sentiment": (
                "\n\n【Language】Keep the exact prefixes "
                "「情感分析：」「情感类别：」 as in the examples; "
                "write the text after each colon in **English**."
            ),
            "intensity": (
                "\n\n【Language】Keep the exact prefixes "
                "「情绪分析：」「情绪值：」 as in the examples; "
                "write the analysis sentence after the colon in **English**; "
                "the integer after 情绪值： must still be -5..5."
            ),
        }
    return notes.get(m, "")


def agent_skill_insight_system_prompt(lang: str) -> str:
    if normalize_preferred_output_lang(lang) == "zh":
        return (
            "根据输入的 Tools / Skills 列表，归纳 3～5 条**具体能力短语**，逗号分隔；"
            "每条**约 8～12 个汉字**（围绕「约 10 字」的信息量），使用**简体中文**。\n"
            "每条必须体现**对象或载体 + 具体动作或场景**（从工具/技能名抽象，但要比原名更可区分，"
            "勿整段复述原名）。\n"
            "**禁止**空洞标签：不得主要靠「处理、操作、读写、执行、调用、获取」等泛化动词凑标签；"
            "不得仅用「文件/数据/内容」而无细分动作或对象。\n"
            "好例：飞书多维表字段批量改写、本地日志按关键字追溯、PDF 表格转结构化字段。\n"
            "反例：文件读写、数据处理、读取与操作。\n"
            "只输出逗号分隔的短语：不要编号、不要解释、不要换行。"
        )
    return (
        "From the Tools / Skills list, write **3–5 capability phrases**, "
        "comma-separated.\n"
        "Each phrase is **about 4–8 words** (one concrete skill), "
        "in **English**.\n"
        "Each phrase must name **what (object, tool, or domain) + a specific "
        "action or scenario** — grounded in the input but more distinctive "
        "than the raw names; do not paste long tool names verbatim.\n"
        "**Forbidden** as stand-alone tags: vague "
        '"processing", "operations", "read/write", "handling", '
        '"calling" without a clear target; "files/data/content" alone.\n'
        "Good: Feishu Bitable bulk field edits, log trace by keyword, "
        "PDF tables to structured fields.\n"
        "Bad: file I/O, data processing, read and manipulate.\n"
        "Output comma-separated phrases only: no numbering, no explanation, "
        "no newlines."
    )

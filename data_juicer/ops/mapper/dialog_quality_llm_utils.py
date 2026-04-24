# Copyright 2025 The Data-Juicer Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prompt builders and JSON helpers for turn-quality LLM mappers."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from data_juicer.ops.mapper.dialog_llm_input_utils import (
    build_dialog_turns_for_prompt,
    clip_query_response_pair,
    clip_text_for_dialog_prompt,
)
from data_juicer.utils.agent_output_locale import dialog_score_json_instruction
from data_juicer.utils.constant import Fields

# Backward-compatible alias (English JSON instruction for 1–5 + reason).
JSON_SCORE_REASON_EN = dialog_score_json_instruction("en")


def extract_json_object(text: str) -> Optional[dict]:
    if not text or not isinstance(text, str):
        return None
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```\s*$", "", s)
    start = s.find("{")
    end = s.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        frag = s[start : end + 1]  # noqa: E203
        return json.loads(frag)
    except json.JSONDecodeError:
        return None


def _normalize_dialog_tail(
    sample: Dict[str, Any],
    history_key: str,
    query_key: str,
    response_key: str,
    max_round: int,
) -> List[Tuple[str, str]]:
    dialog = build_dialog_turns_for_prompt(
        sample,
        history_key=history_key,
        query_key=query_key,
        response_key=response_key,
    )
    if max_round > 0 and len(dialog) > max_round:
        dialog = dialog[-max_round:]
    return dialog


def build_dialog_turn_eval_user_content(
    sample: dict,
    *,
    history_key: str,
    query_key: str,
    response_key: str,
    max_round: int,
    max_query_chars: int,
    max_response_chars: int,
) -> str:
    """Earlier turns + last user message + assistant span to score."""
    turns = _normalize_dialog_tail(
        sample,
        history_key,
        query_key,
        response_key,
        max_round,
    )
    if not turns:
        return ""
    *rest, last = turns
    chunks: List[str] = []
    for u, a in rest:
        qu, au = clip_query_response_pair(
            u,
            a,
            max_query_chars,
            max_response_chars,
        )
        chunks.append(f"[User]\n{qu}\n[Assistant]\n{au}\n")
    lu, lr = clip_query_response_pair(
        last[0],
        last[1],
        max_query_chars,
        max_response_chars,
    )
    early = "".join(chunks)
    return (
        f"### Earlier turns\n{early}"
        f"### Current user message\n{lu}\n"
        f"### Assistant reply to score (evaluate this segment only)\n{lr}\n"
    )


def build_agent_trace_eval_user_content(
    sample: dict,
    *,
    text_key: str,
    max_chars: int,
) -> str:
    """Flattened session text (e.g. after agent_dialog_normalize_mapper)."""
    t = sample.get(text_key)
    if not isinstance(t, str) or not t.strip():
        return ""
    clipped = clip_text_for_dialog_prompt(t, max_chars, "text truncated")
    return f"### Session trace excerpt (may include tool output)\n{clipped}\n"


def build_agent_tool_fit_user_content(
    sample: dict,
    *,
    query_key: str,
    response_key: str,
    tool_types_key: str,
    primary_tool_key: str,
    max_query_chars: int,
    max_response_chars: int,
) -> str:
    q = sample.get(query_key) or ""
    r = sample.get(response_key) or ""
    qu, rs = clip_query_response_pair(
        q,
        r,
        max_query_chars,
        max_response_chars,
    )
    meta = sample.get(Fields.meta) or {}
    if not isinstance(meta, dict):
        meta = {}
    tools = meta.get(tool_types_key)
    primary = meta.get(primary_tool_key)
    tools_s = ""
    if isinstance(tools, list):
        tools_s = ", ".join(str(x) for x in tools[:40])
    elif tools is not None:
        tools_s = str(tools)
    prim_s = "" if primary is None else str(primary)
    tline = f"### Inferred tool list ({tool_types_key})\n{tools_s or '(none)'}\n"
    pline = f"### Primary tool ({primary_tool_key})\n{prim_s or '(none)'}\n"
    return f"### User request\n{qu}\n" f"### Assistant reply (may include tool trace)\n{rs}\n" f"{tline}" f"{pline}"


def normalize_score_1_5(obj: Optional[dict]) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        return {"error": "invalid_json", "score": None, "reason": ""}
    raw = obj.get("score")
    try:
        s = float(raw)
    except (TypeError, ValueError):
        return {
            "error": "bad_score",
            "score": None,
            "reason": str(obj.get("reason") or ""),
        }
    s = max(1.0, min(5.0, s))
    return {
        "score": s,
        "reason": str(obj.get("reason") or "")[:2000],
    }

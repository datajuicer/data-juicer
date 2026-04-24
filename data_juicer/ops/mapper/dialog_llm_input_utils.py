# Copyright 2025 The Data-Juicer Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helpers for dialog LLM mappers (intent / topic / sentiment / intensity)."""

from __future__ import annotations

from typing import List, Tuple


def build_dialog_turns_for_prompt(
    sample: dict,
    *,
    history_key: str,
    query_key: str,
    response_key: str,
) -> List[Tuple[str, str]]:
    """Build (user, assistant) turns for dialog LLM mappers.

    Does not mutate ``sample``. Merge rules match
    ``dialog_quality_llm_utils._normalize_dialog_tail``: after normalize, the last
    turn lives in both ``dialog_history[-1]`` and ``query``/``response``, so those
    fields must not be appended again (would duplicate the final exchange; older
    code that mutated ``dialog_history`` in place corrupted downstream rows).
    """
    dialog: List[Tuple[str, str]] = []
    raw = sample.get(history_key)
    if isinstance(raw, list):
        for turn in raw:
            if isinstance(turn, (list, tuple)) and len(turn) >= 2:
                u0 = "" if turn[0] is None else str(turn[0])
                u1 = "" if turn[1] is None else str(turn[1])
                dialog.append((u0, u1))
    if sample.get(query_key):
        q = sample[query_key]
        r = sample.get(response_key) or ""
        qs = "" if q is None else str(q)
        rs = "" if r is None else str(r)
        if not dialog:
            dialog.append((qs, rs))
        else:
            lu, la = dialog[-1]
            if lu == qs and la == rs:
                pass
            elif lu == qs:
                dialog[-1] = (qs, rs)
            else:
                dialog.append((qs, rs))
    return dialog


def clip_text_for_dialog_prompt(
    text: str,
    max_chars: int,
    note: str = "truncated",
) -> str:
    """Truncate long ``text`` for API prompts when ``max_chars`` > 0.

    Agent traces often concatenate tool outputs into ``response``; formatter
    limits elsewhere do not apply to these mappers' ``history_key`` payloads.
    """
    if max_chars is None or max_chars <= 0:
        return text
    if not text:
        return text
    if len(text) <= max_chars:
        return text
    suffix = f"\n…[{note}]…"
    take = max_chars - len(suffix)
    if take <= 0:
        return suffix.strip()
    return text[:take] + suffix


def clip_query_response_pair(
    q: object,
    r: object,
    max_query_chars: int,
    max_response_chars: int,
) -> Tuple[str, str]:
    qs = "" if q is None else str(q)
    rs = "" if r is None else str(r)
    return (
        clip_text_for_dialog_prompt(qs, max_query_chars, "query truncated"),
        clip_text_for_dialog_prompt(
            rs,
            max_response_chars,
            "response truncated",
        ),
    )

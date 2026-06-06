# Copyright 2025 The Data-Juicer Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Heuristic flags for logging / streaming artifacts
# (sys_log-like incomplete traces).

from __future__ import annotations

import json
from typing import Any, List

from data_juicer.ops.base_op import OPERATORS, TAGGING_OPS, Mapper
from data_juicer.utils.constant import Fields, MetaKeys

OP_NAME = "agent_sys_log_noise_mapper"


def _coerce_message_list(val: Any) -> List[dict]:
    if val is None:
        return []
    if isinstance(val, list):
        return [x for x in val if isinstance(x, dict)]
    if isinstance(val, str) and val.strip():
        try:
            parsed = json.loads(val)
            if isinstance(parsed, list):
                return [x for x in parsed if isinstance(x, dict)]
        except json.JSONDecodeError:
            return []
    return []


def _coerce_choices(val: Any) -> List[dict]:
    if val is None:
        return []
    if isinstance(val, list):
        return [x for x in val if isinstance(x, dict)]
    if isinstance(val, str) and val.strip():
        try:
            parsed = json.loads(val)
            if isinstance(parsed, list):
                return [x for x in parsed if isinstance(x, dict)]
        except json.JSONDecodeError:
            return []
    return []


def _content_to_str(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                t = block.get("text")
                if isinstance(t, str):
                    parts.append(t.strip())
            elif isinstance(block, str):
                parts.append(block.strip())
        return "\n".join(parts).strip()
    if isinstance(content, dict):
        for k in ("text", "content", "value"):
            if k in content and isinstance(content[k], str):
                return str(content[k]).strip()
    return str(content).strip()


@TAGGING_OPS.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
class AgentSysLogNoiseMapper(Mapper):
    """Write ``meta.agent_sys_log_noise`` with cheap structural checks."""

    def __init__(
        self,
        messages_key: str = "messages",
        choices_key: str = "response_choices",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.messages_key = messages_key
        self.choices_key = choices_key

    def process_single(self, sample):
        mk = Fields.meta
        meta = sample.get(mk)
        if not isinstance(meta, dict):
            if isinstance(meta, str) and meta.strip():
                try:
                    meta = json.loads(meta)
                except json.JSONDecodeError:
                    meta = {}
            if not isinstance(meta, dict):
                meta = {}
            sample[mk] = meta
        flags: List[str] = []
        messages = _coerce_message_list(sample.get(self.messages_key))

        if messages and (messages[-1].get("role") or "").lower() == "tool":
            flags.append("trailing_tool_without_assistant")

        pending_calls = 0
        for m in messages:
            if not isinstance(m, dict):
                continue
            role = (m.get("role") or "").lower()
            if role == "assistant":
                tcs = m.get("tool_calls") or m.get("tool_use") or []
                pending_calls = len(tcs) if tcs else 0
            elif role == "tool" and pending_calls > 0:
                pending_calls -= 1
            elif role == "user" and pending_calls > 0:
                flags.append("tool_calls_unresolved_before_next_user")
                pending_calls = 0

        choices = _coerce_choices(sample.get(self.choices_key))
        if choices:
            c0 = choices[0] if isinstance(choices[0], dict) else {}
            msg = c0.get("message") or c0.get("delta") or {}
            if isinstance(msg, dict):
                fr = msg.get("finish_reason") or c0.get("finish_reason")
                if fr and str(fr).lower() not in (
                    "stop",
                    "end_turn",
                    "completed",
                    "none",
                    "",
                ):
                    flags.append("non_terminal_finish_reason")
                body = _content_to_str(msg.get("content"))
                if body == "" and (msg.get("tool_calls") or msg.get("tool_use")):
                    flags.append("empty_assistant_with_tool_calls_only")

        sev = min(1.0, 0.35 * len(set(flags)))
        meta[MetaKeys.agent_sys_log_noise] = {
            "flags": sorted(set(flags)),
            "severity": float(sev),
            "is_likely_noise": bool(flags),
        }
        return sample

# Copyright 2025 The Data-Juicer Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Heuristic flags for harness / eval-runner artifacts mixed into agent logs.

from __future__ import annotations

import json
import re
from typing import Any, List, Optional

from data_juicer.ops.base_op import OPERATORS, TAGGING_OPS, Mapper
from data_juicer.utils.constant import Fields, MetaKeys

OP_NAME = "agent_harness_noise_mapper"


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


def _flatten_message_texts(messages: List[dict], max_chars: int = 8000) -> str:
    parts: List[str] = []
    n = 0
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or "")
        blob = str(m.get("content") or "")
        if isinstance(m.get("content"), (list, dict)):
            blob = str(m.get("content"))
        line = f"{role}:{blob[:2000]}"
        parts.append(line)
        n += len(line)
        if n >= max_chars:
            break
    return "\n".join(parts)


@TAGGING_OPS.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
class AgentHarnessNoiseMapper(Mapper):
    """Write ``meta.agent_harness_noise`` when eval/harness patterns appear in messages."""

    def __init__(
        self,
        messages_key: str = "messages",
        harness_patterns: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.messages_key = messages_key
        pats = harness_patterns or [
            r"(?i)^\s*\[runner\]",
            r"(?i)<eval_output>",
            r"(?i)Observation:\s*Error while",
            r"(?i)##\s*eval\s*case",
            r"(?i)unit\s*test\s*passed",
            r"(?i)score\s*:\s*\d+\s*/\s*\d+",
        ]
        self._compiled = [re.compile(p) for p in pats]

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
        messages = _coerce_message_list(sample.get(self.messages_key))
        blob = _flatten_message_texts(messages)
        hits: List[str] = []
        for i, cre in enumerate(self._compiled):
            if cre.search(blob):
                hits.append(f"pattern[{i}]")
        sev = min(1.0, 0.4 * len(hits))
        meta[MetaKeys.agent_harness_noise] = {
            "flags": hits,
            "severity": float(sev),
            "is_likely_noise": bool(hits),
        }
        return sample

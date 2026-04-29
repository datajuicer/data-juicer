# Copyright 2025 The Data-Juicer Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLM: final reply vs prior constraints/facts (1–5).

Inspired by OpenJudge-style context memory; DJ: single call, truncated turns.

Reference (similar public rubric intent):
https://agentscope-ai.github.io/OpenJudge/built_in_graders/multi_turn/
#contextmemorygrader
"""

from __future__ import annotations

from data_juicer.ops.base_op import OPERATORS, TAGGING_OPS
from data_juicer.ops.mapper.dialog_quality_llm_base import _DialogTurnQualityMapper
from data_juicer.utils.constant import MetaKeys

OP_NAME = "dialog_memory_consistency_mapper"


@TAGGING_OPS.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
class DialogMemoryConsistencyMapper(_DialogTurnQualityMapper):
    """Whether the final assistant turn respects prior user constraints and facts."""

    OP_NAME = OP_NAME
    META_KEY = MetaKeys.dialog_memory_consistency

    def _system_prompt(self) -> str:
        return (
            "You are a dialog-quality judge. From the text only: does the "
            "**Assistant reply to score** honor preferences, constraints, and "
            "stated facts from earlier turns? Any clear forgetting or "
            "self-contradiction lowers the score.\n"
            "1 = severe violation or forgetting; 3 = mostly OK with gaps; "
            "5 = key constraints respected.\n"
            "Do not invent facts that are not supported by the transcript."
        )

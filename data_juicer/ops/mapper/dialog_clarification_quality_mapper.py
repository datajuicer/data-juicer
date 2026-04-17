# Copyright 2025 The Data-Juicer Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLM: clarifying questions on vague asks (1–5).

Inspired by OpenJudge-style clarification; DJ: reward direct solve when clear.

Reference:
https://agentscope-ai.github.io/OpenJudge/built_in_graders/multi_turn/
#instructionclarificationgrader
"""

from __future__ import annotations

from data_juicer.ops.base_op import OPERATORS, TAGGING_OPS
from data_juicer.ops.mapper.dialog_quality_llm_base import _DialogTurnQualityMapper
from data_juicer.utils.constant import MetaKeys

OP_NAME = "dialog_clarification_quality_mapper"


@TAGGING_OPS.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
class DialogClarificationQualityMapper(_DialogTurnQualityMapper):
    """Quality of clarifying questions when the ask is vague; direct solve when clear."""

    OP_NAME = OP_NAME
    META_KEY = MetaKeys.dialog_clarification_quality

    def _system_prompt(self) -> str:
        return (
            "When the user request is vague or missing key parameters, check "
            "whether the assistant asks **few, targeted** clarifications "
            "instead of guessing wildly or over-questioning.\n"
            "When the request is already actionable, completing it directly "
            "deserves a high score.\n"
            "Short follow-ups (e.g. status checks like 'any progress?') often do "
            "**not** require new clarifying questions—a concise progress or "
            "next-step answer can merit 3–5 unless critical parameters are "
            "obviously missing.\n"
            "1 = should clarify but does not, or answers blindly; 5 = right "
            "clarifications or a one-shot good answer."
        )

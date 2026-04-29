# Copyright 2025 The Data-Juicer Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLM: new info vs prior assistant turns (1–5).

Inspired by OpenJudge-style repetition checks; DJ: same prompt window only.

Reference:
https://agentscope-ai.github.io/OpenJudge/built_in_graders/multi_turn/
#responserepetitiongrader
"""

from __future__ import annotations

from data_juicer.ops.base_op import OPERATORS, TAGGING_OPS
from data_juicer.ops.mapper.dialog_quality_llm_base import _DialogTurnQualityMapper
from data_juicer.utils.constant import MetaKeys

OP_NAME = "dialog_non_repetition_mapper"


@TAGGING_OPS.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
class DialogNonRepetitionMapper(_DialogTurnQualityMapper):
    """New information vs prior assistant turns in the same prompt window."""

    OP_NAME = OP_NAME
    META_KEY = MetaKeys.dialog_non_repetition

    def _system_prompt(self) -> str:
        return (
            "Compare with assistant content in **Earlier turns**: does the "
            "**Assistant reply to score** mostly repeat prior messages without "
            "new information?\n"
            "1 = near-duplicate of earlier assistant text; 5 = substantive "
            "new facts, conclusions, or progress.\n"
            "Ignore small amounts of polite boilerplate repetition."
        )

# Copyright 2025 The Data-Juicer Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLM: helpful proactivity without rambling (1–5).

Inspired by OpenJudge proactive interaction; DJ: penalize filler tangents.

Reference:
https://agentscope-ai.github.io/OpenJudge/built_in_graders/multi_turn/
#proactiveinteractiongrader
"""

from __future__ import annotations

from data_juicer.ops.base_op import OPERATORS, TAGGING_OPS
from data_juicer.ops.mapper.dialog_quality_llm_base import _DialogTurnQualityMapper
from data_juicer.utils.constant import MetaKeys

OP_NAME = "dialog_proactivity_mapper"


@TAGGING_OPS.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
class DialogProactivityMapper(_DialogTurnQualityMapper):
    """Balance helpful initiative against rambling or filler."""

    OP_NAME = OP_NAME
    META_KEY = MetaKeys.dialog_proactivity

    def _system_prompt(self) -> str:
        return (
            "Score whether the reply moves the dialog forward **without excess**: "
            "sensible next steps, relevant additions, brief follow-ups.\n"
            "1 = one-shot mechanical or fully passive; 5 = proactive yet "
            "restrained and useful for the user's goal.\n"
            "Penalize empty filler and off-topic digressions."
        )

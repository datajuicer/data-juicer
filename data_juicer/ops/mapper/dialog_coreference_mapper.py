# Copyright 2025 The Data-Juicer Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLM: pronoun/reference resolution for latest user turn (1–5).

Inspired by OpenJudge-style anaphora checks; DJ: one call, short JSON reason.

Reference:
https://agentscope-ai.github.io/OpenJudge/built_in_graders/multi_turn/
#anaphoraresolutiongrader
"""

from __future__ import annotations

from data_juicer.ops.base_op import OPERATORS, TAGGING_OPS
from data_juicer.ops.mapper.dialog_quality_llm_base import _DialogTurnQualityMapper
from data_juicer.utils.constant import MetaKeys

OP_NAME = "dialog_coreference_mapper"


@TAGGING_OPS.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
class DialogCoreferenceMapper(_DialogTurnQualityMapper):
    """Whether the reply resolves pronouns/deictics in the latest user turn."""

    OP_NAME = OP_NAME
    META_KEY = MetaKeys.dialog_coreference

    def _system_prompt(self) -> str:
        return (
            "You judge coreference. Using the **Current user message**, check "
            "whether the **Assistant reply to score** picks the right referents "
            "(may rely on earlier turns).\n"
            "1 = clear mis-resolution; 3 = understandable but ambiguous; "
            "5 = crisp and correct."
        )

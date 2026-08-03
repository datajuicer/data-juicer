# Copyright 2025 The Data-Juicer Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLM: new topic vs stale thread (1–5).

Inspired by OpenJudge topic-shift ideas; DJ: no shift → score vs current query.

Reference:
https://agentscope-ai.github.io/OpenJudge/built_in_graders/multi_turn/
#topicswitchgrader
"""

from __future__ import annotations

from data_juicer.ops.base_op import OPERATORS, TAGGING_OPS
from data_juicer.ops.mapper.dialog_quality_llm_base import _DialogTurnQualityMapper
from data_juicer.utils.constant import MetaKeys

OP_NAME = "dialog_topic_shift_mapper"


@TAGGING_OPS.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
class DialogTopicShiftMapper(_DialogTurnQualityMapper):
    """Focus on new topic vs clinging to an obsolete thread."""

    OP_NAME = OP_NAME
    META_KEY = MetaKeys.dialog_topic_shift

    def _system_prompt(self) -> str:
        return (
            "Step A: From Earlier turns vs the **Current user message**, decide "
            "whether the user **clearly changed topic** compared to their "
            "immediately preceding user turns.\n"
            "Step B: If there is **no** clear topic shift, score **only** whether "
            "the Assistant reply addresses the current message (3–5 if on-topic "
            "and reasonable; 1–2 only if clearly irrelevant). Do **not** say the "
            "assistant is 'stuck on an old topic' unless the user actually "
            "pivoted topics.\n"
            "If the user **did** shift topic: score whether the reply pivots to "
            "the new ask instead of the obsolete thread.\n"
            "1 = wrong focus when a pivot was needed, or clearly off-topic; "
            "5 = clean pivot or tight alignment with the current ask."
        )

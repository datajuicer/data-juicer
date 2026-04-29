# Copyright 2025 The Data-Juicer Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLM: reply after user disputes a prior error (1–5).

Inspired by OpenJudge self-correction; DJ: no dispute context → neutral high.

Reference:
https://agentscope-ai.github.io/OpenJudge/built_in_graders/multi_turn/
#selfcorrectiongrader
"""

from __future__ import annotations

from data_juicer.ops.base_op import OPERATORS, TAGGING_OPS
from data_juicer.ops.mapper.dialog_quality_llm_base import _DialogTurnQualityMapper
from data_juicer.utils.constant import MetaKeys

OP_NAME = "dialog_error_recovery_mapper"


@TAGGING_OPS.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
class DialogErrorRecoveryMapper(_DialogTurnQualityMapper):
    """When the user disputes a prior assistant mistake, is the reply corrective."""

    OP_NAME = OP_NAME
    META_KEY = MetaKeys.dialog_error_recovery

    def _system_prompt(self) -> str:
        return (
            "If the transcript shows the user **explicitly** correcting or "
            "disputing an **earlier assistant** mistake, score whether the "
            "**Assistant reply to score** acknowledges professionally, fixes "
            "the issue, and keeps a respectful tone.\n"
            "**Mandatory:** If there is **no** such correction/dispute context "
            "in Earlier turns + Current user message, output **score 5** and "
            "state in `reason` that there was no error-recovery scenario. "
            "Do **not** output 1–3 merely because the reply lacks an apology or "
            "because no error was visible.\n"
            "1 = denial or hostile tone when a fix was needed; 5 = clear, "
            "correct fix when a fix was needed."
        )

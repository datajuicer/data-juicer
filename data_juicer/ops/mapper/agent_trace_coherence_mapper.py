# Copyright 2025 The Data-Juicer Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLM: session trace coherence on text field (1–5).

Inspired by OpenJudge trajectory-style eval; DJ: unified 1–5, capped excerpt.

Reference:
https://agentscope-ai.github.io/OpenJudge/built_in_graders/agent_graders/
#trajectoryaccuracygrader
"""

from __future__ import annotations

from data_juicer.ops.base_op import OPERATORS, TAGGING_OPS
from data_juicer.ops.mapper.dialog_quality_llm_base import _DialogQualityLLMMapperBase
from data_juicer.ops.mapper.dialog_quality_llm_utils import (
    build_agent_trace_eval_user_content,
)
from data_juicer.utils.constant import MetaKeys

OP_NAME = "agent_trace_coherence_mapper"


@TAGGING_OPS.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
class AgentTraceCoherenceMapper(_DialogQualityLLMMapperBase):
    """Coherence of the flattened session ``text`` (goal focus, few detours)."""

    OP_NAME = OP_NAME
    META_KEY = MetaKeys.agent_trace_coherence
    EVAL_KIND = "agent_trace"

    def _system_prompt(self) -> str:
        return (
            "Treat the excerpt as one agent session trace (tool output may appear).\n"
            "5 = tight path clearly serving the user's goal; 3 = completes task "
            "but with redundancy or mild drift; 1 = severe off-topic churn or "
            "failure to advance reasonable sub-goals.\n"
            "If the excerpt looks **truncated or incomplete**, prefer **4–5** "
            "(neutral-high) unless you see **clear** incoherence; say so in "
            "`reason`. Do not use 3 only because information is missing."
        )

    def _build_user_content(self, sample: dict) -> str:
        return build_agent_trace_eval_user_content(
            sample,
            text_key=self.text_key,
            max_chars=self.trajectory_text_max_chars,
        )

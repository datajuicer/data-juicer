# Copyright 2025 The Data-Juicer Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLM: tool/capability fit vs user task (1–5).

Inspired by OpenJudge tool-selection; DJ: meta tags or text-only, no schema.

Reference:
https://agentscope-ai.github.io/OpenJudge/built_in_graders/agent_graders/
#toolselectiongrader
"""

from __future__ import annotations

from data_juicer.ops.base_op import OPERATORS, TAGGING_OPS
from data_juicer.ops.mapper.dialog_quality_llm_base import _DialogQualityLLMMapperBase
from data_juicer.ops.mapper.dialog_quality_llm_utils import (
    build_agent_tool_fit_user_content,
)
from data_juicer.utils.constant import MetaKeys

OP_NAME = "agent_tool_relevance_mapper"


@TAGGING_OPS.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
class AgentToolRelevanceMapper(_DialogQualityLLMMapperBase):
    """Rough fit between tools/capabilities and the user task (uses meta tool tags)."""

    OP_NAME = OP_NAME
    META_KEY = MetaKeys.agent_tool_relevance
    EVAL_KIND = "agent_tool"

    def _system_prompt(self) -> str:
        return (
            "Using the **User request**, **Assistant reply** (including any tool "
            "trace embedded in that text), and the inferred tool list, judge "
            "whether capability choices are roughly sound: when tools are "
            "needed, were relevant ones used?\n"
            "If the meta tool list matches tool names visible in the assistant "
            "text, treat that as weak evidence of intent—reserve score 1–2 for "
            "**clear** task/tool contradictions, not for underspecified user "
            "wording alone.\n"
            "If no tool list is present, infer only from the text; an empty "
            "list does not automatically imply a bad score.\n"
            "1 = severe mismatch; 5 = apt and efficient."
        )

    def _build_user_content(self, sample: dict) -> str:
        return build_agent_tool_fit_user_content(
            sample,
            query_key=self.query_key,
            response_key=self.response_key,
            tool_types_key=self.tool_types_key,
            primary_tool_key=self.primary_tool_key,
            max_query_chars=self.max_query_chars_for_prompt,
            max_response_chars=self.max_response_chars_for_prompt,
        )

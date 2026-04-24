# Copyright 2025 The Data-Juicer Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Derive primary_tool_type and dominant_tool_types from agent_tool_types meta.
# For multi-dimensional analysis and filtering by tool/skill.

from collections import Counter

from data_juicer.ops.base_op import OPERATORS, TAGGING_OPS, Mapper
from data_juicer.utils.constant import Fields, MetaKeys

OP_NAME = "agent_tool_type_mapper"


@TAGGING_OPS.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
class AgentToolTypeMapper(Mapper):
    """Set primary_tool_type and dominant_tool_types from meta.agent_tool_types."""

    def __init__(
        self,
        tool_types_meta_key: str = MetaKeys.agent_tool_types,
        primary_key: str = MetaKeys.primary_tool_type,
        dominant_key: str = MetaKeys.dominant_tool_types,
        top_k_dominant: int = 5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tool_types_meta_key = tool_types_meta_key
        self.primary_key = primary_key
        self.dominant_key = dominant_key
        self.top_k_dominant = top_k_dominant

    def process_single(self, sample):
        if Fields.meta not in sample:
            sample[Fields.meta] = {}
        meta = sample[Fields.meta]

        tool_list = meta.get(self.tool_types_meta_key)
        if not isinstance(tool_list, list):
            tool_list = []
        tool_list = [str(t).strip() for t in tool_list if str(t).strip()]

        if not tool_list:
            meta[self.primary_key] = None
            meta[self.dominant_key] = []
            return sample

        counts = Counter(tool_list)
        most_common = counts.most_common(self.top_k_dominant)
        primary = most_common[0][0] if most_common else None
        dominant = [name for name, _ in most_common]

        meta[self.primary_key] = primary
        meta[self.dominant_key] = dominant
        return sample

# Copyright 2025 The Data-Juicer Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# LLM-based summarization of agent_tool_types + agent_skill_types into
# short concrete capability phrases (agent_skill_insights).

import re
from typing import Dict, Optional

from loguru import logger
from pydantic import PositiveInt

from data_juicer.ops.base_op import OPERATORS, TAGGING_OPS, Mapper
from data_juicer.utils.agent_output_locale import (
    agent_skill_insight_system_prompt,
    normalize_preferred_output_lang,
)
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.model_utils import get_model, prepare_model

OP_NAME = "agent_skill_insight_mapper"


@TAGGING_OPS.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
class AgentSkillInsightMapper(Mapper):
    """Summarize agent_tool_types and agent_skill_types into insights via LLM.

    Reads ``meta[agent_tool_types]`` and ``meta[agent_skill_types]`` (from
    ``agent_dialog_normalize_mapper``), calls the API for 3–5 **concrete**
    capability phrases (about 10 Chinese characters or ~4–8 English words
    each; avoid vague 'read/write / processing'), and stores them in
    ``meta[agent_skill_insights]``. Run after normalize. Override
    ``system_prompt`` for locale-specific label style.
    """

    def __init__(
        self,
        api_model: str = "gpt-4o",
        *,
        tool_types_key: str = MetaKeys.agent_tool_types,
        skill_types_key: str = MetaKeys.agent_skill_types,
        insights_key: str = MetaKeys.agent_skill_insights,
        api_endpoint: Optional[str] = None,
        response_path: Optional[str] = None,
        system_prompt: Optional[str] = None,
        try_num: PositiveInt = 2,
        model_params: Dict = {},
        sampling_params: Dict = {},
        preferred_output_lang: str = "en",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tool_types_key = tool_types_key
        self.skill_types_key = skill_types_key
        self.insights_key = insights_key
        self.preferred_output_lang = normalize_preferred_output_lang(
            preferred_output_lang,
        )
        self.system_prompt = system_prompt or agent_skill_insight_system_prompt(
            self.preferred_output_lang,
        )
        self.try_num = try_num
        self.sampling_params = sampling_params or {}
        self.model_key = prepare_model(
            model_type="api",
            model=api_model,
            endpoint=api_endpoint,
            response_path=response_path,
            **model_params,
        )

    def process_single(self, sample, rank=None):
        meta = sample.get(Fields.meta)
        if not isinstance(meta, dict):
            return sample
        if self.insights_key in meta:
            return sample

        tools = meta.get(self.tool_types_key) or []
        skills = meta.get(self.skill_types_key) or []
        if not isinstance(tools, list):
            tools = [tools] if tools else []
        if not isinstance(skills, list):
            skills = [skills] if skills else []
        tools = [str(t).strip() for t in tools if str(t).strip()]
        skills = [str(s).strip() for s in skills if str(s).strip()]

        if not tools and not skills:
            meta[self.insights_key] = []
            return sample

        tools_str = ", ".join(str(x) for x in tools[:30])
        skills_str = ", ".join(str(x) for x in skills[:30])
        user_content = f"Tools: {tools_str}\nSkills: {skills_str}"

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]
        raw = ""
        for _ in range(self.try_num):
            try:
                client = get_model(self.model_key, rank=rank)
                raw = client(messages, **self.sampling_params)
                if raw and isinstance(raw, str) and raw.strip():
                    break
            except Exception as e:
                logger.warning("agent_skill_insight_mapper: %s", e)

        if not raw or not isinstance(raw, str):
            meta[self.insights_key] = []
            return sample
        # Split on common separators (LLMs often use Chinese commas /顿号).
        parts = re.split(r"[,，、;；]+", raw.strip())
        labels = [s.strip() for s in parts if s.strip()]
        meta[self.insights_key] = list(dict.fromkeys(labels))
        return sample

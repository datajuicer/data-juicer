# Copyright 2025 The Data-Juicer Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Internal base classes for dialog/agent turn- and trace-quality LLM mappers.

Each concrete mapper performs **one** API call per sample and writes
``meta[META_KEY]`` with ``score`` (1–5), ``reason``, and ``eval_kind``.
Inputs are composable: set ``history_key`` / ``query_key`` / ``response_key``
or ``text_key`` to match your dataset. After ``agent_dialog_normalize_mapper``,
``dialog_history[-1]`` matches ``query``/``response``; ``build_dialog_turn_eval_user_content``
deduplicates so that final turn is not repeated under both "Earlier turns" and
"Current" (which previously biased non-repetition and topic-shift judges).
Override rubrics by subclassing or forking the mapper; rubric text stays English-friendly.
Pass ``preferred_output_lang="zh"`` (YAML) so JSON ``reason`` and instructions use Chinese;
``"en"`` keeps English (default).
"""

from __future__ import annotations

from typing import Dict, Optional

from loguru import logger
from pydantic import NonNegativeInt, PositiveInt

from data_juicer.ops.base_op import Mapper
from data_juicer.ops.mapper.dialog_quality_llm_utils import (
    build_dialog_turn_eval_user_content,
    extract_json_object,
    normalize_score_1_5,
)
from data_juicer.utils.agent_output_locale import (
    dialog_score_json_instruction,
    normalize_preferred_output_lang,
    rubric_reason_language_clause,
)
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.model_utils import get_model, prepare_model


class _DialogQualityLLMMapperBase(Mapper):
    """One API call → meta[META_KEY] with score (1–5) and reason."""

    OP_NAME = ""
    META_KEY = ""
    EVAL_KIND = "dialog_turn"

    def __init__(
        self,
        api_model: str = "qwen-turbo",
        *,
        api_endpoint: Optional[str] = None,
        response_path: Optional[str] = None,
        history_key: str = "dialog_history",
        query_key: str = "query",
        response_key: str = "response",
        text_key: str = "text",
        max_round: NonNegativeInt = 8,
        max_query_chars_for_prompt: NonNegativeInt = 6000,
        max_response_chars_for_prompt: NonNegativeInt = 8000,
        trajectory_text_max_chars: NonNegativeInt = 12000,
        tool_types_key: str = MetaKeys.agent_tool_types,
        primary_tool_key: str = MetaKeys.primary_tool_type,
        try_num: PositiveInt = 2,
        overwrite: bool = False,
        model_params: Optional[Dict] = None,
        sampling_params: Optional[Dict] = None,
        preferred_output_lang: str = "en",
        **kwargs,
    ):
        super().__init__(text_key=text_key, **kwargs)
        self.history_key = history_key
        self.query_key = query_key
        self.response_key = response_key
        self.max_round = int(max_round)
        self.max_query_chars_for_prompt = int(max_query_chars_for_prompt)
        self.max_response_chars_for_prompt = int(max_response_chars_for_prompt)
        self.trajectory_text_max_chars = int(trajectory_text_max_chars)
        self.tool_types_key = tool_types_key
        self.primary_tool_key = primary_tool_key
        self.try_num = int(try_num)
        self.overwrite = bool(overwrite)
        self.sampling_params = dict(sampling_params or {})
        self.sampling_params.setdefault("max_tokens", 384)
        self.sampling_params.setdefault("temperature", 0.2)
        self.model_key = prepare_model(
            model_type="api",
            model=api_model,
            endpoint=api_endpoint,
            response_path=response_path,
            **(model_params or {}),
        )
        self.preferred_output_lang = normalize_preferred_output_lang(preferred_output_lang)

    def _system_prompt(self) -> str:
        raise NotImplementedError

    def _json_instruction(self) -> str:
        return dialog_score_json_instruction(self.preferred_output_lang)

    def _build_user_content(self, sample: dict) -> str:
        raise NotImplementedError

    def process_single(self, sample, rank=None):
        mk = self.META_KEY
        meta = sample.setdefault(Fields.meta, {})
        if not self.overwrite and mk in meta:
            return sample

        user_block = self._build_user_content(sample)
        if not (user_block and str(user_block).strip()):
            meta[mk] = {"skipped": True, "reason": "empty_input"}
            return sample

        system = (
            f"{self._system_prompt()}\n\n{self._json_instruction()}\n\n"
            f"{rubric_reason_language_clause(self.preferred_output_lang)}"
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_block},
        ]
        client = get_model(self.model_key, rank=rank)
        raw = ""
        for attempt in range(self.try_num):
            try:
                raw = client(messages, **self.sampling_params)
                if raw and isinstance(raw, str) and raw.strip():
                    break
            except Exception as e:
                logger.warning(
                    "%s attempt %s: %s",
                    self.OP_NAME,
                    attempt + 1,
                    e,
                )

        if not raw:
            meta[mk] = {"error": "empty_llm_response"}
            return sample

        parsed = extract_json_object(raw)
        if parsed is None:
            meta[mk] = {"error": "json_parse_failed", "raw": raw[:8000]}
            return sample

        out = normalize_score_1_5(parsed)
        out["eval_kind"] = self.EVAL_KIND
        meta[mk] = out
        return sample


class _DialogTurnQualityMapper(_DialogQualityLLMMapperBase):
    """Score the final assistant turn given ``dialog_history`` tail."""

    EVAL_KIND = "dialog_turn"

    def _build_user_content(self, sample: dict) -> str:
        return build_dialog_turn_eval_user_content(
            sample,
            history_key=self.history_key,
            query_key=self.query_key,
            response_key=self.response_key,
            max_round=self.max_round,
            max_query_chars=self.max_query_chars_for_prompt,
            max_response_chars=self.max_response_chars_for_prompt,
        )

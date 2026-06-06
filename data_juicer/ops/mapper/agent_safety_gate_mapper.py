# Copyright 2025 The Data-Juicer Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# LLM yes/no gate before calling external teacher APIs — writes meta only (no filtering).

from __future__ import annotations

from typing import Optional

from loguru import logger
from pydantic import PositiveInt

from data_juicer.ops.base_op import OPERATORS, TAGGING_OPS, Mapper
from data_juicer.utils.constant import Fields, MetaKeys, StatsKeys
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.llm_semantic_ops import InferenceStrategy, condition_filter_one
from data_juicer.utils.model_utils import (
    get_model,
    prepare_model,
    update_sampling_params,
)

vllm = LazyLoader("vllm")

OP_NAME = "agent_safety_gate_mapper"


def _ensure_meta_dict(sample: dict) -> dict:
    mk = Fields.meta
    m = sample.get(mk)
    if isinstance(m, dict):
        return m
    import json

    if isinstance(m, str) and m.strip():
        try:
            parsed = json.loads(m)
            if isinstance(parsed, dict):
                sample[mk] = parsed
                return parsed
        except json.JSONDecodeError:
            pass
    sample[mk] = {}
    return sample[mk]


@TAGGING_OPS.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
class AgentSafetyGateMapper(Mapper):
    """Run a natural-language condition on ``text_key``; store result in meta (no drop)."""

    _accelerator = "cuda"

    def __init__(
        self,
        text_key: str = "text",
        condition: str = "",
        api_or_hf_model: str = "gpt-4o",
        *,
        api_endpoint: Optional[str] = None,
        response_path: Optional[str] = None,
        is_hf_model: bool = False,
        enable_vllm: bool = False,
        strategy: Optional[InferenceStrategy] = None,
        examples: Optional[str] = None,
        try_num: PositiveInt = 2,
        model_params: Optional[dict] = None,
        sampling_params: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.text_key = text_key
        self.condition = condition or (
            "The text is safe for downstream model training: no toxic content, "
            "no clear copyright infringement, and no instructions to exfiltrate secrets."
        )
        self.try_num = try_num
        self.strategy = strategy
        self.examples = examples
        self.is_hf_model = is_hf_model
        self.enable_vllm = enable_vllm
        model_params = model_params or {}
        sampling_params = update_sampling_params(sampling_params or {}, api_or_hf_model, enable_vllm)
        self.sampling_params = sampling_params
        if enable_vllm:
            self.model_key = prepare_model(
                model_type="vllm",
                pretrained_model_name_or_path=api_or_hf_model,
                **model_params,
            )
            self.sampling_params = vllm.SamplingParams(**self.sampling_params)
        elif is_hf_model:
            self.model_key = prepare_model(
                model_type="huggingface",
                pretrained_model_name_or_path=api_or_hf_model,
                return_pipe=True,
                trust_remote_code=True,
                **model_params,
            )
        else:
            self.model_key = prepare_model(
                model_type="api",
                model=api_or_hf_model,
                endpoint=api_endpoint,
                response_path=response_path,
                **model_params,
            )

    def process_single(self, sample: dict, rank=None) -> dict:
        meta = _ensure_meta_dict(sample)
        text = str(sample.get(self.text_key) or "").strip()
        if not text:
            meta[MetaKeys.agent_training_safety_gate] = {"ok": False, "skipped": True, "reason": "empty_text"}
            return sample

        if self.enable_vllm or self.is_hf_model:
            model, _ = get_model(self.model_key, rank, self.use_cuda())
        else:
            model = get_model(self.model_key, rank, self.use_cuda())

        ok = False
        usage = None
        for _ in range(self.try_num):
            try:
                ok, usage = condition_filter_one(
                    text,
                    self.condition,
                    model,
                    strategy=self.strategy,
                    examples=self.examples,
                    enable_vllm=self.enable_vllm,
                    is_hf_model=self.is_hf_model,
                    sampling_params=self.sampling_params,
                )
                break
            except Exception as e:
                logger.warning("agent_safety_gate_mapper: %s", e)

        meta[MetaKeys.agent_training_safety_gate] = {
            "ok": bool(ok),
            "condition": self.condition,
        }
        if usage is not None:
            prev = meta.get(MetaKeys.llm_semantic_usage, {})
            curr = usage.to_dict()
            meta[MetaKeys.llm_semantic_usage] = {
                "prompt_tokens": int(prev.get("prompt_tokens", 0)) + int(curr.get("prompt_tokens", 0)),
                "completion_tokens": int(prev.get("completion_tokens", 0)) + int(curr.get("completion_tokens", 0)),
                "total_tokens": int(prev.get("total_tokens", 0)) + int(curr.get("total_tokens", 0)),
                "cost_estimate": float(prev.get("cost_estimate", 0)) + float(curr.get("cost_estimate", 0)),
            }
        if Fields.stats not in sample:
            sample[Fields.stats] = {}
        sample[Fields.stats][StatsKeys.llm_condition_filter_result] = bool(ok)
        return sample

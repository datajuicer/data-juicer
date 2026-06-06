# Copyright 2025 The Data-Juicer Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# LLM: structured rewrite hints for low-tier rows (orchestrator-friendly; does not mutate messages).

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from loguru import logger
from pydantic import PositiveInt

from data_juicer.ops.base_op import OPERATORS, TAGGING_OPS, Mapper
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.llm_semantic_ops import call_llm_sync
from data_juicer.utils.model_utils import (
    get_model,
    prepare_model,
    update_sampling_params,
)

vllm = LazyLoader("vllm")

OP_NAME = "agent_rewrite_hint_mapper"


def _ensure_meta_dict(sample: dict) -> dict:
    mk = Fields.meta
    m = sample.get(mk)
    if isinstance(m, dict):
        return m
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


def _parse_json_obj(raw: str) -> Optional[dict]:
    if not raw or not isinstance(raw, str):
        return None
    text = raw.strip()
    m = re.search(r"\{[\s\S]*\}\s*$", text)
    if m:
        text = m.group(0)
    try:
        out = json.loads(text)
        return out if isinstance(out, dict) else None
    except (json.JSONDecodeError, TypeError):
        return None


@TAGGING_OPS.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
class AgentRewriteHintMapper(Mapper):
    """Write ``meta.agent_rewrite_hints`` (JSON) for selected tiers."""

    _accelerator = "cuda"

    def __init__(
        self,
        api_model: str = "gpt-4o",
        *,
        api_endpoint: Optional[str] = None,
        response_path: Optional[str] = None,
        is_hf_model: bool = False,
        enable_vllm: bool = False,
        model_params: Optional[dict] = None,
        sampling_params: Optional[dict] = None,
        try_num: PositiveInt = 2,
        query_key: str = "query",
        response_key: str = "response",
        only_for_training_dataset_tiers: Optional[List[str]] = None,
        preferred_output_lang: str = "en",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.query_key = query_key
        self.response_key = response_key
        self.only_for_training_dataset_tiers = [
            str(x).strip().lower() for x in (only_for_training_dataset_tiers or ["bronze"])
        ]
        self.try_num = try_num
        self.preferred_output_lang = (preferred_output_lang or "en").lower()
        self.is_hf_model = is_hf_model
        self.enable_vllm = enable_vllm
        model_params = model_params or {}
        sp = update_sampling_params(sampling_params or {}, api_model, enable_vllm)
        self.sampling_params = sp
        if enable_vllm:
            self.model_key = prepare_model(
                model_type="vllm",
                pretrained_model_name_or_path=api_model,
                **model_params,
            )
            self.sampling_params = vllm.SamplingParams(**self.sampling_params)
        elif is_hf_model:
            self.model_key = prepare_model(
                model_type="huggingface",
                pretrained_model_name_or_path=api_model,
                return_pipe=True,
                trust_remote_code=True,
                **model_params,
            )
        else:
            self.model_key = prepare_model(
                model_type="api",
                model=api_model,
                endpoint=api_endpoint,
                response_path=response_path,
                **model_params,
            )
        self._api_model_label = str(api_model)

    def _system_prompt(self) -> str:
        if self.preferred_output_lang.startswith("zh"):
            return (
                "你是数据清洗顾问。根据 query 与 response 摘要，输出单个 JSON："
                '{"hints":["..."], "risk_flags":["..."], "suggested_next_step": "none|human_review|re_run_tools"} '
                "不要输出除 JSON 以外的文字。"
            )
        return (
            "You are a data-cleaning advisor. From the user query and model response excerpt, "
            "output a single JSON object: "
            '{"hints":["..."], "risk_flags":["..."], "suggested_next_step": "none|human_review|re_run_tools"}. '
            "No prose outside JSON."
        )

    def process_single(self, sample: dict, rank=None) -> dict:
        meta = _ensure_meta_dict(sample)
        tier = str(
            meta.get(MetaKeys.agent_training_dataset_tier) or meta.get(MetaKeys.agent_learnable_value_tier) or ""
        ).lower()
        if tier not in self.only_for_training_dataset_tiers:
            return sample

        q = str(sample.get(self.query_key) or "")[:6000]
        r = str(sample.get(self.response_key) or "")[:6000]
        tax = meta.get(MetaKeys.agent_error_taxonomy)
        pack = {"query": q, "response": r, "agent_error_taxonomy": tax}
        user = json.dumps(pack, ensure_ascii=False)

        if self.enable_vllm or self.is_hf_model:
            model, _ = get_model(self.model_key, rank, self.use_cuda())
        else:
            model = get_model(self.model_key, rank, self.use_cuda())

        messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": user},
        ]
        raw = ""
        usage = None
        for _ in range(self.try_num):
            try:
                raw, usage = call_llm_sync(
                    model,
                    messages,
                    enable_vllm=self.enable_vllm,
                    is_hf_model=self.is_hf_model,
                    sampling_params=self.sampling_params,
                )
                if raw and raw.strip():
                    break
            except Exception as e:
                logger.warning("agent_rewrite_hint_mapper: %s", e)

        parsed = _parse_json_obj(raw) if raw else None
        out: Dict[str, Any] = {
            "teacher_model": self._api_model_label,
            "parse_ok": bool(parsed),
            "raw": raw[:4000] if isinstance(raw, str) else "",
        }
        if parsed:
            out["hints"] = parsed.get("hints")
            out["risk_flags"] = parsed.get("risk_flags")
            out["suggested_next_step"] = parsed.get("suggested_next_step")
        meta[MetaKeys.agent_rewrite_hints] = out
        if usage is not None:
            prev = meta.get(MetaKeys.llm_semantic_usage, {})
            curr = usage.to_dict()
            meta[MetaKeys.llm_semantic_usage] = {
                "prompt_tokens": int(prev.get("prompt_tokens", 0)) + int(curr.get("prompt_tokens", 0)),
                "completion_tokens": int(prev.get("completion_tokens", 0)) + int(curr.get("completion_tokens", 0)),
                "total_tokens": int(prev.get("total_tokens", 0)) + int(curr.get("total_tokens", 0)),
                "cost_estimate": float(prev.get("cost_estimate", 0)) + float(curr.get("cost_estimate", 0)),
            }
        return sample

# Copyright 2025 The Data-Juicer Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Optional teacher LLM: propose a cleaned / improved final reply for SFT-style targets.

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

OP_NAME = "agent_distill_trajectory_mapper"


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


def _coerce_messages(val: Any) -> List[dict]:
    if val is None:
        return []
    if isinstance(val, list):
        return [x for x in val if isinstance(x, dict)]
    if isinstance(val, str) and val.strip():
        try:
            p = json.loads(val)
            if isinstance(p, list):
                return [x for x in p if isinstance(x, dict)]
        except json.JSONDecodeError:
            return []
    return []


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
class AgentDistillTrajectoryMapper(Mapper):
    """Write ``meta.agent_distilled_trajectory`` for selected training-dataset tiers (API)."""

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
        messages_key: str = "messages",
        query_key: str = "query",
        response_key: str = "response",
        only_for_training_dataset_tiers: Optional[List[str]] = None,
        require_safety_gate_ok: bool = True,
        max_context_chars: int = 12000,
        preferred_output_lang: str = "en",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.messages_key = messages_key
        self.query_key = query_key
        self.response_key = response_key
        self.only_for_training_dataset_tiers = [
            str(x).strip().lower() for x in (only_for_training_dataset_tiers or ["gold", "silver"])
        ]
        self.require_safety_gate_ok = bool(require_safety_gate_ok)
        self.max_context_chars = max(2000, int(max_context_chars))
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
                "你是用于构造训练数据的教师模型。阅读用户任务与对话 JSON，"
                "只输出一个 JSON 对象，键为：distilled_final_reply（字符串）、"
                "summary_of_fixes（字符串）、confidence（0到1的小数）。"
                "不要编造日志中未出现的工具结果；若上下文不完整请在 summary_of_fixes 说明。"
            )
        return (
            "You are a teacher model for building training data. Read the user task and "
            "conversation JSON. Reply with ONE JSON object only, keys: "
            "distilled_final_reply (string), summary_of_fixes (string), confidence (0-1 float). "
            "Do not invent tool outcomes not shown in the log; if context is incomplete, say so "
            "in summary_of_fixes."
        )

    def process_single(self, sample: dict, rank=None) -> dict:
        meta = _ensure_meta_dict(sample)
        tier = str(
            meta.get(MetaKeys.agent_training_dataset_tier) or meta.get(MetaKeys.agent_learnable_value_tier) or ""
        ).lower()
        if tier not in self.only_for_training_dataset_tiers:
            return sample
        if self.require_safety_gate_ok:
            gate = meta.get(MetaKeys.agent_training_safety_gate)
            if not isinstance(gate, dict):
                meta[MetaKeys.agent_distilled_trajectory] = {
                    "skipped": True,
                    "reason": "missing_safety_gate",
                }
                return sample
            if gate.get("ok") is False and not gate.get("skipped"):
                meta[MetaKeys.agent_distilled_trajectory] = {
                    "skipped": True,
                    "reason": "safety_gate_not_ok",
                }
                return sample

        msgs = _coerce_messages(sample.get(self.messages_key))
        blob = json.dumps(msgs, ensure_ascii=False)
        if len(blob) > self.max_context_chars:
            h = self.max_context_chars // 2
            blob = blob[:h] + "\n...[truncated]...\n" + blob[-h:]

        q = str(sample.get(self.query_key) or "")[:8000]
        r = str(sample.get(self.response_key) or "")[:8000]
        user = json.dumps(
            {"last_user_query": q, "last_model_response": r, "messages_json": blob},
            ensure_ascii=False,
        )

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
                logger.warning("agent_distill_trajectory_mapper: %s", e)

        parsed = _parse_json_obj(raw) if raw else None
        out: Dict[str, Any] = {
            "teacher_model": getattr(self, "_api_model_label", ""),
            "raw": raw[:8000] if isinstance(raw, str) else "",
        }
        if parsed:
            out.update(parsed)
        meta[MetaKeys.agent_distilled_trajectory] = out
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

"""LLM semantic ops: user-configurable extract/condition helpers.

Supports both structured output (JSON/schema) and unstructured input (e.g. plain
text, jsonl). Shared by llm_extract_mapper, llm_condition_filter; reusable for
DataFrame/SQL/DB by adapting input to (text, schema/condition). Aligns with llm_* naming.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)


# ---- Cost / usage ----


@dataclass
class LLMCallUsage:
    """Token usage (and optional cost) for a single LLM call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_estimate: Optional[float] = None  # optional $ estimate if pricing known

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }
        if self.cost_estimate is not None:
            d["cost_estimate"] = self.cost_estimate
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LLMCallUsage":
        return cls(
            prompt_tokens=int(d.get("prompt_tokens", 0)),
            completion_tokens=int(d.get("completion_tokens", 0)),
            total_tokens=int(d.get("total_tokens", 0)),
            cost_estimate=d.get("cost_estimate"),
        )


# ---- Record row / batch (Pydantic for type check and autocomplete) ----


class RecordRow(BaseModel):
    """Single row of extracted fields; schema aligns with output_schema keys.

    Use model_validate(d) or RecordRow(**d) for dict -> RecordRow.
    Use row.model_dump() for RecordRow -> dict. Extra keys from output_schema
    are allowed (extra='allow').
    """

    model_config = ConfigDict(extra="allow")

    @classmethod
    def from_schema_dict(cls, d: Dict[str, Any], schema_keys: Optional[List[str]] = None) -> "RecordRow":
        """Build RecordRow from dict; optionally restrict to schema_keys."""
        if schema_keys is not None:
            d = {k: d.get(k) for k in schema_keys}
        return cls.model_validate(d)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


# Type alias: batch of extraction records
RecordBatch = List[RecordRow]


def record_batch_from_dicts(items: List[Dict[str, Any]], schema_keys: Optional[List[str]] = None) -> RecordBatch:
    """Convert list of dicts to RecordBatch (list of RecordRow)."""
    return [RecordRow.from_schema_dict(d, schema_keys) for d in items]


def record_batch_to_dicts(batch: RecordBatch) -> List[Dict[str, Any]]:
    """Convert RecordBatch to list of dicts."""
    return [row.to_dict() for row in batch]


class InferenceStrategy(Enum):
    DIRECT = "direct"  # Fast, no reasoning
    COT = "cot"  # Chain-of-Thought reasoning
    FEW_SHOT = "few_shot"  # Direct with examples
    COT_SHOT = "cot_shot"  # Reasoning with examples


# ---- Default prompts for user-configurable schema/condition ----

DEFAULT_EXTRACT_SYSTEM = (
    "You are a precise information extraction assistant. "
    "Given the input text, extract the requested fields and return ONLY a valid "
    "JSON object. Use the exact output keys provided. "
    "If a value cannot be determined, use null."
)  # noqa: E501

DEFAULT_EXTRACT_USER_TEMPLATE = (
    "# Input\n{input_text}\n\n"
    "# Instructions (output keys and what to extract)\n"
    "{output_instructions}\n\n"
    "# Output\nReturn a single JSON object with the above keys. No explanation."
)

DEFAULT_EXTRACT_COT_TEMPLATE = (
    "# Input\n{input_text}\n\n"
    "# Instructions (output keys and what to extract)\n"
    "{output_instructions}\n\n"
    "# Output\nFirst, think step by step about how to extract each field. "
    "Then return a single JSON object with the above keys."
)

DEFAULT_EXTRACT_FEW_SHOT_TEMPLATE = (
    "# Examples\n{examples}\n\n"
    "# Input\n{input_text}\n\n"
    "# Instructions (output keys and what to extract)\n"
    "{output_instructions}\n\n"
    "# Output\nReturn a single JSON object with the above keys. No explanation."
)

DEFAULT_EXTRACT_COT_SHOT_TEMPLATE = (
    "# Examples\n{examples}\n\n"
    "# Input\n{input_text}\n\n"
    "# Instructions (output keys and what to extract)\n"
    "{output_instructions}\n\n"
    "# Output\nFirst, think step by step about how to extract each field based on the examples. "
    "Then return a single JSON object with the above keys."
)

DEFAULT_CONDITION_SYSTEM = "You are a binary classifier. Answer only 'yes' or 'no', nothing else."

DEFAULT_CONDITION_USER_TEMPLATE = (
    "# Text\n{text}\n\n# Condition\n{condition}\n\n" "Does the text satisfy the condition? Answer yes or no."
)  # noqa: E501

DEFAULT_CONDITION_COT_TEMPLATE = (
    "# Text\n{text}\n\n# Condition\n{condition}\n\n"
    "Think step by step: analyze the text and condition, then determine if the text satisfies the condition. "
    "Answer yes or no."
)

DEFAULT_CONDITION_FEW_SHOT_TEMPLATE = (
    "# Examples\n{examples}\n\n"
    "# Text\n{text}\n\n# Condition\n{condition}\n\n"
    "Does the text satisfy the condition? Answer yes or no."
)

DEFAULT_CONDITION_COT_SHOT_TEMPLATE = (
    "# Examples\n{examples}\n\n"
    "# Text\n{text}\n\n# Condition\n{condition}\n\n"
    "Think step by step: analyze the text and condition based on the examples, "
    "then determine if the text satisfies the condition. Answer yes or no."
)


def get_extract_prompt(
    input_text: str,
    output_schema: Dict[str, str],
    knowledge_grounding: Optional[str] = None,
    strategy: Optional[InferenceStrategy] = None,
    examples: Optional[str] = None,
) -> str:
    """Build user prompt for extraction. output_schema: {key: instruction}."""
    instructions = "\n".join(f"- {k}: {v}" for k, v in output_schema.items())
    strategy = strategy or InferenceStrategy.DIRECT

    template_map = {
        InferenceStrategy.DIRECT: lambda: DEFAULT_EXTRACT_USER_TEMPLATE.format(
            input_text=input_text,
            output_instructions=instructions,
        ),
        InferenceStrategy.COT: lambda: DEFAULT_EXTRACT_COT_TEMPLATE.format(
            input_text=input_text,
            output_instructions=instructions,
        ),
        InferenceStrategy.FEW_SHOT: lambda: DEFAULT_EXTRACT_FEW_SHOT_TEMPLATE.format(
            examples=examples or "",
            input_text=input_text,
            output_instructions=instructions,
        ),
        InferenceStrategy.COT_SHOT: lambda: DEFAULT_EXTRACT_COT_SHOT_TEMPLATE.format(
            examples=examples or "",
            input_text=input_text,
            output_instructions=instructions,
        ),
    }

    body = template_map.get(strategy, template_map[InferenceStrategy.DIRECT])()

    if knowledge_grounding:
        body = f"# Background / grounding\n{knowledge_grounding}\n\n" + body
    return body


def get_condition_prompt(
    text: str,
    condition: str,
    knowledge_grounding: Optional[str] = None,
    strategy: Optional[InferenceStrategy] = None,
    examples: Optional[str] = None,
) -> str:
    """Build user prompt for LLM condition filter (yes/no)."""
    strategy = strategy or InferenceStrategy.DIRECT

    template_map = {
        InferenceStrategy.DIRECT: lambda: DEFAULT_CONDITION_USER_TEMPLATE.format(text=text, condition=condition),
        InferenceStrategy.COT: lambda: DEFAULT_CONDITION_COT_TEMPLATE.format(text=text, condition=condition),
        InferenceStrategy.FEW_SHOT: lambda: DEFAULT_CONDITION_FEW_SHOT_TEMPLATE.format(
            examples=examples or "",
            text=text,
            condition=condition,
        ),
        InferenceStrategy.COT_SHOT: lambda: DEFAULT_CONDITION_COT_SHOT_TEMPLATE.format(
            examples=examples or "",
            text=text,
            condition=condition,
        ),
    }

    body = template_map.get(strategy, template_map[InferenceStrategy.DIRECT])()

    if knowledge_grounding:
        body = f"# Background\n{knowledge_grounding}\n\n" + body
    return body


def _parse_usage_from_response(response: Any, is_api: bool) -> LLMCallUsage:
    """Extract token usage from API response when available."""
    usage = LLMCallUsage()
    try:
        if is_api and hasattr(response, "usage"):
            u = response.usage
            pt = getattr(u, "prompt_tokens", 0) or getattr(u, "input_tokens", 0)
            ct = getattr(u, "completion_tokens", 0) or getattr(u, "output_tokens", 0)
            usage = LLMCallUsage(
                prompt_tokens=pt,
                completion_tokens=ct,
                total_tokens=getattr(u, "total_tokens", 0) or (pt + ct),
            )
        elif isinstance(response, dict):
            u = response.get("usage", response)
            if isinstance(u, dict):
                usage = LLMCallUsage(
                    prompt_tokens=u.get("prompt_tokens", u.get("input_tokens", 0)),
                    completion_tokens=u.get("completion_tokens", u.get("output_tokens", 0)),
                    total_tokens=u.get("total_tokens", 0),
                )
    except Exception as e:
        logger.debug("Could not parse LLM usage: %s", e)
    return usage


def call_llm_sync(
    model: Any,
    messages: list,
    *,
    enable_vllm: bool = False,
    is_hf_model: bool = False,
    sampling_params: Optional[Dict] = None,
) -> tuple[str, LLMCallUsage]:
    """Call LLM synchronously; return (content, usage). Compatible with DJ model_utils."""
    sampling_params = sampling_params or {}
    usage = LLMCallUsage()
    try:
        if enable_vllm:
            from data_juicer.utils.lazy_loader import LazyLoader

            vllm_mod = LazyLoader("vllm")
            sp = vllm_mod.SamplingParams(**sampling_params) if isinstance(sampling_params, dict) else sampling_params
            response = model.chat(messages, sp)
            text = (response[0].outputs[0].text or "").strip()
            # vLLM may expose usage on request
            if hasattr(response[0], "usage") and response[0].usage:
                u = response[0].usage
                usage = LLMCallUsage(
                    prompt_tokens=getattr(u, "prompt_tokens", 0),
                    completion_tokens=getattr(u, "completion_tokens", 0),
                    total_tokens=getattr(u, "total_tokens", 0),
                )
            return text, usage
        if is_hf_model:
            out = model(messages, return_full_text=False, **sampling_params)
            text = (out[0].get("generated_text") or "").strip()
            return text, usage
        # API path: model may return (content, response) or content
        result = model(messages, **sampling_params)
        if isinstance(result, tuple) and len(result) >= 2:
            text = (result[0] or "").strip()
            usage = _parse_usage_from_response(result[1], is_api=True)
            return text, usage
        text = (result or "").strip()
        if hasattr(model, "last_response") and getattr(model, "last_response"):
            usage = _parse_usage_from_response(model.last_response, is_api=True)
        return text, usage
    except Exception as e:
        logger.warning("LLM semantic ops call failed: %s", e)
        return "", usage


def extract_one(
    input_text: str,
    output_schema: Dict[str, str],
    model: Any,
    *,
    system_prompt: Optional[str] = None,
    knowledge_grounding: Optional[str] = None,
    strategy: Optional[InferenceStrategy] = None,
    examples: Optional[str] = None,
    enable_vllm: bool = False,
    is_hf_model: bool = False,
    sampling_params: Optional[Dict] = None,
    return_record_row: bool = False,
) -> Union[tuple[Dict[str, Any], LLMCallUsage], tuple[RecordRow, LLMCallUsage]]:
    """Extract structured fields from input_text using the model.

    Returns (result, usage). result is dict by default, or RecordRow if return_record_row=True.
    Compatible with both structured (JSON) and unstructured (e.g. plain text) input.
    """
    user_content = get_extract_prompt(input_text, output_schema, knowledge_grounding, strategy, examples)
    messages = [
        {"role": "system", "content": system_prompt or DEFAULT_EXTRACT_SYSTEM},
        {"role": "user", "content": user_content},
    ]
    raw, usage = call_llm_sync(
        model,
        messages,
        enable_vllm=enable_vllm,
        is_hf_model=is_hf_model,
        sampling_params=sampling_params,
    )
    out_keys = list(output_schema.keys())
    if not raw:
        out = {k: None for k in out_keys}
        return (RecordRow.from_schema_dict(out, out_keys), usage) if return_record_row else (out, usage)
    raw = raw.strip()
    start_obj = raw.find("{")
    end_obj = raw.rfind("}") + 1
    start_arr = raw.find("[")
    end_arr = raw.rfind("]") + 1
    has_obj = start_obj >= 0 and end_obj > start_obj
    has_arr = start_arr >= 0 and end_arr > start_arr
    try:
        if has_obj and (not has_arr or start_obj <= start_arr):
            js = json.loads(raw[start_obj:end_obj])
            if isinstance(js, dict):
                out = {k: js.get(k) for k in out_keys}
                if return_record_row:
                    return RecordRow.from_schema_dict(out, out_keys), usage
                return out, usage
        if has_arr and len(out_keys) == 1:
            js = json.loads(raw[start_arr:end_arr])
            if isinstance(js, list):
                out = {out_keys[0]: js}
                if return_record_row:
                    return RecordRow.from_schema_dict(out, out_keys), usage
                return out, usage
    except json.JSONDecodeError:
        pass
    out = {k: None for k in out_keys}
    if return_record_row:
        return RecordRow.from_schema_dict(out, out_keys), usage
    return out, usage


def condition_filter_one(
    text: str,
    condition: str,
    model: Any,
    *,
    knowledge_grounding: Optional[str] = None,
    strategy: Optional[InferenceStrategy] = None,
    examples: Optional[str] = None,
    enable_vllm: bool = False,
    is_hf_model: bool = False,
    sampling_params: Optional[Dict] = None,
) -> tuple[bool, LLMCallUsage]:
    """True iff the model says the text satisfies the condition (yes/no). Returns (result, usage)."""
    user_content = get_condition_prompt(text, condition, knowledge_grounding, strategy, examples)
    messages = [
        {"role": "system", "content": DEFAULT_CONDITION_SYSTEM},
        {"role": "user", "content": user_content},
    ]
    raw, usage = call_llm_sync(
        model,
        messages,
        enable_vllm=enable_vllm,
        is_hf_model=is_hf_model,
        sampling_params=sampling_params,
    )
    if not raw:
        return False, usage
    low = raw.strip().lower()
    return (low.startswith("yes") or low == "y"), usage

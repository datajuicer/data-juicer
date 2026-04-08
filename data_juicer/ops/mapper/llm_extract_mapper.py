"""LLM extract mapper: user-configurable structured extraction into sample meta.

Part of the llm_* ops family; distinguished by user-provided output_schema
rather than fixed evaluation dimensions.
"""

from typing import Dict, List, Optional

from loguru import logger
from pydantic import PositiveInt

from data_juicer.ops.base_op import OPERATORS, TAGGING_OPS, Mapper
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.llm_semantic_ops import InferenceStrategy, extract_one
from data_juicer.utils.model_utils import (
    get_model,
    prepare_model,
    update_sampling_params,
)

vllm = LazyLoader("vllm")

OP_NAME = "llm_extract_mapper"


@TAGGING_OPS.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
class LLMExtractMapper(Mapper):
    """Extract structured fields from text using an LLM; write results to meta.

    Input: sample[input_keys] -> concatenated as input text.
    Output: meta[meta_output_key] (dict) or meta[out_key] per output_schema key.
    Uses user-provided output_schema (key -> instruction); supports
    knowledge_grounding via sample key or fixed string.
    """

    _accelerator = "cuda"

    def __init__(
        self,
        input_keys: List[str],
        output_schema: Dict[str, str],
        api_or_hf_model: str = "gpt-4o",
        *,
        meta_output_key: Optional[str] = MetaKeys.llm_extract,
        knowledge_grounding_key: Optional[str] = None,
        knowledge_grounding_fixed: Optional[str] = None,
        is_hf_model: bool = False,
        enable_vllm: bool = False,
        api_endpoint: Optional[str] = None,
        response_path: Optional[str] = None,
        system_prompt: Optional[str] = None,
        strategy: Optional[InferenceStrategy] = None,
        examples: Optional[str] = None,
        try_num: PositiveInt = 3,
        model_params: Optional[Dict] = None,
        sampling_params: Optional[Dict] = None,
        **kwargs,
    ):
        """Args:
        input_keys: Sample keys to build input text (e.g. ["text"] or ["query","response"]).
        output_schema: {output_key: "extraction instruction"}.
        api_or_hf_model: Model name for API or HuggingFace.
        meta_output_key: If set, write full result to meta[meta_output_key].
        knowledge_grounding_key: Optional sample key for per-sample grounding.
        knowledge_grounding_fixed: Optional fixed grounding string.
        strategy: Prompt strategy for extraction (direct/cot/few_shot/cot_shot).
        examples: Optional examples text used by few-shot strategies.
        try_num: Retries on parse/API failure.
        """
        super().__init__(**kwargs)
        self.input_keys = input_keys
        self.output_schema = output_schema
        self.meta_output_key = meta_output_key
        self.knowledge_grounding_key = knowledge_grounding_key
        self.knowledge_grounding_fixed = knowledge_grounding_fixed
        self.system_prompt = system_prompt
        self.strategy = strategy
        self.examples = examples
        self.try_num = try_num
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

    def _input_text(self, sample: Dict) -> str:
        parts = []
        for k in self.input_keys:
            if k in sample and sample[k] is not None:
                parts.append(f"{k}: {sample[k]}")
        return "\n".join(parts) if parts else ""

    def _knowledge_grounding(self, sample: Dict) -> Optional[str]:
        if self.knowledge_grounding_fixed:
            return self.knowledge_grounding_fixed
        key = self.knowledge_grounding_key
        if key and key in sample:
            v = sample[key]
            return str(v) if v is not None else None
        return None

    def process_single(self, sample: Dict, rank: Optional[int] = None) -> Dict:
        if Fields.meta not in sample:
            sample[Fields.meta] = {}
        input_text = self._input_text(sample)
        if not input_text:
            empty = {k: None for k in self.output_schema}
            if self.meta_output_key:
                sample[Fields.meta][self.meta_output_key] = empty
            else:
                for k, v in empty.items():
                    sample[Fields.meta][k] = v
            return sample

        kg = self._knowledge_grounding(sample)
        if self.enable_vllm or self.is_hf_model:
            model, _ = get_model(self.model_key, rank, self.use_cuda())
        else:
            model = get_model(self.model_key, rank, self.use_cuda())

        extracted = None
        usage = None
        for _ in range(self.try_num):
            try:
                extracted, usage = extract_one(
                    input_text,
                    self.output_schema,
                    model,
                    system_prompt=self.system_prompt,
                    knowledge_grounding=kg,
                    strategy=self.strategy,
                    examples=self.examples,
                    enable_vllm=self.enable_vllm,
                    is_hf_model=self.is_hf_model,
                    sampling_params=self.sampling_params,
                )
                if extracted is not None:
                    break
            except Exception as e:
                logger.warning("LLMExtractMapper attempt failed: %s", e)
        if extracted is None:
            extracted = {k: None for k in self.output_schema}
            usage = None

        if self.meta_output_key:
            sample[Fields.meta][self.meta_output_key] = extracted
        else:
            for k, v in extracted.items():
                sample[Fields.meta][k] = v
        if usage is not None:
            prev_usage = sample[Fields.meta].get(MetaKeys.llm_semantic_usage, {})
            curr_usage = usage.to_dict()
            sample[Fields.meta][MetaKeys.llm_semantic_usage] = {
                "prompt_tokens": prev_usage.get("prompt_tokens", 0) + curr_usage.get("prompt_tokens", 0),
                "completion_tokens": prev_usage.get("completion_tokens", 0) + curr_usage.get("completion_tokens", 0),
                "total_tokens": prev_usage.get("total_tokens", 0) + curr_usage.get("total_tokens", 0),
                "cost_estimate": prev_usage.get("cost_estimate", 0) + curr_usage.get("cost_estimate", 0),
            }
        return sample

"""LLM condition filter: keep samples satisfying a user-given condition.

Part of the llm_* ops family; yes/no by user-specified condition string
(unlike llm_analysis_filter which uses fixed dimensions).
"""

from typing import Optional

from loguru import logger
from pydantic import PositiveInt

from data_juicer.ops.base_op import OPERATORS, Filter
from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.llm_semantic_ops import InferenceStrategy, condition_filter_one
from data_juicer.utils.model_utils import (
    get_model,
    prepare_model,
    update_sampling_params,
)

vllm = LazyLoader("vllm")

OP_NAME = "llm_condition_filter"


@OPERATORS.register_module(OP_NAME)
class LLMConditionFilter(Filter):
    """Filter by user-given natural language condition (LLM yes/no).

    Uses text_key; writes to stats.llm_condition_filter_result; keeps if True.
    """

    _accelerator = "cuda"

    def __init__(
        self,
        text_key: str = "text",
        condition: str = "",
        api_or_hf_model: str = "gpt-4o",
        *,
        knowledge_grounding_key: Optional[str] = None,
        knowledge_grounding_fixed: Optional[str] = None,
        is_hf_model: bool = False,
        enable_vllm: bool = False,
        api_endpoint: Optional[str] = None,
        response_path: Optional[str] = None,
        strategy: Optional[InferenceStrategy] = None,
        examples: Optional[str] = None,
        try_num: PositiveInt = 3,
        model_params: Optional[dict] = None,
        sampling_params: Optional[dict] = None,
        **kwargs,
    ):
        """Args:
        text_key: Sample key for the text to evaluate.
        condition: Natural language condition (e.g. "contains X").
        api_or_hf_model: Model name.
        knowledge_grounding_key: Optional sample key for per-sample grounding.
        knowledge_grounding_fixed: Optional fixed grounding string.
        strategy: Prompt strategy for condition inference (direct/cot/few_shot/cot_shot).
        examples: Optional examples text used by few-shot strategies.
        try_num: Retries on API/parse failure; treat as False after all fail.
        """
        super().__init__(**kwargs)
        self.text_key = text_key
        self.condition = condition
        self.knowledge_grounding_key = knowledge_grounding_key
        self.knowledge_grounding_fixed = knowledge_grounding_fixed
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

    def _text(self, sample: dict) -> str:
        v = sample.get(self.text_key)
        if v is None:
            return ""
        return str(v).strip()

    def _knowledge_grounding(self, sample: dict) -> Optional[str]:
        if self.knowledge_grounding_fixed:
            return self.knowledge_grounding_fixed
        key = self.knowledge_grounding_key
        if key and key in sample:
            v = sample[key]
            return str(v) if v is not None else None
        return None

    def compute_stats_single(self, sample: dict, rank: Optional[int] = None, context: bool = False):
        if Fields.stats not in sample:
            sample[Fields.stats] = {}
        text = self._text(sample)
        if not text:
            sample[Fields.stats][StatsKeys.llm_condition_filter_result] = False
            return sample
        if not self.condition:
            sample[Fields.stats][StatsKeys.llm_condition_filter_result] = True
            return sample

        kg = self._knowledge_grounding(sample)
        if self.enable_vllm or self.is_hf_model:
            model, _ = get_model(self.model_key, rank, self.use_cuda())
        else:
            model = get_model(self.model_key, rank, self.use_cuda())

        result = False
        usage = None
        for _ in range(self.try_num):
            try:
                result, usage = condition_filter_one(
                    text,
                    self.condition,
                    model,
                    knowledge_grounding=kg,
                    strategy=self.strategy,
                    examples=self.examples,
                    enable_vllm=self.enable_vllm,
                    is_hf_model=self.is_hf_model,
                    sampling_params=self.sampling_params,
                )
                break
            except Exception as e:
                logger.warning("LLMConditionFilter attempt failed: %s", e)
        sample[Fields.stats][StatsKeys.llm_condition_filter_result] = result
        if usage is not None:
            prev_usage = sample[Fields.stats].get(StatsKeys.llm_semantic_usage, {})
            curr_usage = usage.to_dict()
            sample[Fields.stats][StatsKeys.llm_semantic_usage] = {
                "prompt_tokens": prev_usage.get("prompt_tokens", 0) + curr_usage.get("prompt_tokens", 0),
                "completion_tokens": prev_usage.get("completion_tokens", 0) + curr_usage.get("completion_tokens", 0),
                "total_tokens": prev_usage.get("total_tokens", 0) + curr_usage.get("total_tokens", 0),
                "cost_estimate": prev_usage.get("cost_estimate", 0) + curr_usage.get("cost_estimate", 0),
            }
        return sample

    def process_single(self, sample: dict, rank: Optional[int] = None) -> bool:
        stats = sample.get(Fields.stats, {})
        result = stats.get(StatsKeys.llm_condition_filter_result, False)
        return bool(result)

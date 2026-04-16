import json
import re
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger
from pydantic import PositiveInt

from data_juicer.ops.base_op import OPERATORS, Filter
from data_juicer.utils.agent_output_locale import (
    llm_filter_free_text_language_appendix,
    normalize_preferred_output_lang,
)
from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.model_utils import (
    get_model,
    prepare_model,
    update_sampling_params,
)
from data_juicer.utils.ray_utils import is_ray_mode

torch = LazyLoader("torch")
vllm = LazyLoader("vllm")

OP_NAME = "llm_analysis_filter"


@OPERATORS.register_module(OP_NAME)
class LLMAnalysisFilter(Filter):
    """Base filter class for leveraging LLMs to analyze and filter data samples.

    This operator uses an LLM to score and tag each sample across multiple quality
    dimensions. It supports both API-based and Hugging Face models. The LLM evaluates the
    sample on clarity, relevance, usefulness, and fluency, providing scores from 1 to 5.
    Tags are assigned to categorize the sample, and a recommendation is made to keep,
    review, or discard the sample. The average score is computed based on the required
    dimension keys. Samples are kept if their average score falls within the specified min
    and max score thresholds. The key metric 'llm_analysis_score' is cached in the sample's
    stats."""

    # avoid leading whitespace
    DEFAULT_SYSTEM_PROMPT = """You are a meticulous data quality assessor for LLM training. Analyze each data sample across multiple quality dimensions and provide numerical scores, tags, and reasoning. Follow these guidelines:

1. Evaluation Dimensions
Score each dimension (1-5 scale: 1=lowest, 5=highest):
- Clarity: How easy is the sample to understand?
- Relevance: How relevant is the sample to the intended task or topic?
- Usefulness: How helpful or valuable is the information in the sample?
- Fluency: How natural and well-written is the sample (grammar, style)?

2. Tagging:
Assign descriptive tags to categorize the data sample (string or list of string).  Examples include:
- "Topic": The main subject of the sample (e.g., "Machine Learning", "Historical Event").
- "Style":  The writing style or genre (e.g., "Informational", "Narrative", "Technical").
3. Scoring Protocol
- Base scores and tags on concrete evidence from the text.
- Flag samples needing human review (confidence <90%).
- Compare with similar data points for consistency.
- Penalize hallucination/misinformation severely (if applicable).

4. Output Format
json
{
  "dimension_scores": {
    "clarity": ,
    "relevance": ,
    "usefulness": ,
    "fluency":
  },
  "tags": {
    "topic": ,
    "style":
  },
  "flags": ["syntax_error", "insufficient_information", ...],
  "rationale": "Concise analysis of quality dimensions and tagging decisions.",
  "recommendation": ["keep", "review", "discard"]
}

5. Special Instructions
- Prioritize accuracy and relevance over stylistic qualities.
- Contextualize cultural references appropriately.
- Clearly justify your scores, tags, and flags in the rationale.
- Response a json dict

Example Response:

json
{
  "dimension_scores": {
    "clarity": 4,
    "relevance": 5,
    "usefulness": 3,
    "fluency": 4
  },
  "tags": {
    "topic": "Artificial Intelligence",
    "style": "Informational"
  },
  "flags": ["minor_grammar_issues"],
  "rationale": "The text is highly relevant and generally well-written, but suffers from some minor grammar issues and could be more useful with additional examples.  The topic is clearly Artificial Intelligence, and the difficulty is appropriate for an intermediate audience.",
  "recommendation": ["review"]
}
"""  # noqa: E501
    DEFAULT_INPUT_TEMPLATE = "# Data\n'''\n{data}\n'''\n\n# Response\njson\n"
    DEFAULT_FIELD_TEMPLATE = "**{field_name}**\n{field_data}"
    DEFAULT_DIM_REQUIRED_KEYS = ["clarity", "relevance", "usefulness", "fluency"]

    _accelerator = "cuda"

    def __init__(
        self,
        api_or_hf_model: str = "gpt-4o",
        min_score: float = 0.5,
        max_score: float = 1.0,
        is_hf_model: bool = False,
        *,
        api_endpoint: Optional[str] = None,
        response_path: Optional[str] = None,
        input_keys: List[str] = ["text"],
        field_names: List[str] = ["Text"],
        system_prompt: Optional[str] = None,
        input_template: Optional[str] = None,
        field_template: Optional[str] = None,
        try_num: PositiveInt = 3,
        enable_vllm: bool = False,
        model_params: Dict = {},
        sampling_params: Dict = {},
        dim_required_keys: Optional[List[str]] = None,
        preferred_output_lang: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialization method.

        :param api_or_hf_model: API or huggingface model name.
        :param min_score: The min score threshold to keep the sample.
        :param max_score: The max score threshold to keep the sample.
        :param is_hf_model:  If true, use Transformers for loading hugging face or
            local llm.
        :param api_endpoint: URL endpoint for the API.
        :param response_path: Path to extract content from the API response.
            Defaults to 'choices.0.message.content'.
        :param input_keys: Sub set of keys in the sample. Support data with
            multi fields such as 'query', 'analysis' and 'answer' in RFT data.
        :param field_names: Corresponding field names for input keys.
        :param system_prompt: System prompt for the task.
        :param input_template: Template for building the model input.
        :param field_template: Template for each field in the prompt.
        :param try_num: The number of retry attempts when there is an API
            call error or output parsing error.
        :param enable_vllm: If true, use VLLM for loading hugging face or
            local llm.
        :param model_params: Parameters for initializing the API model.
        :param sampling_params: Extra parameters passed to the API call.
            e.g {'temperature': 0.9, 'top_p': 0.95}
        :param dim_required_keys: A list of keys used to calculate the average
            dimension score, only the dimension scores associated with these
            keys are used in the average calculation.
        :param kwargs: Extra keyword arguments.
        """
        super().__init__(**kwargs)

        base_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        if preferred_output_lang is not None and str(preferred_output_lang).strip() != "":
            self.system_prompt = base_prompt + llm_filter_free_text_language_appendix(
                normalize_preferred_output_lang(preferred_output_lang)
            )
        else:
            self.system_prompt = base_prompt
        assert len(input_keys) == len(field_names), "The input_keys and field_names must correspond one-to-one!"
        self.input_keys = input_keys
        self.field_names = field_names
        self.input_template = input_template or self.DEFAULT_INPUT_TEMPLATE
        self.field_template = field_template or self.DEFAULT_FIELD_TEMPLATE
        self.dim_required_keys = dim_required_keys or self.DEFAULT_DIM_REQUIRED_KEYS

        self.min_score = min_score
        self.max_score = max_score
        self.try_num = try_num

        self.enable_vllm = enable_vllm
        self.is_hf_model = is_hf_model

        sampling_params = update_sampling_params(sampling_params, api_or_hf_model, self.enable_vllm)

        if enable_vllm:
            if not is_ray_mode():
                # cannot initialize vllm replicas on different GPUs
                self.num_proc = 1
            self.model_key = prepare_model(
                model_type="vllm", pretrained_model_name_or_path=api_or_hf_model, **model_params
            )
            self.sampling_params = vllm.SamplingParams(**sampling_params)
        elif is_hf_model:
            self.model_key = prepare_model(
                model_type="huggingface",
                pretrained_model_name_or_path=api_or_hf_model,
                return_pipe=True,
                trust_remote_code=True,
                **model_params,
            )
            self.sampling_params = sampling_params
        else:
            self.sampling_params = sampling_params

            self.model_key = prepare_model(
                model_type="api",
                model=api_or_hf_model,
                endpoint=api_endpoint,
                response_path=response_path,
                **model_params,
            )

            # api model. Change the accelerator to cpu
            self.accelerator = "cpu"

    def build_input(self, sample):
        if not set(self.input_keys) <= set(sample.keys()):
            logger.warning(f"Not all input keys {self.input_keys} are in the sample!")
        field_strs = [
            self.field_template.format(field_name=n, field_data=sample[k])
            for (k, n) in zip(self.input_keys, self.field_names)
            if k in sample
        ]
        data_str = "\n\n".join(field_strs)
        input_prompt = self.input_template.format(data=data_str)

        return input_prompt

    @staticmethod
    def _normalize_recommendation_to_str_list(val: Any) -> List[str]:
        """Force stable Arrow/HF schema: ``recommendation`` is always ``list[str]``."""
        if val is None:
            return []
        if isinstance(val, str):
            s = val.strip()
            return [s] if s else []
        if isinstance(val, np.ndarray):
            val = val.tolist()
        if isinstance(val, (list, tuple)):
            out: List[str] = []
            for x in val:
                if x is None:
                    continue
                sx = str(x).strip()
                if sx:
                    out.append(sx)
            return out
        sx = str(val).strip()
        return [sx] if sx else []

    @staticmethod
    def _normalize_tags_to_str(tags) -> str:
        """Serialize tags to a single stable string for Arrow schema.

        Tags are stored under one fixed key (llm_analysis_tags / llm_*_tags) as
        a JSON string so that every sample always writes the same key with the
        same Arrow type (string), regardless of what keys the LLM returned.
        Storing tags as dynamic top-level keys in stats causes Arrow
        string-vs-null schema conflicts when different samples have different tag keys.
        """
        if tags is None:
            return ""
        if isinstance(tags, str):
            return tags
        try:
            return json.dumps(tags, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(tags)

    @staticmethod
    def _normalize_tags_to_dict(tags) -> dict:
        """Normalize tags to dict for stable Arrow schema."""
        if tags is None:
            return {}
        if isinstance(tags, dict):
            # Ensure all values are strings for consistent schema
            return {str(k): str(v) if not isinstance(v, (list, dict)) else str(v) for k, v in tags.items()}
        if isinstance(tags, str):
            return {"tags": tags}
        if isinstance(tags, (list, tuple)):
            return {"tags": ", ".join(str(x) for x in tags)}
        return {"tags": str(tags)}

    @staticmethod
    def _normalize_record(record) -> str:
        """Serialize record to a stable JSON string for Arrow schema.

        Record is stored as a JSON string (not a dict) so that every sample
        always writes the same Arrow column type (string). Storing record as a
        nested dict causes Arrow schema conflicts because:
        - Empty lists [] are inferred as list<null>, non-empty as list<string>
        - dimension_scores values may be int vs float across samples
        - Some samples may have extra keys others don't
        All of these cause 'Couldn't cast array of type string to null' errors
        when shards are merged.
        """
        if record is None:
            return ""
        try:
            return json.dumps(record, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(record)

    def parse_output(self, raw_output):
        def extract_outer_braces(s):
            if s is None or (isinstance(s, str) and not s.strip()):
                return None
            pattern = r"(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})"
            match = re.search(pattern, s)
            if match:
                return match.group(1)
            return None

        if raw_output is None or (isinstance(raw_output, str) and not (raw_output or "").strip()):
            return 0.0, None, None
        json_str = extract_outer_braces(raw_output)
        if json_str is None:
            return 0.0, None, None
        try:
            data = json.loads(json_str)
        except (json.JSONDecodeError, TypeError):
            return 0.0, None, None
        # Model outputs sometimes use "review" vs ["review"], breaking dataset shard alignment
        data["recommendation"] = self._normalize_recommendation_to_str_list(data.get("recommendation"))
        tags = data.get("tags", None)

        dimension_scores = data.get("dimension_scores", None)

        total_score = 0
        if dimension_scores and self.dim_required_keys:
            for key in self.dim_required_keys:
                total_score += dimension_scores[key]
            # div 5 for normalization
            avg_score = float(total_score) / len(self.dim_required_keys) / 5
        else:
            avg_score = 0.0  # Use 0.0 instead of None for stable Arrow float schema
            logger.warning(
                "Either dimension_scores is empty or dim_required_keys "
                "is empty. Dimension score has been set to 0.0, "
                "Ensure this setting is intentional to disable the "
                "dimension score. "
            )

        return avg_score, data, tags

    def generate_llm_analysis(self, sample, rank):
        if self.enable_vllm or self.is_hf_model:
            model, _ = get_model(self.model_key, rank, self.use_cuda())
        else:
            model = get_model(self.model_key, rank, self.use_cuda())

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.build_input(sample)},
        ]
        score, record, tags = 0.0, None, None
        for _ in range(self.try_num):
            try:
                if self.enable_vllm:
                    response = model.chat(messages, self.sampling_params)
                    output = response[0].outputs[0].text
                elif self.is_hf_model:
                    response = model(messages, return_full_text=False, **self.sampling_params)
                    output = response[0]["generated_text"]
                else:
                    output = model(messages, **self.sampling_params)
                if output is None or (isinstance(output, str) and not (output or "").strip()):
                    continue
                score, record, tags = self.parse_output(output)
                if record is not None:
                    break
            except Exception as e:
                logger.warning(f"Exception: {e}")

        return score, record, tags

    def compute_stats_single(self, sample, rank=None, context=False):
        # check if it's computed already
        if StatsKeys.llm_analysis_score in sample[Fields.stats]:
            return sample

        score, record, tags = self.generate_llm_analysis(sample, rank)

        # score is always float (0.0 on failure); store unconditionally for stable Arrow schema
        sample[Fields.stats][StatsKeys.llm_analysis_score] = score
        # Normalize record to ensure stable Arrow schema (handles nested None values)
        sample[Fields.stats][StatsKeys.llm_analysis_record] = self._normalize_record(record)

        # Store all tags under a single fixed key to avoid dynamic key schema conflicts.
        # Dynamic keys (e.g. "topic", "style") vary across samples, causing Arrow
        # string-vs-null schema merge errors when some samples have the key and others don't.
        sample[Fields.stats][StatsKeys.llm_analysis_tags] = self._normalize_tags_to_str(tags)

        return sample

    def process_single(self, sample, rank=None):
        itm_score = sample[Fields.stats].get(StatsKeys.llm_analysis_score, 0.0)
        if itm_score:
            return self.get_keep_boolean(itm_score, self.min_score, self.max_score)
        else:
            # score is 0.0, meaning LLM analysis failed; disable the filter
            return True

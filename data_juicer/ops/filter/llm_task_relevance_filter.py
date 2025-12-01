from typing import Dict, List, Optional

from datasets import Dataset
from loguru import logger

from data_juicer.ops.base_op import ATTRIBUTION_FILTERS, OPERATORS
from data_juicer.ops.filter import LLMAnalysisFilter
from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.lazy_loader import LazyLoader

torch = LazyLoader("torch")
vllm = LazyLoader("vllm")

OP_NAME = "llm_task_relevance_filter"


@OPERATORS.register_module(OP_NAME)
@ATTRIBUTION_FILTERS.register_module(OP_NAME)
class LLMTaskRelevanceFilter(LLMAnalysisFilter):
    """Filter to keep samples with high relevance scores to validation tasks estimated by an
    LLM.

    This operator evaluates the relevance of each sample to a specified validation task
    using an LLM. The LLM scores the sample on multiple dimensions, including topical
    relevance, linguistic style match, task match, knowledge alignment, and potential
    utility. Each dimension is scored on a 1-5 scale, with 5 being the highest. The key
    metric, 'llm_task_relevance', is the average score across these dimensions. Samples are
    kept if their average score meets or exceeds the specified minimum threshold. The
    operator uses either an API or a Hugging Face model for evaluation. If no validation
    dataset or task description is provided, the 'prepare_valid_feature' method must be
    called manually before applying the filter."""

    # TODO: fix dataset cast error

    # avoid leading whitespace
    DEFAULT_SYSTEM_PROMPT = """
You are a meticulous data quality assessor for LLM training. Evaluate whether each data sample is beneficial for improving model performance on a downstream task.
The downstream task will be characterized by a task description or/and some validation data in the user query.

1. Evaluation Dimensions
Score each dimension (1-5 scale: 1=lowest, 5=highest):
- Topical Relevance: Does the content or theme of the sample relate to those seen in the validation set?
- Linguistic Style Match: Does the style, tone, and complexity of the sample resemble those in the validation set?
- Task Match: If the validation examples are from a task (e.g., summarization, classification, etc.), is the sample solving a similar task?
- Knowledge Alignment: Is the type of knowledge or reasoning required in the sample aligned with that in the validation set?
- Potential Utility: If this sample were added to the training data, is it likely to improve generalization to the validation set?

2. Output Format
json
{
  "dimension_scores": {
    "topical_relevance": ,
    "linguistic_style_match": ,
    "task_match": ,
    "knowledge_alignment": ,
    "potential_utility": ,
  },
  "flags": ["topical_mismatch", "task_irrelevant", ...],
  "rationale": "Technical analysis of the relevance",
}
3. Special Instructions
- Focus on **alignment with the validation examples**, not general quality.
- If the sample is entirely unrelated to the validation set (e.g., different topic, domain, or task), assign a score of 1 and explain briefly.
- If the validation examples are ambiguous, make a **conservative judgment** based on their shared patterns.
- Be consistent in your rating scale across evaluations.
- Do **not** make up or reinterpret the sample content; base all reasoning on the actual text.
- Avoid overrating stylistically impressive but **task-irrelevant** samples.

Example Response:

json
{
  "dimension_scores": {"topical_relevance": 2, "linguistic_style_match": 4, "task_match": 2, "knowledge_alignment": 2, "potential_utility": 2},
  "flags": ["topical_mismatch"],
  "rationale": "The text provides rich information about American history, while the validation tasks is on multistep reasoning to solve challenging math problems."
}
"""  # noqa: E501

    DEFAULT_DIM_REQUIRED_KEYS = [
        "topical_relevance",
        "linguistic_style_match",
        "task_match",
        "knowledge_alignment",
        "potential_utility",
    ]

    def __init__(
        self,
        api_or_hf_model: str = "gpt-4o",
        min_score: float = 0.5,
        is_hf_model: bool = False,
        *,
        valid_dataset: Optional[List[Dict]] = None,
        task_desc: Optional[str] = None,
        n_shot: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialization method.

        :param api_or_hf_model: API or huggingface model name.
        :param min_score: The lowest score threshold to keep the sample.
        :param is_hf_model: Indicates if the model is from HuggingFace.
        :param valid_dataset: The dataset to use for validation.
        :param task_desc: The description of the validation task.
            If valid_dataset=None and task_desc=None,
            'self.prepare_valid_feature' should be manually called before applying the filter.
        :param n_shot: The number of shots in validation.
        """
        super().__init__(api_or_hf_model, min_score, is_hf_model, **kwargs)
        self.valid_feature = {}
        if valid_dataset is not None or task_desc is not None:
            self.prepare_valid_feature(Dataset.from_list(valid_dataset), task_desc, n_shot)
        else:
            logger.warning(
                f"valid_dataset and task_desc are both None when initializing {OP_NAME}. \
                'prepare_valid_feature' method should be manually called before applying the filter."
            )

    @property
    def valid_feature_ready(self):
        return "valid_info" in self.valid_feature

    def prepare_valid_feature(self, dataset=None, task_desc=None, n_shot=None, *args, **kwargs):
        if dataset is None:
            valid_data_block = ""
        else:
            n_shot = n_shot or len(dataset)
            sample = dataset[0]
            if not set(self.input_keys) <= set(sample.keys()):
                logger.warning(f"Not all input keys {self.input_keys} are in the sample!")
            data_strs = []
            for i, sample in enumerate(dataset):
                if i + 1 > n_shot:
                    break
                field_strs = [
                    self.field_template.format(field_name=n, field_data=sample[k])
                    for (k, n) in zip(self.input_keys, self.field_names)
                    if k in sample
                ]
                data_str = "\n\n".join(field_strs)
                data_strs.append("'''\n{data}\n'''".format(data=data_str))
            valid_data_block = "# Validation Data\n" + ("\n\n".join(data_strs)) + "\n\n"

        if task_desc is None:
            task_desc_block = ""
        else:
            task_desc_block = f"# Task Description\n{task_desc}\n\n"

        valid_txt = task_desc_block + valid_data_block
        if len(valid_txt) == 0:
            logger.warning("Empty validation information, please provide validation dataset or task description.")
        else:
            self.valid_feature["valid_info"] = valid_txt

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

        return self.valid_feature.get("valid_info", "") + input_prompt

    def compute_stats_single(self, sample, rank=None, context=False):
        # check if it's computed already
        if sample[Fields.stats].get(StatsKeys.llm_task_relevance, -1) >= 0:
            return sample

        assert self.valid_feature_ready, "Validation feature not ready yet. Call prepare_valid_feature first."

        score, record, tags = self.generate_llm_analysis(sample, rank)

        sample[Fields.stats][StatsKeys.llm_task_relevance] = score
        sample[Fields.stats][StatsKeys.llm_task_relevance_record] = record

        if tags and isinstance(tags, dict):
            for key, value in tags.items():
                sample[Fields.stats][key] = value

        return sample

    def process_single(self, sample, rank=None):
        itm_score = sample[Fields.stats][StatsKeys.llm_task_relevance]
        if itm_score is None:
            return True

        return self.get_keep_boolean(itm_score, self.min_score, None)

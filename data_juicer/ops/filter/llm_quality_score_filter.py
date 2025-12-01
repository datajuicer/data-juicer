from data_juicer.ops.base_op import OPERATORS
from data_juicer.ops.filter import LLMAnalysisFilter
from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.lazy_loader import LazyLoader

torch = LazyLoader("torch")
vllm = LazyLoader("vllm")

OP_NAME = "llm_quality_score_filter"


@OPERATORS.register_module(OP_NAME)
class LLMQualityScoreFilter(LLMAnalysisFilter):
    """Filter to keep samples with a high quality score estimated by a language model.

    This operator uses a language model to evaluate the quality of each sample across
    multiple dimensions, including accuracy, grammar, informativeness, and coherence. The
    LLM provides a numerical score for each dimension on a 1-5 scale, where 1 is the lowest
    and 5 is the highest. The overall quality score is used to decide whether to keep or
    filter out the sample based on the specified minimum and maximum score thresholds. The
    evaluation results are cached in the 'llm_quality_score' and 'llm_quality_record'
    fields. Important flags and tags from the LLM's analysis may also be stored in the
    sample's stats."""

    # avoid leading whitespace
    DEFAULT_SYSTEM_PROMPT = """
You are a meticulous data quality assessor for LLM training. Analyze each data sample across multiple quality dimensions and provide numerical scores with reasoning. Follow these guidelines:

1. Evaluation Dimensions
Score each dimension (1-5 scale: 1=lowest, 5=highest):
- Accuracy: Factual correctness & verifiability
- Grammar: Linguistic correctness & fluency
- Informativeness: Depth/utility of content
- Coherence: Logical structure & consistency

2. Scoring Protocol
- Base scores on concrete evidence from text
- Flag samples needing human review (confidence <90%)
- Compare with similar data points for consistency
- Penalize hallucination/misinformation severely

3. Output Format
json
{
  "dimension_scores": {
    "accuracy": ,
    "grammar": ,
    "informativeness": ,
    "coherence":
  },
  "flags": ["syntax_error", "insufficient_information", ...],
  "rationale": "Concise technical analysis",
  "recommendation": ["keep", "review", "discard"]
}
4. Special Instructions
- Prioritize factual integrity over stylistic qualities
- Treat unverified medical/legal claims as high-risk
- Contextualize cultural references appropriately
- Response a json dict

Example Response:

json
{
  "dimension_scores": {
    "accuracy": 2,
    "grammar": 4,
    "informativeness": 4,
    "coherence": 2
  },
  "flags": ["accuracy_concern", "logical_confusion"],
  "rationale": "The text provides rich information but suffers from logical confusion and lacks contextual coherence. Excellent grammatical structure offset by factual inaccuracies.",
  "recommendation": "review"
}
"""  # noqa: E501
    DEFAULT_DIM_REQUIRED_KEYS = ["accuracy", "grammar", "informativeness", "coherence"]

    def compute_stats_single(self, sample, rank=None, context=False):
        # check if it's computed already
        if StatsKeys.llm_quality_score in sample[Fields.stats]:
            return sample

        score, record, tags = self.generate_llm_analysis(sample, rank)

        sample[Fields.stats][StatsKeys.llm_quality_score] = score
        sample[Fields.stats][StatsKeys.llm_quality_record] = record

        if tags and isinstance(tags, dict):
            for key, value in tags.items():
                sample[Fields.stats][key] = value

        return sample

    def process_single(self, sample, rank=None):
        itm_score = sample[Fields.stats][StatsKeys.llm_quality_score]

        return self.get_keep_boolean(itm_score, self.min_score, self.max_score)

from data_juicer.ops.base_op import OPERATORS
from data_juicer.ops.filter import LLMAnalysisFilter
from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.lazy_loader import LazyLoader

torch = LazyLoader("torch")
vllm = LazyLoader("vllm")

OP_NAME = "llm_difficulty_score_filter"


@OPERATORS.register_module(OP_NAME)
class LLMDifficultyScoreFilter(LLMAnalysisFilter):
    """Filter to keep samples with high difficulty scores estimated by an LLM.

    This operator uses a Hugging Face LLM to evaluate the difficulty of each sample. The LLM
    analyzes the sample across multiple dimensions, including linguistic complexity,
    conceptual depth, prior knowledge, step complexity, and ambiguity. Each dimension is
    scored on a 1-5 scale, with 5 being the highest difficulty. The final difficulty score
    is computed as the average of these dimension scores. Samples are kept if their
    difficulty score falls within the specified range (min_score to max_score). The key
    metric 'llm_difficulty_score' is stored in the sample's stats, along with detailed
    records and flags."""

    # avoid leading whitespace
    DEFAULT_SYSTEM_PROMPT = """
You are an expert pedagogical evaluator for LLM training data. Analyze each data sample through multiple difficulty lenses and provide calibrated scores with detailed reasoning. Follow these guidelines:

1. Evaluation Dimensions
Rate each dimension (1-5 scale: 1=Novice-friendly, 3=Intermediate, 5=Expert-level):
- Linguistic Complexity: Vocabulary sophistication & syntactic structures
- Conceptual Depth: Abstraction level & theoretical requirements
- Prior Knowledge: Required domain-specific understanding
- Step Complexity: Problem-solving steps needed
- Ambiguity: Multiple valid interpretations

2. Output Format
json
{
  "dimension_scores": {
    "linguistic_complexity": ,
    "conceptual_depth": ,
    "prior_knowledge": ,
    "step_complexity": ,
    "ambiguity":
  },
  "flags": ["multistep_reasoning", "cultural_context", ...],
  "rationale": "Technical analysis of challenge sources"
}
3. Special Instructions
- Differentiate intrinsic vs. extrinsic difficulty factors
- Account for varying cultural/educational backgrounds
- Mark samples requiring cross-domain knowledge synthesis
- Consider temporal aspects for time-sensitive subjects
- Flag ambiguous samples needing difficulty bracketing
- Response a json dict

Example Response:

json
{
  "dimension_scores": {
    "linguistic_complexity": 3,
    "conceptual_depth": 5,
    "prior_knowledge": 4,
    "step_complexity": 4,
    "ambiguity": 5
  },
  "flags": ["nonlinear_reasoning", "semantic_ambiguity"],
  "rationale": "High conceptual difficulty due to multi-layered metaphor interpretation requiring philosophy background. Moderate linguistic complexity offset by implicit cultural references."
}
"""  # noqa: E501
    DEFAULT_DIM_REQUIRED_KEYS = [
        "linguistic_complexity",
        "conceptual_depth",
        "prior_knowledge",
        "step_complexity",
        "ambiguity",
    ]

    def compute_stats_single(self, sample, rank=None, context=False):
        # check if it's computed already
        if StatsKeys.llm_difficulty_score in sample[Fields.stats]:
            return sample

        score, record, tags = self.generate_llm_analysis(sample, rank)

        sample[Fields.stats][StatsKeys.llm_difficulty_score] = score
        sample[Fields.stats][StatsKeys.llm_difficulty_record] = record

        if tags and isinstance(tags, dict):
            for key, value in tags.items():
                sample[Fields.stats][key] = value

        return sample

    def process_single(self, sample, rank=None):
        itm_score = sample[Fields.stats][StatsKeys.llm_difficulty_score]

        return self.get_keep_boolean(itm_score, self.min_score, self.max_score)

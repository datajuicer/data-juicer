import logging

from data_juicer.ops.base_op import OPERATORS
from data_juicer.ops.filter.llm_perplexity_filter import LLMPerplexityFilter
from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.lazy_loader import LazyLoader

torch = LazyLoader("torch")
transformers = LazyLoader("transformers")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

OP_NAME = "instruction_following_difficulty_filter"


@OPERATORS.register_module(OP_NAME)
class InstructionFollowingDifficultyFilter(LLMPerplexityFilter):
    """Filter to keep texts based on their instruction following difficulty (IFD,
        https://arxiv.org/abs/2308.12032) score.

    This operator computes the IFD score for each sample, which is the ratio of the loss
    with and without the query. It keeps samples where the IFD score falls within a
    specified range. The IFD score is calculated using a Hugging Face tokenizer and model.
    If the IFD score is already cached in the 'ifd_score' field, it will be reused. The
    operator decides to keep or filter samples based on the provided minimum and maximum IFD
    score thresholds."""

    _accelerator = "cuda"

    def compute_stats_single(self, sample, rank=None):

        # check if it's computed already
        if StatsKeys.ifd_score in sample[Fields.stats]:
            return sample

        sample_w_msgs = self.sample_with_messages(sample)
        msgs_wo_query = sample_w_msgs["messages"][-1:]
        sample_w_msg_wo_query = dict(**sample_w_msgs)
        sample_w_msg_wo_query.update({"messages": msgs_wo_query})

        loss_w_query = self._loss(sample_w_msgs, rank)
        loss_wo_query = self._loss(sample_w_msg_wo_query, rank)
        sample[Fields.stats][StatsKeys.ifd_score] = loss_w_query / loss_wo_query

        return sample

    def process_single(self, sample):
        score = sample[Fields.stats][StatsKeys.ifd_score]

        return self.get_keep_boolean(score, self.min_score, self.max_score)

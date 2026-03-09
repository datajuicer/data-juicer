import sys

from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.model_utils import get_model, prepare_model

from ..base_op import OPERATORS, Filter

OP_NAME = "token_num_filter"


@OPERATORS.register_module(OP_NAME)
class TokenNumFilter(Filter):
    """Filter to keep samples with a total token number within a specified range.

    This operator uses a Hugging Face tokenizer to count the number of tokens in each
    sample. It keeps samples where the token count is between the minimum and maximum
    thresholds. The token count is stored in the 'num_token' field of the sample's stats. If
    the token count is not already computed, it will be calculated using the specified
    tokenizer."""

    _batched_op = True

    def __init__(
        self,
        hf_tokenizer: str = "EleutherAI/pythia-6.9b-deduped",
        min_num: int = 10,
        max_num: int = sys.maxsize,
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param hf_tokenizer: the tokenizer name of Hugging Face tokenizers.
        :param min_num: The min filter token number in this op, samples
            will be filtered if their token number is below this
            parameter.
        :param max_num: The max filter token number in this op, samples
            will be filtered if their token number exceeds this
            parameter.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.min_num = min_num
        self.max_num = max_num
        self.hf_tokenizer = hf_tokenizer
        self.model_key = prepare_model(
            model_type="huggingface", pretrained_model_name_or_path=hf_tokenizer, return_model=False
        )

    def compute_stats_batched(self, samples, *args, **kwargs):
        samples_list = samples[self.text_key]
        samples_stats = samples[Fields.stats]

        # Collect indices and texts that need tokenization
        indices = []
        texts = []
        for idx, stat in enumerate(samples_stats):
            if StatsKeys.num_token not in stat:
                indices.append(idx)
                texts.append(samples_list[idx])

        if texts:
            tokenizer = get_model(self.model_key)
            encoded = tokenizer(texts, add_special_tokens=False)
            for i, idx in enumerate(indices):
                samples_stats[idx][StatsKeys.num_token] = len(encoded["input_ids"][i])

        return samples

    def process_batched(self, samples):
        return [
            self.get_keep_boolean(stat[StatsKeys.num_token], self.min_num, self.max_num)
            for stat in samples[Fields.stats]
        ]

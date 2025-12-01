import sys

from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.model_utils import get_model, prepare_model

from ..base_op import OPERATORS, Filter
from ..common import get_words_from_document

OP_NAME = "alphanumeric_filter"


@OPERATORS.register_module("alphanumeric_filter")
class AlphanumericFilter(Filter):
    """Filter to keep samples with an alphabet/numeric ratio within a specific range.

    This operator filters samples based on the ratio of alphanumeric characters or tokens.
    It keeps samples where the ratio of alphanumeric characters (or tokens) to the total
    number of characters (or tokens) is within the specified range. The ratio is computed
    either character-based or token-based, depending on the `tokenization` parameter. If
    `tokenization` is True, it uses a Hugging Face tokenizer to count tokens. The key metric
    used for filtering is 'alpha_token_ratio' if tokenization is enabled, otherwise
    'alnum_ratio'. The operator caches these metrics in the stats field for each sample."""

    _batched_op = True

    def __init__(
        self, tokenization: bool = False, min_ratio: float = 0.25, max_ratio: float = sys.maxsize, *args, **kwargs
    ):
        """
        Initialization method.

        :param tokenization: Whether to count the ratio of alphanumeric
            to the total number of tokens. if tokenization=False, it
            will count the ratio of alphanumeric to the total number of
            characters.
        :param min_ratio: The min filter ratio in alphanumeric op,
            samples will be filtered if their alphabet/numeric ratio is
            below this parameter.
        :param max_ratio: The max filter ratio in alphanumeric op,
            samples will be filtered if their alphabet/numeric ratio
            exceeds this parameter.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.tokenization = tokenization
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.model_key = None

        if tokenization:
            self.model_key = prepare_model(
                model_type="huggingface",
                pretrained_model_name_or_path="EleutherAI/pythia-6.9b-deduped",
                return_model=False,
            )

    def compute_stats_batched(self, samples):
        samples_list = samples[self.text_key]
        samples_stats = samples[Fields.stats]

        for idx, stat in enumerate(samples_stats):
            cur_text = samples_list[idx]
            if self.tokenization:
                if StatsKeys.alpha_token_ratio in stat:
                    continue
                alpha_count = sum(map(lambda char: 1 if char.isalpha() else 0, cur_text))
                tokenizer = get_model(self.model_key)
                token_count = len(
                    get_words_from_document(cur_text, token_func=tokenizer.tokenize if tokenizer else None)
                )
                samples_stats[idx][StatsKeys.alpha_token_ratio] = (
                    (alpha_count / token_count) if token_count != 0 else 0.0
                )
            else:
                if StatsKeys.alnum_ratio in stat:
                    continue
                alnum_count = sum(map(lambda char: 1 if char.isalnum() else 0, cur_text))
                samples_stats[idx][StatsKeys.alnum_ratio] = (alnum_count / len(cur_text)) if len(cur_text) != 0 else 0.0

        return samples

    def process_batched(self, samples):
        ratio_key = StatsKeys.alpha_token_ratio if self.tokenization else StatsKeys.alnum_ratio
        assert isinstance(samples[Fields.stats], list)
        return map(
            lambda stat: self.get_keep_boolean(stat[ratio_key], self.min_ratio, self.max_ratio), samples[Fields.stats]
        )

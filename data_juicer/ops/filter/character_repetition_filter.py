# Some code here has been modified from:
# https://huggingface.co/spaces/huggingface/text-data-filtering
# --------------------------------------------------------

import numpy as np
from pydantic import PositiveInt

from data_juicer.utils.constant import Fields, StatsKeys

from ..base_op import OPERATORS, Filter


@OPERATORS.register_module("character_repetition_filter")
class CharacterRepetitionFilter(Filter):
    """Filter to keep samples with character-level n-gram repetition ratio within a specific
    range.

    This operator calculates the character-level n-gram repetition ratio for each sample and
    filters out samples that do not fall within the specified range. The repetition ratio is
    computed based on the frequency of n-grams in the text. The key metric 'char_rep_ratio'
    is cached in the stats field. Samples are kept if their 'char_rep_ratio' is between the
    specified min and max ratios. The n-gram length, minimum, and maximum ratios are
    configurable."""

    _batched_op = True

    def __init__(self, rep_len: PositiveInt = 10, min_ratio: float = 0.0, max_ratio: float = 0.5, *args, **kwargs):
        """
        Initialization method.

        :param rep_len: Repetition length for char-level n-gram.
        :param min_ratio: The min filter ratio in this op, samples will
            be filtered if their char-level n-gram repetition ratio is
            below this parameter.
        :param max_ratio: The max filter ratio in this op, samples will
            be filtered if their char-level n-gram repetition ratio
            exceeds this parameter.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.n = rep_len
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def _compute_char_rep_ratio(self, text):
        """Compute character repetition ratio for a given text."""
        char_ngrams = [text[i : i + self.n] for i in range(len(text) - self.n + 1)]
        freq_char_ngrams = {}
        for char_ngram in char_ngrams:
            freq_char_ngrams[char_ngram] = freq_char_ngrams.get(char_ngram, 0) + 1

        if len(freq_char_ngrams) == 0:
            return 0.0

        freq_char_ngrams = sorted(list(freq_char_ngrams.values()), reverse=True)
        num_no_rep_char_ngrams = len([el for el in freq_char_ngrams if el == 1])
        num_rep_char_ngrams = min(
            int(np.sqrt(len(freq_char_ngrams))),
            len(freq_char_ngrams) - num_no_rep_char_ngrams,
        )
        return (
            (sum(freq_char_ngrams[:num_rep_char_ngrams]) / sum(freq_char_ngrams)) if sum(freq_char_ngrams) != 0 else 0.0
        )

    def compute_stats_batched(self, samples, *args, **kwargs):
        samples_list = samples[self.text_key]
        samples_stats = samples[Fields.stats]

        for idx, stat in enumerate(samples_stats):
            # check if it's computed already
            if StatsKeys.char_rep_ratio in stat:
                continue

            cur_text = samples_list[idx]
            char_ngrams = [cur_text[i : i + self.n] for i in range(len(cur_text) - self.n + 1)]
            freq_char_ngrams = {}
            for char_ngram in char_ngrams:
                freq_char_ngrams[char_ngram] = freq_char_ngrams.get(char_ngram, 0) + 1

            if len(freq_char_ngrams) == 0:
                samples_stats[idx][StatsKeys.char_rep_ratio] = 0.0
                continue

            freq_char_ngrams = sorted(list(freq_char_ngrams.values()), reverse=True)
            num_no_rep_char_ngrams = len([el for el in freq_char_ngrams if el == 1])
            num_rep_char_ngrams = min(
                int(np.sqrt(len(freq_char_ngrams))),
                len(freq_char_ngrams) - num_no_rep_char_ngrams,
            )
            samples_stats[idx][StatsKeys.char_rep_ratio] = (
                (sum(freq_char_ngrams[:num_rep_char_ngrams]) / sum(freq_char_ngrams))
                if sum(freq_char_ngrams) != 0
                else 0.0
            )

        return samples

    def process_batched(self, samples):
        assert isinstance(samples[Fields.stats], list)
        return map(
            lambda stat: self.get_keep_boolean(stat[StatsKeys.char_rep_ratio], self.min_ratio, self.max_ratio),
            samples[Fields.stats],
        )

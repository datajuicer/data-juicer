# Some code here has been modified from:
# https://huggingface.co/spaces/huggingface/text-data-filtering
# --------------------------------------------------------

from data_juicer.utils.constant import Fields, StatsKeys

from ..base_op import OPERATORS, Filter
from ..common import SPECIAL_CHARACTERS


@OPERATORS.register_module("special_characters_filter")
class SpecialCharactersFilter(Filter):
    """Filter to keep samples with special-character ratio within a specific range.

    This operator filters out samples based on the ratio of special characters in the text.
    It keeps samples where the special-character ratio is within the specified minimum and
    maximum thresholds. The special-character ratio is computed as the number of special
    characters divided by the total number of characters in the text. If the
    'special_char_ratio' is already cached in the stats, it will be reused. Otherwise, it
    will be computed and stored in the 'special_char_ratio' field."""

    _batched_op = True

    def __init__(self, min_ratio: float = 0.0, max_ratio: float = 0.25, *args, **kwargs):
        """
        Initialization method.

        :param min_ratio: The min filter ratio in this op, samples will
            be filtered if their special-char ratio is below this
            parameter.
        :param max_ratio: The max filter ratio in this op, samples will
            be filtered if their special-char ratio exceeds this
            parameter.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def compute_stats_batched(self, samples):
        samples_list = samples[self.text_key]
        samples_stats = samples[Fields.stats]

        for idx, stat in enumerate(samples_stats):
            # check if it's computed already
            if StatsKeys.special_char_ratio in stat:
                continue
            cur_text = samples_list[idx]
            # get ratio of special characters
            samples_stats[idx][StatsKeys.special_char_ratio] = (
                (len([c for c in cur_text if c in SPECIAL_CHARACTERS]) / len(cur_text)) if len(cur_text) != 0 else 0.0
            )

        return samples

    def process_batched(self, samples):
        assert isinstance(samples[Fields.stats], list)
        return map(
            lambda stat: self.get_keep_boolean(stat[StatsKeys.special_char_ratio], self.min_ratio, self.max_ratio),
            samples[Fields.stats],
        )

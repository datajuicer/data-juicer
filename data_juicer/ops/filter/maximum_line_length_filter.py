import sys

from data_juicer.utils.constant import Fields, InterVars, StatsKeys

from ..base_op import OPERATORS, Filter
from ..op_fusion import INTER_LINES

OP_NAME = "maximum_line_length_filter"


@OPERATORS.register_module(OP_NAME)
@INTER_LINES.register_module(OP_NAME)
class MaximumLineLengthFilter(Filter):
    """Filter to keep samples with a maximum line length within a specified range.

    This operator filters out samples based on the length of their longest line. It retains
    samples where the maximum line length is within the specified `min_len` and `max_len`
    range. The maximum line length is computed by splitting the text into lines and
    measuring the length of each line. If the context is provided, it uses precomputed lines
    stored under the key 'lines' in the context. The maximum line length is cached in the
    'max_line_length' field of the stats."""

    _batched_op = True

    def __init__(self, min_len: int = 10, max_len: int = sys.maxsize, *args, **kwargs):
        """
        Initialization method.

        :param min_len: The min filter length in this op, samples will
            be filtered if their maximum line length is below this
            parameter.
        :param max_len: The max filter length in this op, samples will
            be filtered if their maximum line length exceeds this
            parameter.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.min_len = min_len
        self.max_len = max_len

    def compute_stats_batched(self, samples, context=False):
        samples_list = samples[self.text_key]
        samples_stats = samples[Fields.stats]
        context_key = f"{InterVars.lines}"

        for idx, stat in enumerate(samples_stats):
            # check if it's computed already
            if StatsKeys.max_line_length in stat:
                continue

            if context and context_key in samples[Fields.context][idx]:
                lines = samples[Fields.context][idx][context_key]
            else:
                lines = samples_list[idx].splitlines()
                if context:
                    samples[Fields.context][idx][context_key] = lines
            line_lengths = list(map(len, lines))
            samples_stats[idx][StatsKeys.max_line_length] = max(line_lengths) if line_lengths else 0

        return samples

    def process_batched(self, samples):
        assert isinstance(samples[Fields.stats], list)
        return map(
            lambda stat: self.get_keep_boolean(stat[StatsKeys.max_line_length], self.min_len, self.max_len),
            samples[Fields.stats],
        )

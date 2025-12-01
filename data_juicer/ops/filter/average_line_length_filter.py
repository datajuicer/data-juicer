import sys

from data_juicer.utils.constant import Fields, InterVars, StatsKeys

from ..base_op import OPERATORS, Filter
from ..op_fusion import INTER_LINES

OP_NAME = "average_line_length_filter"


@OPERATORS.register_module(OP_NAME)
@INTER_LINES.register_module(OP_NAME)
class AverageLineLengthFilter(Filter):
    """Filter to keep samples with average line length within a specific range.

    This operator filters out samples based on their average line length. It keeps samples
    where the average line length is between the specified minimum and maximum values. The
    average line length is calculated as the total text length divided by the number of
    lines. If the context is provided, it uses precomputed lines from the context. The
    computed average line length is stored in the 'avg_line_length' key in the stats field."""

    _batched_op = True

    def __init__(self, min_len: int = 10, max_len: int = sys.maxsize, *args, **kwargs):
        """
        Initialization method.

        :param min_len: The min filter length in this op, samples will
            be filtered if their average line length is below this
            parameter.
        :param max_len: The max filter length in this op, samples will
            be filtered if their average line length exceeds this
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
            if StatsKeys.avg_line_length in stat:
                continue

            cur_text = samples_list[idx]
            if context and context_key in samples[Fields.context][idx]:
                lines = samples[Fields.context][idx][context_key]
            else:
                lines = cur_text.splitlines()
                if context:
                    samples[Fields.context][idx][context_key] = lines
            samples_stats[idx][StatsKeys.avg_line_length] = len(cur_text) / len(lines) if len(lines) != 0 else 0.0
        return samples

    def process_batched(self, samples):
        assert isinstance(samples[Fields.stats], list)
        return map(
            lambda stat: self.get_keep_boolean(stat[StatsKeys.avg_line_length], self.min_len, self.max_len),
            samples[Fields.stats],
        )

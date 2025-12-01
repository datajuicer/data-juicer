from typing import List, Union

from data_juicer.utils.constant import Fields

from ..base_op import NON_STATS_FILTERS, OPERATORS, Filter

OP_NAME = "suffix_filter"


@NON_STATS_FILTERS.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
class SuffixFilter(Filter):
    """Filter to keep samples with specified suffix.

    This operator retains samples that have a suffix matching any of the provided suffixes.
    If no suffixes are specified, all samples are kept. The key metric 'keep' is computed
    based on whether the sample's suffix matches the specified list. The 'suffix' field of
    each sample is checked against the list of allowed suffixes. If the suffix matches, the
    sample is kept; otherwise, it is filtered out."""

    def __init__(self, suffixes: Union[str, List[str]] = [], *args, **kwargs):
        """
        Initialization method.

        :param suffixes: the suffix of text that will be keep.
            For example: '.txt', 'txt' or ['txt', '.pdf', 'docx']
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        if suffixes is None:
            self.suffixes = []
        elif isinstance(suffixes, str):
            self.suffixes = [suffixes]
        else:
            self.suffixes = suffixes

    def compute_stats_single(self, sample):
        return sample

    def process_single(self, sample):
        if self.suffixes:
            res_bool = sample[Fields.suffix] in self.suffixes
            if self.reversed_range:
                res_bool = not res_bool
            return res_bool
        else:
            return True

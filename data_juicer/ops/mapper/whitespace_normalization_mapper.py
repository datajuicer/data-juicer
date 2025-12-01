# Most of the code here has been modified from:
# https://github.com/bigscience-workshop/data-preparation
# --------------------------------------------------------

from ..base_op import OPERATORS, Mapper
from ..common.special_characters import VARIOUS_WHITESPACES


@OPERATORS.register_module("whitespace_normalization_mapper")
class WhitespaceNormalizationMapper(Mapper):
    """Normalizes various types of whitespace characters to standard spaces in text samples.

    This mapper converts all non-standard whitespace characters, such as tabs and newlines,
    to the standard space character (' ', 0x20). It also trims leading and trailing
    whitespace from the text. This ensures consistent spacing across all text samples,
    improving readability and consistency. The normalization process is based on a
    comprehensive list of whitespace characters, which can be found at
    https://en.wikipedia.org/wiki/Whitespace_character."""

    _batched_op = True

    def __init__(self, *args, **kwargs):
        """
        Initialization method.

        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)

    def process_batched(self, samples):
        for idx, text in enumerate(samples[self.text_key]):
            # remove whitespaces before and after the main content
            text = text.strip()

            # replace all kinds of whitespaces with ' '
            samples[self.text_key][idx] = "".join([char if char not in VARIOUS_WHITESPACES else " " for char in text])

        return samples

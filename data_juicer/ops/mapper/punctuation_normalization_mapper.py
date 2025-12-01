# Some code here has been modified from:
# https://github.com/bigscience-workshop/data-preparation
# --------------------------------------------------------

from ..base_op import OPERATORS, Mapper


@OPERATORS.register_module("punctuation_normalization_mapper")
class PunctuationNormalizationMapper(Mapper):
    """Normalizes unicode punctuations to their English equivalents in text samples.

    This operator processes a batch of text samples and replaces any unicode punctuation
    with its corresponding English punctuation. The mapping includes common substitutions
    like "，" to ",", "。" to ".", and "“" to ". It iterates over each character in the text,
    replacing it if it is found in the predefined punctuation map. The result is a set of
    text samples with consistent punctuation formatting."""

    _batched_op = True

    def __init__(self, *args, **kwargs):
        """
        Initialization method.

        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.punctuation_unicode = {
            "，": ",",
            "。": ".",
            "、": ",",
            "„": '"',
            "”": '"',
            "“": '"',
            "«": '"',
            "»": '"',
            "１": '"',
            "」": '"',
            "「": '"',
            "《": '"',
            "》": '"',
            "´": "'",
            "∶": ":",
            "：": ":",
            "？": "?",
            "！": "!",
            "（": "(",
            "）": ")",
            "；": ";",
            "–": "-",
            "—": " - ",
            "．": ". ",
            "～": "~",
            "’": "'",
            "…": "...",
            "━": "-",
            "〈": "<",
            "〉": ">",
            "【": "[",
            "】": "]",
            "％": "%",
            "►": "-",
        }

    def process_batched(self, samples):
        samples[self.text_key] = [
            "".join([self.punctuation_unicode.get(c, c) for c in text]) for text in samples[self.text_key]
        ]
        return samples

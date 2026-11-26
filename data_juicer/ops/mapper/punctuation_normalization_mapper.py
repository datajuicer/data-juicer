# Some code here has been modified from:
# https://github.com/bigscience-workshop/data-preparation
# --------------------------------------------------------

from ..base_op import OPERATORS, Mapper


@OPERATORS.register_module("punctuation_normalization_mapper")
class PunctuationNormalizationMapper(Mapper):
    """Normalizes unicode punctuations to their English equivalents in text samples.

    This operator processes a batch of text samples and replaces any unicode punctuation
    with its corresponding English punctuation. The mapping includes common substitutions
    like "，" to ",", "。" to ".", and "“" to ". It replaces only the characters found in the
    predefined punctuation map."""

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
            "＂": '"',
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
        replacements = {char: replacement for char, replacement in self.punctuation_unicode.items() if len(char) == 1}
        # Independent replacements can use native string scans. If a customized
        # mapping can introduce another key, preserve single-pass replacement.
        if not replacements.keys().isdisjoint(char for value in replacements.values() for char in value):
            samples[self.text_key] = [
                "".join(replacements.get(char, char) for char in text) for text in samples[self.text_key]
            ]
            return samples

        only_non_ascii = not any(char.isascii() for char in replacements)
        normalized = []
        for text in samples[self.text_key]:
            if not (only_non_ascii and text.isascii()):
                for char, replacement in replacements.items():
                    text = text.replace(char, replacement)
            normalized.append(text)
        samples[self.text_key] = normalized
        return samples

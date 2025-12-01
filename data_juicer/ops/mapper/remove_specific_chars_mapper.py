from typing import List, Union

import regex as re

from ..base_op import OPERATORS, Mapper


@OPERATORS.register_module("remove_specific_chars_mapper")
class RemoveSpecificCharsMapper(Mapper):
    """Removes specific characters from text samples.

    This operator removes specified characters from the text. The characters to be removed
    can be provided as a string or a list of strings. If no characters are specified, the
    default set includes special and non-alphanumeric characters. The operator processes the
    text using a regular expression pattern that matches any of the specified characters and
    replaces them with an empty string. This is done in a batched manner for efficiency."""

    _batched_op = True

    def __init__(self, chars_to_remove: Union[str, List[str]] = "◆●■►▼▲▴∆▻▷❖♡□", *args, **kwargs):
        """
        Initialization method.

        :param chars_to_remove: a list or a string including all
            characters that need to be removed from text.
        :param args: extra args
        :param kwargs: extra args
        """

        super().__init__(*args, **kwargs)
        if chars_to_remove:
            self.pattern = "[" + "|".join(chars_to_remove) + "]"
        else:
            self.pattern = None

    def process_batched(self, samples):
        if self.pattern is None:
            return samples

        samples[self.text_key] = [
            re.sub(pattern=self.pattern, repl=r"", string=text, flags=re.DOTALL) for text in samples[self.text_key]
        ]
        return samples

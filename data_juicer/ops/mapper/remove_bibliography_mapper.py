# Some code here has been modified from:
# https://github.com/togethercomputer/RedPajama-Data/tree/rp_v1/
# --------------------------------------------------------

import regex as re

from ..base_op import OPERATORS, Mapper


@OPERATORS.register_module("remove_bibliography_mapper")
class RemoveBibliographyMapper(Mapper):
    """Removes bibliography sections at the end of LaTeX documents.

    This operator identifies and removes bibliography sections in LaTeX documents. It uses a
    regular expression to match common bibliography commands such as \\appendix,
    \\begin{references}, \\begin{thebibliography}, and \\bibliography. The matched sections are
    removed from the text. The operator processes samples in batch mode for efficiency."""

    _batched_op = True

    def __init__(self, *args, **kwargs):
        """
        Initialization method.

        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        # \appendix, \appendices (IEEEtran) or \appendixpage (the appendix package), each as a
        # whole command: \appendixname and other longer commands must not remove the rest of the text
        self.pattern = r"(\\(?:appendix|appendices|appendixpage)(?![a-zA-Z])|"
        self.pattern += r"\\begin\{references\}|"
        self.pattern += r"\\begin\{REFERENCES\}|"
        self.pattern += r"\\begin\{thebibliography\}|"
        self.pattern += r"\\bibliography\{.*\}"
        self.pattern += r").*$"
        self.bibliography_pattern = re.compile(self.pattern, re.DOTALL)

    def process_batched(self, samples):
        # `.*$` under DOTALL swallows everything from the first match to the end of
        # the text, so substituting it with an empty string is the same as keeping
        # the prefix before that match. Slicing avoids building a match object for
        # the whole tail.
        samples[self.text_key] = [
            text if (match := self.bibliography_pattern.search(text)) is None else text[: match.start()]
            for text in samples[self.text_key]
        ]

        return samples

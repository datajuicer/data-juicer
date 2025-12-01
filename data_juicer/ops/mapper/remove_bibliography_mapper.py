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
        self.pattern = r"(\\appendix|"
        self.pattern += r"\\begin\{references\}|"
        self.pattern += r"\\begin\{REFERENCES\}|"
        self.pattern += r"\\begin\{thebibliography\}|"
        self.pattern += r"\\bibliography\{.*\}"
        self.pattern += r").*$"

    def process_batched(self, samples):
        samples[self.text_key] = [
            re.sub(pattern=self.pattern, repl=r"", string=text, flags=re.DOTALL) for text in samples[self.text_key]
        ]

        return samples

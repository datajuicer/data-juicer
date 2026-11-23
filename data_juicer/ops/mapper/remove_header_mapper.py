# Some code here has been modified from:
# https://github.com/togethercomputer/RedPajama-Data/tree/rp_v1/
# --------------------------------------------------------

import regex as re

from ..base_op import OPERATORS, Mapper


@OPERATORS.register_module("remove_header_mapper")
class RemoveHeaderMapper(Mapper):
    """Removes headers at the beginning of documents in LaTeX samples.

    This operator identifies and removes headers such as chapter, part, section, subsection,
    subsubsection, paragraph, and subparagraph. It uses a regular expression to match these
    headers. If a sample does not contain any headers and `drop_no_head` is set to True, the
    sample text will be removed. Otherwise, the sample remains unchanged. The operator
    processes samples in batches for efficiency."""

    _batched_op = True

    def __init__(self, drop_no_head: bool = True, *args, **kwargs):
        """
        Initialization method.

        :param drop_no_head: whether to drop sample texts without
            headers.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        headers = (
            r"\\\bchapter\b\*?(?:\[(.*?)\])?\{(.*?)\}|"
            r"\\\bpart\b\*?(?:\[(.*?)\])?\{(.*?)\}|"
            r"\\\bsection\b\*?(?:\[(.*?)\])?\{(.*?)\}|"
            r"\\\bsubsection\b\*?(?:\[(.*?)\])?\{(.*?)\}|"
            r"\\\bsubsubsection\b\*?(?:\[(.*?)\])?\{(.*?)\}|"
            r"\\\bparagraph\b\*?(?:\[(.*?)\])?\{(.*?)\}|"
            r"\\\bsubparagraph\b\*?(?:\[(.*?)\])?\{(.*?)\}"
        )
        # `^(.*?)(...)` anchors the match to the start, so the engine has to
        # retry from every offset and cannot use its literal-prefix fast
        # search. Searching the headers on their own and slicing the text
        # gives the same result with a cheaper scan.
        # Keep the plain-string attribute that existed before this change;
        # subclass code and callers may read it.
        self.pattern = headers
        self.header_pattern = re.compile(headers, re.DOTALL)

        self.drop_no_head = drop_no_head

    def process_batched(self, samples):
        for idx, text in enumerate(samples[self.text_key]):
            # Every alternative starts with a literal backslash, so text without
            # one cannot contain a header. This is a C-level scan and skips the
            # regex entirely on plain text.
            if "\\" not in text:
                if self.drop_no_head:
                    samples[self.text_key][idx] = ""
                continue
            match = self.header_pattern.search(text)
            if match is None:
                if self.drop_no_head:
                    samples[self.text_key][idx] = ""
                continue
            samples[self.text_key][idx] = text[match.start() :]

        return samples

# Some code here has been modified from:
# https://github.com/kallewesterling/CleanText/
# --------------------------------------------------------
from typing import Optional

import regex as re

from ..base_op import OPERATORS, Mapper


@OPERATORS.register_module("clean_links_mapper")
class CleanLinksMapper(Mapper):
    """Mapper to clean links like http/https/ftp in text samples.

    This operator removes or replaces URLs and other web links in the text. It uses a
    regular expression pattern to identify and remove links. By default, it replaces the
    identified links with an empty string, effectively removing them. The operator can be
    customized with a different pattern and replacement string. It processes samples in
    batches and modifies the text in place. If no links are found in a sample, it is left
    unchanged."""

    _batched_op = True

    def __init__(self, pattern: Optional[str] = None, repl: str = "", *args, **kwargs):
        """
        Initialization method.

        :param pattern: regular expression pattern to search for within text.
        :param repl: replacement string, default is empty string.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        if pattern is None:
            self.pattern = r"(?i)\b("
            self.pattern += r"(?:[a-z][\w-]+:(?:\/{1,3}|"
            self.pattern += r"[a-z0-9%])|www\d{0,3}[.]|"
            self.pattern += r"[a-z0-9.\-]+[.][a-z]{2,4}\/)"
            self.pattern += r"(?:[^\s()<>]+|\(([^\s()<>]+|"
            self.pattern += r"(\([^\s()<>]+\)))*\))"
            self.pattern += r"+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|"
            self.pattern += r"[^\s`!()\[\]{};:\'\".,<>?«»“”‘’])"
            self.pattern += r")"
        else:
            self.pattern = pattern
            if (len(pattern) > 2) and (
                pattern.startswith("r'") and pattern.endswith("'") or pattern.startswith('r"') and pattern.endswith('"')
            ):
                self.pattern = pattern[2:-1]
        self.repl = repl

    def process_batched(self, samples):
        for idx, text in enumerate(samples[self.text_key]):
            if not re.search(self.pattern, text, flags=re.DOTALL):
                continue

            samples[self.text_key][idx] = re.sub(pattern=self.pattern, repl=self.repl, string=text, flags=re.DOTALL)
        return samples

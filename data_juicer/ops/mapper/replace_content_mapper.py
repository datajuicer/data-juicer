from typing import List, Union

import regex as re

from ..base_op import OPERATORS, Mapper


@OPERATORS.register_module("replace_content_mapper")
class ReplaceContentMapper(Mapper):
    """Replaces content in the text that matches a specific regular expression pattern with a
    designated replacement string.

    This operator processes text by searching for patterns defined in `pattern` and
    replacing them with the corresponding `repl` string. If multiple patterns and
    replacements are provided, each pattern is replaced by its respective replacement. The
    operator supports both single and multiple patterns and replacements. The regular
    expressions are compiled with the `re.DOTALL` flag to match across multiple lines. If
    the length of the patterns and replacements do not match, a `ValueError` is raised. This
    operation is batched, meaning it processes multiple samples at once."""

    _batched_op = True

    def __init__(self, pattern: Union[str, List[str], None] = None, repl: Union[str, List[str]] = "", *args, **kwargs):
        """
        Initialization method.

        :param pattern: regular expression pattern(s) to search for within text
        :param repl: replacement string(s), default is empty string
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.pattern = pattern
        self.repl = repl
        self.compiled_patterns = []
        if isinstance(pattern, str):
            self.compiled_patterns.append(self._prepare_pattern(pattern))
        elif isinstance(pattern, list):
            for p in pattern:
                self.compiled_patterns.append(self._prepare_pattern(p))

    def _prepare_pattern(self, pattern: str) -> re.Pattern:
        """Prepare the regular expression pattern."""
        if (pattern is not None and len(pattern) > 2) and (
            pattern.startswith("r'") and pattern.endswith("'") or pattern.startswith('r"') and pattern.endswith('"')
        ):
            pattern = pattern[2:-1]
        return re.compile(pattern, flags=re.DOTALL)

    def process_batched(self, samples):
        if self.pattern is None:
            return samples

        for idx, text in enumerate(samples[self.text_key]):
            for i, pattern in enumerate(self.compiled_patterns):
                if isinstance(self.repl, list) and i < len(self.repl):
                    replacement = self.repl[i]
                elif isinstance(self.repl, list) and i >= len(self.repl):
                    raise ValueError(
                        f"pattern length: {len(self.pattern)} '" f"must be equal to " f"repl length: {len(self.repl)}"
                    )
                else:
                    replacement = self.repl

                text = pattern.sub(replacement, text)

            samples[self.text_key][idx] = text

        return samples

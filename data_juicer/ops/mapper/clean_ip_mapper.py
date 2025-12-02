from typing import Optional

import regex as re

from ..base_op import OPERATORS, Mapper


@OPERATORS.register_module("clean_ip_mapper")
class CleanIpMapper(Mapper):
    """Cleans IPv4 and IPv6 addresses from text samples.

    This operator removes or replaces IPv4 and IPv6 addresses in the text. It uses a regular
    expression to identify and clean the IP addresses. By default, it replaces the IP
    addresses with an empty string, effectively removing them. The operator can be
    configured with a custom pattern and replacement string. If no pattern is provided, a
    default pattern for both IPv4 and IPv6 addresses is used. The operator processes samples
    in batches.

    - Uses a regular expression to find and clean IP addresses.
    - Replaces found IP addresses with a specified replacement string.
    - Default replacement string is an empty string, which removes the IP addresses.
    - Can use a custom regular expression pattern if provided.
    - Processes samples in batches for efficiency."""

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
            self.pattern = r"(?:(?:1[0-9][0-9]\.)|(?:2[0-4][0-9]\.)|"
            self.pattern += r"(?:25[0-5]\.)|(?:[1-9][0-9]\.)|(?:[0-9]\.))"
            self.pattern += r"{3}(?:(?:1[0-9][0-9])|(?:2[0-4][0-9])|"
            self.pattern += r"(?:25[0-5])|(?:[1-9][0-9])|(?:[0-9]))|"
            self.pattern += r"([\da-fA-F]{1,4}:){7}[\da-fA-F]{1,4}"  # ipv6
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

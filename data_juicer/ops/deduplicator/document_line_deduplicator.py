import hashlib
import re
import string
from collections import defaultdict
from typing import Dict, Set

from data_juicer.utils.constant import HashKeys

from ..base_op import OPERATORS, Deduplicator

BRACKETS_ONLY = frozenset("{}[]();")
LATEX_ENV_RE = re.compile(r"^\s*\\(begin|end)\{")
HTML_TAG_RE = re.compile(r"^\s*</?[a-z][a-z0-9]*[^>]*>?\s*$", re.IGNORECASE)
SPECIAL_CHAR_RE = re.compile(rf"\s+|\d+|[{re.escape(string.punctuation)}]")


@OPERATORS.register_module("document_line_deduplicator")
class DocumentLineDeduplicator(Deduplicator):
    """Deduplicates at the line level across documents.

    This operator identifies lines that appear in many documents (boilerplate
    text, copyright notices, navigation bars, etc.) and removes them.  It works
    in two phases:

    1. **compute_hash** – splits each document into lines, applies configurable
       skip rules, and computes an MD5 hash for every non-skipped line.
    2. **process** – counts in how many *distinct* documents each line hash
       appears.  Lines whose document frequency exceeds
       ``frequency_threshold`` are removed from every document.
    """

    def __init__(
        self,
        frequency_threshold: int = 6,
        lowercase: bool = False,
        ignore_special_character: bool = False,
        min_line_length: int = 2,
        skip_brackets: bool = True,
        skip_markdown_headers: bool = True,
        skip_latex_env: bool = True,
        skip_html_tags: bool = True,
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param frequency_threshold: document-frequency threshold.  Lines
            appearing in **more than** this many documents are removed.
        :param lowercase: whether to lower-case a line before hashing.
        :param ignore_special_character: whether to strip whitespace, digits,
            and punctuation before hashing.
        :param min_line_length: lines whose stripped length is below this
            value are skipped (never considered for dedup).
        :param skip_brackets: skip lines consisting solely of bracket /
            semicolon characters such as ``{ } [ ] ( ) ;``.
        :param skip_markdown_headers: skip lines that start with ``#``
            (Markdown headings).
        :param skip_latex_env: skip LaTeX ``\\begin{…}`` / ``\\end{…}``
            environment declarations.
        :param skip_html_tags: skip lines that are pure HTML / XML tags.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.frequency_threshold = frequency_threshold
        self.lowercase = lowercase
        self.ignore_special_character = ignore_special_character
        self.min_line_length = min_line_length
        self.skip_brackets = skip_brackets
        self.skip_markdown_headers = skip_markdown_headers
        self.skip_latex_env = skip_latex_env
        self.skip_html_tags = skip_html_tags

    # ------------------------------------------------------------------
    # Skip-rule helpers
    # ------------------------------------------------------------------

    def _should_skip_line(self, line: str) -> bool:
        """Return True if *line* should be exempt from deduplication."""
        stripped = line.strip()

        if len(stripped) < self.min_line_length:
            return True

        if self.skip_brackets and all(ch in BRACKETS_ONLY for ch in stripped):
            return True

        if self.skip_markdown_headers and stripped.startswith("#"):
            return True

        if self.skip_latex_env and LATEX_ENV_RE.match(stripped):
            return True

        if self.skip_html_tags and HTML_TAG_RE.match(stripped):
            return True

        return False

    # ------------------------------------------------------------------
    # Phase 1 – per-document hash computation
    # ------------------------------------------------------------------

    def compute_hash(self, sample):
        """
        Compute per-line MD5 hashes for a single document.

        Skipped lines receive an empty-string hash so that the list of
        hashes stays aligned with the original lines.

        :param sample: input sample
        :return: sample with ``HashKeys.line_hashes`` populated.
        """
        if HashKeys.line_hashes in sample:
            return sample

        text = sample[self.text_key]
        lines = text.split("\n")
        line_hashes = []

        for line in lines:
            if self._should_skip_line(line):
                line_hashes.append("")
            else:
                norm = line
                if self.lowercase:
                    norm = norm.lower()
                if self.ignore_special_character:
                    norm = SPECIAL_CHAR_RE.sub("", norm)
                md5 = hashlib.md5(norm.strip().encode("utf-8")).hexdigest()
                line_hashes.append(md5)

        sample[HashKeys.line_hashes] = line_hashes
        return sample

    # ------------------------------------------------------------------
    # Phase 2 – global frequency counting & line removal
    # ------------------------------------------------------------------

    def process(self, dataset, show_num=0):
        """
        Remove high-frequency lines from the dataset.

        :param dataset: input dataset (already hash-annotated).
        :param show_num: number of traced duplicate pairs for inspection.
        :return: (dataset, dup_pairs) where *dup_pairs* maps a line hash
            to sample texts that contained it.
        """
        if len(dataset) <= 1:
            return dataset, {}

        # --- count document frequency for each line hash ---------------
        doc_freq: Dict[str, int] = defaultdict(int)
        for row in dataset:
            unique_hashes = {h for h in row[HashKeys.line_hashes] if h}
            for h in unique_hashes:
                doc_freq[h] += 1

        high_freq: Set[str] = {h for h, cnt in doc_freq.items() if cnt > self.frequency_threshold}

        if not high_freq:
            # nothing to remove – drop the temporary column and return
            dataset = dataset.remove_columns([HashKeys.line_hashes])
            return dataset, {}

        # --- optionally collect duplicate pairs for tracing ------------
        dup_pairs: dict = {}
        if show_num > 0:
            sorted_hashes = sorted(
                [(h, doc_freq[h]) for h in high_freq],
                key=lambda x: x[1],
                reverse=True,
            )[:show_num]
            dup_pairs = {h: [] for h, _ in sorted_hashes}

        # --- remove high-frequency lines from every document -----------
        def _remove_high_freq_lines(sample):
            line_hashes = sample[HashKeys.line_hashes]
            text = sample[self.text_key]
            lines = text.split("\n")
            kept_lines = []

            for line, h in zip(lines, line_hashes):
                if h and h in high_freq:
                    # collect tracing info
                    if show_num > 0 and h in dup_pairs:
                        if len(dup_pairs[h]) < 2:
                            dup_pairs[h].append(sample[self.text_key])
                else:
                    kept_lines.append(line)

            sample[self.text_key] = "\n".join(kept_lines)
            return sample

        dataset = dataset.map(
            _remove_high_freq_lines,
            num_proc=self.runtime_np() if show_num == 0 else 1,
            load_from_cache_file=False if show_num > 0 else True,
        )

        # clean up temporary column
        dataset = dataset.remove_columns([HashKeys.line_hashes])
        return dataset, dup_pairs

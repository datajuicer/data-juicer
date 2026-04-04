import unittest

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.deduplicator.document_line_deduplicator import (
    DocumentLineDeduplicator,
)
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class DocumentLineDeduplicatorTest(DataJuicerTestCaseBase):

    def _run_line_dedup(self, dataset, target_list, op, show_num=0):
        dataset = dataset.map(op.compute_hash)
        dataset, dup_pairs = op.process(dataset, show_num=show_num)
        dataset = dataset.select_columns(column_names=["text"])
        res_list = dataset.to_list()
        self.assertEqual(res_list, target_list)
        return dup_pairs

    def test_basic_dedup(self):
        """High-frequency lines are removed; low-frequency lines kept."""
        ds_list = [
            {"text": "Hello\nBoilerplate\nWorld"},
            {"text": "Foo\nBoilerplate\nBar"},
            {"text": "Baz\nBoilerplate\nQux"},
        ]
        tgt_list = [
            {"text": "Hello\nWorld"},
            {"text": "Foo\nBar"},
            {"text": "Baz\nQux"},
        ]
        dataset = Dataset.from_list(ds_list)
        op = DocumentLineDeduplicator(frequency_threshold=2)
        self._run_line_dedup(dataset, tgt_list, op)

    def test_below_threshold(self):
        """Lines under the frequency threshold are not removed."""
        ds_list = [
            {"text": "Hello\nBoilerplate\nWorld"},
            {"text": "Foo\nBoilerplate\nBar"},
            {"text": "Baz\nBoilerplate\nQux"},
        ]
        # threshold=3, and "Boilerplate" appears in exactly 3 docs => kept
        tgt_list = [
            {"text": "Hello\nBoilerplate\nWorld"},
            {"text": "Foo\nBoilerplate\nBar"},
            {"text": "Baz\nBoilerplate\nQux"},
        ]
        dataset = Dataset.from_list(ds_list)
        op = DocumentLineDeduplicator(frequency_threshold=3)
        self._run_line_dedup(dataset, tgt_list, op)

    def test_skip_short_lines(self):
        """Short lines are never removed even if high-frequency."""
        ds_list = [
            {"text": "A\nHello"},
            {"text": "A\nWorld"},
            {"text": "A\nFoo"},
        ]
        # "A" is short (len=1 < min_line_length=2) so skipped
        tgt_list = [
            {"text": "A\nHello"},
            {"text": "A\nWorld"},
            {"text": "A\nFoo"},
        ]
        dataset = Dataset.from_list(ds_list)
        op = DocumentLineDeduplicator(frequency_threshold=2)
        self._run_line_dedup(dataset, tgt_list, op)

    def test_skip_brackets(self):
        """Bracket-only lines are never removed even if high-frequency."""
        ds_list = [
            {"text": "{\nHello\n}"},
            {"text": "{\nWorld\n}"},
            {"text": "{\nFoo\n}"},
        ]
        tgt_list = [
            {"text": "{\nHello\n}"},
            {"text": "{\nWorld\n}"},
            {"text": "{\nFoo\n}"},
        ]
        dataset = Dataset.from_list(ds_list)
        op = DocumentLineDeduplicator(frequency_threshold=2)
        self._run_line_dedup(dataset, tgt_list, op)

    def test_skip_markdown_headers(self):
        """Markdown headers are never removed even if high-frequency."""
        ds_list = [
            {"text": "# Title\nHello"},
            {"text": "# Title\nWorld"},
            {"text": "# Title\nFoo"},
        ]
        tgt_list = [
            {"text": "# Title\nHello"},
            {"text": "# Title\nWorld"},
            {"text": "# Title\nFoo"},
        ]
        dataset = Dataset.from_list(ds_list)
        op = DocumentLineDeduplicator(frequency_threshold=2)
        self._run_line_dedup(dataset, tgt_list, op)

    def test_skip_latex_env(self):
        r"""LaTeX \begin/\end lines are never removed."""
        ds_list = [
            {"text": "\\begin{document}\nHello\n\\end{document}"},
            {"text": "\\begin{document}\nWorld\n\\end{document}"},
            {"text": "\\begin{document}\nFoo\n\\end{document}"},
        ]
        tgt_list = [
            {"text": "\\begin{document}\nHello\n\\end{document}"},
            {"text": "\\begin{document}\nWorld\n\\end{document}"},
            {"text": "\\begin{document}\nFoo\n\\end{document}"},
        ]
        dataset = Dataset.from_list(ds_list)
        op = DocumentLineDeduplicator(frequency_threshold=2)
        self._run_line_dedup(dataset, tgt_list, op)

    def test_skip_html_tags(self):
        """HTML/XML tag lines are never removed."""
        ds_list = [
            {"text": "<div>\nHello\n</div>"},
            {"text": "<div>\nWorld\n</div>"},
            {"text": "<div>\nFoo\n</div>"},
        ]
        tgt_list = [
            {"text": "<div>\nHello\n</div>"},
            {"text": "<div>\nWorld\n</div>"},
            {"text": "<div>\nFoo\n</div>"},
        ]
        dataset = Dataset.from_list(ds_list)
        op = DocumentLineDeduplicator(frequency_threshold=2)
        self._run_line_dedup(dataset, tgt_list, op)

    def test_lowercase(self):
        """With lowercase=True, lines differing only in case are deduped."""
        ds_list = [
            {"text": "Boilerplate\nHello"},
            {"text": "boilerplate\nWorld"},
            {"text": "BOILERPLATE\nFoo"},
        ]
        tgt_list = [
            {"text": "Hello"},
            {"text": "World"},
            {"text": "Foo"},
        ]
        dataset = Dataset.from_list(ds_list)
        op = DocumentLineDeduplicator(frequency_threshold=2, lowercase=True)
        self._run_line_dedup(dataset, tgt_list, op)

    def test_ignore_special_character(self):
        """With ignore_special_character, punctuation/digits are stripped."""
        ds_list = [
            {"text": "Hello, World!\nContent A"},
            {"text": "Hello World\nContent B"},
            {"text": "Hello  World\nContent C"},
        ]
        tgt_list = [
            {"text": "Content A"},
            {"text": "Content B"},
            {"text": "Content C"},
        ]
        dataset = Dataset.from_list(ds_list)
        op = DocumentLineDeduplicator(frequency_threshold=2, ignore_special_character=True)
        self._run_line_dedup(dataset, tgt_list, op)

    def test_single_document(self):
        """A single document is returned unchanged."""
        ds_list = [{"text": "Hello\nWorld"}]
        tgt_list = [{"text": "Hello\nWorld"}]
        dataset = Dataset.from_list(ds_list)
        op = DocumentLineDeduplicator(frequency_threshold=2)
        self._run_line_dedup(dataset, tgt_list, op)

    def test_show_num(self):
        """show_num returns duplicate pair information."""
        ds_list = [
            {"text": "Hello\nBoilerplate\nWorld"},
            {"text": "Foo\nBoilerplate\nBar"},
            {"text": "Baz\nBoilerplate\nQux"},
        ]
        tgt_list = [
            {"text": "Hello\nWorld"},
            {"text": "Foo\nBar"},
            {"text": "Baz\nQux"},
        ]
        dataset = Dataset.from_list(ds_list)
        op = DocumentLineDeduplicator(frequency_threshold=2)
        dup_pairs = self._run_line_dedup(dataset, tgt_list, op, show_num=1)
        self.assertEqual(len(dup_pairs), 1)


if __name__ == "__main__":
    unittest.main()

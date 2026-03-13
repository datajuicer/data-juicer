import unittest

from data_juicer.ops.mapper.arxiv_to_markdown_mapper import (
    ARXIV_HTML_URL_TEMPLATE,
    _normalize_arxiv_id,
    _normalize_arxiv_id_keep_version,
    ArxivToMarkdownMapper,
)
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class ArxivToMarkdownMapperTest(DataJuicerTestCaseBase):
    def setUp(self):
        super().setUp()
        self.op = ArxivToMarkdownMapper(arxiv_id_key="arxiv_id", backend="pdfplumber")

    def test_normalize_arxiv_id(self):
        self.assertEqual(_normalize_arxiv_id("2501.14755"), "2501.14755")
        self.assertEqual(_normalize_arxiv_id(" 2501.14755 "), "2501.14755")
        self.assertEqual(
            _normalize_arxiv_id("https://arxiv.org/abs/2501.14755"),
            "2501.14755",
        )
        self.assertEqual(
            _normalize_arxiv_id("https://arxiv.org/pdf/2501.14755.pdf"),
            "2501.14755",
        )
        self.assertEqual(_normalize_arxiv_id("hep-th/9901001"), "hep-th/9901001")
        self.assertEqual(_normalize_arxiv_id("2501.14755v1"), "2501.14755")
        self.assertEqual(_normalize_arxiv_id("hep-th/9901001v2"), "hep-th/9901001")
        self.assertIsNone(_normalize_arxiv_id(""))
        self.assertIsNone(_normalize_arxiv_id(None))

    def test_normalize_arxiv_id_keep_version(self):
        self.assertEqual(_normalize_arxiv_id_keep_version("2501.14755v1"), "2501.14755v1")
        self.assertEqual(_normalize_arxiv_id_keep_version("2501.14755"), "2501.14755")
        self.assertEqual(
            _normalize_arxiv_id_keep_version("https://arxiv.org/html/2501.14755v1"),
            "2501.14755v1",
        )
        self.assertEqual(
            _normalize_arxiv_id_keep_version("https://arxiv.org/abs/2501.14755v1"),
            "2501.14755v1",
        )
        self.assertIsNone(_normalize_arxiv_id_keep_version(""))

    def test_html_url_template(self):
        self.assertEqual(
            ARXIV_HTML_URL_TEMPLATE.format(arxiv_id="2501.14755v1"),
            "https://arxiv.org/html/2501.14755v1",
        )

    def test_missing_arxiv_id(self):
        sample = {"text": "original"}
        out = self.op.process_single(sample)
        self.assertIn("text", out)
        self.assertEqual(out["text"], "original")

    def test_output_key(self):
        op = ArxivToMarkdownMapper(
            arxiv_id_key="id",
            backend="pdfplumber",
            output_key="markdown",
        )
        sample = {"id": "invalid_blank", "text": "x"}
        out = op.process_single(sample)
        self.assertIn("markdown", out)


if __name__ == "__main__":
    unittest.main()

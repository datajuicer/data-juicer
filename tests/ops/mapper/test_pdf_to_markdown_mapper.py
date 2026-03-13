import unittest

from data_juicer.ops.mapper.pdf_to_markdown_mapper import (
    PdfToMarkdownMapper,
    _ensure_pdf_bytes,
    _pdf_to_markdown_pdfplumber,
)
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


def _make_minimal_pdf_bytes() -> bytes:
    """Create minimal valid PDF bytes for testing."""
    # Minimal PDF with one empty page (pdfplumber can open it)
    return b"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >> endobj
xref
0 4
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
trailer << /Size 4 /Root 1 0 R >>
startxref
190
%%EOF"""


class PdfToMarkdownMapperTest(DataJuicerTestCaseBase):
    def setUp(self):
        super().setUp()
        self.op = PdfToMarkdownMapper(pdf_key="pdf", backend="pdfplumber")

    def test_ensure_pdf_bytes_from_bytes(self):
        raw = b"%PDF-1.4 dummy"
        self.assertEqual(_ensure_pdf_bytes(raw), raw)
        self.assertIsNone(_ensure_pdf_bytes(b""))

    def test_ensure_pdf_bytes_from_path(self):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"%PDF-1.4 x")
            path = f.name
        try:
            out = _ensure_pdf_bytes(path)
            self.assertIsInstance(out, bytes)
            self.assertEqual(out, b"%PDF-1.4 x")
        finally:
            import os
            os.unlink(path)

    def test_missing_pdf(self):
        sample = {"text": "original"}
        out = self.op.process_single(sample)
        self.assertEqual(out["text"], "original")

    def test_pdfplumber_empty_page(self):
        pdf_bytes = _make_minimal_pdf_bytes()
        md = _pdf_to_markdown_pdfplumber(pdf_bytes)
        # May be empty or have "Page 1" depending on pdfplumber
        self.assertIsInstance(md, str)

    def test_output_key(self):
        op = PdfToMarkdownMapper(pdf_key="content", backend="pdfplumber", output_key="md")
        sample = {"content": _make_minimal_pdf_bytes(), "text": "x"}
        out = op.process_single(sample)
        self.assertIn("md", out)
        self.assertIsInstance(out["md"], str)


if __name__ == "__main__":
    unittest.main()

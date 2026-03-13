# Copyright 2025 The DataJuicer Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Mapper to convert PDF content to high-precision structured Markdown.

Reads PDF from a field (file path or bytes), converts to a single structured
Markdown string. Supports mineru (MinerU/magic-pdf) for high precision or
pdfplumber as built-in fallback. Images can be discarded for chunking/LLM use.
"""

import io
import os
import re
import tempfile
from typing import Optional, Union

import pdfplumber
from loguru import logger

from data_juicer.utils.lazy_loader import LazyLoader

from ..base_op import OPERATORS, Mapper

magic_pdf = LazyLoader("magic_pdf", "magic-pdf")

OP_NAME = "pdf_to_markdown_mapper"


def _ensure_pdf_bytes(pdf_field_value: Union[str, bytes]) -> Optional[bytes]:
    """Return PDF bytes from path or bytes."""
    if isinstance(pdf_field_value, bytes):
        return pdf_field_value if pdf_field_value else None
    if isinstance(pdf_field_value, str) and os.path.isfile(pdf_field_value):
        try:
            with open(pdf_field_value, "rb") as f:
                return f.read()
        except Exception as e:
            logger.warning(f"Failed to read PDF file {pdf_field_value}: {e}")
            return None
    return None


def _pdf_to_markdown_pdfplumber(pdf_bytes: bytes) -> str:
    """Convert PDF bytes to plain text with minimal structure using pdfplumber."""
    try:
        with io.BytesIO(pdf_bytes) as f:
            with pdfplumber.open(f) as pdf:
                parts = []
                for i, page in enumerate(pdf.pages):
                    tables = page.find_tables()
                    for table in tables:
                        page = page.outside_bbox(table.bbox)
                    text = page.extract_text()
                    if not text:
                        continue
                    page_num = str(page.page_number)
                    if text.rstrip().endswith(page_num):
                        text = text.rstrip()[: -len(page_num)]
                    if text.strip():
                        parts.append(f"## Page {i + 1}\n\n{text.strip()}")
                return "\n\n".join(parts) if parts else ""
    except Exception as e:
        logger.warning(f"pdfplumber failed to parse PDF: {e}")
        return ""


def _pdf_to_markdown_mineru(pdf_bytes: bytes, keep_images: bool = False) -> str:
    """Convert PDF bytes to structured Markdown using MinerU (magic-pdf)."""
    try:
        from magic_pdf.data.data_reader_writer import FileBasedDataWriter
        from magic_pdf.data.dataset import PymuDocDataset
        from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
        from magic_pdf.config.enums import SupportedPdfParseMethod
    except ImportError as e:
        logger.warning(f"magic-pdf not available, fallback to pdfplumber: {e}")
        return _pdf_to_markdown_pdfplumber(pdf_bytes)

    with tempfile.TemporaryDirectory() as tmpdir:
        image_dir = os.path.join(tmpdir, "images")
        os.makedirs(image_dir, exist_ok=True)
        image_writer = FileBasedDataWriter(image_dir)

        try:
            ds = PymuDocDataset(pdf_bytes)
            if ds.classify() == SupportedPdfParseMethod.OCR:
                infer_result = ds.apply(doc_analyze, ocr=True)
                pipe_result = infer_result.pipe_ocr_mode(image_writer)
            else:
                infer_result = ds.apply(doc_analyze, ocr=False)
                pipe_result = infer_result.pipe_txt_mode(image_writer)

            image_dir_basename = "images"
            md_content = pipe_result.get_markdown(image_dir_basename)
            if not keep_images and md_content:
                md_content = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", md_content)
                md_content = re.sub(r"\n{3,}", "\n\n", md_content).strip()
            return md_content or _pdf_to_markdown_pdfplumber(pdf_bytes)
        except Exception as e:
            logger.warning(f"MinerU conversion failed, fallback to pdfplumber: {e}")
            return _pdf_to_markdown_pdfplumber(pdf_bytes)


@OPERATORS.register_module(OP_NAME)
class PdfToMarkdownMapper(Mapper):
    """Convert PDF (path or bytes) to a single structured Markdown document.

    Use with download_file_mapper (save_field as bytes or path) for URLs, or
    with datasets that already contain PDF paths/bytes. Images are dropped by
    default for easier chunking and LLM consumption.
    """

    _batched_op = False

    _requirements = ["magic-pdf"]

    def __init__(
        self,
        pdf_key: str = "pdf",
        backend: str = "mineru",
        keep_images: bool = False,
        output_key: Optional[str] = None,
        *args,
        **kwargs,
    ):
        """
        :param pdf_key: Field name containing PDF file path (str) or PDF bytes (bytes).
        :param backend: "mineru" (MinerU/magic-pdf, high precision) or "pdfplumber" (built-in).
        :param keep_images: If True, keep image references in Markdown (mineru only).
        :param output_key: Field to write Markdown to; default is text_key.
        """
        super().__init__(*args, **kwargs)
        self._init_parameters = self.remove_extra_parameters(locals())
        self.pdf_key = pdf_key
        self.backend = backend.lower()
        self.keep_images = keep_images
        self.output_key = output_key or self.text_key
        if self.backend not in ("mineru", "pdfplumber"):
            raise ValueError("backend must be one of mineru, pdfplumber")

    def process_single(self, sample):
        raw = sample.get(self.pdf_key)
        pdf_bytes = _ensure_pdf_bytes(raw)
        if not pdf_bytes:
            logger.warning(f"Missing or invalid PDF at key {self.pdf_key}")
            sample[self.output_key] = sample.get(self.text_key, "")
            return sample

        if self.backend == "mineru":
            md = _pdf_to_markdown_mineru(pdf_bytes, keep_images=self.keep_images)
        else:
            md = _pdf_to_markdown_pdfplumber(pdf_bytes)
        sample[self.output_key] = md
        return sample

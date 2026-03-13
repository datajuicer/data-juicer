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
Mapper to convert arXiv paper ID to high-precision structured Markdown.

Given an arXiv paper ID (e.g. 2501.14755 or hep-th/9901001), downloads the PDF
or crawls the HTML/abstract page, converts to a single structured Markdown string
(images can be discarded), suitable for chunking or feeding to LLM / knowledge base.
"""

import io
import re
import tempfile
import time
import xml.etree.ElementTree as ET
from typing import Optional

import pdfplumber
import requests
from loguru import logger

from data_juicer.utils.lazy_loader import LazyLoader

from ..base_op import OPERATORS, Mapper

# Optional backends for high-precision conversion
magic_pdf = LazyLoader("magic_pdf", "magic-pdf")
crawl4ai = LazyLoader("crawl4ai", "crawl4ai")

OP_NAME = "arxiv_to_markdown_mapper"

# arXiv API, PDF, and HTML URLs
ARXIV_PDF_URL_TEMPLATE = "https://arxiv.org/pdf/{arxiv_id}.pdf"
ARXIV_ABS_URL_TEMPLATE = "https://arxiv.org/abs/{arxiv_id}"
ARXIV_HTML_URL_TEMPLATE = "https://arxiv.org/html/{arxiv_id}"
ARXIV_API_QUERY = "http://export.arxiv.org/api/query?id_list={arxiv_id}"

# Default request settings
DEFAULT_TIMEOUT = 60
DEFAULT_DOWNLOAD_DELAY = 1.0  # be nice to arXiv


def _normalize_arxiv_id(raw: str) -> Optional[str]:
    """Normalize input to arXiv ID without version (e.g. 2501.14755 or hep-th/9901001)."""
    s = _normalize_arxiv_id_keep_version(raw)
    if not s:
        return None
    if re.search(r"v\d+$", s):
        s = re.sub(r"v\d+$", "", s)
    return s if s else None


def _normalize_arxiv_id_keep_version(raw: str) -> Optional[str]:
    """Normalize to arXiv ID but keep version suffix for HTML URL (e.g. 2501.14755v1)."""
    if not raw or not isinstance(raw, str):
        return None
    s = raw.strip()
    if "arxiv.org/abs/" in s:
        s = s.split("arxiv.org/abs/")[-1].split("?")[0].split("#")[0].strip("/")
    if "arxiv.org/pdf/" in s:
        s = s.split("arxiv.org/pdf/")[-1].replace(".pdf", "").split("?")[0].strip("/")
    if "arxiv.org/html/" in s:
        s = s.split("arxiv.org/html/")[-1].split("?")[0].split("#")[0].strip("/")
    return s if s else None


def _fetch_arxiv_metadata(arxiv_id: str, timeout: int = 15) -> dict:
    """Fetch title and authors from arXiv API (XML)."""
    out = {"title": "", "authors": [], "abstract": ""}
    try:
        url = ARXIV_API_QUERY.format(arxiv_id=arxiv_id)
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entry = root.find("atom:entry", ns)
        if entry is None:
            return out
        title_el = entry.find("atom:title", ns)
        if title_el is not None and title_el.text:
            out["title"] = title_el.text.strip().replace("\n", " ")
        summary_el = entry.find("atom:summary", ns)
        if summary_el is not None and summary_el.text:
            out["abstract"] = summary_el.text.strip().replace("\n", " ")
        for author in entry.findall("atom:author", ns):
            name_el = author.find("atom:name", ns)
            if name_el is not None and name_el.text:
                out["authors"].append(name_el.text.strip())
    except Exception as e:
        logger.debug(f"arXiv metadata fetch failed for {arxiv_id}: {e}")
    return out


def _download_arxiv_pdf(arxiv_id: str, timeout: int, delay: float) -> Optional[bytes]:
    """Download PDF bytes from arXiv."""
    url = ARXIV_PDF_URL_TEMPLATE.format(arxiv_id=arxiv_id)
    if delay > 0:
        time.sleep(delay)
    try:
        r = requests.get(url, timeout=timeout, stream=True)
        r.raise_for_status()
        return r.content
    except Exception as e:
        logger.error(f"Failed to download arXiv PDF {arxiv_id}: {e}")
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
        import os

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
                # Remove image markdown refs: ![](images/...) or ![alt](images/...)
                md_content = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", md_content)
                md_content = re.sub(r"\n{3,}", "\n\n", md_content).strip()
            return md_content or _pdf_to_markdown_pdfplumber(pdf_bytes)
        except Exception as e:
            logger.warning(f"MinerU conversion failed, fallback to pdfplumber: {e}")
            return _pdf_to_markdown_pdfplumber(pdf_bytes)


def _crawl_url_to_markdown(url: str, timeout: int, delay: float) -> Optional[str]:
    """Crawl a URL with crawl4ai and return raw markdown."""
    if delay > 0:
        time.sleep(delay)
    try:
        import asyncio

        async def _crawl():
            from crawl4ai import AsyncWebCrawler
            from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

            config = {"markdown_generator": DefaultMarkdownGenerator()}
            async with AsyncWebCrawler() as crawler:
                result = await crawler.arun(url, config=config)
            md = getattr(result, "markdown", None)
            if md is None:
                return None
            if hasattr(md, "raw_markdown"):
                return md.raw_markdown
            return str(md) if md else None

        return asyncio.run(_crawl())
    except Exception as e:
        logger.debug(f"crawl4ai failed for {url}: {e}")
        return None


def _arxiv_html_to_markdown_crawl4ai(arxiv_id_html: str, timeout: int, delay: float) -> Optional[str]:
    """Crawl arXiv full HTML version (e.g. https://arxiv.org/html/2501.14755v1) to Markdown."""
    url = ARXIV_HTML_URL_TEMPLATE.format(arxiv_id=arxiv_id_html)
    body = _crawl_url_to_markdown(url, timeout, delay)
    if body and len(body.strip()) > 500:
        return body
    return None


def _arxiv_abs_to_markdown_crawl4ai(arxiv_id: str, timeout: int, delay: float) -> Optional[str]:
    """Crawl arXiv abstract page and return as Markdown (abstract + metadata only)."""
    url = ARXIV_ABS_URL_TEMPLATE.format(arxiv_id=arxiv_id)
    return _crawl_url_to_markdown(url, timeout, delay)


@OPERATORS.register_module(OP_NAME)
class ArxivToMarkdownMapper(Mapper):
    """Convert arXiv paper ID to a single high-precision structured Markdown document.

    Downloads the paper PDF (or crawls the abstract page when backend is crawl4ai),
    converts to Markdown with optional metadata header. Images are dropped by default
    for smaller output and easier chunking/LLM consumption.
    """

    _batched_op = False

    _requirements = ["magic-pdf", "crawl4ai"]

    def __init__(
        self,
        arxiv_id_key: str = "arxiv_id",
        backend: str = "mineru",
        add_metadata_header: bool = True,
        keep_images: bool = False,
        timeout: int = DEFAULT_TIMEOUT,
        download_delay: float = DEFAULT_DOWNLOAD_DELAY,
        output_key: Optional[str] = None,
        *args,
        **kwargs,
    ):
        """
        :param arxiv_id_key: Field name containing arXiv ID (or URL).
        :param backend: Conversion backend: "mineru" (recommended; MinerU/magic-pdf, best quality),
            "pdfplumber" (built-in fallback, text-only), "crawl4ai" (HTML full page if available,
            else abstract page; fallback to PDF if both fail).
        :param add_metadata_header: Prepend title, authors, abstract from arXiv API.
        :param keep_images: If True, keep image references in Markdown (mineru only).
        :param timeout: Request timeout in seconds.
        :param download_delay: Delay in seconds before each download (rate limiting).
        :param output_key: Field to write Markdown to; default is text_key.
        """
        super().__init__(*args, **kwargs)
        self._init_parameters = self.remove_extra_parameters(locals())
        self.arxiv_id_key = arxiv_id_key
        self.backend = backend.lower()
        self.add_metadata_header = add_metadata_header
        self.keep_images = keep_images
        self.timeout = timeout
        self.download_delay = download_delay
        self.output_key = output_key or self.text_key
        if self.backend not in ("mineru", "pdfplumber", "crawl4ai"):
            raise ValueError(f"backend must be one of mineru, pdfplumber, crawl4ai, got {backend}")

    def process_single(self, sample):
        raw_id = sample.get(self.arxiv_id_key)
        arxiv_id = _normalize_arxiv_id(raw_id)
        if not arxiv_id:
            logger.warning(f"Missing or invalid {self.arxiv_id_key}: {raw_id}")
            sample[self.output_key] = sample.get(self.text_key, "")
            return sample
        arxiv_id_for_html = _normalize_arxiv_id_keep_version(raw_id) or arxiv_id

        md_parts = []

        if self.add_metadata_header:
            meta = _fetch_arxiv_metadata(arxiv_id, timeout=min(15, self.timeout))
            if meta["title"]:
                md_parts.append(f"# {meta['title']}\n")
            if meta["authors"]:
                md_parts.append("**Authors:** " + ", ".join(meta["authors"]) + "\n")
            md_parts.append(f"**arXiv:** {arxiv_id}  \n")
            if meta["abstract"]:
                md_parts.append("## Abstract\n\n" + meta["abstract"] + "\n")
            md_parts.append("---\n\n")

        if self.backend == "crawl4ai":
            body = _arxiv_html_to_markdown_crawl4ai(
                arxiv_id_for_html, self.timeout, self.download_delay
            )
            if not body:
                body = _arxiv_abs_to_markdown_crawl4ai(
                    arxiv_id, self.timeout, self.download_delay
                )
            if not body:
                # Use pdfplumber for fallback so crawl4ai users need not install magic-pdf
                body = _pdf_to_markdown_pdfplumber(
                    _download_arxiv_pdf(arxiv_id, self.timeout, self.download_delay) or b""
                )
            md_parts.append(body or "")
        else:
            pdf_bytes = _download_arxiv_pdf(arxiv_id, self.timeout, self.download_delay)
            if not pdf_bytes:
                sample[self.output_key] = "\n".join(md_parts).strip() if md_parts else ""
                return sample
            if self.backend == "mineru":
                body = _pdf_to_markdown_mineru(pdf_bytes, keep_images=self.keep_images)
            else:
                body = _pdf_to_markdown_pdfplumber(pdf_bytes)
            md_parts.append(body)

        sample[self.output_key] = "\n".join(md_parts).strip()
        return sample

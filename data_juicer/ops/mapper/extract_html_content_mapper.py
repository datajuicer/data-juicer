from typing import Optional

from loguru import logger

from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.lazy_loader import LazyLoader

from ..base_op import OPERATORS, TAGGING_OPS, Mapper

trafilatura = LazyLoader("trafilatura")

OP_NAME = "extract_html_content_mapper"

_OUTPUT_FORMATS = ("txt", "markdown")
_FOCUS_MODES = ("balanced", "precision", "recall")
_STR_METADATA_FIELDS = (
    "title",
    "author",
    "date",
    "url",
    "hostname",
    "sitename",
    "description",
)
_LIST_METADATA_FIELDS = ("categories", "tags")


@TAGGING_OPS.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
class ExtractHtmlContentMapper(Mapper):
    """Extracts the main content and metadata of web pages from raw HTML.

    Unlike `clean_html_mapper`, which keeps all visible text on the page, this
    operator uses Trafilatura to separate the main content from boilerplate such
    as navigation bars, headers, footers, sidebars and cookie notices. The
    extracted content replaces the HTML in the text field, either as plain text
    or as Markdown. Page metadata can optionally be stored in the sample's meta
    field. Samples where no main content can be found get an empty string, so
    they can be removed by a downstream length filter."""

    def __init__(
        self,
        output_format: str = "txt",
        focus: str = "balanced",
        include_tables: bool = True,
        include_links: bool = False,
        include_comments: bool = False,
        fast: bool = False,
        url_key: Optional[str] = None,
        extract_metadata: bool = False,
        metadata_field_name: str = MetaKeys.html_metadata,
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param output_format: Format of the extracted content, 'txt' for plain
            text or 'markdown' to keep headings, lists, tables and emphasis.
        :param focus: Extraction trade-off. 'precision' drops more uncertain
            parts to reduce noise, 'recall' keeps more of them to avoid missing
            content, and 'balanced' sits in between.
        :param include_tables: Whether to keep tables in the output text.
        :param include_links: Whether to keep link targets. Only effective
            when `output_format` is 'markdown'.
        :param include_comments: Whether to keep the user comment section of
            the page in addition to the main content.
        :param fast: Whether to skip the fallback extraction algorithms
            (jusText and readability), which is faster but may extract less
            content from unusual page layouts.
        :param url_key: Optional field name holding the page URL. The URL helps
            to resolve relative links and fills the url-related metadata.
        :param extract_metadata: Whether to extract page metadata and store it
            in the sample's meta field.
        :param metadata_field_name: Field name in meta to store the metadata.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        if output_format not in _OUTPUT_FORMATS:
            raise ValueError(f"output_format must be one of {_OUTPUT_FORMATS}, " f"got [{output_format}].")
        if focus not in _FOCUS_MODES:
            raise ValueError(f"focus must be one of {_FOCUS_MODES}, got [{focus}].")
        self.output_format = output_format
        self.focus = focus
        self.include_tables = include_tables
        self.include_links = include_links
        self.include_comments = include_comments
        self.fast = fast
        self.url_key = url_key
        self.extract_metadata = extract_metadata
        self.metadata_field_name = metadata_field_name

    def _build_options(self, url, with_metadata):
        return trafilatura.settings.Extractor(
            output_format=self.output_format,
            fast=self.fast,
            precision=self.focus == "precision",
            recall=self.focus == "recall",
            comments=self.include_comments,
            links=self.include_links,
            tables=self.include_tables,
            url=url,
            with_metadata=with_metadata,
        )

    @staticmethod
    def _render_text(document, options):
        original_with_metadata = options.with_metadata
        options.with_metadata = False
        try:
            return trafilatura.core.determine_returnstring(document, options)
        finally:
            options.with_metadata = original_with_metadata

    @staticmethod
    def _empty_metadata():
        metadata = {key: "" for key in _STR_METADATA_FIELDS}
        metadata.update({key: [] for key in _LIST_METADATA_FIELDS})
        return metadata

    @classmethod
    def _extract_metadata(cls, document):
        metadata = cls._empty_metadata()
        for key in _STR_METADATA_FIELDS:
            metadata[key] = getattr(document, key, None) or ""
        for key in _LIST_METADATA_FIELDS:
            metadata[key] = list(getattr(document, key, None) or [])
        return metadata

    def _initialize_metadata(self, sample):
        metadata = sample.get(Fields.meta)
        if not self.extract_metadata:
            return metadata, False
        if not isinstance(metadata, dict):
            metadata = {}
            sample[Fields.meta] = metadata
        write_metadata = metadata.get(self.metadata_field_name) is None
        if write_metadata:
            metadata[self.metadata_field_name] = self._empty_metadata()
        return metadata, write_metadata

    def process_single(self, sample):
        html = sample[self.text_key]
        metadata, write_metadata = self._initialize_metadata(sample)
        if not isinstance(html, str):
            logger.warning("Invalid input text! Expected a string.")
            return sample

        url = sample.get(self.url_key) if self.url_key else None
        if url is not None and not isinstance(url, str):
            logger.warning("Invalid page URL! Expected a string.")
            url = None

        options = self._build_options(url, with_metadata=write_metadata)
        document = trafilatura.bare_extraction(html, options=options, as_dict=False)
        if document is None:
            sample[self.text_key] = ""
            return sample

        if write_metadata:
            metadata[self.metadata_field_name] = self._extract_metadata(document)
        sample[self.text_key] = self._render_text(document, options)
        return sample

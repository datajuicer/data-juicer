"""
Stream local JSONL with stdlib :func:`json.loads`, skipping bad lines.

Used when HuggingFace's JSON builder (ujson) fails on some rows or when you
need per-line fault tolerance. Output is a normal :class:`datasets.Dataset`,
so downstream operators behave the same as with the default JSONL loader.
"""

from __future__ import annotations

import gzip
import io
import json
import os
from typing import Any, Dict, Iterator, List, Tuple

import zstandard as zstd
from datasets import Dataset

from data_juicer.utils.constant import Fields

# Keys produced by :func:`data_juicer.utils.file_utils.find_files_with_suffix`
JSONL_LENIENT_EXTENSIONS = frozenset({".jsonl", ".jsonl.gz", ".jsonl.zst"})


def _iter_text_lines(path: str) -> Iterator[str]:
    if path.endswith(".jsonl.gz"):
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            yield from handle
    elif path.endswith(".jsonl.zst"):
        with open(path, "rb") as compressed:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(compressed) as reader:
                text_stream = io.TextIOWrapper(
                    reader,
                    encoding="utf-8",
                    newline="",
                )
                yield from text_stream
    else:
        with open(path, "r", encoding="utf-8", newline="") as handle:
            yield from handle


def iter_lenient_jsonl_records(
    file_ext_pairs: List[Tuple[str, str]],
    *,
    add_suffix_column: bool,
) -> Iterator[Dict[str, Any]]:
    """
    Yield one dict per valid JSON object line.

    :param file_ext_pairs: ``(file_path, ext_key)`` where ``ext_key`` is the
        suffix key from ``find_files_with_suffix`` (e.g. ``\".jsonl\"``).
    :param add_suffix_column: if True, set ``Fields.suffix`` to match the
        default HF loader (``\".\" + ext_key.strip(\".\")``).
    """
    # Use stdlib logging inside generator for pickle compatibility with dill.
    # loguru Logger contains multiprocessing.Condition which cannot be pickled.
    import logging

    _logger = logging.getLogger(__name__)

    skipped = 0
    for path, ext_key in file_ext_pairs:
        if not os.path.isfile(path):
            _logger.warning(f"[lenient jsonl] missing file, skip: {path}")
            continue
        suffix_val = ("." + ext_key.strip(".")) if add_suffix_column else None
        try:
            line_iter = _iter_text_lines(path)
        except OSError as exc:
            _logger.error(f"[lenient jsonl] cannot open {path}: {exc}")
            continue
        for lineno, line in enumerate(line_iter, 1):
            chunk = line.strip()
            if not chunk:
                continue
            try:
                obj = json.loads(chunk)
            except (json.JSONDecodeError, ValueError) as exc:
                skipped += 1
                _logger.warning(f"[lenient jsonl] skip {path}:{lineno}: {exc}")
                continue
            if not isinstance(obj, dict):
                skipped += 1
                typ = type(obj).__name__
                _logger.warning(f"[lenient jsonl] skip {path}:{lineno}: " f"expected JSON object, got {typ}")
                continue
            if suffix_val is not None:
                row = dict(obj)
                row[Fields.suffix] = suffix_val
                yield row
            else:
                yield obj

    if skipped:
        _logger.info(f"[lenient jsonl] finished with {skipped} skipped line(s) " "(see warnings above)")


def dataset_from_lenient_jsonl_files(
    file_ext_pairs: List[Tuple[str, str]],
    *,
    add_suffix_column: bool,
) -> Dataset:
    """Build a :class:`datasets.Dataset` by streaming all given JSONL files."""

    def _gen(pairs, add_suffix):
        yield from iter_lenient_jsonl_records(
            pairs,
            add_suffix_column=add_suffix,
        )

    return Dataset.from_generator(
        _gen,
        gen_kwargs={
            "pairs": file_ext_pairs,
            "add_suffix": add_suffix_column,
        },
    )

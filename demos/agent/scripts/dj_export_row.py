"""Helpers for Data-Juicer jsonl exports.

Main ``processed.jsonl`` often omits ``__dj__meta__`` / ``__dj__stats__``
when ``keep_stats_in_res_ds: false``; they go to ``processed_stats.jsonl``.
Resolves meta/stats from either shape and can merge main + stats lines.
"""

from __future__ import annotations

import json
import os
from itertools import zip_longest
from typing import Any, Dict, Iterator, Optional, Tuple

# Match data_juicer.utils.constant.Fields (no package import in demos scripts)
DJ_META = "__dj__meta__"
DJ_STATS = "__dj__stats__"


def _parse_if_string(val: Any) -> Any:
    if isinstance(val, str):
        try:
            return json.loads(val)
        except json.JSONDecodeError:
            return val
    return val


def get_dj_meta(row: Dict[str, Any]) -> Dict[str, Any]:
    for key in (DJ_META, "meta"):
        m = row.get(key)
        m = _parse_if_string(m)
        if isinstance(m, dict):
            return m
    return {}


def get_dj_stats(row: Dict[str, Any]) -> Dict[str, Any]:
    for key in (DJ_STATS, "stats"):
        s = row.get(key)
        s = _parse_if_string(s)
        if isinstance(s, dict):
            return s
    return {}


def infer_stats_jsonl_path(main_export_path: str) -> str:
    """``foo.jsonl`` -> ``foo_stats.jsonl`` (Data-Juicer exporter)."""
    base, ext = os.path.splitext(main_export_path)
    if ext.lower() == ".jsonl":
        return f"{base}_stats.jsonl"
    return main_export_path + "_stats"


def merge_main_with_stats_line(
    main_row: Dict[str, Any],
    stats_row: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Copy ``__dj__meta__`` / ``__dj__stats__`` from stats line if main lacks them."""
    if not stats_row:
        return main_row
    out = dict(main_row)
    if DJ_META not in out and DJ_META in stats_row:
        out[DJ_META] = _parse_if_string(stats_row[DJ_META])
        if not isinstance(out[DJ_META], dict):
            out[DJ_META] = {}
    if DJ_STATS not in out and DJ_STATS in stats_row:
        out[DJ_STATS] = _parse_if_string(stats_row[DJ_STATS])
        if not isinstance(out[DJ_STATS], dict):
            out[DJ_STATS] = {}
    return out


def iter_jsonl_lines(path: str) -> Iterator[Tuple[int, Dict[str, Any]]]:
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield i, json.loads(line)
            except json.JSONDecodeError:
                continue


def iter_merged_export_rows(
    main_path: str,
) -> Iterator[Tuple[int, Dict[str, Any]]]:
    """Yield (line_no, row); merge ``*_stats.jsonl`` when main omits meta."""
    stats_path = infer_stats_jsonl_path(main_path)
    if not os.path.isfile(stats_path):
        for i, row in iter_jsonl_lines(main_path):
            yield i, row
        return

    with open(main_path, encoding="utf-8") as fm, open(
        stats_path,
        encoding="utf-8",
    ) as fs:
        for line_no, (l1, l2) in enumerate(
            zip_longest(fm, fs, fillvalue=""),
            start=1,
        ):
            l1 = (l1 or "").strip()
            if not l1:
                continue
            try:
                main = json.loads(l1)
            except json.JSONDecodeError:
                continue
            stats_row: Optional[Dict[str, Any]] = None
            l2 = (l2 or "").strip()
            if l2:
                try:
                    stats_row = json.loads(l2)
                except json.JSONDecodeError:
                    stats_row = None
            merged = merge_main_with_stats_line(main, stats_row)
            yield line_no, merged

# Copyright 2025 The Data-Juicer Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Deduplicates conversation samples at the session level.
# Groups samples by (session_id, user_id) and keeps only the one with the latest timestamp.
# Supports both direct field extraction (with nested path support) and regex extraction.

import re
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from loguru import logger

from data_juicer.utils.constant import Fields, HashKeys

from ..base_op import OPERATORS, Deduplicator

OP_NAME = "agent_session_deduplicator"


def _extract_nested_field(row: dict, field_path: str) -> Any:
    """Extract value from nested field path (e.g. 'meta.session_id', 'messages.0.content')."""
    if not field_path:
        return None
    keys = field_path.split(".")
    val = row
    for key in keys:
        if isinstance(val, dict):
            val = val.get(key)
        elif isinstance(val, (list, tuple)):
            try:
                val = val[int(key)]
            except (ValueError, IndexError):
                return None
        else:
            return None
        if val is None:
            return None
    return val


@OPERATORS.register_module(OP_NAME)
class AgentSessionDeduplicator(Deduplicator):
    """Deduplicates conversation samples at the session level to keep the latest ones.

    Groups samples by (session_id, user_id) and keeps only the sample with the latest
    timestamp within each group. Both ``session_id`` and ``user_id`` must be non-empty
    for a sample to participate in deduplication; otherwise the sample is always kept.

    Field extraction supports two modes:

    - ``field`` (default): extract from existing fields with nested path support
      (e.g. ``"session_id"``, ``"meta.session_id"``, ``"messages.0.content"``).
    - ``regex_extract``: extract via regex from a text field. The regex must contain
      named groups ``(?P<session_id>...)`` and ``(?P<user_id>...)``.

    The operator writes an ``agent_session_dedup_info`` dict into ``meta`` for each
    sample, recording the group key, group size, and whether the sample was kept.
    """

    def __init__(
        self,
        group_key_mode: str = "field",
        session_id_field: str = "session_id",
        user_id_field: str = "user_id",
        timestamp_field: str = "timestamp",
        regex_pattern: str = "",
        regex_search_target: str = "",
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param group_key_mode: field extraction mode, ``"field"`` or ``"regex_extract"``.
        :param session_id_field: field path for session id (used when mode is ``"field"``).
        :param user_id_field: field path for user id (used when mode is ``"field"``).
        :param timestamp_field: field path for timestamp; larger = newer.
        :param regex_pattern: regex with named groups ``session_id`` and ``user_id``
            (used when mode is ``"regex_extract"``).
        :param regex_search_target: field path to search the regex in, or the shortcut
            ``"first_system_message"`` to search the first message with role ``"system"``.
        :param args: extra args
        :param kwargs: extra args.
        """
        super().__init__(*args, **kwargs)

        mode = (group_key_mode or "field").strip().lower()
        if mode not in ("field", "regex_extract"):
            mode = "field"
        self.group_key_mode = mode

        self.session_id_field = str(session_id_field or "").strip()
        self.user_id_field = str(user_id_field or "").strip()
        self.timestamp_field = str(timestamp_field or "timestamp").strip()

        self.regex_pattern = str(regex_pattern or "").strip()
        if self.group_key_mode == "regex_extract" and not self.regex_pattern:
            logger.warning(
                f"{OP_NAME}: group_key_mode='regex_extract' but regex_pattern is empty; "
                "falling back to 'field' mode."
            )
            self.group_key_mode = "field"

        self.regex_search_target = str(regex_search_target or "").strip()
        if self.regex_search_target:
            self._regex_re = re.compile(self.regex_pattern)
        else:
            self._regex_re = None

    def _extract_ids(self, sample) -> Tuple[Optional[str], Optional[str]]:
        """Return (session_id, user_id) from a sample."""
        if self.group_key_mode == "regex_extract":
            return self._extract_ids_regex(sample)

        sid = _extract_nested_field(sample, self.session_id_field)
        uid = _extract_nested_field(sample, self.user_id_field)
        sid_str = str(sid).strip() if sid is not None else ""
        uid_str = str(uid).strip() if uid is not None else ""
        return sid_str if sid_str else None, uid_str if uid_str else None

    def _extract_ids_regex(self, sample) -> Tuple[Optional[str], Optional[str]]:
        if self._regex_re is None:
            return None, None

        if self.regex_search_target == "first_system_message":
            messages = sample.get("messages") or sample.get(self.text_key, [])
            if isinstance(messages, list):
                first_sys = next((m for m in messages if isinstance(m, dict) and m.get("role") == "system"), None)
                text = first_sys.get("content", "") if first_sys else ""
            else:
                text = ""
        else:
            raw = _extract_nested_field(sample, self.regex_search_target)
            text = str(raw) if raw is not None else ""

        if not text:
            return None, None

        match = self._regex_re.search(text)
        if not match:
            return None, None
        sid = match.group("session_id")
        uid = match.group("user_id")
        return sid.strip() if sid else None, uid.strip() if uid else None

    def _get_timestamp(self, sample) -> float:
        """Extract timestamp as float; supports numeric timestamps and date strings.

        Handles:
        - Numeric timestamps (int/float): e.g., 1711152628.73
        - Date strings: e.g., "2026-03-23 00:30:28.73"

        Returns -inf if unavailable or unparseable.
        """
        raw = _extract_nested_field(sample, self.timestamp_field)
        if raw is None:
            return float("-inf")

        # Case 1: Already numeric (int or float)
        if isinstance(raw, (int, float)):
            return float(raw)

        # Case 2: String - try to parse as numeric first, then as date
        if isinstance(raw, str):
            raw = raw.strip()
            if not raw:
                return float("-inf")

            # Try numeric string
            try:
                return float(raw)
            except (TypeError, ValueError):
                pass

            # Try common date formats
            date_formats = [
                "%Y-%m-%d %H:%M:%S.%f",  # 2026-03-23 00:30:28.73
                "%Y-%m-%d %H:%M:%S",  # 2026-03-23 00:30:28
                "%Y-%m-%dT%H:%M:%S.%f",  # ISO format with microseconds
                "%Y-%m-%dT%H:%M:%S",  # ISO format without timezone
                "%Y-%m-%d",  # Date only
            ]

            for fmt in date_formats:
                try:
                    dt = datetime.strptime(raw, fmt)
                    return dt.timestamp()
                except ValueError:
                    continue

        return float("-inf")

    def compute_hash(self, sample):
        """
        Compute a group key representing the (session_id, user_id) pair.

        :param sample: input sample
        :return: sample with group key stored in ``HashKeys.agent_session_group``.
        """
        # check if it's computed already
        if HashKeys.agent_session_group in sample:
            return sample

        sid, uid = self._extract_ids(sample)
        if sid and uid:
            group_key = f"{sid}::{uid}"
        else:
            # No valid pair: mark as ungrouped so it is always kept
            group_key = ""

        sample[HashKeys.agent_session_group] = group_key
        return sample

    def process(self, dataset, show_num=0):
        """
        For doc-level, dataset --> dataset.

        Within each (session_id, user_id) group, keeps only the sample with the largest
        timestamp. Writes ``meta.agent_session_dedup_info`` for tracing.

        Uses a two-pass approach:
        1. First pass: find the index of the newest sample in each group.
        2. Second pass: filter to keep only those indices.

        :param dataset: input dataset
        :param show_num: number of traced samples used when tracer is open.
        :return: deduplicated dataset and sampled duplicate groups.
        """
        if len(dataset) <= 1:
            return dataset, {}

        # === Pass 1: Find the newest sample index for each group ===
        best_idx_by_group: Dict[str, Tuple[int, float]] = {}  # group_key -> (idx, timestamp)
        group_sizes: Dict[str, int] = defaultdict(int)

        for idx, sample in enumerate(dataset):
            group_key = sample.get(HashKeys.agent_session_group, "") or ""
            ts = self._get_timestamp(sample)

            if not group_key:
                # Ungrouped samples: always kept, no tracking needed
                continue

            group_sizes[group_key] += 1

            if group_key not in best_idx_by_group or ts > best_idx_by_group[group_key][1]:
                best_idx_by_group[group_key] = (idx, ts)

        # Build the set of indices to keep
        kept_indices = {idx for idx, ts in best_idx_by_group.values()}

        # Identify groups with duplicates for tracing
        dup_groups_to_show = None
        if show_num > 0:
            dup_groups = {g for g, size in group_sizes.items() if size > 1}
            if dup_groups:
                sorted_dups = sorted(dup_groups, key=lambda g: group_sizes[g], reverse=True)
                dup_groups_to_show = set(sorted_dups[:show_num])

        dup_groups = {g: [] for g in (dup_groups_to_show or [])}

        # Collect dup pairs for tracing (second pass over dataset)
        if dup_groups_to_show:
            for idx, sample in enumerate(dataset):
                group_key = sample.get(HashKeys.agent_session_group, "") or ""
                if group_key in dup_groups_to_show and len(dup_groups[group_key]) < 2:
                    dup_groups[group_key].append(sample)

        # === Pass 2: Select only the kept indices ===
        # Collect all indices that should be kept (including ungrouped samples)
        all_kept_indices = []
        for idx, sample in enumerate(dataset):
            group_key = sample.get(HashKeys.agent_session_group, "") or ""
            if not group_key:
                all_kept_indices.append(idx)  # Ungrouped always kept
            elif idx in kept_indices:
                all_kept_indices.append(idx)

        # Use select() instead of filter() for direct index-based selection
        dataset = dataset.select(all_kept_indices)

        # Write dedup info to meta for kept samples (post-filter)
        def _add_dedup_info(sample):
            group_key = sample.get(HashKeys.agent_session_group, "") or ""
            ts = self._get_timestamp(sample)

            meta = sample.get(Fields.meta, {})
            if isinstance(meta, str):
                import json

                try:
                    meta = json.loads(meta)
                    sample[Fields.meta] = meta
                except json.JSONDecodeError:
                    meta = {}
                    sample[Fields.meta] = meta
            elif not isinstance(meta, dict):
                meta = {}
                sample[Fields.meta] = meta

            if group_key and group_key in group_sizes:
                meta["agent_session_dedup_info"] = {
                    "group_key": group_key,
                    "group_size": group_sizes[group_key],
                    "is_kept": True,
                    "timestamp": ts,
                }
            return sample

        if kept_indices:  # Only add info if there was actual deduplication
            dataset = dataset.map(_add_dedup_info, desc="add_dedup_info")

        return dataset, dup_groups

# Copyright 2025 The Data-Juicer Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Extract token usage from agent response (choices, usage, response_metadata).
# Multi-format: OpenAI, Anthropic, generic usage objects.

from typing import Any, List, Set, Tuple

from data_juicer.ops.base_op import OPERATORS, TAGGING_OPS, Mapper
from data_juicer.utils.constant import Fields, MetaKeys

OP_NAME = "usage_counter_mapper"


def _usage_dedup_key(u: dict) -> Tuple[int, int, int]:
    """Stable key so duplicate copies of the same usage (e.g. top-level + choice) merge."""
    try:
        p = int(u.get("prompt_tokens") or 0)
    except (TypeError, ValueError):
        p = 0
    try:
        c = int(u.get("completion_tokens") or 0)
    except (TypeError, ValueError):
        c = 0
    raw_t = u.get("total_tokens")
    if raw_t is not None:
        try:
            t = int(raw_t)
        except (TypeError, ValueError):
            t = p + c
    else:
        t = p + c
    return (p, c, t)


def _dedupe_usages_preserve_order(usages: List[dict]) -> List[dict]:
    seen: Set[Tuple[int, int, int]] = set()
    out: List[dict] = []
    for u in usages:
        key = _usage_dedup_key(u)
        if key in seen:
            continue
        seen.add(key)
        out.append(u)
    return out


def _get_usage_from_obj(obj: Any) -> dict:
    """Extract prompt_tokens, completion_tokens, total_tokens from a dict.
    Supports: nested obj.usage / obj.usage_metadata, or obj as the usage dict.
    """
    if not isinstance(obj, dict):
        return {}
    usage = obj.get("usage") or obj.get("usage_metadata") or {}
    if not isinstance(usage, dict):
        usage = {}
    # Top-level usage (e.g. response_usage with prompt_tokens directly)
    if usage and (usage.get("prompt_tokens") is not None or usage.get("completion_tokens") is not None):
        pass
    elif obj.get("prompt_tokens") is not None or obj.get("completion_tokens") is not None:
        usage = obj
    p = usage.get("prompt_tokens") or usage.get("input_tokens", 0)
    c = usage.get("completion_tokens") or usage.get("output_tokens", 0)
    return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": usage.get("total_tokens")}


def _aggregate_usage(usages: List[dict]) -> tuple:
    """Sum prompt/completion; total if present else prompt+completion."""
    p = sum(u.get("prompt_tokens") or 0 for u in usages)
    c = sum(u.get("completion_tokens") or 0 for u in usages)
    totals = [u.get("total_tokens") for u in usages if u.get("total_tokens") is not None]
    t = totals[0] if totals else None
    if t is None and (p or c):
        t = p + c
    return p, c, t


@TAGGING_OPS.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
class UsageCounterMapper(Mapper):
    """Write token usage to meta from choices/usage (OpenAI/Anthropic-style).

    Collects every non-empty usage dict found (top-level ``usage_key``,
    ``response_metadata``, each ``choices[]`` entry, nested message usage).
    By default, **deduplicates** identical usage snapshots before summing: same
    ``(prompt_tokens, completion_tokens, total_tokens or prompt+completion)``
    only counts once (typical when ``response_usage`` mirrors ``choices[0].usage``).
    Set ``dedupe_identical_usage: false`` to restore legacy double-counting.
    """

    def __init__(
        self,
        choices_key: str = "choices",
        usage_key: str = "usage",
        response_metadata_key: str = "response_metadata",
        dedupe_identical_usage: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.choices_key = choices_key
        self.usage_key = usage_key
        self.response_metadata_key = response_metadata_key
        self.dedupe_identical_usage = bool(dedupe_identical_usage)

    def process_single(self, sample):
        usages = []

        # Top-level usage (e.g. usage / response_usage)
        if self.usage_key in sample:
            u = _get_usage_from_obj(sample.get(self.usage_key)) or _get_usage_from_obj(sample)
            if u:
                usages.append(u)

        # response_metadata.usage
        meta = sample.get(self.response_metadata_key) or {}
        if isinstance(meta, dict):
            u = _get_usage_from_obj(meta)
            if u:
                usages.append(u)

        # choices[].usage or choices[].message.usage
        choices = sample.get(self.choices_key) or []
        if isinstance(choices, list):
            for c in choices:
                if not isinstance(c, dict):
                    continue
                u = _get_usage_from_obj(c)
                if u:
                    usages.append(u)
                msg = c.get("message") or c.get("delta")
                if isinstance(msg, dict):
                    u = _get_usage_from_obj(msg)
                    if u:
                        usages.append(u)

        if self.dedupe_identical_usage:
            usages = _dedupe_usages_preserve_order(usages)

        if Fields.meta not in sample:
            sample[Fields.meta] = {}
        meta = sample[Fields.meta]
        p, c, t = _aggregate_usage(usages)
        meta[MetaKeys.prompt_tokens] = p
        meta[MetaKeys.completion_tokens] = c
        meta[MetaKeys.total_tokens] = t
        return sample

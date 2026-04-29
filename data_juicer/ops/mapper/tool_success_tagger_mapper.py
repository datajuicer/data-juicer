# Copyright 2025 The Data-Juicer Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Tag tool call success/failure from messages (role=tool content).
# Configurable success/error patterns for multi-agent and multi-language.

import json
import re
from typing import Any, List, Optional

from data_juicer.ops.base_op import OPERATORS, TAGGING_OPS, Mapper
from data_juicer.utils.constant import Fields, MetaKeys

OP_NAME = "tool_success_tagger_mapper"

# Default: success = Wrote/Success/OK; error = Error/Exception/failed
DEFAULT_SUCCESS_PATTERNS = [
    r"(?i)wrote\s+\d+\s*bytes",
    r"(?i)^(?:success|ok)(?:\s|$)",
    r"(?i)\bsuccessfully\b",
    r"(?i)^ok$",
]
DEFAULT_ERROR_PATTERNS = [
    r"(?i)\berror\s*:",
    r"(?i)\bexception\b",
    r"(?i)\bfailed\b",
    r"(?i)\bfailure\b",
    r"(?i)not\s+found",
    r"(?i)permission\s+denied",
    # Chinese agent/tool logs (avoid bare「失败」— matches「未失败」等)
    r"(?:执行|操作|请求|调用|接口)失败",
    r"(?:发生|出现)错误",
    r"错误码",
    r"无法访问",
    r"权限不足",
    r"无权限",
    r"找不到(?:文件|路径)?",
]


def _content_to_str(content: Any) -> str:
    """Normalize tool message content to string.

    Some runtimes return JSON objects or multimodal lists; regex classifiers
    need a stable string form.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        try:
            return json.dumps(content, ensure_ascii=False)[:10000]
        except (TypeError, ValueError):
            return str(content).strip()
    if isinstance(content, list):
        if not content:
            return ""
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                t = block.get("text")
                if isinstance(t, str):
                    parts.append(t.strip())
                elif isinstance(t, dict):
                    parts.append(_content_to_str(t))
                else:
                    parts.append(_content_to_str(t))
            elif isinstance(block, str):
                parts.append(block.strip())
            elif isinstance(block, dict):
                parts.append(_content_to_str(block))
        return "\n".join(parts).strip()
    return str(content).strip()


def _classify_tool_content(
    content: str,
    success_patterns: List[re.Pattern],
    error_patterns: List[re.Pattern],
) -> str:
    """Return 'success', 'error', or 'unknown'."""
    if not content:
        return "unknown"
    for pat in error_patterns:
        if pat.search(content):
            return "error"
    for pat in success_patterns:
        if pat.search(content):
            return "success"
    # Non-empty content without error often means success (e.g. tool returned data)
    return "success"


@TAGGING_OPS.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
class ToolSuccessTaggerMapper(Mapper):
    """Set meta tool_success_count, tool_fail_count, tool_success_ratio.

    Scans messages for role=tool; configurable success/error patterns.
    """

    def __init__(
        self,
        messages_key: str = "messages",
        tool_role_names: Optional[List[str]] = None,
        success_patterns: Optional[List[str]] = None,
        error_patterns: Optional[List[str]] = None,
        store_per_tool_results: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.messages_key = messages_key
        self.tool_role_names = tool_role_names or ["tool", "tool_use"]
        self.store_per_tool_results = store_per_tool_results
        self._success_pats = [re.compile(p) for p in (success_patterns or DEFAULT_SUCCESS_PATTERNS)]
        self._error_pats = [re.compile(p) for p in (error_patterns or DEFAULT_ERROR_PATTERNS)]

    def process_single(self, sample):
        messages = sample.get(self.messages_key) or []
        if not isinstance(messages, list):
            messages = []

        results = []
        success_count = 0
        fail_count = 0
        unknown_count = 0

        for m in messages:
            role = (m.get("role") or "").lower()
            if role not in self.tool_role_names:
                continue
            content = _content_to_str(m.get("content"))
            label = _classify_tool_content(content, self._success_pats, self._error_pats)
            results.append({"content_preview": content[:200], "result": label})
            if label == "success":
                success_count += 1
            elif label == "error":
                fail_count += 1
            else:
                unknown_count += 1

        total = success_count + fail_count
        ratio = (success_count / total) if total else None

        if Fields.meta not in sample:
            sample[Fields.meta] = {}
        meta = sample[Fields.meta]
        meta[MetaKeys.tool_success_count] = success_count
        meta[MetaKeys.tool_fail_count] = fail_count
        meta[MetaKeys.tool_unknown_count] = unknown_count
        meta[MetaKeys.tool_success_ratio] = ratio
        if self.store_per_tool_results:
            meta[MetaKeys.tool_results] = results
        return sample

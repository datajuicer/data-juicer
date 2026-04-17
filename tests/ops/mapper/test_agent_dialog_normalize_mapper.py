# Copyright 2025 The Data-Juicer Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for agent_dialog_normalize_mapper multi-step tool / assistant chains."""

from data_juicer.ops.mapper.agent_dialog_normalize_mapper import (
    AgentDialogNormalizeMapper,
    _list_str_for_hf_meta,
    _messages_to_history,
)
from data_juicer.utils.constant import Fields, MetaKeys


def test_messages_to_history_accumulates_multiple_assistant_turns():
    """Same user turn: assistant → tool → assistant must not drop the first leg."""
    messages = [
        {"role": "user", "content": "List the folder"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"type": "function", "function": {"name": "ls", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "content": "Error: not found"},
        {"role": "assistant", "content": "Retrying with another path."},
        {"role": "tool", "content": "file.txt"},
        {"role": "assistant", "content": "Found file.txt."},
    ]
    history = _messages_to_history(messages)
    assert len(history) == 1
    q, r = history[0]
    assert q == "List the folder"
    assert "[Tool calls:" in r
    assert "Retrying with another path." in r
    assert "Found file.txt." in r
    assert "[Tool result]" in r
    assert "not found" in r


def test_messages_to_history_tool_result_head_tail_cap():
    """Long tool bodies: cap preserves prefix/suffix and records compression."""
    head = "ERROR: upstream timeout\n"
    mid = "x" * 5000
    tail = "\nstack: final line"
    payload = head + mid + tail
    messages = [
        {"role": "user", "content": "run"},
        {"role": "assistant", "content": "ok", "tool_calls": []},
        {"role": "tool", "content": payload},
    ]
    flag = [False]
    history = _messages_to_history(
        messages,
        history_tool_result_max_chars=800,
        compressed_ref=flag,
    )
    assert flag[0] is True
    _q, r = history[0]
    assert head.strip() in r or head[:20] in r
    assert "final line" in r
    assert "omitted from middle" in r


def test_process_single_sets_meta_when_history_compressed():
    op = AgentDialogNormalizeMapper(
        text_key="text",
        history_key="dialog_history",
        query_key="query",
        response_key="response",
        history_tool_result_max_chars=120,
    )
    long_tool = "A" * 400
    sample = {
        "id": "1",
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "calling"},
            {"role": "tool", "content": long_tool},
        ],
    }
    out = op.process_single(sample)
    assert out[Fields.meta][MetaKeys.agent_dialog_history_compressed] is True
    rsp = out["response"]
    assert "omitted from middle" in rsp or "truncated" in rsp


def test_list_str_for_hf_meta_empty_uses_string_placeholder():
    assert _list_str_for_hf_meta([]) == [""]
    assert _list_str_for_hf_meta(["", None, "  "]) == [""]
    assert _list_str_for_hf_meta(["a", "b"]) == ["a", "b"]


def test_process_single_always_sets_dialog_history_compressed_bool():
    op = AgentDialogNormalizeMapper(
        text_key="text",
        history_key="dialog_history",
        query_key="query",
        response_key="response",
    )
    sample = {"id": "1", "messages": [{"role": "user", "content": "hi"}]}
    out = op.process_single(sample)
    assert out[Fields.meta][MetaKeys.agent_dialog_history_compressed] is False
    assert out[Fields.meta][MetaKeys.agent_tool_types] == [""]
    assert out[Fields.meta][MetaKeys.agent_skill_types] == [""]

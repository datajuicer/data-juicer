# Copyright 2025 The Data-Juicer Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

from data_juicer.ops.mapper.agent_dialog_normalize_mapper import (
    AgentDialogNormalizeMapper,
    _choices_to_text,
    _compress_head_tail,
    _content_to_text,
    _extract_skill_types,
    _extract_tool_types,
    _first_non_empty_str,
    _flatten_history_to_text,
    _get_tool_name_from_call,
    _last_user_assistant_msg_indices,
    _list_str_for_hf_meta,
    _messages_to_history,
    _tool_calls_summary,
)
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class AgentDialogNormalizeMapperTest(DataJuicerTestCaseBase):

    def test_content_to_text_handles_multimodal_nested_and_reasoning_blocks(self):
        content = [
            {"type": "text", "text": {"value": " hello "}},
            {"type": "input_text", "text": [" world ", {"content": "!"}]},
            {"type": "thinking", "thinking": {"text": "plan"}},
            {"reasoning_content": ["step", {"value": "two"}]},
            {"type": "image_url", "image_url": "ignored"},
            " tail ",
        ]

        self.assertEqual(_content_to_text(content), "hello\nworld\n!\nplan\nstep\ntwo\ntail")
        self.assertEqual(_content_to_text({"content": {"text": "dict text"}}), "dict text")
        self.assertEqual(_content_to_text(None), "")
        self.assertEqual(_content_to_text(7), "7")

    def test_tool_name_and_summary_support_multiple_tool_formats(self):
        calls = [
            {"function": {"name": "search"}},
            {"function_call": {"name": "read_file"}},
            {"name": "search"},
            {"name": "write_file"},
            {"name": "extra"},
        ]

        self.assertEqual(_get_tool_name_from_call(calls[0]), "search")
        self.assertEqual(_get_tool_name_from_call(calls[1]), "read_file")
        self.assertIsNone(_get_tool_name_from_call("not a dict"))
        self.assertEqual(_tool_calls_summary(calls, max_names=2), "[Tool calls: search, read_file, ...+2]")
        self.assertEqual(_tool_calls_summary([], max_names=2), "")

    def test_messages_to_history_accumulates_multiple_assistant_turns(self):
        messages = [
            {"role": "user", "content": "List the folder"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"type": "function", "function": {"name": "ls", "arguments": "{}"}}],
            },
            {"role": "tool", "content": "Error: not found"},
            {"role": "assistant", "content": "Retrying with another path."},
            {"role": "tool", "content": "file.txt"},
            {"role": "assistant", "content": "Found file.txt."},
        ]

        history = _messages_to_history(messages)

        self.assertEqual(len(history), 1)
        q, r = history[0]
        self.assertEqual(q, "List the folder")
        self.assertIn("[Tool calls:", r)
        self.assertIn("Retrying with another path.", r)
        self.assertIn("Found file.txt.", r)
        self.assertIn("[Tool result]", r)
        self.assertIn("not found", r)

    def test_messages_to_history_system_lone_assistant_tool_and_caps(self):
        flag = [False]
        history = _messages_to_history(
            [
                {"role": "system", "content": "You are terse."},
                {"role": "user", "content": "U" * 500},
                {"role": "assistant", "content": "A" * 500},
                {"role": "tool", "content": "T" * 500},
                {"role": "assistant", "content": "standalone assistant"},
                {"role": "tool", "content": "orphan tool"},
            ],
            include_system_in_first_user=True,
            history_tool_result_max_chars=120,
            history_max_assistant_trace_chars=180,
            history_max_user_chars=140,
            compressed_ref=flag,
        )

        self.assertTrue(flag[0])
        self.assertEqual(len(history), 1)
        self.assertIn("You are terse.", history[0][0])
        self.assertLessEqual(len(history[0][0]), 140)
        self.assertLessEqual(len(history[0][1]), 180)
        self.assertTrue("omitted" in history[0][1] or "truncated" in history[0][1])

    def test_messages_to_history_tool_result_head_tail_cap(self):
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

        history = _messages_to_history(messages, history_tool_result_max_chars=800, compressed_ref=flag)

        self.assertTrue(flag[0])
        _q, r = history[0]
        self.assertTrue(head.strip() in r or head[:20] in r)
        self.assertIn("final line", r)
        self.assertIn("omitted from middle", r)

    def test_compress_head_tail_small_budget_uses_short_marker(self):
        text = "abcdef" * 100
        out = _compress_head_tail(text, max_chars=80)

        self.assertLessEqual(len(out), 120)
        self.assertTrue(out.startswith("abcdef"))
        self.assertIn("truncated", out)

    def test_extract_tool_and_skill_types_are_unique_and_ordered(self):
        messages = [
            {
                "role": "assistant",
                "content": "Check tools/search/SKILL.md\n## browser.search\n## 123ignore",
                "tool_calls": [{"function": {"name": "search"}}, {"name": "search"}, {"name": "read"}],
            },
            {"role": "assistant", "content": "More docs: c:\\agents\\review\\SKILL.md\n## review"},
        ]

        self.assertEqual(_extract_tool_types(messages), ["search", "read"])
        self.assertEqual(_extract_skill_types(messages), ["search", "browser.search", "review"])

    def test_choices_indices_first_non_empty_and_flatten_helpers(self):
        messages = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            "bad",
        ]
        choices = [
            {"message": {"content": ""}},
            {"delta": {"content": [{"type": "text", "text": "from delta"}]}},
        ]

        self.assertEqual(_last_user_assistant_msg_indices(messages), (3, 2))
        self.assertEqual(_first_non_empty_str({"a": " ", "b": 3}, ["a", "b"]), "3")
        self.assertIsNone(_first_non_empty_str({"a": None}, ["a"]))
        self.assertEqual(_choices_to_text(choices), "from delta")
        self.assertEqual(_choices_to_text([{"message": {"content": " final "}}]), "final")
        self.assertEqual(_choices_to_text("bad"), "")
        self.assertEqual(
            _flatten_history_to_text([("q", "r"), ("q2", "")], "Human", "Bot"),
            "Human: q\n\nBot: r\n\nHuman: q2",
        )

    def test_list_str_for_hf_meta_empty_uses_string_placeholder(self):
        self.assertEqual(_list_str_for_hf_meta([]), [""])
        self.assertEqual(_list_str_for_hf_meta(["", None, "  "]), [""])
        self.assertEqual(_list_str_for_hf_meta(["a", " b "]), ["a", "b"])

    def test_process_single_populates_text_history_meta_lineage_and_choices_fallback(self):
        op = AgentDialogNormalizeMapper(
            text_key="text",
            history_key="dialog_history",
            query_key="query",
            response_key="response",
            include_system_in_first_user=True,
            user_label="Human",
            assistant_label="Bot",
        )
        sample = {
            "id": "root-id",
            "request_id": " req-1 ",
            "request_model": "m1",
            "pt": "2026-06-29",
            "total_cost_time": 321,
            "messages": [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "search"}}]},
                {"role": "tool", "content": "ok"},
                {"role": "user", "content": "final question"},
            ],
            "choices": [{"message": {"content": "final answer"}}],
        }

        out = op.process_single(sample)
        meta = out[Fields.meta]

        self.assertEqual(out["query"], "final question")
        self.assertEqual(out["response"], "final answer")
        self.assertEqual(out["dialog_history"][0][0], "system prompt\n\nquestion")
        self.assertIn("Human: system prompt", out["text"])
        self.assertIn("Bot: [Tool calls: search]", out["text"])
        self.assertFalse(meta[MetaKeys.agent_dialog_history_compressed])
        self.assertEqual(meta[MetaKeys.agent_turn_count], 2)
        self.assertEqual(meta[MetaKeys.agent_tool_types], ["search"])
        self.assertEqual(meta[MetaKeys.agent_skill_types], [""])
        self.assertEqual(meta[MetaKeys.agent_last_user_msg_idx], 4)
        self.assertEqual(meta[MetaKeys.agent_last_assistant_msg_idx], 2)
        self.assertEqual(meta[MetaKeys.agent_request_id], "req-1")
        self.assertEqual(meta[MetaKeys.agent_request_model], "m1")
        self.assertEqual(meta[MetaKeys.agent_pt], "2026-06-29")
        self.assertEqual(meta[MetaKeys.agent_total_cost_time_ms], 321)

    def test_process_single_sets_compression_bool_and_can_disable_tag_extraction(self):
        op = AgentDialogNormalizeMapper(
            text_key="text",
            history_key="dialog_history",
            query_key="query",
            response_key="response",
            extract_tool_skill_tags=False,
            copy_lineage_fields=False,
            copy_request_id=False,
            history_tool_result_max_chars=120,
        )
        sample = {
            "id": "1",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "calling"},
                {"role": "tool", "content": "A" * 400},
            ],
        }

        out = op.process_single(sample)

        self.assertTrue(out[Fields.meta][MetaKeys.agent_dialog_history_compressed])
        self.assertNotIn(MetaKeys.agent_tool_types, out[Fields.meta])
        self.assertNotIn(MetaKeys.agent_skill_types, out[Fields.meta])
        self.assertNotIn(MetaKeys.agent_request_id, out[Fields.meta])
        self.assertTrue("omitted from middle" in out["response"] or "truncated" in out["response"])

    def test_process_single_non_list_messages_still_sets_stable_meta_defaults(self):
        op = AgentDialogNormalizeMapper()

        out = op.process_single({"messages": "bad"})

        self.assertEqual(out["text"], "")
        self.assertEqual(out["dialog_history"], [])
        self.assertEqual(out["query"], "")
        self.assertEqual(out["response"], "")
        self.assertEqual(out[Fields.meta][MetaKeys.agent_dialog_history_compressed], False)
        self.assertEqual(out[Fields.meta][MetaKeys.agent_tool_types], [""])
        self.assertEqual(out[Fields.meta][MetaKeys.agent_skill_types], [""])


if __name__ == "__main__":
    unittest.main()

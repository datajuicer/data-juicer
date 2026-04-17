# Copyright 2025 The Data-Juicer Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import patch

from data_juicer.ops.mapper.agent_tool_relevance_mapper import AgentToolRelevanceMapper
from data_juicer.ops.mapper.agent_trace_coherence_mapper import AgentTraceCoherenceMapper
from data_juicer.ops.mapper.dialog_clarification_quality_mapper import (
    DialogClarificationQualityMapper,
)
from data_juicer.ops.mapper.dialog_coreference_mapper import DialogCoreferenceMapper
from data_juicer.ops.mapper.dialog_error_recovery_mapper import DialogErrorRecoveryMapper
from data_juicer.ops.mapper.dialog_memory_consistency_mapper import (
    DialogMemoryConsistencyMapper,
)
from data_juicer.ops.mapper.dialog_non_repetition_mapper import DialogNonRepetitionMapper
from data_juicer.ops.mapper.dialog_proactivity_mapper import DialogProactivityMapper
from data_juicer.ops.mapper.dialog_topic_shift_mapper import DialogTopicShiftMapper
from data_juicer.ops.mapper.dialog_quality_llm_utils import (
    build_dialog_turn_eval_user_content,
    extract_json_object,
    normalize_score_1_5,
)
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

_M = "data_juicer.ops.mapper.dialog_quality_llm_base"

_DIALOG_TURN_MAPPERS = (
    (DialogMemoryConsistencyMapper, MetaKeys.dialog_memory_consistency),
    (DialogCoreferenceMapper, MetaKeys.dialog_coreference),
    (DialogTopicShiftMapper, MetaKeys.dialog_topic_shift),
    (DialogErrorRecoveryMapper, MetaKeys.dialog_error_recovery),
    (DialogClarificationQualityMapper, MetaKeys.dialog_clarification_quality),
    (DialogProactivityMapper, MetaKeys.dialog_proactivity),
    (DialogNonRepetitionMapper, MetaKeys.dialog_non_repetition),
)


class TestDialogQualityLlmUtils(DataJuicerTestCaseBase):
    def test_extract_json(self):
        raw = 'x {"score": 4, "reason": "ok"} y'
        self.assertEqual(extract_json_object(raw).get("score"), 4)

    def test_normalize_score(self):
        out = normalize_score_1_5({"score": 10, "reason": "x"})
        self.assertEqual(out["score"], 5.0)

    def test_build_dialog_turn_eval_user_content(self):
        sample = {
            "dialog_history": [("a", "b")],
            "query": "q",
            "response": "r",
        }
        s = build_dialog_turn_eval_user_content(
            sample,
            history_key="dialog_history",
            query_key="query",
            response_key="response",
            max_round=5,
            max_query_chars=100,
            max_response_chars=100,
        )
        self.assertIn("Assistant reply to score", s)
        self.assertIn("r", s)

    def test_build_dialog_turn_eval_dedupes_last_history_turn(self):
        """query/response must not repeat dialog_history[-1] (normalize keeps both)."""
        sample = {
            "dialog_history": [("u1", "a1"), ("u2", "a2")],
            "query": "u2",
            "response": "a2",
        }
        s = build_dialog_turn_eval_user_content(
            sample,
            history_key="dialog_history",
            query_key="query",
            response_key="response",
            max_round=10,
            max_query_chars=1000,
            max_response_chars=1000,
        )
        iy = s.find("### Current user message")
        self.assertGreater(iy, 0)
        earlier = s[s.find("### Earlier turns") : iy]
        self.assertNotIn(
            "a2",
            earlier,
            "final assistant text must not appear under Earlier turns when it is "
            "the scored reply (duplicate biases non_repetition / topic_shift).",
        )
        self.assertIn("a2", s[iy:])


class TestDialogQualityMappersSmoke(DataJuicerTestCaseBase):
    @patch(_M + ".prepare_model")
    @patch(_M + ".get_model")
    def test_dialog_turn_mappers_write_meta(self, mock_get, mock_prepare):
        def fake_client(messages, **kwargs):
            return '{"score": 3, "reason": "test"}'

        mock_prepare.return_value = lambda device="cpu": None
        mock_get.return_value = fake_client

        base_sample = {
            Fields.meta: {},
            "dialog_history": [("u1", "a1")],
            "query": "u2",
            "response": "a2",
        }
        for cls, mkey in _DIALOG_TURN_MAPPERS:
            with self.subTest(mapper=cls.__name__):
                op = cls(api_model="qwen-turbo")
                sample = {**base_sample, Fields.meta: {}}
                out = op.process_single(sample)
                meta = out[Fields.meta][mkey]
                self.assertEqual(meta.get("score"), 3.0)
                self.assertEqual(meta.get("eval_kind"), "dialog_turn")

    @patch(_M + ".prepare_model")
    @patch(_M + ".get_model")
    def test_agent_trace_coherence_mapper(self, mock_get, mock_prepare):
        def fake_client(messages, **kwargs):
            return '{"score": 4, "reason": "ok"}'

        mock_prepare.return_value = lambda device="cpu": None
        mock_get.return_value = fake_client

        op = AgentTraceCoherenceMapper(api_model="qwen-turbo")
        sample = {Fields.meta: {}, "text": "user asks\nassistant replies with tools"}
        out = op.process_single(sample)
        meta = out[Fields.meta][MetaKeys.agent_trace_coherence]
        self.assertEqual(meta.get("score"), 4.0)
        self.assertEqual(meta.get("eval_kind"), "agent_trace")

    @patch(_M + ".prepare_model")
    @patch(_M + ".get_model")
    def test_agent_trace_skips_empty_text(self, mock_get, mock_prepare):
        mock_prepare.return_value = lambda device="cpu": None
        mock_get.return_value = lambda *a, **k: '{"score": 1}'
        op = AgentTraceCoherenceMapper(api_model="qwen-turbo")
        out = op.process_single({Fields.meta: {}, "text": "  "})
        self.assertTrue(out[Fields.meta][MetaKeys.agent_trace_coherence].get("skipped"))

    @patch(_M + ".prepare_model")
    @patch(_M + ".get_model")
    def test_agent_tool_relevance_mapper(self, mock_get, mock_prepare):
        def fake_client(messages, **kwargs):
            return '{"score": 2, "reason": "mismatch"}'

        mock_prepare.return_value = lambda device="cpu": None
        mock_get.return_value = fake_client

        op = AgentToolRelevanceMapper(api_model="qwen-turbo")
        sample = {
            Fields.meta: {
                MetaKeys.agent_tool_types: ["read_file"],
                MetaKeys.primary_tool_type: "read_file",
            },
            "query": "list files",
            "response": "done",
        }
        out = op.process_single(sample)
        meta = out[Fields.meta][MetaKeys.agent_tool_relevance]
        self.assertEqual(meta.get("score"), 2.0)
        self.assertEqual(meta.get("eval_kind"), "agent_tool")


if __name__ == "__main__":
    unittest.main()

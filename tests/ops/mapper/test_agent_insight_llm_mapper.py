"""Unit tests for AgentInsightLLMMapper (no live API)."""

import unittest

from data_juicer.ops.mapper.agent_insight_llm_mapper import (
    AgentInsightLLMMapper,
    _build_evidence_pack,
    _dialog_quality_llm_pack,
    _json_safe,
    _parse_llm_json,
    _truncate_record,
)
from data_juicer.utils.constant import Fields, MetaKeys, StatsKeys
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class TestAgentInsightLLMMapper(DataJuicerTestCaseBase):
    def test_skips_when_tier_not_in_run_for_tiers(self):
        op = AgentInsightLLMMapper(api_model="gpt-4o", run_for_tiers=["watchlist"])
        sample = {
            Fields.meta: {MetaKeys.agent_bad_case_tier: "none"},
            Fields.stats: {},
            "query": "q",
            "response": "r",
        }
        out = op.process_single(sample)
        self.assertNotIn(MetaKeys.agent_insight_llm, out[Fields.meta])

    def test_evidence_pack_dialog_quality_llm(self):
        sample = {
            Fields.meta: {
                MetaKeys.agent_bad_case_tier: "watchlist",
                MetaKeys.dialog_memory_consistency: {"score": 2.0, "reason": "forgot constraint"},
            },
            Fields.stats: {},
            "query": "hello",
            "response": "world",
        }
        pack = _build_evidence_pack(sample, "query", "response", 100, 100)
        dq = pack.get("dialog_quality_llm")
        self.assertIsInstance(dq, dict)
        self.assertIn(MetaKeys.dialog_memory_consistency, dq)

    def test_dialog_quality_pack_handles_skipped_error_and_reason(self):
        pack = _dialog_quality_llm_pack(
            {
                MetaKeys.dialog_memory_consistency: {"skipped": True},
                MetaKeys.dialog_coreference: {"error": "bad json"},
                MetaKeys.dialog_topic_shift: {"score": 4, "reason": "x" * 500},
                "ignored": {"score": 1},
            }
        )
        self.assertEqual(pack[MetaKeys.dialog_memory_consistency], {"skipped": True})
        self.assertEqual(pack[MetaKeys.dialog_coreference], {"error": "bad json"})
        self.assertEqual(pack[MetaKeys.dialog_topic_shift]["score"], 4)
        self.assertEqual(len(pack[MetaKeys.dialog_topic_shift]["reason"]), 400)
        self.assertIsNone(_dialog_quality_llm_pack({}))

    def test_json_safe_and_truncate_record_edge_cases(self):
        class ToListBad:
            def tolist(self):
                raise RuntimeError("no")

        class ItemBad:
            def item(self):
                raise RuntimeError("no")

        nested = {"x": [1, (2, 3)], "obj": ToListBad(), "item": ItemBad()}
        safe = _json_safe(nested)
        self.assertEqual(safe["x"], [1, [2, 3]])
        self.assertIn("ToListBad", safe["obj"])
        deep = _json_safe({"deep": {"a": {"b": {"c": {"d": {"e": {"f": 1}}}}}}})
        self.assertEqual(deep["deep"]["a"]["b"]["c"]["d"]["e"]["f"], "...")
        self.assertIsNone(_truncate_record(None))
        self.assertIsNone(_truncate_record(""))
        self.assertEqual(_truncate_record('{"a": 1}'), {"a": 1})
        self.assertEqual(_truncate_record("plain", max_chars=10), "plain")
        self.assertEqual(_truncate_record("x" * 15, max_chars=5), "xxxxx\u2026")
        self.assertTrue(str(_truncate_record({"x": "y" * 20}, max_chars=12)).endswith("\u2026"))

    def test_parse_llm_json_extracts_trailing_object(self):
        self.assertIsNone(_parse_llm_json(""))
        self.assertIsNone(_parse_llm_json("not json"))
        self.assertEqual(_parse_llm_json("prefix\n{\"headline\": \"ok\"}"), {"headline": "ok"})

    def test_evidence_pack_truncates_lists_and_records(self):
        sample = {
            Fields.meta: {
                MetaKeys.agent_request_model: "model",
                MetaKeys.agent_pt: "pt",
                MetaKeys.agent_total_cost_time_ms: 123,
                MetaKeys.prompt_tokens: 1,
                MetaKeys.completion_tokens: 2,
                MetaKeys.total_tokens: 3,
                MetaKeys.tool_success_count: 2,
                MetaKeys.tool_fail_count: 1,
                MetaKeys.tool_success_ratio: 0.66,
                MetaKeys.primary_tool_type: "search",
                MetaKeys.dominant_tool_types: list(range(20)),
                MetaKeys.agent_skill_insights: list(range(20)),
                MetaKeys.dialog_intent_labels: list(range(20)),
                MetaKeys.agent_turn_count: 4,
                MetaKeys.agent_bad_case_signals: ["latency"],
                MetaKeys.agent_bad_case_tier: "watchlist",
            },
            Fields.stats: {
                StatsKeys.llm_analysis_score: 1,
                StatsKeys.llm_quality_record: {"text": "x" * 2000},
                StatsKeys.text_len: 10,
            },
            "query": "q" * 10,
            "response": "r" * 10,
        }

        pack = _build_evidence_pack(sample, "query", "response", 4, 5)

        self.assertEqual(pack["lineage"]["agent_request_model"], "model")
        self.assertEqual(pack["tools"]["dominant_tool_types"], list(range(8)))
        self.assertEqual(pack["query_preview"], "qqqq")
        self.assertEqual(pack["response_preview"], "rrrrr")
        self.assertTrue(str(pack["llm_eval_support"]["llm_quality_record"]).endswith("\u2026"))


if __name__ == "__main__":
    unittest.main()

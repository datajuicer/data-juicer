"""Unit tests for AgentInsightLLMMapper (no live API)."""

import unittest

from data_juicer.ops.mapper.agent_insight_llm_mapper import (
    AgentInsightLLMMapper,
    _build_evidence_pack,
)
from data_juicer.utils.constant import Fields, MetaKeys
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


if __name__ == "__main__":
    unittest.main()

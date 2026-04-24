"""Unit tests for AgentBadCaseSignalMapper."""

import json
import tempfile
import unittest
from pathlib import Path

from data_juicer.ops.mapper.agent_bad_case_signal_mapper import AgentBadCaseSignalMapper
from data_juicer.utils.constant import Fields, MetaKeys, StatsKeys
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class TestAgentBadCaseSignalMapper(DataJuicerTestCaseBase):

    def test_tool_fail_is_high_precision(self):
        op = AgentBadCaseSignalMapper()
        sample = {
            Fields.meta: {
                MetaKeys.tool_fail_count: 1,
                MetaKeys.total_tokens: 100,
            },
            Fields.stats: {},
            "query": "short",
            "response": "ok",
        }
        out = op.process_single(sample)
        sigs = out[Fields.meta][MetaKeys.agent_bad_case_signals]
        self.assertTrue(any(s["code"] == "tool_message_error_pattern" for s in sigs))
        self.assertEqual(out[Fields.meta][MetaKeys.agent_bad_case_tier], "high_precision")

    def test_empty_response_medium_watchlist(self):
        op = AgentBadCaseSignalMapper(
            signal_on_tool_fail=False,
            min_query_len_for_empty_check=10,
            max_response_len_for_empty_check=5,
            min_medium_signals_for_watchlist=1,
        )
        sample = {
            Fields.meta: {MetaKeys.tool_fail_count: 0},
            Fields.stats: {},
            "query": "x" * 50,
            "response": "",
        }
        out = op.process_single(sample)
        self.assertEqual(out[Fields.meta][MetaKeys.agent_bad_case_tier], "watchlist")

    def test_llm_discard_strict_low_score_high_tier(self):
        op = AgentBadCaseSignalMapper(
            signal_on_tool_fail=False,
            signal_on_suspect_empty_response=False,
            llm_analysis_score_max_for_bad=0.5,
            high_precision_llm_analysis_discard_threshold=0.5,
        )
        sample = {
            Fields.meta: {},
            Fields.stats: {
                StatsKeys.llm_analysis_score: 0.2,
                StatsKeys.llm_analysis_record: json.dumps({
                    "recommendation": "discard",
                    "dimension_scores": {},
                }),
            },
            "query": "q",
            "response": "r",
        }
        out = op.process_single(sample)
        self.assertEqual(out[Fields.meta][MetaKeys.agent_bad_case_tier], "high_precision")

    def test_calibration_json_tokens_and_perplexity(self):
        cal = {
            "version": 1,
            "percentile": 95,
            "default": {
                "max_total_tokens": 100,
                "max_latency_ms": 1000,
                "perplexity_high_threshold": 50.0,
            },
            "by_request_model": {
                "m-big": {"max_total_tokens": 200},
            },
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "cal.json"
            path.write_text(json.dumps(cal), encoding="utf-8")
            op = AgentBadCaseSignalMapper(
                signal_on_tool_fail=False,
                auto_calibrate_thresholds=True,
                calibration_json_path=str(path),
                signal_on_high_perplexity=False,
                auto_enable_perplexity_from_calibration=True,
            )
            s1 = {
                Fields.meta: {
                    MetaKeys.agent_request_model: "m-small",
                    MetaKeys.total_tokens: 150,
                },
                Fields.stats: {StatsKeys.perplexity: 60.0},
                "query": "q",
                "response": "r",
            }
            out1 = op.process_single(dict(s1))
            codes1 = [x["code"] for x in out1[Fields.meta][MetaKeys.agent_bad_case_signals]]
            self.assertIn("high_token_usage", codes1)
            self.assertIn("high_perplexity", codes1)

            s2 = {
                Fields.meta: {
                    MetaKeys.agent_request_model: "m-big",
                    MetaKeys.total_tokens: 150,
                },
                Fields.stats: {},
                "query": "q",
                "response": "r",
            }
            out2 = op.process_single(dict(s2))
            codes2 = [x["code"] for x in out2[Fields.meta][MetaKeys.agent_bad_case_signals]]
            self.assertNotIn("high_token_usage", codes2)

    def test_manual_max_tokens_overrides_calibration(self):
        cal = {
            "version": 1,
            "default": {"max_total_tokens": 50},
            "by_request_model": {},
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "cal.json"
            path.write_text(json.dumps(cal), encoding="utf-8")
            op = AgentBadCaseSignalMapper(
                signal_on_tool_fail=False,
                max_total_tokens=200,
                auto_calibrate_thresholds=True,
                calibration_json_path=str(path),
                calibration_manual_overrides_auto=True,
            )
            sample = {
                Fields.meta: {
                    MetaKeys.total_tokens: 100,
                },
                Fields.stats: {},
                "query": "q",
                "response": "r",
            }
            out = op.process_single(sample)
            codes = [x["code"] for x in out[Fields.meta][MetaKeys.agent_bad_case_signals]]
            self.assertNotIn("high_token_usage", codes)

    def test_dialog_quality_meta_low_watchlist(self):
        op = AgentBadCaseSignalMapper(
            signal_on_tool_fail=False,
            signal_on_suspect_empty_response=False,
            signal_on_negative_sentiment_hint=False,
            signal_on_llm_analysis_low=False,
            signal_on_llm_text_quality_low=False,
            signal_hard_query_poor_reply=False,
            signal_on_low_dialog_quality_meta=True,
            dialog_quality_low_score_threshold=2.0,
            min_dialog_quality_low_axes_for_signal=1,
        )
        sample = {
            Fields.meta: {
                MetaKeys.dialog_memory_consistency: {"score": 1.5, "reason": "x"},
            },
            Fields.stats: {},
            "query": "q",
            "response": "r",
        }
        out = op.process_single(sample)
        codes = [s["code"] for s in out[Fields.meta][MetaKeys.agent_bad_case_signals]]
        self.assertIn("dialog_turn_quality_meta_low", codes)
        self.assertEqual(out[Fields.meta][MetaKeys.agent_bad_case_tier], "watchlist")

    def test_dialog_quality_meta_signal_disabled(self):
        op = AgentBadCaseSignalMapper(
            signal_on_tool_fail=False,
            signal_on_suspect_empty_response=False,
            signal_on_negative_sentiment_hint=False,
            signal_on_llm_analysis_low=False,
            signal_on_llm_text_quality_low=False,
            signal_hard_query_poor_reply=False,
            signal_on_low_dialog_quality_meta=False,
        )
        sample = {
            Fields.meta: {
                MetaKeys.dialog_memory_consistency: {"score": 1.0, "reason": "x"},
            },
            Fields.stats: {},
            "query": "q",
            "response": "r",
        }
        out = op.process_single(sample)
        codes = [s["code"] for s in out[Fields.meta][MetaKeys.agent_bad_case_signals]]
        self.assertNotIn("dialog_turn_quality_meta_low", codes)


if __name__ == "__main__":
    unittest.main()

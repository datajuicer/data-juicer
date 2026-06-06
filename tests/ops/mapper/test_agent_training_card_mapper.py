# Copyright 2025 The Data-Juicer Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import unittest

from data_juicer.ops.mapper.agent_training_card_mapper import AgentTrainingCardMapper
from data_juicer.utils.constant import Fields, MetaKeys, StatsKeys


class TestAgentTrainingCardMapper(unittest.TestCase):
    def test_card_basic_tier_and_sft_ready(self):
        sample = {
            Fields.meta: {
                MetaKeys.agent_training_dataset_tier: "gold",
                MetaKeys.agent_tool_chain_complete: True,
                MetaKeys.agent_error_taxonomy: {
                    "hard_drop_recommended": False,
                    "buckets": {
                        "timeout": {"severity": 0.4},
                        "safety": {"severity": 0.1},
                    },
                },
                MetaKeys.agent_learnable_value: {"score": 0.8},
                MetaKeys.agent_cross_model_pair: {"has_pairwise_contrast": True, "delta_to_best": 0.05},
                MetaKeys.agent_training_safety_gate: {"ok": True},
            },
            Fields.stats: {
                StatsKeys.llm_difficulty_score: 0.7,
                StatsKeys.llm_analysis_score: 0.6,
            },
        }
        out = AgentTrainingCardMapper().process_single(dict(sample))
        card = json.loads(out[Fields.meta][MetaKeys.agent_training_card])
        self.assertEqual(card["training_dataset_tier"], "gold")
        self.assertIsNotNone(card.get("learnable_value_json"))
        self.assertIn("score", json.loads(card["learnable_value_json"]))
        self.assertTrue(card["training_ready"]["sft"])
        self.assertTrue(card["training_ready"]["preference"])
        self.assertEqual(card["safety_gate_ok"], "true")
        self.assertIn("timeout", card["target_capabilities"])

    def test_card_with_distill_and_rewrite_flags(self):
        sample = {
            Fields.meta: {
                MetaKeys.agent_training_dataset_tier: "silver",
                MetaKeys.agent_tool_chain_complete: True,
                MetaKeys.agent_error_taxonomy: {"hard_drop_recommended": False, "buckets": {}},
                MetaKeys.agent_distilled_trajectory: {
                    "skipped": False,
                    "distilled_final_reply": "ok",
                },
                MetaKeys.agent_rewrite_hints: {"hints": ["x"], "parse_ok": True},
            },
            Fields.stats: {},
        }
        out = AgentTrainingCardMapper().process_single(dict(sample))
        card = json.loads(out[Fields.meta][MetaKeys.agent_training_card])
        self.assertTrue(card["distilled_present"])
        self.assertTrue(card["rewrite_hints_present"])
        self.assertIn("has teacher distilled reply", card["recommended_usage"])


if __name__ == "__main__":
    unittest.main()

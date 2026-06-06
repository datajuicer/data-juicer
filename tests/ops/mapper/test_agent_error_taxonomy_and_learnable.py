# Copyright 2025 The Data-Juicer Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import unittest

from data_juicer.ops.mapper.agent_error_taxonomy_mapper import AgentErrorTaxonomyMapper
from data_juicer.ops.mapper.agent_learnable_value_mapper import AgentLearnableValueMapper
from data_juicer.utils.constant import Fields, MetaKeys, StatsKeys


class TestAgentErrorTaxonomyAndLearnable(unittest.TestCase):
    def test_taxonomy_and_learnable_smoke(self):
        sample = {
            Fields.meta: {
                MetaKeys.dialog_memory_consistency: {"score": 2.0},
                MetaKeys.agent_tool_relevance: {"score": 2.0},
                MetaKeys.tool_success_ratio: 0.5,
                MetaKeys.agent_tool_chain_complete: True,
                MetaKeys.agent_turn_count: 3,
                MetaKeys.dialog_intent_labels: ["a", "b"],
                MetaKeys.dialog_topic_labels: ["t1"],
                MetaKeys.agent_cross_model_pair: {"has_pairwise_contrast": True, "delta_to_best": 0.2},
            },
            Fields.stats: {
                StatsKeys.llm_difficulty_score: 0.8,
                StatsKeys.llm_analysis_record: json.dumps(
                    {
                        "dimension_scores": {
                            "instruction_following": 2.0,
                            "response_quality": 2.0,
                            "helpfulness": 3.0,
                            "safety": 4.0,
                        }
                    }
                ),
                StatsKeys.llm_quality_record: json.dumps({"dimension_scores": {"grammar": 3.0}}),
            },
        }
        sample = AgentErrorTaxonomyMapper().process_single(dict(sample))
        sample = AgentLearnableValueMapper().process_single(sample)
        self.assertIn("buckets", sample[Fields.meta][MetaKeys.agent_error_taxonomy])
        self.assertIn(MetaKeys.agent_learnable_value, sample[Fields.meta])
        self.assertIn(MetaKeys.agent_learnable_value_tier, sample[Fields.meta])
        self.assertIn(MetaKeys.agent_training_dataset_tier, sample[Fields.meta])


if __name__ == "__main__":
    unittest.main()

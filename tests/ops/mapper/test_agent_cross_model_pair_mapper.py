# Copyright 2025 The Data-Juicer Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import unittest

from data_juicer.ops.mapper.agent_cross_model_pair_mapper import AgentCrossModelPairMapper
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.unittest_utils import TEST_TAG, DataJuicerTestCaseBase


class AgentCrossModelPairMapperTest(DataJuicerTestCaseBase):
    def test_requires_full_dataset_pass_flag(self):
        self.assertIs(AgentCrossModelPairMapper.REQUIRES_FULL_DATASET_PASS, True)

    def test_apply_full_dataset_annotations_exact_pair(self):
        op = AgentCrossModelPairMapper(
            pair_key_meta="agent_lineage_sample_id",
            model_meta="agent_lineage_tag_model",
            version_meta="agent_lineage_tag_version",
            score_meta="agent_lineage_tag_quality",
            min_group_size=2,
            group_key_mode="exact",
        )
        rows = [
            {
                "id": "1",
                Fields.meta: {
                    "agent_lineage_sample_id": "s1",
                    "agent_lineage_tag_model": "weak",
                    "agent_lineage_tag_version": "1",
                    "agent_lineage_tag_quality": 0.4,
                },
            },
            {
                "id": "2",
                Fields.meta: {
                    "agent_lineage_sample_id": "s1",
                    "agent_lineage_tag_model": "strong",
                    "agent_lineage_tag_version": "1",
                    "agent_lineage_tag_quality": 0.9,
                },
            },
        ]
        rows = copy.deepcopy(rows)
        op.apply_full_dataset_annotations(rows)
        p0 = rows[0][Fields.meta][MetaKeys.agent_cross_model_pair]
        p1 = rows[1][Fields.meta][MetaKeys.agent_cross_model_pair]
        self.assertTrue(p0["has_pairwise_contrast"])
        self.assertEqual(p0["best_model"], "strong")
        self.assertEqual(p0["my_model"], "weak")
        self.assertEqual(p1["my_model"], "strong")
        self.assertEqual(p0["group_key"], p1["group_key"])


@TEST_TAG("ray")
class AgentCrossModelPairMapperRayPathTest(DataJuicerTestCaseBase):
    """RayData must use take_all + apply, not map_batches(process)."""

    def test_process_op_writes_pair_meta(self):
        import ray
        from data_juicer.core.data.ray_dataset import RayDataset

        rows = [
            {
                "id": "1",
                "query": "q",
                "response": "r1",
                Fields.meta: {
                    "agent_lineage_sample_id": "s1",
                    "agent_lineage_tag_model": "a",
                    "agent_lineage_tag_version": "1",
                    "agent_lineage_tag_quality": 0.2,
                },
            },
            {
                "id": "2",
                "query": "q",
                "response": "r2",
                Fields.meta: {
                    "agent_lineage_sample_id": "s1",
                    "agent_lineage_tag_model": "b",
                    "agent_lineage_tag_version": "1",
                    "agent_lineage_tag_quality": 0.8,
                },
            },
        ]
        ds = RayDataset(ray.data.from_items(rows))
        op = AgentCrossModelPairMapper(
            min_group_size=2,
            group_key_mode="exact",
        )
        out = ds.process([op])
        out_rows = out.data.take_all()
        self.assertEqual(len(out_rows), 2)
        for r in out_rows:
            self.assertIsInstance(r, dict)
            m = r[Fields.meta]
            if isinstance(m, str) and m.strip():
                import json

                m = json.loads(m)
            self.assertIn(MetaKeys.agent_cross_model_pair, m)
            self.assertTrue(m[MetaKeys.agent_cross_model_pair]["has_pairwise_contrast"])


if __name__ == "__main__":
    unittest.main()

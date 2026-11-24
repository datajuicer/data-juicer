import json
import unittest
from unittest.mock import patch

from datasets import Dataset

from data_juicer.core.data import NestedDataset
from data_juicer.ops.mapper.agent_cross_model_pair_mapper import (
    AgentCrossModelPairMapper,
)
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.unittest_utils import TEST_TAG


def examples():
    return [
        {
            "id": "a-low",
            "query": " Task A ",
            "text": "long trajectory " * 200,
            Fields.meta: {
                "agent_lineage_sample_id": "a",
                "agent_lineage_tag_model": "qwen",
                "agent_lineage_tag_version": "3.5",
                "agent_lineage_tag_quality": 0.2,
            },
        },
        {
            "id": "b",
            "query": "task b",
            "text": "other trajectory",
            Fields.meta: {
                "agent_lineage_sample_id": "b",
                "agent_lineage_tag_model": "other",
                "agent_lineage_tag_version": "1",
                "agent_lineage_tag_quality": 0.4,
            },
        },
        {
            "id": "a-high",
            "query": "task   a",
            "text": "better trajectory",
            Fields.meta: {
                "agent_lineage_sample_id": "a",
                "agent_lineage_tag_model": "qwen",
                "agent_lineage_tag_version": "3.6",
                "agent_lineage_tag_quality": 0.8,
            },
        },
    ]


class AgentPairStreamingTest(unittest.TestCase):
    def test_same_model_different_versions_form_contrast(self):
        rows = examples()
        AgentCrossModelPairMapper().apply_full_dataset_annotations(rows)
        pair = rows[0][Fields.meta][MetaKeys.agent_cross_model_pair]
        self.assertTrue(pair["has_pairwise_contrast"])
        self.assertEqual(pair["peer_models"], ["qwen"])
        self.assertEqual(pair["best_version"], "3.6")
        self.assertAlmostEqual(pair["delta_to_best"], 0.6)

    def test_unknown_or_identical_versions_are_not_contrast(self):
        for versions in [(None, "1"), ("", "1"), ("1", "1")]:
            rows = [examples()[0], examples()[2]]
            for row, version in zip(rows, versions):
                row[Fields.meta]["agent_lineage_tag_version"] = version
            AgentCrossModelPairMapper().apply_full_dataset_annotations(rows)
            self.assertFalse(rows[0][Fields.meta][MetaKeys.agent_cross_model_pair]["has_pairwise_contrast"])

    def test_nonfinite_scores_do_not_override_valid_scores(self):
        for invalid in [float("nan"), float("inf"), float("-inf")]:
            rows = [examples()[0], examples()[2]]
            rows[0][Fields.meta]["agent_lineage_tag_quality"] = invalid
            AgentCrossModelPairMapper().apply_full_dataset_annotations(rows)
            pair = rows[0][Fields.meta][MetaKeys.agent_cross_model_pair]
            self.assertEqual(pair["best_quality"], 0.8)
            self.assertIsNone(pair["my_quality"])

    def test_exact_and_normalized_run_without_whole_table_conversion(self):
        for mode in ["exact", "normalized_query"]:
            with self.subTest(mode=mode):
                rows = examples()
                source = NestedDataset.from_list(rows)
                with patch.object(Dataset, "to_list", side_effect=AssertionError("whole-table conversion")):
                    result = AgentCrossModelPairMapper(group_key_mode=mode, num_proc=1).run(source)
                actual = list(result)
                self.assertEqual([r["id"] for r in actual], [r["id"] for r in rows])
                self.assertEqual([r["text"] for r in actual], [r["text"] for r in rows])
                self.assertEqual(actual[0][Fields.meta][MetaKeys.agent_cross_model_pair]["group_size"], 2)
                self.assertNotIn(MetaKeys.agent_cross_model_pair, source[0][Fields.meta])
                self.assertEqual(result.column_names, source.column_names)

    def test_missing_keys_and_null_first_group_remain_serializable(self):
        rows = examples()
        rows[0][Fields.meta]["agent_lineage_sample_id"] = ""
        rows[0][Fields.meta]["agent_lineage_tag_quality"] = None
        result = AgentCrossModelPairMapper(num_proc=1).run(NestedDataset.from_list(rows))
        self.assertEqual(result[0][Fields.meta][MetaKeys.agent_cross_model_pair]["group_size"], 0)
        self.assertEqual(result[2][Fields.meta][MetaKeys.agent_cross_model_pair]["best_quality"], 0.8)
        self.assertEqual(len(list(result)), 3)

    def test_json_metadata_is_parsed_and_fields_preserved(self):
        for mode in ["exact", "normalized_query"]:
            with self.subTest(mode=mode):
                rows = examples()
                for row in rows:
                    row[Fields.meta] = json.dumps(row[Fields.meta])
                result = AgentCrossModelPairMapper(group_key_mode=mode, num_proc=1).run(NestedDataset.from_list(rows))
                self.assertEqual(result[0][Fields.meta][MetaKeys.agent_cross_model_pair]["best_version"], "3.6")
                self.assertEqual(result[0][Fields.meta]["agent_lineage_sample_id"], "a")

    def test_query_only_rows_without_metadata_are_preserved(self):
        rows = [{"query": "same task", "response": "a"}, {"query": "same task", "response": "b"}]
        result = AgentCrossModelPairMapper(group_key_mode="normalized_query", num_proc=1).run(
            NestedDataset.from_list(rows)
        )
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][Fields.meta][MetaKeys.agent_cross_model_pair]["group_size"], 2)
        self.assertFalse(result[0][Fields.meta][MetaKeys.agent_cross_model_pair]["has_pairwise_contrast"])

    def test_empty_dataset_keeps_schema(self):
        source = NestedDataset.from_list(examples()).select([])
        result = AgentCrossModelPairMapper(num_proc=1).run(source)
        self.assertEqual(len(result), 0)
        self.assertEqual(result.features, source.features)

    def test_internal_column_names_do_not_replace_input_fields(self):
        rows = examples()
        for row in rows:
            row["__dj_agent_pair_key__"] = "user value"
            row["__dj_agent_pair_position__"] = 42
        result = AgentCrossModelPairMapper(num_proc=1).run(NestedDataset.from_list(rows))
        self.assertEqual(result[0]["__dj_agent_pair_key__"], "user value")
        self.assertEqual(result[0]["__dj_agent_pair_position__"], 42)

    def test_minimum_group_size_still_applies_to_versions(self):
        rows = examples()
        AgentCrossModelPairMapper(min_group_size=3).apply_full_dataset_annotations(rows)
        self.assertFalse(rows[0][Fields.meta][MetaKeys.agent_cross_model_pair]["has_pairwise_contrast"])

    def test_simhash_mode_retains_grouping(self):
        rows = examples()
        op = AgentCrossModelPairMapper(group_key_mode="simhash_lsh", simhash_max_hamming=0, num_proc=1)
        result = op.run(NestedDataset.from_list(rows))
        self.assertEqual(result[0][Fields.meta][MetaKeys.agent_cross_model_pair]["group_size"], 2)
        self.assertEqual(result[0][Fields.meta][MetaKeys.agent_cross_model_pair]["match_basis"], "simhash_lsh")

    def test_null_first_writer_batch_has_explicit_pair_schema(self):
        rows = [examples()[1] for _ in range(1001)]
        for row in rows:
            row[Fields.meta]["agent_lineage_sample_id"] = ""
            row[Fields.meta]["agent_lineage_tag_quality"] = None
        rows.append(examples()[2])
        result = AgentCrossModelPairMapper(num_proc=1).run(NestedDataset.from_list(rows))
        self.assertIsInstance(result, NestedDataset)
        self.assertEqual(len(result), 1002)
        self.assertEqual(result[-1][Fields.meta][MetaKeys.agent_cross_model_pair]["best_quality"], 0.8)

    def test_cpu_pipeline_reaches_training_cards(self):
        from data_juicer.ops.mapper.agent_dialog_normalize_mapper import (
            AgentDialogNormalizeMapper,
        )
        from data_juicer.ops.mapper.agent_error_taxonomy_mapper import (
            AgentErrorTaxonomyMapper,
        )
        from data_juicer.ops.mapper.agent_learnable_value_mapper import (
            AgentLearnableValueMapper,
        )
        from data_juicer.ops.mapper.agent_training_card_mapper import (
            AgentTrainingCardMapper,
        )

        rows = []
        for version, score in [("1", 0.2), ("2", 0.8)]:
            rows.append(
                {
                    "id": version,
                    "sample_id": "task",
                    "tag": {"model": "qwen", "version": version, "quality": score},
                    "messages": [
                        {"role": "user", "content": "Solve task"},
                        {"role": "assistant", "content": "Answer " + version},
                    ],
                }
            )
        ops = [
            AgentDialogNormalizeMapper(
                lineage_extra_keys=["sample_id", "tag.model", "tag.version", "tag.quality"], num_proc=1
            ),
            AgentCrossModelPairMapper(num_proc=1),
            AgentErrorTaxonomyMapper(num_proc=1),
            AgentLearnableValueMapper(num_proc=1),
            AgentTrainingCardMapper(num_proc=1),
        ]
        dataset = NestedDataset.from_list(rows)
        for op in ops:
            dataset = op.run(dataset)
        cards = [json.loads(row[Fields.meta][MetaKeys.agent_training_card]) for row in dataset]
        self.assertTrue(cards[0]["training_ready"]["preference"])
        self.assertFalse(cards[1]["training_ready"]["preference"])
        self.assertGreater(float(cards[0]["learnable_value_json"]), float(cards[1]["learnable_value_json"]))


class AgentPairRayStreamingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import ray

        cls.started_ray = not ray.is_initialized()
        if cls.started_ray:
            ray.init(num_cpus=2, include_dashboard=False, object_store_memory=100 * 1024 * 1024)

    @classmethod
    def tearDownClass(cls):
        if cls.started_ray:
            import ray

            ray.shutdown()

    @TEST_TAG("ray")
    def test_distributed_cohorts_without_collecting_source_rows(self):
        import ray

        from data_juicer.core.data.ray_dataset import RayDataset

        for mode in ["exact", "normalized_query"]:
            with self.subTest(mode=mode):
                inputs = examples()
                orphan = examples()[1]
                orphan.update(id="unpaired", query="")
                orphan[Fields.meta]["agent_lineage_sample_id"] = ""
                inputs.append(orphan)
                dataset = RayDataset(ray.data.from_items(inputs, override_num_blocks=2))
                with patch.object(ray.data.Dataset, "take_all", side_effect=AssertionError("driver collection")):
                    result = dataset.process([AgentCrossModelPairMapper(group_key_mode=mode, num_proc=1)])
                    result.data = result.data.materialize()
                rows = {r["id"]: r for r in result.data.take_all()}
                self.assertEqual(set(rows), {"a-low", "a-high", "b", "unpaired"})
                self.assertTrue(rows["a-low"][Fields.meta][MetaKeys.agent_cross_model_pair]["has_pairwise_contrast"])
                self.assertEqual(rows["a-low"]["text"], examples()[0]["text"])
                self.assertEqual(rows["unpaired"][Fields.meta][MetaKeys.agent_cross_model_pair]["group_size"], 0)

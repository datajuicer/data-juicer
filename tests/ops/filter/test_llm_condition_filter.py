"""Unit tests for LLMConditionFilter (llm_semantic_ops).

Use ``op.compute_stats`` / ``op.process`` with ``NestedDataset.map`` / ``filter``,
not ``*_single`` — HF passes dict-of-lists batches; wrappers expand/collapse them.
"""

import unittest
from unittest.mock import patch

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.filter.llm_condition_filter import LLMConditionFilter
from data_juicer.utils.llm_semantic_ops import LLMCallUsage
from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase, FROM_FORK


@unittest.skipIf(
    FROM_FORK, "Skipping API-based test because running from a fork repo"
)
class TestLLMConditionFilter(DataJuicerTestCaseBase):
    api_or_hf_model = "gpt-4o"

    def _run_test(self, dataset: Dataset, op: LLMConditionFilter):
        if Fields.stats not in dataset.features:
            dataset = dataset.add_column(
                name=Fields.stats, column=[{}] * dataset.num_rows
            )
        dataset = dataset.map(op.compute_stats, batch_size=op.batch_size)
        for sample in dataset:
            self.assertIn(Fields.stats, sample)
            self.assertIn(
                StatsKeys.llm_condition_filter_result, sample[Fields.stats]
            )
            if StatsKeys.llm_semantic_usage in sample[Fields.stats]:
                usage = sample[Fields.stats][StatsKeys.llm_semantic_usage]
                self.assertIn("prompt_tokens", usage)
                self.assertIn("completion_tokens", usage)
                self.assertIn("total_tokens", usage)
        filtered = dataset.filter(op.process, batch_size=op.batch_size)
        return filtered

    def test_condition_question(self):
        """Samples satisfying 'contains a question' are kept."""
        ds_list = [
            {"text": "What is the capital of France?"},
            {"text": "The capital of France is Paris."},
            {"text": "How do I install Python?"},
        ]
        dataset = Dataset.from_list(ds_list)
        op = LLMConditionFilter(
            text_key="text",
            condition="The text contains a clear question.",
            api_or_hf_model=self.api_or_hf_model,
            try_num=2,
        )
        filtered = self._run_test(dataset, op)
        out_list = filtered.to_list()
        self.assertLessEqual(len(out_list), 3)
        self.assertGreater(len(out_list), 0)

    def test_empty_text(self):
        """Empty text gets False and no LLM call for condition."""
        ds_list = [{"text": ""}]
        dataset = Dataset.from_list(ds_list)
        dataset = dataset.add_column(
            name=Fields.stats, column=[{}] * dataset.num_rows
        )
        op = LLMConditionFilter(
            text_key="text",
            condition="The text is non-empty.",
            api_or_hf_model=self.api_or_hf_model,
        )
        dataset = dataset.map(op.compute_stats, batch_size=1)
        first = dataset.to_list()[0]
        self.assertFalse(
            first[Fields.stats][StatsKeys.llm_condition_filter_result]
        )
        filtered = dataset.filter(op.process, batch_size=1)
        self.assertEqual(len(filtered.to_list()), 0)

    def test_empty_condition(self):
        """Empty condition string keeps all samples (no LLM call)."""
        ds_list = [{"text": "Some content."}]
        dataset = Dataset.from_list(ds_list)
        dataset = dataset.add_column(
            name=Fields.stats, column=[{}] * dataset.num_rows
        )
        op = LLMConditionFilter(
            text_key="text",
            condition="",
            api_or_hf_model=self.api_or_hf_model,
        )
        dataset = dataset.map(op.compute_stats, batch_size=1)
        first = dataset.to_list()[0]
        self.assertTrue(
            first[Fields.stats][StatsKeys.llm_condition_filter_result]
        )
        filtered = dataset.filter(op.process, batch_size=1)
        self.assertEqual(len(filtered.to_list()), 1)


class TestLLMConditionFilterUsageAccumulation(unittest.TestCase):
    @patch("data_juicer.ops.filter.llm_condition_filter.get_model", return_value=object())
    @patch("data_juicer.ops.filter.llm_condition_filter.condition_filter_one")
    def test_usage_is_accumulated_in_stats(self, mock_condition_filter_one, _mock_get_model):
        op = LLMConditionFilter(
            text_key="text",
            condition="contains question",
            api_or_hf_model="gpt-4o",
            try_num=1,
        )
        sample = {
            "text": "What is AI?",
            Fields.stats: {
                StatsKeys.llm_semantic_usage: {
                    "prompt_tokens": 2,
                    "completion_tokens": 3,
                    "total_tokens": 5,
                    "cost_estimate": 0.1,
                }
            },
        }
        mock_condition_filter_one.return_value = (
            True,
            LLMCallUsage(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
                cost_estimate=0.5,
            ),
        )

        out = op.compute_stats_single(sample)
        usage = out[Fields.stats][StatsKeys.llm_semantic_usage]
        self.assertEqual(usage["prompt_tokens"], 12)
        self.assertEqual(usage["completion_tokens"], 23)
        self.assertEqual(usage["total_tokens"], 35)
        self.assertEqual(usage["cost_estimate"], 0.6)


if __name__ == "__main__":
    unittest.main()

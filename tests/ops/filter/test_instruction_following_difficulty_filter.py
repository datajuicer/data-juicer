import unittest

from data_juicer.core.data import NestedDataset as Dataset

from data_juicer.ops.filter.instruction_following_difficulty_filter import InstructionFollowingDifficultyFilter
from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class InstructionFollowingDifficultyFilterTest(DataJuicerTestCaseBase):
    _hf_model = "Qwen/Qwen2.5-0.5B"
    # _hf_model = "/your/local/path/to/Qwen2.5-0.5B"

    def _run_test(self, dataset: Dataset, target_list, op):
        if Fields.stats not in dataset.features:
            dataset = dataset.add_column(name=Fields.stats,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(
            op.compute_stats,
            batch_size=op.batch_size,
            )
        dataset = dataset.filter(op.process, batch_size=op.batch_size)
        dataset_test = dataset.select_columns(column_names=['text'])
        res_list = dataset_test.to_list()
        self.assertEqual(res_list, target_list)
        return dataset

    def test_rft_data(self):
        ds_list = [
            {
                "text": "Explain gravity.",
                "answer": "Gravity is a fundamental force pulling objects toward each other."
            },
            {
                "text": "What is the capital of France?",
                "answer": "The capital of France is Paris."
            },
            {
                "text": "How does chocolate taste?",
                "answer": "The capital of France is Paris."
            },
        ]
        tgt_list = [
            {
                "text": "Explain gravity.",
            }
        ]
        dataset = Dataset.from_list(ds_list)
        op = InstructionFollowingDifficultyFilter(
            hf_model=self._hf_model,
            min_score=0.2,
            max_score=0.9,
            query_template="Question: {text}",
            response_template="Answer: {answer}",
        )
        self._run_test(dataset, tgt_list, op)


if __name__ == '__main__':
    unittest.main()

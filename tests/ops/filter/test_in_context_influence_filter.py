import unittest

from data_juicer.core.data import NestedDataset as Dataset

from data_juicer.ops.filter.in_context_influence_filter import InContextInfluenceFilter
from data_juicer.utils.constant import Fields
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class InContextInfluenceFilterTest(DataJuicerTestCaseBase):
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

    def test_sample_as_demo(self):
        ds_list = [
            {
                "text": "What is the capital of France?",
                "answer": "The capital of France is Paris."
            },
            {
                "text": "Explain gravity.",
                "answer": "Gravity is a fundamental force pulling objects toward each other."
            },

        ]
        vs_list = [
            {
                "text": "What is the capital of France?",
                "answer": "The capital of France is Paris."
            }
        ]
        tgt_list = [
            {
                "text": "What is the capital of France?",
            }
        ]
        dataset = Dataset.from_list(ds_list)
        valid_dataset = Dataset.from_list(vs_list)
        op = InContextInfluenceFilter(
            hf_model=self._hf_model,
            min_score=1.0,
            max_score=100.0,
            query_template="{text}",
            response_template="{answer}",
            valid_as_demo=False,
        )
        op.prepare_valid_feature(valid_dataset)
        self._run_test(dataset, tgt_list, op)


if __name__ == '__main__':
    unittest.main()

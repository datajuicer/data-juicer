import unittest

from data_juicer.core.data import NestedDataset as Dataset

from data_juicer.ops.filter.llm_perplexity_filter import LLMPerplexityFilter
from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class LLMPerplexityFilterTest(DataJuicerTestCaseBase):
    _hf_model = "Qwen/Qwen2.5-0.5B"
    # _hf_model = "/your/local/path/to/Qwen2.5-0.5B"

    def _run_perplexity_filter(self, dataset: Dataset, target_list, op):
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

    def test_hf_model(self):

        ds_list = [{
            'text': "Today is Sunday and it's a happy day!"
        }, {
            'text':
            "Today is Sund Sund Sund Sund Sunda and it's a happy day!"
        }, {
            'text': 'a v s e c s f e f g a qkc'
        }, {
            'text': '，。、„”“«»１」「《》´∶：？！（）；–—．～’…━〈〉【】％►'
        }, {
            'text': 'Do you need a cup of coffee?'
        }, {
            'text': 'emoji表情测试下😊，😸31231'
        }]
        tgt_list = [{
            'text': "Today is Sunday and it's a happy day!"
        }, {
            'text': 'Do you need a cup of coffee?'
        }]
        dataset = Dataset.from_list(ds_list)
        op = LLMPerplexityFilter(
            hf_model=self._hf_model,
            min_score=1,
            max_score=50
        )
        self._run_perplexity_filter(dataset, tgt_list, op)

    def test_rft_data(self):
        ds_list = [
            {
                "text": "What is the capital of France?",
                "answer": "The capital of France is Paris."
            },
            {
                "text": "What is the capital of China?",
                "answer": "The capital of China is Paris."
            },

        ]
        tgt_list = [
            {
                "text": "What is the capital of France?",
            }
        ]
        dataset = Dataset.from_list(ds_list)
        op = LLMPerplexityFilter(
            hf_model=self._hf_model,
            min_score=1,
            max_score=5,
            query_template="Question: {text}",
            response_template="Answer: {answer}",
        )
        self._run_perplexity_filter(dataset, tgt_list, op)


if __name__ == '__main__':
    unittest.main()

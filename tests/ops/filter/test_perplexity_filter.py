import unittest

from data_juicer.core.data import NestedDataset as Dataset

from data_juicer.ops.filter.perplexity_filter import PerplexityFilter
from data_juicer.utils.constant import Fields
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class PerplexityFilterTest(DataJuicerTestCaseBase):

    def _run_perplexity_filter(self, dataset: Dataset, target_list, op, context=False):
        if Fields.stats not in dataset.features:
            # TODO:
            # this is a temp solution,
            # only add stats when calling filter op
            dataset = dataset.add_column(name=Fields.stats,
                                         column=[{}] * dataset.num_rows)
        if context:
            dataset = dataset.add_column(name=Fields.context,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(
            op.compute_stats,
            batch_size=op.batch_size,
            fn_kwargs={'context': context}
            )
        dataset = dataset.filter(op.process, batch_size=op.batch_size)
        dataset_test = dataset.select_columns(column_names=['text'])
        res_list = dataset_test.to_list()
        self.assertEqual(res_list, target_list)
        return dataset

    def _test_en_case(self, context=False):

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
        op = PerplexityFilter(lang='en', max_ppl=900, batch_size=2)
        dataset= self._run_perplexity_filter(dataset, tgt_list, op, context)
        if context:
            dataset = dataset.select_columns(column_names=[Fields.context])
            context_list = dataset.to_list()
            res_words_list = [list(context_list[i][Fields.context].values()) \
                for i in range(len(context_list))]
            tgt_words_list = [
                [['▁Today', '▁is', '▁Sunday', '▁and', '▁it', "'", 's', '▁a', '▁happy', '▁day', '!']], 
                [['▁Do', '▁you', '▁need', '▁a', '▁cup', '▁of', '▁coffee', '?']]
            ]
            self.assertListEqual(res_words_list, tgt_words_list)

    def test_en_case_default(self):
        self._test_en_case(context=False)

    def test_en_case_context(self):
        self._test_en_case(context=True)


if __name__ == '__main__':
    unittest.main()

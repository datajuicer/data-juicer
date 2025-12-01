import unittest

from data_juicer.core.data import NestedDataset as Dataset

from data_juicer.ops.filter.text_length_filter import TextLengthFilter
from data_juicer.utils.constant import Fields
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class TextLengthFilterTest(DataJuicerTestCaseBase):

    def _run_text_length_filter(self, dataset: Dataset, target_list, op):
        if Fields.stats not in dataset.features:
            # TODO:
            # this is a temp solution,
            # only add stats when calling filter op
            dataset = dataset.add_column(name=Fields.stats,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.compute_stats, batch_size=3)
        dataset = dataset.filter(op.process, batch_size=2)
        dataset = dataset.select_columns(column_names=['text'])
        res_list = dataset.to_list()
        self.assertEqual(res_list, target_list)

    def test_case(self):

        ds_list = [{
            'text': 'Today is'
        }, {
            'text':
            "Today is Sund Sund Sund Sund Sund Sunda and it's a happy day!"
        }, {
            'text': 'a v s e c s f e f g a a a  '
        }, {
            'text': '，。、„”“«»１」「《》´∶：？！（）；–—．～’…━〈〉【】％►'
        }, {
            'text': '中文也是一个字算一个长度'
        }]
        tgt_list = [{
            'text': 'a v s e c s f e f g a a a  '
        }, {
            'text': '，。、„”“«»１」「《》´∶：？！（）；–—．～’…━〈〉【】％►'
        }, {
            'text': '中文也是一个字算一个长度'
        }]
        dataset = Dataset.from_list(ds_list)
        op = TextLengthFilter(min_len=10, max_len=50)
        self._run_text_length_filter(dataset, tgt_list, op)


if __name__ == '__main__':
    unittest.main()

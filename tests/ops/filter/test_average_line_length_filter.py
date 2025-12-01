import unittest

from data_juicer.core.data import NestedDataset as Dataset

from data_juicer.ops.filter.average_line_length_filter import \
    AverageLineLengthFilter
from data_juicer.utils.constant import Fields, InterVars
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class AverageLineLengthFilterTest(DataJuicerTestCaseBase):
    text_key = 'text'
    ds_list = [{
            text_key: 'a=1\nb\nc=1+2+3+5\nd=6'
        }, {
            text_key:
            "Today is Sund Sund Sunda and it's a happy day!\nYou know"
        }, {
            text_key: 'a v s e e f g a qkc'
        }, {
            text_key: '，。、„”“«»１」「《》´∶：？！（）；–—．～’…━〈〉【】％►'
        }, {
            text_key: 'Do you need a cup of coffee?'
        }, {
            text_key: 'emoji表情测试下😊，😸31231\n'
        }]
    tgt_list = [{
            text_key: 'a v s e e f g a qkc'
        }, {
            text_key: 'emoji表情测试下😊，😸31231\n'
        }]

    def _run_average_line_length_filter(self, dataset: Dataset, target_list,
                                        op, context=False):
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
        dataset_test = dataset.select_columns(column_names=[self.text_key])
        res_list = dataset_test.to_list()
        self.assertEqual(res_list, target_list)

        return dataset

    def test_case_default(self):
        dataset = Dataset.from_list(self.ds_list)
        op = AverageLineLengthFilter(min_len=10, max_len=20, batch_size=3)
        self._run_average_line_length_filter(dataset, self.tgt_list, op, context=False)

    def test_case_context(self):
        dataset = Dataset.from_list(self.ds_list)
        op = AverageLineLengthFilter(min_len=10, max_len=20, batch_size=2)
        dataset = self._run_average_line_length_filter(dataset, self.tgt_list, op, context=True)

        dataset = dataset.select_columns(column_names=[Fields.context])
        res_list = dataset.to_list()

        tgt_context_list = [
            {
                Fields.context: {
                    InterVars.lines: tgt[self.text_key].splitlines()
                    }
            } for tgt in self.tgt_list
        ]

        self.assertEqual(res_list, tgt_context_list)


if __name__ == '__main__':
    unittest.main()

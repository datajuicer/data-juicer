import unittest

from data_juicer.core.data import NestedDataset as Dataset

from data_juicer.ops.filter.average_line_length_filter import \
    AverageLineLengthFilter
from data_juicer.utils.constant import Fields, InterVars, StatsKeys
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

    def test_standalone_stats_match_splitlines(self):
        texts = [
            "", "\n", "\r\n", "alpha\n\nbeta\n", "alpha\rbeta",
            "alpha\r\nbeta\n", "\r\r\n", "a\vb\fc\x1cd\x1de\x1ef",
            "中文\x85é\u2028字\u2029", "single line",
        ]
        samples = {"text": texts, Fields.stats: [{} for _ in texts]}
        AverageLineLengthFilter().compute_stats_batched(samples)
        expected = [len(text) / len(text.splitlines()) if text else 0.0 for text in texts]
        self.assertEqual(
            [stat[StatsKeys.avg_line_length] for stat in samples[Fields.stats]], expected
        )

    def test_case_context(self):
        dataset = Dataset.from_list(self.ds_list)
        op = AverageLineLengthFilter(min_len=10, max_len=20, batch_size=2)
        dataset = self._run_average_line_length_filter(dataset, self.tgt_list, op, context=True)

        dataset = dataset.select_columns(column_names=[Fields.context])
        res_list = dataset.to_list()

        tgt_context_list = [
            {
                Fields.context: {
                    f'{InterVars.lines}-{self.text_key}': tgt[self.text_key].splitlines()
                    }
            } for tgt in self.tgt_list
        ]

        self.assertEqual(res_list, tgt_context_list)

    def test_reuses_fused_context_lines(self):
        key = f"{InterVars.lines}-{self.text_key}"
        lines = ["already", "split"]
        sample = {
            self.text_key: ["abcdefgh"],
            Fields.stats: [{}],
            Fields.context: [{key: lines}],
        }
        AverageLineLengthFilter().compute_stats_batched(sample, context=True)
        self.assertEqual(sample[Fields.stats][0][StatsKeys.avg_line_length], 4.0)
        self.assertIs(sample[Fields.context][0][key], lines)


if __name__ == '__main__':
    unittest.main()

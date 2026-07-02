import unittest

from data_juicer.core.data import NestedDataset as Dataset

from data_juicer.ops.filter.suffix_filter import SuffixFilter
from data_juicer.utils.constant import Fields
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class SuffixFilterTest(DataJuicerTestCaseBase):

    def _run_suffix_filter(self, dataset: Dataset, target_list, op):
        dataset = dataset.map(op.compute_stats)
        dataset = dataset.filter(op.process)
        res_list = dataset.to_list()
        self.assertEqual(res_list, target_list)

    def test_case(self):

        ds_list = [{
            'text': 'Today is Sun',
            Fields.suffix: '.pdf'
        }, {
            'text': 'a v s e c s f e f g a a a  ',
            Fields.suffix: '.docx'
        }, {
            'text': '中文也是一个字算一个长度',
            Fields.suffix: '.txt'
        }, {
            'text': '，。、„”“«»１」「《》´∶：？！',
            Fields.suffix: '.html'
        }, {
            'text': 'dasdasdasdasdasdasdasd',
            Fields.suffix: '.py'
        }]
        tgt_list = [{
            'text': 'Today is Sun',
            Fields.suffix: '.pdf'
        }, {
            'text': '中文也是一个字算一个长度',
            Fields.suffix: '.txt'
        }]
        dataset = Dataset.from_list(ds_list)
        op = SuffixFilter(suffixes=['.txt', '.pdf'])
        self._run_suffix_filter(dataset, tgt_list, op)

    def test_none_case(self):

        ds_list = [{
            'text': 'Today is Sun',
            Fields.suffix: '.pdf'
        }, {
            'text': 'a v s e c s f e f g a a a  ',
            Fields.suffix: '.docx'
        }, {
            'text': '中文也是一个字算一个长度',
            Fields.suffix: '.txt'
        }, {
            'text': '，。、„”“«»１」「《》´∶：？！',
            Fields.suffix: '.html'
        }, {
            'text': 'dasdasdasdasdasdasdasd',
            Fields.suffix: '.py'
        }]
        tgt_list = [{
            'text': 'Today is Sun',
            Fields.suffix: '.pdf'
        }, {
            'text': 'a v s e c s f e f g a a a  ',
            Fields.suffix: '.docx'
        }, {
            'text': '中文也是一个字算一个长度',
            Fields.suffix: '.txt'
        }, {
            'text': '，。、„”“«»１」「《》´∶：？！',
            Fields.suffix: '.html'
        }, {
            'text': 'dasdasdasdasdasdasdasd',
            Fields.suffix: '.py'
        }]
        dataset = Dataset.from_list(ds_list)
        op = SuffixFilter()
        self._run_suffix_filter(dataset, tgt_list, op)


    def test_string_suffix(self):
        """Single string suffix (not list) is handled correctly."""
        ds_list = [
            {'text': 'hello', Fields.suffix: '.txt'},
            {'text': 'world', Fields.suffix: '.py'},
        ]
        tgt_list = [{'text': 'hello', Fields.suffix: '.txt'}]
        dataset = Dataset.from_list(ds_list)
        op = SuffixFilter(suffixes='.txt')
        self._run_suffix_filter(dataset, tgt_list, op)

    def test_none_suffixes(self):
        """suffixes=None keeps all samples."""
        ds_list = [
            {'text': 'hello', Fields.suffix: '.txt'},
            {'text': 'world', Fields.suffix: '.py'},
        ]
        dataset = Dataset.from_list(ds_list)
        op = SuffixFilter(suffixes=None)
        self._run_suffix_filter(dataset, ds_list, op)

    def test_reversed_range(self):
        """reversed_range=True inverts the filter logic."""
        ds_list = [
            {'text': 'hello', Fields.suffix: '.txt'},
            {'text': 'world', Fields.suffix: '.py'},
        ]
        tgt_list = [{'text': 'world', Fields.suffix: '.py'}]
        dataset = Dataset.from_list(ds_list)
        op = SuffixFilter(suffixes=['.txt'], reversed_range=True)
        self._run_suffix_filter(dataset, tgt_list, op)


if __name__ == '__main__':
    unittest.main()

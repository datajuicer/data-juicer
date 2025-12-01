import unittest

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.filter.general_field_filter import GeneralFieldFilter
from data_juicer.utils.constant import Fields
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class GeneralFieldFilterTest(DataJuicerTestCaseBase):

    def _run_general_field_filter(self, dataset: Dataset, op, target: list):
        if Fields.stats not in dataset.features:
            dataset = dataset.add_column(name=Fields.stats,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.compute_stats)
        dataset = dataset.filter(op.process)
        dataset = dataset.select_columns(column_names=['text'])
        res_list = dataset.to_list()
        self.assertEqual(res_list, target)

    def test_simple_comparison(self):
        ds_list = [
            {'text': 'sample1', 'num': 5},
            {'text': 'sample2', 'num': 15},
            {'text': 'sample3', 'num': 25},
        ]
        target_list = [{'text': 'sample2'}]
        dataset = Dataset.from_list(ds_list)
        op = GeneralFieldFilter(filter_condition="10 < num < 20")
        self._run_general_field_filter(dataset, op, target_list)

    def test_logical_and(self):
        ds_list = [
            {'text': 'sample1', 'num': 5, 'flag': True},
            {'text': 'sample2', 'num': 15, 'flag': False},
            {'text': 'sample3', 'num': 25, 'flag': True},
        ]
        target_list = [{'text': 'sample1'}]
        dataset = Dataset.from_list(ds_list)
        op = GeneralFieldFilter(filter_condition="num < 10 and flag == True")
        self._run_general_field_filter(dataset, op, target_list)

    def test_logical_or(self):
        ds_list = [
            {'text': 'sample1', 'num': 5},
            {'text': 'sample2', 'num': 15},
            {'text': 'sample3', 'num': 25},
        ]
        target_list = [{'text': 'sample1'}, {'text': 'sample3'}]
        dataset = Dataset.from_list(ds_list)
        op = GeneralFieldFilter(filter_condition="num < 10 or num > 20")
        self._run_general_field_filter(dataset, op, target_list)

    def test_field_missing(self):
        ds_list = [
            {'text': 'sample1', 'num': 5},
            {'text': 'sample2'},
            {'text': 'sample3', 'num': 25},
        ]
        target_list = [{'text': 'sample1'}]
        dataset = Dataset.from_list(ds_list)
        op = GeneralFieldFilter(filter_condition="num <= 5")
        self._run_general_field_filter(dataset, op, target_list)

    def test_nested_field(self):
        ds_list = [
            {'text': 'sample1', '__dj__meta__': {'a': 1}},
            {'text': 'sample2', '__dj__meta__': {'a': 2}},
            {'text': 'sample3', '__dj__meta__': {'a': 3}},
        ]
        target_list = [{'text': 'sample2'}]
        dataset = Dataset.from_list(ds_list)
        op = GeneralFieldFilter(filter_condition="__dj__meta__.a == 2")
        self._run_general_field_filter(dataset, op, target_list)

    def test_combined_conditions(self):
        ds_list = [
            {'text': 'sample1', 'num': 5, '__dj__meta__': {'a': 1}},
            {'text': 'sample2', 'num': 15, '__dj__meta__': {'a': 2}},
            {'text': 'sample3', 'num': 25, '__dj__meta__': {'a': 3}},
        ]
        target_list = [{'text': 'sample2'}]
        dataset = Dataset.from_list(ds_list)
        op = GeneralFieldFilter(filter_condition="10 < num < 20 and __dj__meta__.a == 2")
        self._run_general_field_filter(dataset, op, target_list)

    def test_invalid_condition(self):
        ds_list = [{'text': 'sample1', 'num': 5}]
        dataset = Dataset.from_list(ds_list)
        with self.assertRaises(ValueError):
            GeneralFieldFilter(filter_condition="invalid syntax")

    def test_empty_condition(self):
        ds_list = [
            {'text': 'sample1', 'num': 5},
            {'text': 'sample2', 'num': 15},
        ]
        target_list = [{'text': 'sample1'}, {'text': 'sample2'}]
        dataset = Dataset.from_list(ds_list)
        op = GeneralFieldFilter(filter_condition="")
        self._run_general_field_filter(dataset, op, target_list)

    def test_complex_condition(self):
        ds_list = [
            {'text': 'sample1', 'num': 5, 'flag': True},
            {'text': 'sample2', 'num': 15, 'flag': False},
            {'text': 'sample3', 'num': 25, 'flag': True},
            {'text': 'sample4', 'num': 30, 'flag': False},
        ]
        target_list = [{'text': 'sample1'}]
        dataset = Dataset.from_list(ds_list)
        op = GeneralFieldFilter(filter_condition="(num < 10 or num > 20) and flag == True and text!='sample3'")
        self._run_general_field_filter(dataset, op, target_list)


if __name__ == '__main__':
    unittest.main()

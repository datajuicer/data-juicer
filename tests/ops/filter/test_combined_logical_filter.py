import unittest

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.filter.combined_logical_filter import CombinedLogicalFilter
from data_juicer.utils.constant import Fields
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class CombinedLogicalFilterTest(DataJuicerTestCaseBase):

    def _run_combined_logical_filter(self, dataset: Dataset, target_list, op):
        if Fields.stats not in dataset.features:
            dataset = dataset.add_column(name=Fields.stats, column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.compute_stats, batch_size=3)
        dataset = dataset.filter(op.process, batch_size=2)
        dataset = dataset.select_columns(column_names=['text'])
        res_list = dataset.to_list()
        self.assertEqual(res_list, target_list)

    def test_and_operation(self):
        """Test AND operation: sample must satisfy both filters."""
        ds_list = [
            {'text': 'short'},  # Too short, filtered by first filter
            {'text': 'This is a medium length text that should pass both filters'},  # Passes both
            {'text': 'This is a very long text that exceeds the maximum length limit and should be filtered'},  # Too long, filtered by first filter
        ]
        target_list = [
            {'text': 'This is a medium length text that should pass both filters'}
        ]
        dataset = Dataset.from_list(ds_list)
        op = CombinedLogicalFilter(
            filter_ops=[
                {"text_length_filter": {"min_len": 20, "max_len": 100}},
                {"text_length_filter": {"min_len": 10, "max_len": 200}}
            ],
            logical_op="and"
        )
        self._run_combined_logical_filter(dataset, target_list, op)

    def test_or_operation(self):
        """Test OR operation: sample must satisfy at least one filter."""
        ds_list = [
            {'text': 'short'},  # Too short for first filter, but passes second
            {'text': 'This is a medium length text'},  # Passes both
            {'text': 'This is a very long text that exceeds the maximum length limit'},  # Too long for first, but passes second
            {'text': 'x'},  # Too short for both, filtered
        ]
        target_list = [
            {'text': 'short'},
            {'text': 'This is a medium length text'},
            {'text': 'This is a very long text that exceeds the maximum length limit'}
        ]
        dataset = Dataset.from_list(ds_list)
        op = CombinedLogicalFilter(
            filter_ops=[
                {"text_length_filter": {"min_len": 20, "max_len": 50}},
                {"text_length_filter": {"min_len": 5, "max_len": 200}}
            ],
            logical_op="or"
        )
        self._run_combined_logical_filter(dataset, target_list, op)

    def test_single_filter(self):
        """Test with a single filter (should work like a normal filter)."""
        ds_list = [
            {'text': 'short'},
            {'text': 'This is a medium length text'},
            {'text': 'This is a very long text that exceeds the maximum length limit'},
        ]
        target_list = [
            {'text': 'This is a medium length text'}
        ]
        dataset = Dataset.from_list(ds_list)
        op = CombinedLogicalFilter(
            filter_ops=[
                {"text_length_filter": {"min_len": 20, "max_len": 50}}
            ],
            logical_op="and"
        )
        self._run_combined_logical_filter(dataset, target_list, op)

    def test_default_and(self):
        """Test that default logical_op is 'and'."""
        ds_list = [
            {'text': 'short'},
            {'text': 'This is a medium length text'},
        ]
        target_list = [
            {'text': 'This is a medium length text'}
        ]
        dataset = Dataset.from_list(ds_list)
        # Don't specify logical_op, should default to "and"
        op = CombinedLogicalFilter(
            filter_ops=[
                {"text_length_filter": {"min_len": 20, "max_len": 50}}
            ]
        )
        self._run_combined_logical_filter(dataset, target_list, op)

    def test_empty_filter_ops(self):
        """Test that empty filter_ops raises ValueError."""
        with self.assertRaises(ValueError) as context:
            CombinedLogicalFilter(filter_ops=[])
        self.assertIn("cannot be empty", str(context.exception))

    def test_invalid_logical_op(self):
        """Test that invalid logical_op raises ValueError."""
        with self.assertRaises(ValueError) as context:
            CombinedLogicalFilter(
                filter_ops=[{"text_length_filter": {"min_len": 10}}],
                logical_op="xor"
            )
        self.assertIn("must be 'and' or 'or'", str(context.exception))

    def test_case_insensitive_logical_op(self):
        """Test that logical_op is case-insensitive."""
        ds_list = [
            {'text': 'short'},
            {'text': 'This is a medium length text'},
        ]
        target_list = [
            {'text': 'This is a medium length text'}
        ]
        dataset = Dataset.from_list(ds_list)
        # Test uppercase
        op = CombinedLogicalFilter(
            filter_ops=[{"text_length_filter": {"min_len": 20, "max_len": 50}}],
            logical_op="AND"
        )
        self._run_combined_logical_filter(dataset, target_list, op)

    def test_multiple_filters_and(self):
        """Test AND operation with multiple filters."""
        ds_list = [
            {'text': 'short'},  # Fails length check
            {'text': 'This is a medium length text'},  # Passes all
            {'text': 'This is a very long text that exceeds the maximum length limit'},  # Fails length check
        ]
        target_list = [
            {'text': 'This is a medium length text'}
        ]
        dataset = Dataset.from_list(ds_list)
        op = CombinedLogicalFilter(
            filter_ops=[
                {"text_length_filter": {"min_len": 20, "max_len": 100}},
                {"text_length_filter": {"min_len": 10, "max_len": 200}},
                {"text_length_filter": {"min_len": 15, "max_len": 150}}
            ],
            logical_op="and"
        )
        self._run_combined_logical_filter(dataset, target_list, op)

    def test_multiple_filters_or(self):
        """Test OR operation with multiple filters."""
        ds_list = [
            {'text': 'short'},  # Fails all
            {'text': 'This is a medium length text'},  # Passes all
            {'text': 'This is a very long text that exceeds the maximum length limit'},  # Passes some
        ]
        target_list = [
            {'text': 'This is a medium length text'},
            {'text': 'This is a very long text that exceeds the maximum length limit'}
        ]
        dataset = Dataset.from_list(ds_list)
        op = CombinedLogicalFilter(
            filter_ops=[
                {"text_length_filter": {"min_len": 20, "max_len": 50}},  # medium passes, long fails
                {"text_length_filter": {"min_len": 5, "max_len": 200}},  # all pass
                {"text_length_filter": {"min_len": 100, "max_len": 300}}  # only long passes
            ],
            logical_op="or"
        )
        self._run_combined_logical_filter(dataset, target_list, op)


if __name__ == '__main__':
    unittest.main()

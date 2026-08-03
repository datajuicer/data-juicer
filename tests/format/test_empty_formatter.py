import os
import unittest

from data_juicer.format.empty_formatter import EmptyFormatter
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class EmptyFormatterTest(DataJuicerTestCaseBase):

    text_key = 'text'

    def test_empty_dataset(self):
        ds_len = 10
        formatter = EmptyFormatter(length=ds_len, feature_keys=[self.text_key])
        ds = formatter.load_dataset()

        self.assertEqual(len(ds), ds_len)
        self.assertEqual(list(ds.features.keys()), [self.text_key])

        for item in ds:
            self.assertDictEqual(item, {self.text_key: None})

        # test map
        update_column = {self.text_key: 1}

        def map_fn(sample):
            sample.update(update_column)
            return sample

        ds = ds.map(map_fn)
        self.assertEqual(len(ds), ds_len)
        for item in ds:
            self.assertDictEqual(item, update_column)

        # test filter
        def filter_fn(sample):
            return sample[self.text_key] > 2
        
        ds = ds.filter(filter_fn)
        self.assertEqual(len(ds), 0)


    def test_multiple_feature_keys(self):
        """Multiple feature_keys should create columns for each key."""
        keys = ['text', 'meta', 'label']
        ds_len = 5
        formatter = EmptyFormatter(length=ds_len, feature_keys=keys)
        ds = formatter.load_dataset()

        self.assertEqual(len(ds), ds_len)
        self.assertEqual(sorted(ds.features.keys()), sorted(keys))
        for item in ds:
            for key in keys:
                self.assertIsNone(item[key])

    def test_string_feature_keys_converted_to_list(self):
        """A single string feature_key should be auto-wrapped into a list."""
        formatter = EmptyFormatter(length=3, feature_keys='text')
        self.assertIsInstance(formatter.feature_keys, list)
        self.assertEqual(formatter.feature_keys, ['text'])

        ds = formatter.load_dataset()
        self.assertEqual(len(ds), 3)
        self.assertEqual(list(ds.features.keys()), ['text'])

    def test_null_value_property(self):
        """EmptyFormatter.null_value should return None."""
        formatter = EmptyFormatter(length=1, feature_keys=['text'])
        self.assertIsNone(formatter.null_value)

    def test_zero_length(self):
        """length=0 should produce an empty dataset with correct schema."""
        formatter = EmptyFormatter(length=0, feature_keys=['text'])
        ds = formatter.load_dataset()
        self.assertEqual(len(ds), 0)
        self.assertIn('text', ds.features)

    def test_no_feature_keys(self):
        """Empty feature_keys list produces a 0-row dataset (no columns → no rows)."""
        formatter = EmptyFormatter(length=3, feature_keys=[])
        ds = formatter.load_dataset()
        # Dataset.from_dict({}) yields 0 rows regardless of requested length
        self.assertEqual(len(ds), 0)
        self.assertEqual(list(ds.features.keys()), [])


if __name__ == '__main__':
    unittest.main()

import os
import unittest

from data_juicer.format.json_formatter import JsonFormatter
from data_juicer.format.load import load_formatter
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class JsonFormatterTest(DataJuicerTestCaseBase):

    def setUp(self):
        super().setUp()

        self._path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                  'data', 'structured')
        self._file = os.path.join(self._path, 'demo-dataset.jsonl')
        print(self._file)

    def test_json_file(self):
        formatter = JsonFormatter(self._file)
        ds = formatter.load_dataset()
        self.assertEqual(len(ds), 6)
        self.assertEqual(list(ds.features.keys()), ['text', 'meta'])

    def test_json_path(self):
        formatter = JsonFormatter(self._path)
        ds = formatter.load_dataset()
        self.assertEqual(len(ds), 6)
        self.assertEqual(list(ds.features.keys()), ['text', 'meta'])

    def test_load_formatter_with_file(self):
        """Test load_formatter with a direct file path"""
        formatter = load_formatter(self._file)
        self.assertIsInstance(formatter, JsonFormatter)
        ds = formatter.load_dataset()
        self.assertEqual(len(ds), 6)
        self.assertEqual(list(ds.features.keys()), ['text', 'meta'])

    def test_load_formatter_with_specified_suffix(self):
        """Test load_formatter with specified suffixes"""
        formatter = load_formatter(self._path, suffixes=['.jsonl'])
        self.assertIsInstance(formatter, JsonFormatter)
        ds = formatter.load_dataset()
        self.assertEqual(len(ds), 6)
        self.assertEqual(list(ds.features.keys()), ['text', 'meta'])


if __name__ == '__main__':
    unittest.main() 
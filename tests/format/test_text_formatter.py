import os
import unittest

from data_juicer.format.text_formatter import TextFormatter
from data_juicer.format.load import load_formatter
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class TextFormatterTest(DataJuicerTestCaseBase):

    def setUp(self):
        super().setUp()

        self._path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "text")
        self._file = os.path.join(self._path, "sample1.txt")

    def test_text_file(self):
        formatter = TextFormatter(self._file)
        ds = formatter.load_dataset()
        self.assertEqual(len(ds), 1)
        self.assertIn("text", list(ds.features.keys()))

    def test_text_path(self):
        formatter = TextFormatter(self._path)
        ds = formatter.load_dataset()
        self.assertEqual(len(ds), 6)
        self.assertIn("text", list(ds.features.keys()))

    def test_text_path_with_suffixes(self):
        formatter = TextFormatter(self._path, suffixes=[".txt"])
        ds = formatter.load_dataset()
        self.assertEqual(len(ds), 6)
        self.assertIn("text", list(ds.features.keys()))

    def test_text_path_with_add_suffix(self):
        formatter = TextFormatter(self._path, add_suffix=True)
        ds = formatter.load_dataset()
        self.assertEqual(len(ds), 6)
        self.assertIn("text", list(ds.features.keys()))
        self.assertIn("__dj__suffix__", list(ds.features.keys()))

    def test_load_formatter_with_file(self):
        """Test load_formatter with a direct text file path"""
        formatter = load_formatter(self._file)
        self.assertIsInstance(formatter, TextFormatter)
        ds = formatter.load_dataset()
        self.assertEqual(len(ds), 1)
        self.assertIn("text", list(ds.features.keys()))

    def test_load_formatter_with_specified_suffix(self):
        """Test load_formatter with specified suffixes"""
        formatter = load_formatter(self._path, suffixes=[".txt"])
        self.assertIsInstance(formatter, TextFormatter)
        ds = formatter.load_dataset()
        self.assertEqual(len(ds), 6)
        self.assertIn("text", list(ds.features.keys()))


if __name__ == "__main__":
    unittest.main()

import os
import unittest

from data_juicer.format.csv_formatter import CsvFormatter
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class CsvFormatterTest(DataJuicerTestCaseBase):

    def setUp(self):
        super().setUp()

        self._path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                  'data', 'structured')
        self._file = os.path.join(self._path, 'demo-dataset.csv')
        print(self._file)

    def test_csv_file(self):
        formatter = CsvFormatter(self._file)
        ds = formatter.load_dataset()
        self.assertEqual(len(ds), 6)
        self.assertEqual(list(ds.features.keys()), ['text', 'meta'])

    def test_csv_path(self):
        formatter = CsvFormatter(self._path)
        ds = formatter.load_dataset()
        self.assertEqual(len(ds), 6)
        self.assertEqual(list(ds.features.keys()), ['text', 'meta'])


if __name__ == '__main__':
    unittest.main()

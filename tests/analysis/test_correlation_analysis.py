import os
import unittest
import pandas as pd
import numpy as np

from data_juicer.analysis.correlation_analysis import is_numeric_list_series, CorrelationAnalysis
from data_juicer.utils.constant import Fields

from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

class CorrelationAnalysisTest(DataJuicerTestCaseBase):

    def setUp(self):
        super().setUp()

        self.df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [[1, 2, 3], [4, 5], [6]],
            'C': ['a', 'b', 'c'],
            'D': [[1, 'a'], [2, 3], [4]],
            'E': [[], [7, 8, 9], [10.5]],
            'F': [True, False, True],
            'G': [[1, 2], None, [3, 4]],
            'H': [1.1, 2.2, 3.3],
            'I': [[1, 2], [3, 4], [5, 6]],
            'J': [np.nan, np.nan, np.nan],
            'K': [[], [], []],
        })
        self.temp_output_path = 'tmp/test_correlation_analysis/'

    def tearDown(self):
        if os.path.exists(self.temp_output_path):
            os.system(f'rm -rf {self.temp_output_path}')

        super().tearDown()

    def test_is_numeric_list_series(self):
        res = {
            'A': False,
            'B': True,
            'C': False,
            'D': False,
            'E': True,
            'F': False,
            'G': True,
            'H': False,
            'I': True,
            'J': False,
            'K': False,
        }
        temp = self.df.copy()
        for col in temp.columns:
            self.assertEqual(is_numeric_list_series(temp[col]), res[col])

    def test_correlation_analysis(self):
        corr_analyzer = CorrelationAnalysis({Fields.stats: self.df}, self.temp_output_path)
        self.assertEqual(set(corr_analyzer.stats.columns), {'A', 'B', 'E', 'G', 'H', 'I', 'J'})

        ret = corr_analyzer.analyze(skip_export=True)
        self.assertFalse(os.path.exists(os.path.join(self.temp_output_path, 'stats-corr-pearson.png')))
        self.assertIsInstance(ret, pd.DataFrame)

        ret = corr_analyzer.analyze()
        self.assertTrue(os.path.exists(os.path.join(self.temp_output_path, 'stats-corr-pearson.png')))
        self.assertIsInstance(ret, pd.DataFrame)

        with self.assertRaises(AssertionError):
            _ = corr_analyzer.analyze(method='unknown_method')

        corr_analyzer = CorrelationAnalysis({Fields.stats: {}}, self.temp_output_path)
        ret = corr_analyzer.analyze()
        self.assertIsNone(ret)


if __name__ == '__main__':
    unittest.main()

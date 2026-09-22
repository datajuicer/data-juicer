import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

ROOT = Path(__file__).resolve().parents[2]


class AppTest(DataJuicerTestCaseBase):
    def setUp(self):
        super().setUp()
        spec = importlib.util.spec_from_file_location("dj_app", ROOT / "app.py")
        self.app = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.app)

    def test_combine_conds_without_cond(self):
        # None of the OPs in the config is covered by op_stats_dict, so no
        # condition is collected. The combined mask must still cover every
        # sample: a scalar would be read as a label by DataFrame.loc and raise
        # KeyError instead of selecting rows.
        dataframe = pd.DataFrame({"text": ["a", "b", "c"]})
        all_conds = self.app.combine_conds([], len(dataframe))
        self.assertEqual(len(all_conds), len(dataframe))
        self.assertTrue(np.all(all_conds))
        self.assertEqual(len(dataframe.loc[all_conds]), 3)
        self.assertEqual(len(dataframe.loc[np.invert(all_conds)]), 0)

    def test_combine_conds_with_conds(self):
        dataframe = pd.DataFrame({"text": ["a", "b", "c"]})
        conds = [
            {("1 text_length_filter", "text_len"): [True, True, False]},
            {("2 words_num_filter", "num_words"): [True, False, True]},
        ]
        all_conds = self.app.combine_conds(conds, len(dataframe))
        self.assertEqual(list(all_conds), [True, False, False])
        self.assertEqual(len(dataframe.loc[all_conds]), 1)


if __name__ == "__main__":
    unittest.main()

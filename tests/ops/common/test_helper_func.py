import unittest

from data_juicer.ops.common.helper_func import split_on_whitespace
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

class HelperFuncTest(DataJuicerTestCaseBase):

    def test_split_on_whitespace_literal_separators(self):
        text = "  a\nb\tc\rd\u00a0e\u2003f\vg  h  "
        cases = {
            (False, False): ["a\nb\tc\rd\u00a0e\u2003f\vg", "h"],
            (True, False): ["a", "b\tc\rd\u00a0e\u2003f\vg", "h"],
            (False, True): ["a\nb", "c\rd\u00a0e\u2003f\vg", "h"],
            (True, True): ["a", "b", "c\rd\u00a0e\u2003f\vg", "h"],
        }
        for (new_line, tab), expected in cases.items():
            with self.subTest(new_line=new_line, tab=tab):
                self.assertEqual(split_on_whitespace(text, new_line, tab), expected)
                self.assertEqual(split_on_whitespace("", new_line, tab), [])
        self.assertEqual(split_on_whitespace(" \t\n\t \n ", True, True), [])


if __name__ == '__main__':
    unittest.main()

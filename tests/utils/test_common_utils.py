import unittest
import sys

from data_juicer.utils.common_utils import (
    stats_to_number, dict_to_hash, nested_access, is_string_list,
    avg_split_string_list_under_limit, is_float
)

from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

class CommonUtilsTest(DataJuicerTestCaseBase):

    def test_stats_to_number(self):
        self.assertEqual(stats_to_number('1.0'), 1.0)
        self.assertEqual(stats_to_number([1.0, 2.0, 3.0]), 2.0)

        self.assertEqual(stats_to_number([]), -sys.maxsize)
        self.assertEqual(stats_to_number(None), -sys.maxsize)
        self.assertEqual(stats_to_number([], reverse=False), sys.maxsize)
        self.assertEqual(stats_to_number(None, reverse=False), sys.maxsize)

    def test_dict_to_hash(self):
        self.assertEqual(len(dict_to_hash({'a': 1, 'b': 2})), 64)
        self.assertEqual(len(dict_to_hash({'a': 1, 'b': 2}, hash_length=32)), 32)

    def test_nested_access(self):
        self.assertEqual(nested_access({'a': {'b': 1}}, 'a.b'), 1)
        self.assertEqual(nested_access({'a': [{'b': 1}]}, 'a.0.b', digit_allowed=True), 1)
        self.assertEqual(nested_access({'a': [{'b': 1}]}, 'a.0.b', digit_allowed=False), None)

    def test_is_string_list(self):
        self.assertTrue(is_string_list(['a', 'b', 'c']))
        self.assertFalse(is_string_list([1, 2, 3]))
        self.assertFalse(is_string_list(['a', 2, 'c']))

    def test_is_float(self):
        self.assertTrue(is_float('1.0'))
        self.assertTrue(is_float(1.0))
        self.assertTrue(is_float('1e-4'))
        self.assertFalse(is_float('a'))

    def test_avg_split_string_list_under_limit(self):
        test_data = [
            (['a', 'b', 'c'], [1, 2, 3], None, [['a', 'b', 'c']]),
            (['a', 'b', 'c'], [1, 2, 3], 3, [['a', 'b'], ['c']]),
            (['a', 'b', 'c'], [1, 2, 3], 2, [['a'], ['b'], ['c']]),
            (['a', 'b', 'c', 'd', 'e'], [1, 2, 3, 1, 1], 3, [['a', 'b'], ['c'], ['d', 'e']]),
            (['a', 'b', 'c'], [1, 2], 3, [['a', 'b', 'c']]),
            (['a', 'b', 'c'], [1, 2, 3], 100, [['a', 'b', 'c']]),
        ]

        for str_list, token_nums, max_token_num, expected_result in test_data:
            self.assertEqual(avg_split_string_list_under_limit(str_list, token_nums, max_token_num), expected_result)


if __name__ == '__main__':
    unittest.main()

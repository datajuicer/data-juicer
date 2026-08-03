import sys
import unittest
import warnings

import numpy as np

from data_juicer.utils.common_utils import (
    avg_split_string_list_under_limit,
    check_op_method_param,
    deprecated,
    dict_to_hash,
    is_float,
    is_string_list,
    nested_access,
    stats_to_number,
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


class CheckOpMethodParamTest(DataJuicerTestCaseBase):
    """Test check_op_method_param: checks if method has named param or **kwargs."""

    def test_finds_named_param(self):
        def example(x, target_param, y):
            pass
        self.assertTrue(check_op_method_param(example, 'target_param'))

    def test_missing_param_returns_false(self):
        def example(x, y):
            pass
        self.assertFalse(check_op_method_param(example, 'missing'))

    def test_finds_var_keyword(self):
        """If method has **kwargs, any param name should return True."""
        def example(x, **kwargs):
            pass
        self.assertTrue(check_op_method_param(example, 'anything'))

    def test_no_params(self):
        def example():
            pass
        self.assertFalse(check_op_method_param(example, 'x'))

    def test_self_param_on_method(self):
        class Dummy:
            def method(self, context):
                pass
        self.assertTrue(check_op_method_param(Dummy.method, 'context'))
        self.assertFalse(check_op_method_param(Dummy.method, 'missing'))


class DeprecatedDecoratorTest(DataJuicerTestCaseBase):
    """Test deprecated decorator: marks functions as deprecated with warnings."""

    def test_bare_decorator(self):
        """@deprecated without arguments should emit DeprecationWarning."""
        @deprecated
        def old_func():
            return 42

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = old_func()

        self.assertEqual(result, 42)
        self.assertEqual(len(caught), 1)
        self.assertTrue(issubclass(caught[0].category, DeprecationWarning))
        self.assertIn("old_func", str(caught[0].message))

    def test_with_reason(self):
        @deprecated(reason="Use new_func instead")
        def old_func():
            return 1

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            old_func()

        self.assertIn("Use new_func instead", str(caught[0].message))

    def test_with_version(self):
        @deprecated(reason="Outdated", version="2.0")
        def old_func():
            return 1

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            old_func()

        msg = str(caught[0].message)
        self.assertIn("Outdated", msg)
        self.assertIn("2.0", msg)

    def test_preserves_function_name(self):
        @deprecated(reason="old")
        def my_special_func():
            pass
        self.assertEqual(my_special_func.__name__, "my_special_func")

    def test_invalid_reason_type_raises(self):
        with self.assertRaises(TypeError):
            @deprecated(reason=123)
            def func():
                pass

    def test_invalid_version_type_raises(self):
        with self.assertRaises(TypeError):
            @deprecated(version=123)
            def func():
                pass

    def test_bare_decorator_no_parens_works(self):
        """@deprecated without parens should work and return a wrapper."""
        @deprecated
        def old_func():
            return "result"

        self.assertEqual(old_func.__name__, "old_func")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertEqual(old_func(), "result")
        self.assertEqual(len(caught), 1)


class CommonUtilsEdgeCaseTest(DataJuicerTestCaseBase):
    """Additional edge-case tests for common_utils functions."""

    def test_stats_to_number_float_input(self):
        """Pass a float directly (not a string)."""
        self.assertEqual(stats_to_number(3.14), 3.14)

    def test_stats_to_number_numpy_array(self):
        """Pass a numpy array -- should return the mean."""
        arr = np.array([2.0, 4.0, 6.0])
        self.assertAlmostEqual(stats_to_number(arr), 4.0)

    def test_avg_split_string_list_under_limit_single_token_exceeds(self):
        """A single token exceeds max_len, triggering the warning path."""
        result = avg_split_string_list_under_limit(
            ['big', 'small'], [100, 1], max_token_num=10)
        # 'big' alone exceeds 10, but still gets placed in a group
        # Verify we get groups and all items are present
        flat = [item for group in result for item in group]
        self.assertEqual(sorted(flat), ['big', 'small'])
        # The big token should be in its own group
        self.assertTrue(any('big' in group for group in result))

    def test_deprecated_empty_parens(self):
        """@deprecated() with empty parentheses should work."""
        @deprecated()
        def old_func():
            return 99

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = old_func()

        self.assertEqual(result, 99)
        self.assertEqual(len(caught), 1)
        self.assertTrue(issubclass(caught[0].category, DeprecationWarning))
        self.assertIn("old_func", str(caught[0].message))

    def test_deprecated_version_only(self):
        """@deprecated(version='2.0') without reason should include version."""
        @deprecated(version="2.0")
        def old_func():
            return 7

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = old_func()

        self.assertEqual(result, 7)
        self.assertEqual(len(caught), 1)
        msg = str(caught[0].message)
        self.assertIn("2.0", msg)
        self.assertIn("old_func", msg)

    def test_nested_access_deep(self):
        """Test 3+ levels of nesting."""
        data = {'a': {'b': {'c': {'d': 42}}}}
        self.assertEqual(nested_access(data, 'a.b.c.d'), 42)

    def test_nested_access_missing_intermediate(self):
        """Test where an intermediate key does not exist."""
        data = {'a': {'b': 1}}
        result = nested_access(data, 'a.x.y')
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()

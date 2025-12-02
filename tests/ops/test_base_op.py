import unittest

from data_juicer.ops.filter import AudioDurationFilter
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class BaseOPTest(DataJuicerTestCaseBase):

    def test_filter_get_keep_boolean(self):
        # test cases with tuple (min_closed_interval, max_closed_interval, reversed_range, val, min_val, max_val, tgt)
        test_cases = [
            # normal ranges
            (True, True, False, 5, 1, 10, True),
            (True, True, False, 5, None, 10, True),
            (True, True, False, 5, 1, None, True),
            (True, True, False, 5, None, None, True),
            # marginal cases
            (True, True, False, 5, 1, 5, True),
            (True, True, False, 5, 5, 10, True),
            (True, True, False, 5, 5, 5, True),
            (True, True, False, 5, 1, 4, False),
            (True, True, False, 5, 6, 10, False),
            # open intervals
            (True, False, False, 5, 1, 10, True),
            (True, False, False, 5, 5, 10, True),
            (True, False, False, 5, 1, 5, False),
            (False, True, False, 5, 1, 10, True),
            (False, True, False, 5, 5, 10, False),
            (False, True, False, 5, 1, 5, True),
            # reversed ranges
            (True, True, True, 5, 1, 10, False),
            (True, True, True, 5, None, 10, False),
            (True, True, True, 5, 1, None, False),
            (True, True, True, 5, None, None, False),
            (True, True, True, 5, 1, 5, True),
            (True, True, True, 5, 5, 10, True),
            (True, True, True, 5, 5, 5, True),
            (False, True, True, 5, 1, 5, True),
            (False, True, True, 5, 5, 10, False),
            (False, True, True, 5, 5, 5, True),
            (True, False, True, 5, 1, 5, False),
            (True, False, True, 5, 5, 10, True),
            (True, False, True, 5, 5, 5, True),
            (False, False, True, 5, 1, 5, False),
            (False, False, True, 5, 5, 10, False),
            (False, False, True, 5, 5, 5, False),
        ]
        for tc in test_cases:
            min_closed_interval, max_closed_interval, reversed_range, val, min_val, max_val, tgt = tc
            op = AudioDurationFilter(min_closed_interval=min_closed_interval,
                                     max_closed_interval=max_closed_interval,
                                     reversed_range=reversed_range)
            self.assertEqual(
                op.get_keep_boolean(val, min_val, max_val), tgt)


if __name__ == '__main__':
    unittest.main()

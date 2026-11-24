import unittest

import data_juicer.utils.unittest_utils as unittest_utils
from data_juicer.utils.unittest_utils import TEST_TAG, skip_if_from_fork


class SkipIfFromForkTest(unittest.TestCase):
    def setUp(self):
        self._from_fork = unittest_utils.FROM_FORK

    def tearDown(self):
        unittest_utils.set_from_fork_flag(self._from_fork)

    def test_class_decorator_preserves_testcase_discovery(self):
        @skip_if_from_fork("skip")
        class DecoratedCase(unittest.TestCase):
            def test_example(self):
                pass

        suite = unittest.defaultTestLoader.loadTestsFromTestCase(DecoratedCase)

        self.assertTrue(isinstance(DecoratedCase, type))
        self.assertTrue(issubclass(DecoratedCase, unittest.TestCase))
        self.assertEqual(suite.countTestCases(), 1)

    def test_class_decorator_skips_before_class_setup_when_from_fork(self):
        calls = []

        @skip_if_from_fork("skip")
        class DecoratedCase(unittest.TestCase):
            @classmethod
            def setUpClass(cls):
                calls.append("setUpClass")

            def test_example(self):
                calls.append("test_example")

        unittest_utils.set_from_fork_flag(True)
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(DecoratedCase)
        result = unittest.TestResult()
        suite.run(result)

        self.assertEqual(calls, [])
        self.assertEqual(len(result.skipped), 1)
        self.assertEqual(result.failures, [])
        self.assertEqual(result.errors, [])

    def test_method_decorator_preserves_test_tags(self):
        @skip_if_from_fork("skip")
        @TEST_TAG("ray")
        def tagged_test():
            pass

        self.assertEqual(tagged_test.__test_tags__, ("ray",))


if __name__ == '__main__':
    unittest.main()

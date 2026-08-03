import unittest

import datasets

from data_juicer.utils.cache_utils import DatasetCacheControl, dataset_cache_control

from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

class DatasetCacheControlTest(DataJuicerTestCaseBase):

    def setUp(self):
        # in default, the cache is enabled
        datasets.enable_caching()

        super().setUp()

    def tearDown(self) -> None:
        # reset the cache
        datasets.enable_caching()

        super().tearDown()

    def test_basic_func(self):
        self.assertTrue(datasets.is_caching_enabled())
        with DatasetCacheControl(on=False):
            self.assertFalse(datasets.is_caching_enabled())
        self.assertTrue(datasets.is_caching_enabled())

        with DatasetCacheControl(on=False):
            self.assertFalse(datasets.is_caching_enabled())
            with DatasetCacheControl(on=True):
                self.assertTrue(datasets.is_caching_enabled())
            self.assertFalse(datasets.is_caching_enabled())
        self.assertTrue(datasets.is_caching_enabled())

    def test_decorator(self):

        @dataset_cache_control(on=False)
        def check():
            return datasets.is_caching_enabled()

        self.assertTrue(datasets.is_caching_enabled())
        self.assertFalse(check())
        self.assertTrue(datasets.is_caching_enabled())

    def test_dataset_cache_control_enable_from_disabled(self):
        """Start with cache disabled, use DatasetCacheControl(on=True)
        to re-enable, then verify restore."""
        datasets.disable_caching()
        self.assertFalse(datasets.is_caching_enabled())

        with DatasetCacheControl(on=True):
            self.assertTrue(datasets.is_caching_enabled())

        # Should be restored to disabled
        self.assertFalse(datasets.is_caching_enabled())

    def test_dataset_cache_control_decorator_enable(self):
        """Use @dataset_cache_control(on=True) decorator on a function
        that checks caching is enabled inside."""
        datasets.disable_caching()
        self.assertFalse(datasets.is_caching_enabled())

        @dataset_cache_control(on=True)
        def check_enabled():
            return datasets.is_caching_enabled()

        result = check_enabled()
        self.assertTrue(result)
        # After the decorated function returns, cache should be
        # restored to disabled
        self.assertFalse(datasets.is_caching_enabled())


if __name__ == '__main__':
    unittest.main()

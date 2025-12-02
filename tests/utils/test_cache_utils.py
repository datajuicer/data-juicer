import unittest

import datasets

from data_juicer.utils.cache_utils import DatasetCacheControl, dataset_cache_control

from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

class DatasetCacheControlTest(DataJuicerTestCaseBase):

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


if __name__ == '__main__':
    unittest.main()

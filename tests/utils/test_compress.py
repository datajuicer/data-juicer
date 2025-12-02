import os
import json
import unittest

from datasets import config, load_dataset

from data_juicer.core import NestedDataset
from data_juicer.utils.compress import compress, decompress, cleanup_compressed_cache_files, CompressionOff
from data_juicer.utils import cache_utils
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

class CacheCompressTest(DataJuicerTestCaseBase):

    def setUp(self) -> None:
        super().setUp()
        self.temp_output_path = 'tmp/test_compress/'
        self.test_data_path = self.temp_output_path + 'test.json'
        os.makedirs(self.temp_output_path, exist_ok=True)
        with open(self.test_data_path, 'w') as fout:
            json.dump([{'test_key_1': 'test_val_1'}], fout)
        self.ori_cache_dir = config.HF_DATASETS_CACHE
        config.HF_DATASETS_CACHE = self.temp_output_path

    def tearDown(self):
        if os.path.exists(self.temp_output_path):
            os.system(f'rm -rf {self.temp_output_path}')
        config.HF_DATASETS_CACHE = self.ori_cache_dir

        super().tearDown()

    def test_basic_func(self):
        cache_utils.CACHE_COMPRESS = 'zstd'
        ds = load_dataset('json', data_files=self.test_data_path, split='train')
        prev_ds = ds.map(lambda s: {'test_key_2': 'test_val_2', **s})
        curr_ds = prev_ds.map(lambda s: {'test_key_3': 'test_val_3', **s})
        for fn in ds.cache_files:
            self.assertTrue(os.path.exists(fn['filename']))
        for fn in prev_ds.cache_files:
            self.assertTrue(os.path.exists(fn['filename']))
        for fn in curr_ds.cache_files:
            self.assertTrue(os.path.exists(fn['filename']))

        # won't compress original dataset
        compress(ds, prev_ds)
        # cache files of the original dataset always exist
        for fn in ds.cache_files:
            self.assertTrue(os.path.exists(fn['filename']))
            self.assertFalse(os.path.exists(fn['filename'] + f'.{cache_utils.CACHE_COMPRESS}'))
        # cache files of the previous dataset are deleted
        for fn in prev_ds.cache_files:
            self.assertTrue(os.path.exists(fn['filename']))
        # cache files of the current dataset are kept
        for fn in curr_ds.cache_files:
            self.assertTrue(os.path.exists(fn['filename']))

        # compress previous dataset
        compress(prev_ds, curr_ds)
        # cache files of the original dataset always exist
        for fn in ds.cache_files:
            self.assertTrue(os.path.exists(fn['filename']))
        # cache files of the previous dataset are deleted
        for fn in prev_ds.cache_files:
            self.assertFalse(os.path.exists(fn['filename']))
            self.assertTrue(os.path.exists(fn['filename'] + f'.{cache_utils.CACHE_COMPRESS}'))
        # cache files of the current dataset are kept
        for fn in curr_ds.cache_files:
            self.assertTrue(os.path.exists(fn['filename']))

        # decompress the previous dataset
        decompress(prev_ds)
        for fn in prev_ds.cache_files:
            self.assertTrue(os.path.exists(fn['filename']))
            self.assertTrue(os.path.exists(fn['filename'] + f'.{cache_utils.CACHE_COMPRESS}'))

        # clean up the compressed cache files of the previous dataset
        cleanup_compressed_cache_files(prev_ds)
        for fn in prev_ds.cache_files:
            self.assertTrue(os.path.exists(fn['filename']))
            self.assertFalse(os.path.exists(fn['filename'] + f'.{cache_utils.CACHE_COMPRESS}'))

    def test_dif_compress_method(self):
        cache_utils.CACHE_COMPRESS = 'gzip'
        ds = load_dataset('json', data_files=self.test_data_path, split='train')
        prev_ds = ds.map(lambda s: {'test_key_2': 'test_val_2', **s})
        curr_ds = prev_ds.map(lambda s: {'test_key_3': 'test_val_3', **s})
        for fn in ds.cache_files:
            self.assertTrue(os.path.exists(fn['filename']))
        for fn in prev_ds.cache_files:
            self.assertTrue(os.path.exists(fn['filename']))
        for fn in curr_ds.cache_files:
            self.assertTrue(os.path.exists(fn['filename']))

        # won't compress original dataset
        compress(ds, prev_ds)
        # cache files of the original dataset always exist
        for fn in ds.cache_files:
            self.assertTrue(os.path.exists(fn['filename']))
            self.assertFalse(os.path.exists(fn['filename'] + f'.{cache_utils.CACHE_COMPRESS}'))
        # cache files of the previous dataset are deleted
        for fn in prev_ds.cache_files:
            self.assertTrue(os.path.exists(fn['filename']))
        # cache files of the current dataset are kept
        for fn in curr_ds.cache_files:
            self.assertTrue(os.path.exists(fn['filename']))

        # compress previous dataset
        compress(prev_ds, curr_ds)
        # cache files of the original dataset always exist
        for fn in ds.cache_files:
            self.assertTrue(os.path.exists(fn['filename']))
        # cache files of the previous dataset are deleted
        for fn in prev_ds.cache_files:
            self.assertFalse(os.path.exists(fn['filename']))
            self.assertTrue(os.path.exists(fn['filename'] + f'.{cache_utils.CACHE_COMPRESS}'))
        # cache files of the current dataset are kept
        for fn in curr_ds.cache_files:
            self.assertTrue(os.path.exists(fn['filename']))

        # decompress the previous dataset
        decompress(prev_ds)
        for fn in prev_ds.cache_files:
            self.assertTrue(os.path.exists(fn['filename']))
            self.assertTrue(os.path.exists(fn['filename'] + f'.{cache_utils.CACHE_COMPRESS}'))

        # clean up the compressed cache files of the previous dataset
        cleanup_compressed_cache_files(prev_ds)
        for fn in prev_ds.cache_files:
            self.assertTrue(os.path.exists(fn['filename']))
            self.assertFalse(os.path.exists(fn['filename'] + f'.{cache_utils.CACHE_COMPRESS}'))

    def test_multiprocessing(self):
        cache_utils.CACHE_COMPRESS = 'zstd'
        ds = load_dataset('json', data_files=self.test_data_path, split='train')
        prev_ds = ds.map(lambda s: {'test_key_2': 'test_val_2', **s})
        curr_ds = prev_ds.map(lambda s: {'test_key_3': 'test_val_3', **s})
        for fn in ds.cache_files:
            self.assertTrue(os.path.exists(fn['filename']))
        for fn in prev_ds.cache_files:
            self.assertTrue(os.path.exists(fn['filename']))
        for fn in curr_ds.cache_files:
            self.assertTrue(os.path.exists(fn['filename']))
        compress(prev_ds, curr_ds, num_proc=2)
        # cache files of the original dataset always exist
        for fn in ds.cache_files:
            self.assertTrue(os.path.exists(fn['filename']))
        # cache files of the previous dataset are deleted
        for fn in prev_ds.cache_files:
            self.assertFalse(os.path.exists(fn['filename']))
            self.assertTrue(os.path.exists(fn['filename'] + f'.{cache_utils.CACHE_COMPRESS}'))
        # cache files of the current dataset are kept
        for fn in curr_ds.cache_files:
            self.assertTrue(os.path.exists(fn['filename']))

        # decompress the previous dataset
        decompress(prev_ds, num_proc=2)
        for fn in prev_ds.cache_files:
            self.assertTrue(os.path.exists(fn['filename']))
            self.assertTrue(os.path.exists(fn['filename'] + f'.{cache_utils.CACHE_COMPRESS}'))

        # clean up the compressed cache files of the previous dataset
        cleanup_compressed_cache_files(prev_ds)
        for fn in prev_ds.cache_files:
            self.assertTrue(os.path.exists(fn['filename']))
            self.assertFalse(os.path.exists(fn['filename'] + f'.{cache_utils.CACHE_COMPRESS}'))

    def test_compression_off(self):
        cache_utils.CACHE_COMPRESS = 'lz4'
        ds = load_dataset('json', data_files=self.test_data_path, split='train')
        prev_ds = ds.map(lambda s: {'test_key_2': 'test_val_2', **s})
        curr_ds = prev_ds.map(lambda s: {'test_key_3': 'test_val_3', **s})
        for fn in ds.cache_files:
            self.assertTrue(os.path.exists(fn['filename']))
        for fn in prev_ds.cache_files:
            self.assertTrue(os.path.exists(fn['filename']))
        for fn in curr_ds.cache_files:
            self.assertTrue(os.path.exists(fn['filename']))

        # disable cache compression
        with CompressionOff():
            compress(prev_ds, curr_ds)
        # cache files of the original dataset always exist
        for fn in ds.cache_files:
            self.assertTrue(os.path.exists(fn['filename']))
        # cache files of the previous dataset are deleted
        for fn in prev_ds.cache_files:
            self.assertTrue(os.path.exists(fn['filename']))
            self.assertFalse(os.path.exists(fn['filename'] + f'.{cache_utils.CACHE_COMPRESS}'))
        # cache files of the current dataset are kept
        for fn in curr_ds.cache_files:
            self.assertTrue(os.path.exists(fn['filename']))

        # re-enable cache compression
        compress(prev_ds, curr_ds)
        # cache files of the original dataset always exist
        for fn in ds.cache_files:
            self.assertTrue(os.path.exists(fn['filename']))
        # cache files of the previous dataset are deleted
        for fn in prev_ds.cache_files:
            self.assertFalse(os.path.exists(fn['filename']))
            self.assertTrue(os.path.exists(fn['filename'] + f'.{cache_utils.CACHE_COMPRESS}'))
        # cache files of the current dataset are kept
        for fn in curr_ds.cache_files:
            self.assertTrue(os.path.exists(fn['filename']))

    def test_dataset_without_cache(self):
        prev_ds = NestedDataset.from_list([{'test_key': 'test_val'}])
        curr_ds = prev_ds.map(lambda s: {'test_key_2': 'test_val_2', **s})
        # dataset from list does not have cache files
        self.assertTrue(len(prev_ds.cache_files) == 0)
        self.assertTrue(len(curr_ds.cache_files) == 0)
        compress(prev_ds, curr_ds)
        decompress(prev_ds)
        cleanup_compressed_cache_files(prev_ds)
        self.assertTrue(len(prev_ds.cache_files) == 0)
        self.assertTrue(len(curr_ds.cache_files) == 0)


if __name__ == '__main__':
    unittest.main()

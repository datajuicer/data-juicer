import os
import json
import unittest

from data_juicer.utils.asset_utils import load_words_asset

from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

class LoadWordsAssetTest(DataJuicerTestCaseBase):

    def setUp(self) -> None:
        super().setUp()
        self.temp_output_path = 'tmp/test_asset_utils/'

    def tearDown(self):
        if os.path.exists(self.temp_output_path):
            os.system(f'rm -rf {self.temp_output_path}')
        super().tearDown()

    def test_basic_func(self):
        # download assets from the remote server
        words_dict = load_words_asset(self.temp_output_path, 'stopwords')
        self.assertTrue(len(words_dict) > 0)
        self.assertTrue(os.path.exists(os.path.join(self.temp_output_path, 'stopwords.json')))

        words_dict = load_words_asset(self.temp_output_path, 'flagged_words')
        self.assertTrue(len(words_dict) > 0)
        self.assertTrue(os.path.exists(os.path.join(self.temp_output_path, 'flagged_words.json')))

        # non-existing asset
        with self.assertRaises(ValueError):
            load_words_asset(self.temp_output_path, 'non_existing_asset')

    def test_load_from_existing_file(self):
        os.makedirs(self.temp_output_path, exist_ok=True)
        temp_asset = os.path.join(self.temp_output_path, 'temp_asset.json')
        with open(temp_asset, 'w') as fout:
            json.dump({'test_key': ['test_val']}, fout)

        words_list = load_words_asset(self.temp_output_path, 'temp_asset')
        self.assertEqual(len(words_list), 1)
        self.assertEqual(len(words_list['test_key']), 1)

    def test_load_from_serial_files(self):
        os.makedirs(self.temp_output_path, exist_ok=True)
        temp_asset = os.path.join(self.temp_output_path, 'temp_asset_v1.json')
        with open(temp_asset, 'w') as fout:
            json.dump({'test_key': ['test_val_1']}, fout)
        temp_asset = os.path.join(self.temp_output_path, 'temp_asset_v2.json')
        with open(temp_asset, 'w') as fout:
            json.dump({'test_key': ['test_val_2']}, fout)

        words_list = load_words_asset(self.temp_output_path, 'temp_asset')
        self.assertEqual(len(words_list), 1)
        self.assertEqual(len(words_list['test_key']), 2)


if __name__ == '__main__':
    unittest.main()

import os
import unittest
import tempfile
import json

from data_juicer.core import NestedDataset
from data_juicer.config import init_configs
from data_juicer.utils.constant import StatsKeys
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

test_yaml_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                              '..',
                              'config',
                              'demo_4_test.yaml')

class StatsKeysTest(DataJuicerTestCaseBase):

    def setUp(self) -> None:
        super().setUp()
        StatsKeys._accessed_by = {}
        
        # Create a temporary jsonl file for testing
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_jsonl = os.path.join(self.temp_dir.name, "test-dataset.jsonl")
        with open(self.temp_jsonl, 'w', encoding='utf-8') as f:
            f.write(json.dumps({"text": "hello world"}) + "\n")

    def tearDown(self) -> None:
        super().tearDown()
        StatsKeys._accessed_by = {}
        if hasattr(self, 'temp_dir'):
            self.temp_dir.cleanup()

    def test_basic_func(self):
        # Create a temporary config with the test dataset path
        args = f'--config {test_yaml_path} --dataset_path {self.temp_jsonl}'.split()
        cfg = init_configs(args=args)
        
        res = StatsKeys.get_access_log(cfg)
        self.assertEqual(len(res), 1)  # only 1 filter
        self.assertIn('language_id_score_filter', res)
        self.assertEqual(res['language_id_score_filter'], {'lang', 'lang_score'})

        # obtain again
        res_2 = StatsKeys.get_access_log(cfg)
        self.assertEqual(res, res_2)

    def test_basic_func_with_dataset(self):
        dataset = NestedDataset.from_list([{'text': 'hello world'}])
        args = f'--config {test_yaml_path} --dataset_path {self.temp_jsonl}'.split()
        cfg = init_configs(args=args)
        
        res = StatsKeys.get_access_log(cfg, dataset)
        self.assertEqual(len(res), 1)  # only 1 filter
        self.assertIn('language_id_score_filter', res)
        self.assertEqual(res['language_id_score_filter'], {'lang', 'lang_score'})


if __name__ == '__main__':
    unittest.main()

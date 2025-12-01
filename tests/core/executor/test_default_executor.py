import os
import unittest
from datasets import load_dataset
from data_juicer.core import DefaultExecutor, NestedDataset
from data_juicer.config import init_configs
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

root_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..')
test_yaml_path = os.path.join(root_path,
                              'config',
                              'demo_4_test.yaml')

class DefaultExecutorTest(DataJuicerTestCaseBase):
    test_file = 'text_only_2.3k.jsonl'

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        # test dataset
        download_link = f'http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/data_juicer/unittest_data/{cls.test_file}'
        os.system(f'wget {download_link}')

        print(f"CURRENT DIR: {os.getcwd()}")

    @classmethod
    def tearDownClass(cls, hf_model_name=None) -> None:
        # remove test dataset
        os.system(f'rm -f {cls.test_file}')

        super().tearDownClass(hf_model_name)

    def setUp(self) -> None:
        super().setUp()
        # tmp dir
        self.tmp_dir = 'tmp/test_default_executor/'
        os.makedirs(self.tmp_dir, exist_ok=True)

    def tearDown(self) -> None:
        super().tearDown()
        if os.path.exists(self.tmp_dir):
            os.system(f'rm -rf {self.tmp_dir}')

    def test_end2end_execution(self):
        cfg = init_configs(['--config', test_yaml_path])
        cfg.export_path = os.path.join(self.tmp_dir, 'test_end2end_execution', 'res.jsonl')
        cfg.work_dir = os.path.join(self.tmp_dir, 'test_end2end_execution')
        executor = DefaultExecutor(cfg)
        executor.run()

        # check result files
        self.assertTrue(os.path.exists(cfg.export_path))

    def test_end2end_execution_skip_export(self):
        cfg = init_configs(['--config', test_yaml_path])
        cfg.export_path = os.path.join(self.tmp_dir, 'test_end2end_execution_skip_export', 'res.jsonl')
        cfg.work_dir = os.path.join(self.tmp_dir, 'test_end2end_execution_skip_export')
        executor = DefaultExecutor(cfg)
        executor.run(skip_export=True)

        # check result files
        self.assertFalse(os.path.exists(cfg.export_path))

    def test_end2end_execution_with_existing_dataset(self):
        cfg = init_configs(['--config', test_yaml_path])
        cfg.export_path = os.path.join(self.tmp_dir, 'test_end2end_execution_with_existing_dataset', 'res.jsonl')
        cfg.work_dir = os.path.join(self.tmp_dir, 'test_end2end_execution')
        ds = NestedDataset(load_dataset('json', data_files=self.test_file, split='train'))
        # open optional modules
        cfg.open_tracer = False
        cfg.use_checkpoint = True
        cfg.op_fusion = True
        cfg.adaptive_batch_size = True
        cfg.cache_compress = 'gzip'
        executor = DefaultExecutor(cfg)
        executor.run(ds)

        # check result files
        self.assertTrue(os.path.exists(cfg.export_path))
        # check tracer outputs
        self.assertFalse(os.path.exists(os.path.join(cfg.work_dir, 'trace')))
        # check checkpoints
        self.assertTrue(os.path.exists(os.path.join(cfg.work_dir, 'ckpt')))

    def test_sample_data(self):
        ds_length = 6
        cfg = init_configs(['--config', test_yaml_path])
        executor = DefaultExecutor(cfg)
        res_ds = executor.sample_data(sample_ratio=0.5)
        self.assertEqual(len(res_ds) * 2, ds_length)

    def test_sample_data_with_existing_dataset(self):
        cfg = init_configs(['--config', test_yaml_path])
        ds = NestedDataset(load_dataset('json', data_files=self.test_file, split='train'))
        executor = DefaultExecutor(cfg)
        res_ds = executor.sample_data(ds, sample_algo='frequency_specified_field_selector', field_key='id', top_ratio=0.5)
        self.assertEqual(len(res_ds) * 2, len(ds))

    def test_sample_data_with_existing_dataset_topk(self):
        cfg = init_configs(['--config', test_yaml_path])
        ds = NestedDataset(load_dataset('json', data_files=self.test_file, split='train'))
        executor = DefaultExecutor(cfg)
        res_ds = executor.sample_data(ds, sample_algo='topk_specified_field_selector', field_key='id', topk=100)
        self.assertEqual(len(res_ds), 100)

    def test_sample_data_unknow_algo(self):
        cfg = init_configs(['--config', test_yaml_path])
        executor = DefaultExecutor(cfg)
        with self.assertRaises(ValueError):
            executor.sample_data(sample_algo='unknown_algo')


if __name__ == '__main__':
    unittest.main()

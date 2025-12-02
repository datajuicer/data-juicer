import os
import unittest
from datasets import load_dataset
from data_juicer.core import Analyzer, NestedDataset
from data_juicer.config import init_configs
from data_juicer.utils.file_utils import add_suffix_to_filename
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

root_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..')
test_yaml_path = os.path.join(root_path,
                              'tests',
                              'config',
                              'demo_4_test.yaml')

class AnalyzerTest(DataJuicerTestCaseBase):

    def setUp(self) -> None:
        super().setUp()
        # tmp dir
        self.tmp_dir = 'tmp/test_analyzer/'
        os.makedirs(self.tmp_dir, exist_ok=True)

    def tearDown(self) -> None:
        super().tearDown()
        if os.path.exists(self.tmp_dir):
            os.system(f'rm -rf {self.tmp_dir}')

    def test_end2end_analysis(self):
        cfg = init_configs(['--config', test_yaml_path])
        cfg.export_path = os.path.join(self.tmp_dir, 'test_end2end_analysis', 'res.jsonl')
        cfg.work_dir = os.path.join(self.tmp_dir, 'test_end2end_analysis')
        analyzer = Analyzer(cfg)
        analyzer.run()

        # check result files
        self.assertTrue(os.path.exists(cfg.work_dir))
        self.assertTrue(os.path.exists(os.path.join(cfg.work_dir, 'analysis')))
        self.assertTrue(os.path.exists(add_suffix_to_filename(cfg.export_path, '_stats')))

    def test_auto_analysis_with_existing_dataset(self):
        cfg = init_configs(['--config', test_yaml_path])
        cfg.export_path = os.path.join(self.tmp_dir, 'test_end2end_analysis', 'res.jsonl')
        cfg.work_dir = os.path.join(self.tmp_dir, 'test_end2end_analysis')
        cfg.auto = True
        cfg.auto_num = 3
        cfg.op_fusion = True
        cfg.use_cache = True
        cfg.cache_compress = 'gzip'
        ds = NestedDataset(load_dataset('json', data_files=os.path.join(root_path, 'demos/data/demo-dataset.jsonl'), split='train'))
        analyzer = Analyzer(cfg)
        analyzer.run(ds)

        # check result files
        self.assertTrue(os.path.exists(cfg.work_dir))
        self.assertTrue(os.path.exists(os.path.join(cfg.work_dir, 'analysis')))
        self.assertTrue(os.path.exists(add_suffix_to_filename(cfg.export_path, '_stats')))

    def test_analysis_without_stats(self):
        cfg = init_configs(['--config', test_yaml_path])
        cfg.export_path = os.path.join(self.tmp_dir, 'test_end2end_analysis', 'res.jsonl')
        cfg.work_dir = os.path.join(self.tmp_dir, 'test_end2end_analysis')
        cfg.process = []
        analyzer = Analyzer(cfg)
        analyzer.run()

        # check result files
        self.assertFalse(os.path.exists(os.path.join(cfg.work_dir, 'analysis')))
        self.assertFalse(os.path.exists(add_suffix_to_filename(cfg.export_path, '_stats')))


if __name__ == '__main__':
    unittest.main()

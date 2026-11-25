import os
import unittest

from data_juicer.config import init_configs
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase, TEST_TAG

root_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..')
test_yaml_path = os.path.join(root_path, 'tests', 'config', 'demo_4_test.yaml')


class RayAnalyzerTest(DataJuicerTestCaseBase):

    def setUp(self):
        super().setUp()
        self.tmp_dir = 'tmp/test_ray_analyzer/'
        os.makedirs(self.tmp_dir, exist_ok=True)

    def tearDown(self):
        super().tearDown()
        if os.path.exists(self.tmp_dir):
            import shutil
            shutil.rmtree(self.tmp_dir)

    def _get_cfg(self, subdir):
        cfg = init_configs(
            ['--config', test_yaml_path],
            allow_auto=True,
        )
        cfg.executor_type = 'ray'
        cfg.export_path = os.path.join(self.tmp_dir, subdir, 'res.jsonl')
        cfg.work_dir = os.path.join(self.tmp_dir, subdir)
        return cfg

    @TEST_TAG('ray')
    def test_end2end_analysis(self):
        cfg = self._get_cfg('test_end2end')
        # use only text filters for basic test
        cfg.process = [
            {'text_length_filter': {'min_len': 10, 'max_len': 10000}},
            {'words_num_filter': {'lang': 'en', 'min_num': 1, 'max_num': 10000}},
        ]

        from data_juicer.core import RayAnalyzer
        analyzer = RayAnalyzer(cfg)
        analyzer.run()

        analysis_dir = os.path.join(cfg.work_dir, 'analysis')
        self.assertTrue(os.path.exists(analysis_dir))
        self.assertTrue(os.path.exists(os.path.join(analysis_dir, 'overall.csv')))
        self.assertTrue(os.path.exists(os.path.join(analysis_dir, 'overall.md')))

        # verify overall_result has expected columns
        self.assertIsNotNone(analyzer.overall_result)
        self.assertGreater(len(analyzer.overall_result.columns), 0)

    @TEST_TAG('ray')
    def test_analysis_without_stats(self):
        cfg = self._get_cfg('test_no_stats')
        cfg.process = []

        from data_juicer.core import RayAnalyzer
        analyzer = RayAnalyzer(cfg)
        analyzer.run()

        analysis_dir = os.path.join(cfg.work_dir, 'analysis')
        self.assertFalse(os.path.exists(os.path.join(analysis_dir, 'overall.csv')))

    @TEST_TAG('ray')
    def test_analysis_with_list_numeric_columns(self):
        """Test that list-valued numeric stats (e.g. image_width) are
        correctly flattened and aggregated."""
        cfg = self._get_cfg('test_list_numeric')
        cfg.dataset_path = os.path.join(root_path, 'demos/data/demo-dataset-images.jsonl')
        cfg.process = [
            {'text_length_filter': {'min_len': 1, 'max_len': 10000}},
            {'image_shape_filter': {
                'min_width': 1, 'max_width': 10000,
                'min_height': 1, 'max_height': 10000,
            }},
        ]

        from data_juicer.core import RayAnalyzer
        analyzer = RayAnalyzer(cfg)
        analyzer.run()

        overall = analyzer.overall_result
        self.assertIsNotNone(overall)
        # image_width and image_height are list-typed stats
        self.assertIn('image_width', overall.columns)
        self.assertIn('image_height', overall.columns)
        # text_len is a scalar stat
        self.assertIn('text_len', overall.columns)
        # verify count/mean/std/min/max rows exist
        for row in ['count', 'mean', 'std', 'min', 'max']:
            self.assertIn(row, overall.index)

    @TEST_TAG('ray')
    def test_auto_mode(self):
        cfg = self._get_cfg('test_auto')
        cfg.auto = True
        cfg.auto_num = 3
        cfg.process = [
            {'text_length_filter': {'min_len': 1, 'max_len': 10000}},
        ]

        from data_juicer.core import RayAnalyzer
        analyzer = RayAnalyzer(cfg)
        analyzer.run()

        analysis_dir = os.path.join(cfg.work_dir, 'analysis')
        self.assertTrue(os.path.exists(analysis_dir))
        self.assertIsNotNone(analyzer.overall_result)


if __name__ == '__main__':
    unittest.main()

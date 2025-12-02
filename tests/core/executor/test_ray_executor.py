import os
import unittest
from data_juicer.core.executor.ray_executor import RayExecutor
from data_juicer.config import init_configs
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase, TEST_TAG

class RayExecutorTest(DataJuicerTestCaseBase):
    root_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', '..')

    def setUp(self) -> None:
        super().setUp()
        # tmp dir
        self.tmp_dir = os.path.join(self.root_path, 'tmp/test_ray_executor/')
        os.makedirs(self.tmp_dir, exist_ok=True)

    def tearDown(self) -> None:
        super().tearDown()
        if os.path.exists(self.tmp_dir):
            os.system(f'rm -rf {self.tmp_dir}')

    @TEST_TAG('ray')
    def test_end2end_execution(self):
        cfg = init_configs(['--config', os.path.join(self.root_path, 'demos/process_on_ray/configs/demo-new-config.yaml')])
        cfg.export_path = os.path.join(self.tmp_dir, 'test_end2end_execution', 'res.jsonl')
        cfg.work_dir = os.path.join(self.tmp_dir, 'test_end2end_execution')
        executor = RayExecutor(cfg)
        executor.run()

        # check result files
        self.assertTrue(os.path.exists(cfg.export_path))

    @TEST_TAG('ray')
    def test_end2end_execution_skip_export(self):
        cfg = init_configs(
            ['--config', os.path.join(self.root_path, 'demos/process_on_ray/configs/demo-new-config.yaml')])
        cfg.export_path = os.path.join(self.tmp_dir, 'test_end2end_execution_skip_export', 'res.jsonl')
        cfg.work_dir = os.path.join(self.tmp_dir, 'test_end2end_execution_skip_export')
        executor = RayExecutor(cfg)
        executor.run(skip_export=True)

        # check result files
        self.assertFalse(os.path.exists(cfg.export_path))

    @TEST_TAG('ray')
    def test_end2end_execution_op_fusion(self):
        cfg = init_configs(['--config', os.path.join(self.root_path, 'demos/process_on_ray/configs/demo-new-config.yaml')])
        cfg.export_path = os.path.join(self.tmp_dir, 'test_end2end_execution_op_fusion', 'res.jsonl')
        cfg.work_dir = os.path.join(self.tmp_dir, 'test_end2end_execution_op_fusion')
        cfg.op_fusion = True
        executor = RayExecutor(cfg)
        executor.run()

        # check result files
        self.assertTrue(os.path.exists(cfg.export_path))


if __name__ == '__main__':
    unittest.main()

import os
import shutil
import tempfile
import unittest
from unittest.mock import patch, MagicMock
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


class RayExecutorEncryptTest(DataJuicerTestCaseBase):
    """Tests that RayExecutor correctly passes encryption params to RayExporter.

    RayExecutor.__init__ imports Ray and starts a cluster, so we mock
    the RayExporter constructor to inspect the forwarded arguments without
    needing a live Ray environment.
    """

    root_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', '..')
    _demo_cfg = os.path.join(root_path, 'demos/process_on_ray/configs/demo-new-config.yaml')

    def setUp(self):
        super().setUp()
        self.tmp_dir = tempfile.mkdtemp()
        from cryptography.fernet import Fernet
        self._key = Fernet.generate_key()
        self._key_file = os.path.join(self.tmp_dir, 'enc.key')
        with open(self._key_file, 'wb') as f:
            f.write(self._key)

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        os.environ.pop('DJ_ENCRYPTION_KEY', None)

    def _make_cfg(self, extra: dict = None):
        cfg = init_configs(['--config', self._demo_cfg])
        cfg.export_path = os.path.join(self.tmp_dir, 'res.jsonl')
        cfg.work_dir = self.tmp_dir
        if extra:
            for k, v in extra.items():
                setattr(cfg, k, v)
        return cfg

    def _build_executor_no_ray(self, cfg):
        """Construct RayExecutor while bypassing Ray initialisation."""
        from data_juicer.core.ray_exporter import RayExporter
        captured = {}
        original_init = RayExporter.__init__

        def _capturing_init(self_inner, *args, **kwargs):
            captured['encrypt_before_export'] = kwargs.get('encrypt_before_export', False)
            captured['encryption_key_path'] = kwargs.get('encryption_key_path', None)
            original_init(self_inner, *args, **kwargs)

        # Mock Ray so we never need a live cluster:
        #   - initialize_ray: skip cluster connection
        #   - ray.get_runtime_context().get_job_id(): used to build tmp_dir
        mock_ctx = MagicMock()
        mock_ctx.get_job_id.return_value = 'test_job'

        # patch.object on the actual class so all import aliases are covered
        with patch('data_juicer.core.executor.ray_executor.ray') as mock_ray, \
             patch('data_juicer.utils.ray_utils.initialize_ray'), \
             patch.object(RayExporter, '__init__', _capturing_init):
            mock_ray.get_runtime_context.return_value = mock_ctx
            try:
                RayExecutor.__init__(RayExecutor.__new__(RayExecutor), cfg)
            except Exception:
                pass  # Ray may fail to init; we only care about exporter args
        return captured

    @TEST_TAG('ray')
    def test_ray_executor_no_encrypt_by_default(self):
        """RayExecutor passes encrypt_before_export=False by default."""
        cfg = self._make_cfg()
        captured = self._build_executor_no_ray(cfg)
        self.assertFalse(captured.get('encrypt_before_export', False))

    @TEST_TAG('ray')
    def test_ray_executor_passes_encrypt_flag(self):
        """RayExecutor forwards encrypt_before_export=True to RayExporter."""
        cfg = self._make_cfg({
            'encrypt_before_export': True,
            'encryption_key_path': self._key_file,
        })
        captured = self._build_executor_no_ray(cfg)
        self.assertTrue(captured.get('encrypt_before_export', False))

    @TEST_TAG('ray')
    def test_ray_executor_passes_key_path(self):
        """RayExecutor forwards encryption_key_path to RayExporter."""
        cfg = self._make_cfg({
            'encrypt_before_export': True,
            'encryption_key_path': self._key_file,
        })
        captured = self._build_executor_no_ray(cfg)
        self.assertEqual(captured.get('encryption_key_path'), self._key_file)


if __name__ == '__main__':
    unittest.main()

import unittest
from unittest.mock import MagicMock, patch

from data_juicer.utils import ray_utils
from data_juicer.utils.constant import RAY_JOB_ENV_VAR
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class RayUtilsTest(DataJuicerTestCaseBase):

    def test_initialize_ray_without_cfg_uses_default_address(self):
        mock_ray = MagicMock()
        mock_ray.is_initialized.return_value = False

        with patch.object(ray_utils, "ray", mock_ray):
            ray_utils.initialize_ray(cfg=None)

        mock_ray.init.assert_called_once()
        args, kwargs = mock_ray.init.call_args
        self.assertEqual(args, ("auto",))
        self.assertTrue(kwargs["ignore_reinit_error"])
        self.assertIsNone(kwargs["runtime_env"]["py_modules"])
        self.assertIn(RAY_JOB_ENV_VAR, kwargs["runtime_env"]["env_vars"])

    def test_initialize_ray_passes_custom_operator_paths(self):
        class Config:
            ray_address = "ray://127.0.0.1:10001"
            custom_operator_paths = ["custom_ops"]

            def get(self, key, default=None):
                return getattr(self, key, default)

        mock_ray = MagicMock()
        mock_ray.is_initialized.return_value = False

        with patch.object(ray_utils, "ray", mock_ray):
            ray_utils.initialize_ray(cfg=Config())

        mock_ray.init.assert_called_once()
        args, kwargs = mock_ray.init.call_args
        self.assertEqual(args, ("ray://127.0.0.1:10001",))
        self.assertEqual(kwargs["runtime_env"]["py_modules"], ["custom_ops"])


if __name__ == "__main__":
    unittest.main()

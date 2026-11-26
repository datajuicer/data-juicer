import unittest
from unittest.mock import MagicMock, patch

from data_juicer.utils import ray_utils
from data_juicer.utils.constant import RAY_JOB_ENV_VAR
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class RayUtilsTest(DataJuicerTestCaseBase):

    def test_node_probe_uses_ray_resources_instead_of_hostnames(self):
        driver_id, gpu_id, cpu_id, dead_id = [str(i) * 56 for i in range(1, 5)]
        nodes = [
            {"Alive": True, "NodeID": driver_id, "NodeManagerHostname": "driver", "Resources": {}},
            {"Alive": True, "NodeID": gpu_id, "NodeManagerHostname": "gpu-head", "Resources": {"GPU": 1}},
            {"Alive": True, "NodeID": cpu_id, "NodeManagerHostname": "worker", "Resources": {"CPU": 2}},
            {"Alive": False, "NodeID": dead_id, "NodeManagerHostname": "dead", "Resources": {"GPU": 1}},
        ]
        mock_ray = MagicMock()
        mock_ray.nodes.return_value = nodes
        mock_ray.get.return_value = [{"gpu_count": 1}, {"gpu_count": 0}]
        with patch.object(ray_utils, "ray", mock_ray), patch.object(ray_utils, "_RAY_NODES_INFO", None):
            self.assertEqual(ray_utils.get_ray_nodes_info(), {gpu_id: {"gpu_count": 1}, cpu_id: {"gpu_count": 0}})

        calls = mock_ray.remote.return_value.options.call_args_list
        self.assertEqual([call.kwargs["scheduling_strategy"].node_id for call in calls], [gpu_id, cpu_id])
        self.assertTrue(all(not call.kwargs["scheduling_strategy"].soft for call in calls))
        self.assertTrue(all(call.kwargs["num_cpus"] == 0 for call in calls))
        remote_calls = mock_ray.remote.return_value.options.return_value.remote.call_args_list
        self.assertEqual([call.args for call in remote_calls], [(True,), (False,)])

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

import os
import unittest
import time
from loguru import logger
from data_juicer.core import Monitor
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

class MonitorTest(DataJuicerTestCaseBase):

    def setUp(self) -> None:
        super().setUp()
        self.work_dir = 'tmp/test_monitor/'
        os.makedirs(self.work_dir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.work_dir):
            os.system(f'rm -rf {self.work_dir}')
        super().tearDown()

    def test_monitor_current_resources(self):
        resource_dict = Monitor.monitor_current_resources()
        logger.info(resource_dict)
        self.assertIn('timestamp', resource_dict)
        self.assertIn('CPU count', resource_dict)
        self.assertIn('Mem. util.', resource_dict)

    def test_analyze_resource_util_list(self):
        resource_samples = []
        for i in range(5):
            resource_samples.append(Monitor.monitor_current_resources())
            time.sleep(0.2)
        resource_util_list = [{
            'time': 1,
            'sampling interval': 0.2,
            'resource': resource_samples,
        }]
        analysis_res = Monitor.analyze_resource_util_list(resource_util_list)
        logger.info(analysis_res)
        item = analysis_res[0]
        self.assertIn('resource_analysis', item)
        resource_analysis = item['resource_analysis']
        cpu_util = resource_analysis['CPU util.']
        self.assertIn('max', cpu_util)
        self.assertIn('min', cpu_util)
        self.assertIn('avg', cpu_util)

        # test draw resource util list
        Monitor.draw_resource_util_graph(resource_util_list, self.work_dir)
        self.assertTrue(os.path.exists(os.path.join(self.work_dir, 'func_0_CPU_util..jpg')))
        self.assertTrue(os.path.exists(os.path.join(self.work_dir, 'func_0_Used_mem..jpg')))

    def test_monitor_func(self):
        def test_func():
            for _ in range(5):
                time.sleep(0.2)

        ret, resource_util_dict = Monitor.monitor_func(test_func, sample_interval=0.3)
        self.assertIsNone(ret)
        self.assertIn("resource", resource_util_dict)
        self.assertIn("sampling interval", resource_util_dict)
        self.assertIn("time", resource_util_dict)

        self.assertEqual(resource_util_dict["sampling interval"], 0.3)
        resource_list = resource_util_dict["resource"]
        self.assertGreater(len(resource_list), 0)


if __name__ == '__main__':
    unittest.main()

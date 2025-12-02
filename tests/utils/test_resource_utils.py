import unittest
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.resource_utils import query_cuda_info, query_mem_info, cpu_count, get_cpu_utilization
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class RegistryTest(DataJuicerTestCaseBase):

    def test_query_cuda_info(self):
        torch = LazyLoader('torch')
        if torch.cuda.is_available():
            self.assertIsNotNone(query_cuda_info('memory.used'))
        else:
            self.assertIsNone(query_cuda_info('memory.used'))

    def test_query_mem_info(self):
        self.assertIsInstance(query_mem_info('total'), float)
        self.assertIsNone(query_mem_info('invalid key'))

    def test_cpu_count(self):
        self.assertIsInstance(cpu_count(), int)

    def test_get_cpu_utilization(self):
        self.assertIsInstance(get_cpu_utilization(), float)


if __name__ == '__main__':
    unittest.main()

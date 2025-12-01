import unittest
from data_juicer.core.executor import ExecutorFactory, DefaultExecutor
from data_juicer.core.executor.ray_executor import RayExecutor
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class TestFactory(DataJuicerTestCaseBase):

    def test_factory(self):
        self.assertEqual(ExecutorFactory.create_executor('default'), DefaultExecutor)
        self.assertEqual(ExecutorFactory.create_executor('ray'), RayExecutor)
        with self.assertRaises(ValueError):
            ExecutorFactory.create_executor('invalid')


if __name__ == '__main__':
    unittest.main()

import unittest

from data_juicer.utils.registry import Registry
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

class RegistryTest(DataJuicerTestCaseBase):

    def test_basic_func(self):
        registry = Registry('test')

        class A:
            pass
        registry.register_module('module_a', A)

        @registry.register_module('module_b')
        class B:
            pass

        self.assertEqual(registry.name, 'test')
        self.assertEqual(registry.modules, {'module_a': A, 'module_b': B})
        self.assertEqual(registry.list(), ['module_a', 'module_b'])
        self.assertEqual(registry.get('module_a'), A)
        self.assertEqual(registry.get('module_b'), B)

        with self.assertRaises(KeyError):
            registry.register_module('module_b', B)

        with self.assertRaises(TypeError):
            registry.register_module(1, A)


if __name__ == '__main__':
    unittest.main()

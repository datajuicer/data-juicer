import unittest

from data_juicer.utils.availability_utils import _is_package_available
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

class AvailabilityUtilsTest(DataJuicerTestCaseBase):

    def test_is_package_available(self):
        exist = _is_package_available('fsspec')
        self.assertTrue(exist)
        exist, version = _is_package_available('fsspec', return_version=True)
        self.assertTrue(exist)
        self.assertEqual(version, '2023.5.0')

        exist = _is_package_available('non_existing_package')
        self.assertFalse(exist)
        exist, version = _is_package_available('non_existing_package', return_version=True)
        self.assertFalse(exist)
        self.assertEqual(version, 'N/A')


if __name__ == '__main__':
    unittest.main()

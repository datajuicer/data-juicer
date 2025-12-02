import unittest

from data_juicer.core import NestedDataset
from data_juicer.utils.fingerprint_utils import generate_fingerprint
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

class FingerprintUtilsTest(DataJuicerTestCaseBase):

    def test_generate_fingerprint(self):
        dataset = NestedDataset.from_list([{'text_key': 'test_val'}])
        fingerprint = generate_fingerprint(dataset)
        self.assertLessEqual(len(fingerprint), 64)

        # with func args
        new_fingerprint = generate_fingerprint(dataset, lambda x: x['text_key'])
        self.assertLessEqual(len(new_fingerprint), 64)
        self.assertNotEqual(new_fingerprint, fingerprint)


if __name__ == '__main__':
    unittest.main()

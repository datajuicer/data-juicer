import unittest

from data_juicer.ops.deduplicator.document_simhash_deduplicator import DocumentSimhashDeduplicator


class DocumentSimhashParamValidationTest(unittest.TestCase):

    def test_valid_params_construct(self):
        op = DocumentSimhashDeduplicator(num_blocks=6, hamming_distance=4)
        self.assertEqual(op.num_blocks, 6)
        self.assertEqual(op.hamming_distance, 4)

    def test_hamming_distance_equal_to_num_blocks_raises(self):
        with self.assertRaises(ValueError):
            DocumentSimhashDeduplicator(num_blocks=6, hamming_distance=6)

    def test_hamming_distance_greater_than_num_blocks_raises(self):
        with self.assertRaises(ValueError):
            DocumentSimhashDeduplicator(num_blocks=6, hamming_distance=7)

    def test_num_blocks_over_64_raises(self):
        with self.assertRaises(ValueError):
            DocumentSimhashDeduplicator(num_blocks=65, hamming_distance=4)


if __name__ == "__main__":
    unittest.main()

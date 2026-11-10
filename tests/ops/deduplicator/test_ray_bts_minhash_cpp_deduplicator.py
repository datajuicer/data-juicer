import unittest


class RayBTSMinhashCppDeduplicatorTest(unittest.TestCase):

    def test_calc_minhash_batch_with_multiple_threads(self):
        from data_juicer.ops.deduplicator.minhash import calc_minhash_batch_c

        pairs = calc_minhash_batch_c(
            [["a"], []],
            10,
            [1, 3],
            [0, 0],
            b"empty",
            [(0, 1), (1, 2)],
            4,
            2,
        )

        self.assertCountEqual(
            pairs,
            [
                (0, b"\x00\x00\x00\x00\x00\x00\x00\x00", 10),
                (0, b"\x01\x00\x00\x00\x00\x00\x00\x00", 10),
                (3, b"empty", 11),
            ],
        )


if __name__ == "__main__":
    unittest.main()

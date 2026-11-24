import unittest

import numpy as np

from data_juicer.ops.deduplicator.document_minhash_deduplicator import (
    MAX_HASH,
    MERSENNE_PRIME,
    sha1_hash32,
)
from data_juicer.ops.deduplicator.ray_bts_minhash_deduplicator import (
    _minhash_values_from_tokens,
)


def _reference_minhash_values(tokens, perm_a, perm_b):
    """The original full-matrix implementation, kept as the reference."""
    hv = np.array([sha1_hash32(token) for token in tokens], dtype=np.uint64)
    phv = ((hv[:, None] * perm_a[None, :] + perm_b)
           % MERSENNE_PRIME).astype(np.uint32)
    return phv.min(axis=0)


def _make_perms(num_permutation, seed=42):
    gen = np.random.RandomState(seed)
    perm_a, perm_b = np.array(
        [
            (
                gen.randint(1, MERSENNE_PRIME, dtype=np.uint64),
                gen.randint(0, MERSENNE_PRIME, dtype=np.uint64),
            )
            for _ in range(num_permutation)
        ],
        dtype=np.uint64,
    ).T
    return perm_a, perm_b


class MinhashFoldTest(unittest.TestCase):

    def _check(self, tokens, num_permutation, block_bytes=None):
        perm_a, perm_b = _make_perms(num_permutation)
        expected = _reference_minhash_values(tokens, perm_a, perm_b)
        if block_bytes is None:
            actual = _minhash_values_from_tokens(
                tokens, perm_a, perm_b, num_permutation)
        else:
            actual = _minhash_values_from_tokens(
                tokens, perm_a, perm_b, num_permutation,
                block_bytes=block_bytes)
        self.assertEqual(actual.dtype, np.uint32)
        np.testing.assert_array_equal(actual, expected)

    def test_small_random_tokens(self):
        rng = np.random.default_rng(0)
        vocab = [f"tok{i}".encode() for i in range(64)]
        for size in (1, 2, 10, 1000):
            tokens = [vocab[int(i)]
                      for i in rng.integers(0, len(vocab), size=size)]
            for perms in (16, 256):
                self._check(tokens, perms)

    def test_duplicate_heavy(self):
        tokens = [b"a"] * 500 + [b"b"] * 300 + [b"c"] * 200
        for perms in (16, 256):
            self._check(tokens, perms)

    def test_block_boundaries(self):
        tokens = [f"t{i}".encode() for i in range(257)]
        for perms in (16, 256):
            bytes_per_token = perms * 8
            for block_bytes in (bytes_per_token, bytes_per_token * 3,
                                bytes_per_token * 64):
                self._check(tokens, perms, block_bytes=block_bytes)

    def test_empty_tokens(self):
        perm_a, perm_b = _make_perms(16)
        actual = _minhash_values_from_tokens([], perm_a, perm_b, 16)
        self.assertEqual(actual.dtype, np.uint32)
        np.testing.assert_array_equal(
            actual, np.full(16, MAX_HASH, dtype=np.uint32))

    def test_matches_full_matrix_for_long_input(self):
        rng = np.random.default_rng(1)
        tokens = [f"w{i}".encode()
                  for i in rng.integers(0, 4096, size=20000)]
        self._check(tokens, 256)


if __name__ == "__main__":
    unittest.main()

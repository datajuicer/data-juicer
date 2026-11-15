import random
import unittest

import numpy as np

from data_juicer.ops.common.helper_func import split_on_whitespace
from data_juicer.ops.deduplicator.document_minhash_deduplicator import (
    _MINHASH_PERMUTATION_BLOCK_BYTES,
    MAX_HASH,
    MERSENNE_PRIME,
    DocumentMinhashDeduplicator,
    sha1_hash32,
)
from data_juicer.utils.constant import HashKeys
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


def reference_minhash(op, text):
    """The previous set-based implementation, kept as the oracle."""
    if op.lowercase:
        text = text.lower()
    if op.ignore_pattern:
        text = op.ignore_pattern.sub("", text)

    window_size = op.window_size
    if op.tokenization == "character":
        tokens = {str.encode(text[i : i + window_size]) for i in range(len(text) - window_size + 1)}
    elif op.tokenization == "punctuation":
        parts = op.punctuation_pattern.split(text)
        tokens = {str.encode(" ".join(parts[i : i + window_size])) for i in range(len(parts) - window_size + 1)}
    elif op.tokenization == "space":
        parts = split_on_whitespace(text)
        tokens = {str.encode(" ".join(parts[i : i + window_size])) for i in range(len(parts) - window_size + 1)}
    else:
        raise NotImplementedError(op.tokenization)

    hv = np.fromiter((sha1_hash32(token) for token in tokens), dtype=np.uint64, count=len(tokens))
    if len(hv) == 0:
        hash_values = np.full(op.num_permutation, MAX_HASH, dtype=np.uint64)
    else:
        bytes_per_token = op.num_permutation * np.dtype(np.uint64).itemsize
        tokens_per_block = max(1, _MINHASH_PERMUTATION_BLOCK_BYTES // bytes_per_token)
        hash_values = None
        for start in range(0, len(hv), tokens_per_block):
            block = hv[start : start + tokens_per_block]
            phv = np.bitwise_and(
                (block[:, None] * op.perm_a + op.perm_b) % MERSENNE_PRIME,
                MAX_HASH,
            )
            block_min = phv.min(axis=0)
            if hash_values is None:
                hash_values = block_min
            else:
                np.minimum(hash_values, block_min, out=hash_values)
    return [bytes(hash_values[start:end].byteswap().data) for start, end in op.hash_ranges]


class DocumentMinhashStreamingShinglesTest(DataJuicerTestCaseBase):

    def _make_op(self, tokenization="space", **kwargs):
        return DocumentMinhashDeduplicator(
            tokenization=tokenization,
            num_permutations=16,
            num_bands=4,
            num_rows_per_band=4,
            **kwargs,
        )

    def _assert_matches_reference(self, op, texts):
        for text in texts:
            result = op.compute_hash({"text": text})
            self.assertEqual(
                result[HashKeys.minhash],
                reference_minhash(op, text),
                msg=f"mismatch for tokenization={op.tokenization!r}, text={text[:60]!r}",
            )

    def test_space_tokenization_matches_reference(self):
        rng = random.Random(20261115)
        vocab = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
        texts = [" ".join(rng.choice(vocab) for _ in range(rng.randint(0, 80))) for _ in range(40)]
        texts += ["", "alpha", "alpha alpha alpha alpha alpha", "alpha beta"]
        self._assert_matches_reference(self._make_op("space"), texts)

    def test_character_tokenization_matches_reference(self):
        texts = [
            "注意力机制是深度学习中的一个重要组成部分",
            "aaaaaaaaaaaaaaaaaaaa",
            "abcde",
            "abc",
            "",
        ]
        self._assert_matches_reference(self._make_op("character"), texts)

    def test_punctuation_tokenization_matches_reference(self):
        texts = [
            "Hello, world! This is a test.",
            "a,b,c,d,e,f,g",
            "one two three",
            "",
        ]
        self._assert_matches_reference(self._make_op("punctuation"), texts)

    def test_lowercase_and_ignore_pattern_match_reference(self):
        op = self._make_op("space", lowercase=True, ignore_pattern=r"<[^>]+>")
        self._assert_matches_reference(
            op,
            [
                "<b>Alpha</b> beta gamma delta epsilon zeta",
                "ALPHA BETA GAMMA DELTA",
                "<tag>only</tag>",
            ],
        )

    def test_duplicate_shingles_do_not_change_result(self):
        op = self._make_op("space")
        unique = op.compute_hash({"text": "alpha alpha alpha alpha alpha"})
        repeated = op.compute_hash({"text": "alpha " * 200})
        self.assertEqual(unique[HashKeys.minhash], repeated[HashKeys.minhash])

    def test_empty_text_uses_max_hash(self):
        op = self._make_op("space")
        result = op.compute_hash({"text": ""})
        expected = bytes(np.full(4, MAX_HASH, dtype=np.uint64).byteswap().data)
        self.assertEqual(result[HashKeys.minhash][0], expected)

    def test_existing_hash_is_not_recomputed(self):
        op = self._make_op("space")
        result = op.compute_hash({"text": "alpha beta gamma", HashKeys.minhash: ["sentinel"]})
        self.assertEqual(result[HashKeys.minhash], ["sentinel"])

    def test_invalid_tokenization_raises(self):
        op = self._make_op("space")
        op.tokenization = "unknown"
        with self.assertRaises(NotImplementedError):
            op.compute_hash({"text": "alpha beta gamma"})


if __name__ == "__main__":
    unittest.main()

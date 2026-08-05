import random
import unittest
from unittest.mock import patch

import numpy as np

from data_juicer.ops.common.helper_func import split_on_whitespace
from data_juicer.ops.deduplicator import document_minhash_deduplicator as minhash_module
from data_juicer.ops.deduplicator.document_minhash_deduplicator import (
    MAX_HASH,
    MERSENNE_PRIME,
    DocumentMinhashDeduplicator,
    sha1_hash32,
)
from data_juicer.utils.constant import HashKeys


class _FakeSentencePieceTokenizer:
    def encode(self, text, out_type):
        if out_type is not str:
            raise AssertionError("unexpected tokenizer output type")
        return list(text)


class MinhashBlockwisePermutationTest(unittest.TestCase):
    @staticmethod
    def _new_op(**kwargs):
        defaults = {
            "tokenization": "space",
            "window_size": 1,
            "num_permutations": 32,
            "num_bands": 8,
            "num_rows_per_band": 4,
        }
        defaults.update(kwargs)
        return DocumentMinhashDeduplicator(**defaults)

    @staticmethod
    def _legacy_hash(op, text):
        if op.lowercase:
            text = text.lower()
        if op.ignore_pattern:
            text = op.ignore_pattern.sub("", text)
        if op.tokenization == "character":
            tokens = {
                str.encode(text[index : index + op.window_size]) for index in range(len(text) - op.window_size + 1)
            }
        elif op.tokenization == "punctuation":
            pieces = op.punctuation_pattern.split(text)
            tokens = {
                str.encode(" ".join(pieces[index : index + op.window_size]))
                for index in range(len(pieces) - op.window_size + 1)
            }
        elif op.tokenization == "space":
            pieces = split_on_whitespace(text)
            tokens = {
                str.encode(" ".join(pieces[index : index + op.window_size]))
                for index in range(len(pieces) - op.window_size + 1)
            }
        elif op.tokenization == "sentencepiece":
            pieces = op.tokenizer.encode(text, out_type=str)
            tokens = {
                str.encode("".join(pieces[index : index + op.window_size]))
                for index in range(len(pieces) - op.window_size + 1)
            }
        else:
            raise AssertionError(f"unsupported test tokenization: {op.tokenization}")

        hv = np.fromiter(
            (sha1_hash32(token) for token in tokens),
            dtype=np.uint64,
            count=len(tokens),
        )
        phv = np.bitwise_and(
            (hv[:, None] * op.perm_a + op.perm_b) % MERSENNE_PRIME,
            MAX_HASH,
        )
        hash_values = phv.min(axis=0)
        return [bytes(hash_values[start:end].byteswap().data) for start, end in op.hash_ranges]

    def test_compute_hash_bounds_the_token_permutation_workspace(self):
        op = self._new_op()
        sample = {"text": " ".join(f"token-{index}" for index in range(64))}
        original_bitwise_and = minhash_module.np.bitwise_and
        observed_shapes = []

        def recording_bitwise_and(values, mask):
            observed_shapes.append(values.shape)
            return original_bitwise_and(values, mask)

        with (
            patch.object(
                minhash_module,
                "_MINHASH_PERMUTATION_BLOCK_BYTES",
                op.num_permutation * np.dtype(np.uint64).itemsize * 7,
                create=True,
            ),
            patch.object(
                minhash_module.np,
                "bitwise_and",
                side_effect=recording_bitwise_and,
            ),
        ):
            op.compute_hash(sample)

        self.assertGreater(len(observed_shapes), 1)
        self.assertTrue(all(shape[0] <= 7 for shape in observed_shapes))
        self.assertTrue(all(shape[1] == op.num_permutation for shape in observed_shapes))

    def test_blockwise_hash_is_byte_exact_with_full_matrix(self):
        op = self._new_op(
            window_size=2,
            lowercase=False,
            ignore_pattern=r"[0-9]",
            num_permutations=64,
            num_bands=8,
            num_rows_per_band=8,
        )
        texts = [
            "One TWO three four five six",
            "punctuation, Unicode-世界, and 12345",
            "repeat repeat repeat distinct suffix",
        ]

        for budget in (64 * 8, 64 * 8 * 3, 8 * 1024 * 1024):
            for text in texts:
                with self.subTest(budget=budget, text=text):
                    expected = self._legacy_hash(op, text)
                    with patch.object(
                        minhash_module,
                        "_MINHASH_PERMUTATION_BLOCK_BYTES",
                        budget,
                        create=True,
                    ):
                        actual = op.compute_hash({"text": text})[HashKeys.minhash]
                    self.assertEqual(actual, expected)

    def test_all_tokenization_modes_match_full_matrix(self):
        sentencepiece_op = self._new_op(window_size=3)
        sentencepiece_op.tokenization = "sentencepiece"
        sentencepiece_op.tokenizer = _FakeSentencePieceTokenizer()
        cases = [
            (
                self._new_op(tokenization="character", window_size=3, ignore_pattern=r"\p{P}"),
                "标点，Unicode🙂与重复重复文本。",
            ),
            (
                self._new_op(tokenization="punctuation", window_size=2),
                "alpha,beta.gamma!delta?epsilon:zeta",
            ),
            (self._new_op(window_size=2, lowercase=False), "ONE two THREE four five"),
            (sentencepiece_op, "abCD世界🙂"),
        ]

        for op, text in cases:
            with self.subTest(tokenization=op.tokenization):
                expected = self._legacy_hash(op, text)
                with patch.object(
                    minhash_module,
                    "_MINHASH_PERMUTATION_BLOCK_BYTES",
                    op.num_permutation * np.dtype(np.uint64).itemsize * 2,
                    create=True,
                ):
                    sample = {"text": text, "ordinal": 17, "marker": b"\x00\xff"}
                    original = sample.copy()
                    returned = op.compute_hash(sample)
                self.assertEqual(returned[HashKeys.minhash], expected)
                self.assertEqual(returned["ordinal"], 17)
                self.assertEqual(returned["marker"], b"\x00\xff")
                self.assertEqual(sample, original)

    def test_empty_shingles_preserve_legacy_reduction_error(self):
        op = self._new_op(window_size=5, num_permutations=16, num_bands=4, num_rows_per_band=4)

        with self.assertRaisesRegex(
            ValueError,
            "zero-size array to reduction operation minimum which has no identity",
        ):
            op.compute_hash({"text": "too short"})

    def test_randomized_blockwise_hash_matches_full_matrix(self):
        generator = random.Random(20260728)
        op = self._new_op(
            lowercase=False,
            num_permutations=33,
            num_bands=3,
            num_rows_per_band=11,
        )
        vocabulary = [
            "alpha",
            "BETA",
            "punctuation,",
            "世界",
            "🙂",
            "repeat",
            "12345",
        ]

        for case_index in range(500):
            text = " ".join(
                generator.choice(vocabulary) + (f"-{index}" if generator.random() < 0.4 else "")
                for index in range(generator.randint(1, 80))
            )
            expected = self._legacy_hash(op, text)
            block_rows = generator.randint(1, 9)
            with patch.object(
                minhash_module,
                "_MINHASH_PERMUTATION_BLOCK_BYTES",
                block_rows * op.num_permutation * np.dtype(np.uint64).itemsize,
                create=True,
            ):
                actual = op.compute_hash({"text": text})[HashKeys.minhash]
            self.assertEqual(actual, expected, f"case {case_index} differed with block size {block_rows}")

    def test_workspace_smaller_than_one_permutation_row_still_progresses(self):
        op = self._new_op(num_permutations=16, num_bands=4, num_rows_per_band=4)
        text = "alpha beta gamma delta"
        expected = self._legacy_hash(op, text)

        with patch.object(
            minhash_module,
            "_MINHASH_PERMUTATION_BLOCK_BYTES",
            1,
            create=True,
        ):
            actual = op.compute_hash({"text": text})[HashKeys.minhash]

        self.assertEqual(actual, expected)

    def test_existing_minhash_is_returned_without_recomputation(self):
        existing = [b"already-computed"]
        sample = {"text": "unused", HashKeys.minhash: existing}
        op = self._new_op(num_permutations=16, num_bands=4, num_rows_per_band=4)

        with patch.object(
            minhash_module,
            "sha1_hash32",
            side_effect=AssertionError("precomputed hash must not be recomputed"),
        ):
            returned = op.compute_hash(sample)

        self.assertEqual(returned, sample)
        self.assertEqual(returned[HashKeys.minhash], existing)


if __name__ == "__main__":
    unittest.main()

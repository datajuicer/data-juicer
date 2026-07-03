# Reproduction test for batch processing bug in NlpaugEnMapper
# Bug: samples[self.text_key][0] only processes the first sample in a batch.
# In production (batch_size=1000), 999 out of 1000 samples are never augmented.

import unittest
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase
from data_juicer.ops.mapper.nlpaug_en_mapper import NlpaugEnMapper


class NlpaugEnMapperBatchBugTest(DataJuicerTestCaseBase):
    """Demonstrate that process_batched only augments the first sample."""

    def test_batch_only_augments_first_sample(self):
        """With batch_size > 1, only the first sample should be augmented
        (before the fix). After the fix, all samples should be augmented."""
        op = NlpaugEnMapper(
            sequential=False,
            aug_num=1,
            keep_original_sample=True,
            delete_random_word=True,
        )

        # Simulate a batch of 3 samples (dict-of-lists format)
        samples = {
            'text': [
                'The quick brown fox jumps over the lazy dog',
                'Machine learning is transforming the world today',
                'Natural language processing enables computers to understand text',
            ],
            'meta': ['meta1', 'meta2', 'meta3'],
        }

        result = op.process_batched(samples)

        # With 3 input samples, 1 aug method, aug_num=1, keep_original=True:
        # Each sample should produce 1 original + 1 augmented = 2 texts per sample
        # Total expected: 3 originals + 3 augmented = 6
        num_texts = len(result['text'])
        num_metas = len(result['meta'])

        # Assert that ALL 3 original texts are present in the output
        for original_text in samples['text']:
            self.assertIn(original_text, result['text'],
                          f"Original text missing from output: {original_text}")

        # Assert correct total count: 3 originals + 3 augmented = 6
        self.assertEqual(num_texts, 6,
                         f"Expected 6 texts (3 original + 3 augmented), got {num_texts}")
        self.assertEqual(num_metas, num_texts,
                         f"Meta count ({num_metas}) should match text count ({num_texts})")

    def test_batch_without_keep_original(self):
        """Without keeping originals, all samples should still be augmented."""
        op = NlpaugEnMapper(
            sequential=False,
            aug_num=1,
            keep_original_sample=False,
            delete_random_word=True,
        )

        samples = {
            'text': [
                'The quick brown fox jumps over the lazy dog',
                'Machine learning is transforming the world today',
                'Natural language processing enables computers to understand text',
            ],
            'meta': ['meta1', 'meta2', 'meta3'],
        }

        result = op.process_batched(samples)
        num_texts = len(result['text'])

        # Should have exactly 3 augmented texts (one per input sample)
        self.assertEqual(num_texts, 3,
                         f"Expected 3 augmented texts, got {num_texts}")

    def test_batch_sequential_mode(self):
        """Sequential mode should also process all samples in the batch."""
        op = NlpaugEnMapper(
            sequential=True,
            aug_num=2,
            keep_original_sample=True,
            delete_random_word=True,
            swap_random_char=True,
        )

        samples = {
            'text': [
                'The quick brown fox jumps over the lazy dog',
                'Machine learning is transforming the world today',
            ],
            'meta': ['meta1', 'meta2'],
        }

        result = op.process_batched(samples)
        num_texts = len(result['text'])

        # 2 originals + 2 samples * 2 aug_num = 6
        self.assertEqual(num_texts, 6,
                         f"Expected 6 texts (2 original + 4 augmented), got {num_texts}")


if __name__ == '__main__':
    unittest.main()

"""
Reproduction test for GeneralFusedOP bug: Mapper results assigned to
wrong variable (op_fusion.py:240).

Bug: In GeneralFusedOP.process_batched(), line 240 does:
    samples = op.process_batched(tmp_samples, **process_args)
but the loop continues using tmp_samples. This means:
1. For in-place-mutating mappers: bug is masked (they mutate tmp_samples directly)
2. For mappers returning new dicts: results are silently discarded
3. The method returns tmp_samples (line 264), not samples

This test demonstrates the bug by using two consecutive mappers in a
GeneralFusedOP. The first mapper uppercases text, the second appends a
suffix. Without the fix, the first mapper's results are discarded when
the second mapper returns a new dict.
"""
import unittest

from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase
from data_juicer.ops.base_op import Mapper
from data_juicer.ops.op_fusion import GeneralFusedOP
from data_juicer.utils.constant import Fields


class MockUpperCaseMapper(Mapper):
    """A mapper that uppercases text and returns a NEW dict (not in-place)."""
    _batched_op = True

    def __init__(self, text_key='text', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_key = text_key
        self._name = 'mock_upper_case_mapper'

    def process_batched(self, samples, **kwargs):
        # Create a NEW dict instead of mutating in-place
        # This is what triggers the bug
        new_samples = samples.copy()
        new_samples[self.text_key] = [t.upper() for t in samples[self.text_key]]
        return new_samples


class MockSuffixMapper(Mapper):
    """A mapper that appends a suffix to text and returns a NEW dict."""
    _batched_op = True

    def __init__(self, suffix='_DONE', text_key='text', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_key = text_key
        self.suffix = suffix
        self._name = 'mock_suffix_mapper'

    def process_batched(self, samples, **kwargs):
        # Also returns a new dict
        new_samples = samples.copy()
        new_samples[self.text_key] = [t + self.suffix for t in samples[self.text_key]]
        return new_samples


class MockInPlaceMapper(Mapper):
    """A mapper that mutates in-place (common pattern, masks the bug)."""
    _batched_op = True

    def __init__(self, text_key='text', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_key = text_key
        self._name = 'mock_inplace_mapper'

    def process_batched(self, samples, **kwargs):
        # Mutates samples in-place - this masks the bug
        samples[self.text_key] = [t.lower() for t in samples[self.text_key]]
        return samples


class TestGeneralFusedOPMapperBug(DataJuicerTestCaseBase):
    """Test that demonstrates the mapper result discard bug in GeneralFusedOP."""

    def _make_fused_op(self, ops):
        """Create a GeneralFusedOP with pre-built op instances."""
        fused = GeneralFusedOP.__new__(GeneralFusedOP)
        fused._name = 'GeneralFusedOP:test'
        fused.fused_ops = ops
        fused.accelerator = 'cpu'
        fused.batch_size = 10
        fused.num_proc = 1
        return fused

    def test_two_new_dict_mappers_results_chained(self):
        """
        Two mappers that return NEW dicts. The first mapper's result
        should be visible to the second mapper.

        Expected: 'hello' -> 'HELLO' -> 'HELLO_DONE'
        Bug gives: 'hello' -> (HELLO discarded) -> 'hello_DONE'
        """
        upper_mapper = MockUpperCaseMapper()
        suffix_mapper = MockSuffixMapper(suffix='_DONE')

        fused_op = self._make_fused_op([upper_mapper, suffix_mapper])

        samples = {
            'text': ['hello', 'world'],
            Fields.stats: [{}, {}],
        }

        result = fused_op.process_batched(samples)

        # With correct behavior: HELLO_DONE, WORLD_DONE
        # With bug: hello_DONE, world_DONE (uppercase is lost)
        self.assertEqual(result['text'], ['HELLO_DONE', 'WORLD_DONE'],
                         "BUG CONFIRMED: First mapper's results were discarded! "
                         "Got lowercase because upper_mapper's return value was "
                         "assigned to 'samples' but loop uses 'tmp_samples'.")

    def test_single_mapper_result_returned(self):
        """
        A single mapper that returns a NEW dict. The result should be
        in the returned value.

        Expected: 'hello' -> 'HELLO'
        Bug gives: 'hello' (unchanged, because return is tmp_samples not samples)
        """
        upper_mapper = MockUpperCaseMapper()
        fused_op = self._make_fused_op([upper_mapper])

        samples = {
            'text': ['hello', 'world'],
            Fields.stats: [{}, {}],
        }

        result = fused_op.process_batched(samples)

        self.assertEqual(result['text'], ['HELLO', 'WORLD'],
                         "BUG CONFIRMED: Mapper returned new dict, but "
                         "process_batched returns tmp_samples (original), "
                         "not the mapper's output.")

    def test_inplace_mapper_masks_bug(self):
        """
        An in-place mapper followed by a new-dict mapper.
        The in-place mapper's changes survive (they mutate tmp_samples),
        but a subsequent new-dict mapper's results are lost.
        """
        inplace_mapper = MockInPlaceMapper()  # lowercases in-place
        suffix_mapper = MockSuffixMapper(suffix='_END')

        fused_op = self._make_fused_op([inplace_mapper, suffix_mapper])

        samples = {
            'text': ['HELLO', 'WORLD'],
            Fields.stats: [{}, {}],
        }

        result = fused_op.process_batched(samples)

        # In-place mapper works: 'HELLO' -> 'hello' (mutates tmp_samples)
        # Suffix mapper creates new dict: 'hello' -> 'hello_END'
        # But bug: suffix mapper's result goes to 'samples', return is tmp_samples
        # So we get 'hello' (without _END suffix)
        self.assertEqual(result['text'], ['hello_END', 'world_END'],
                         "BUG CONFIRMED: In-place mapper worked, but the "
                         "subsequent new-dict mapper's results were discarded.")


if __name__ == '__main__':
    unittest.main(verbosity=2)

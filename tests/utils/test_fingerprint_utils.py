import unittest

import dill

from data_juicer.core import NestedDataset
from data_juicer.ops.filter.text_length_filter import TextLengthFilter
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


class FingerprintCacheStabilityTest(DataJuicerTestCaseBase):
    """Tests that execution-only attributes do not poison cache fingerprints."""

    def _make_op(self, work_dir='/tmp/run_a', num_proc=4, **extra):
        return TextLengthFilter(
            min_len=10,
            max_len=100,
            work_dir=work_dir,
            num_proc=num_proc,
            **extra,
        )

    def test_fingerprint_stable_across_work_dirs(self):
        """Same OP with different work_dir must produce identical hash."""
        op_a = self._make_op(work_dir='/tmp/run_a')
        op_b = self._make_op(work_dir='/tmp/run_b')
        self.assertEqual(dill.dumps(op_a), dill.dumps(op_b))

    def test_fingerprint_changes_when_data_params_change(self):
        """Different data-affecting params must produce different hash."""
        op_a = TextLengthFilter(min_len=10, max_len=100)
        op_b = TextLengthFilter(min_len=20, max_len=100)
        self.assertNotEqual(dill.dumps(op_a), dill.dumps(op_b))

    def test_fingerprint_stable_across_num_proc(self):
        """Same OP with different parallelism must produce identical hash."""
        op_a = self._make_op(num_proc=1)
        op_b = self._make_op(num_proc=8)
        self.assertEqual(dill.dumps(op_a), dill.dumps(op_b))

    def test_end_to_end_generate_fingerprint(self):
        """generate_fingerprint(dataset, op.process) stable across work_dirs."""
        dataset = NestedDataset.from_list([
            {'text': 'hello world', 'stats': {}},
        ])
        op_a = self._make_op(work_dir='/tmp/run_a')
        op_b = self._make_op(work_dir='/tmp/run_b')
        fp_a = generate_fingerprint(dataset, op_a.compute_stats)
        fp_b = generate_fingerprint(dataset, op_b.compute_stats)
        self.assertEqual(fp_a, fp_b)

    def test_serialization_round_trip(self):
        """dill round-trip restores data attrs; excluded attrs get defaults."""
        op = self._make_op(
            work_dir='/tmp/run_x',
            num_proc=16,
            skip_op_error=True,
        )
        restored = dill.loads(dill.dumps(op))

        # Data-affecting attributes preserved
        self.assertEqual(restored.min_len, 10)
        self.assertEqual(restored.max_len, 100)

        # Excluded attributes get safe defaults, not original values
        self.assertIsNone(restored.work_dir)
        self.assertIsNone(restored.num_proc)
        self.assertFalse(restored.skip_op_error)


if __name__ == '__main__':
    unittest.main()

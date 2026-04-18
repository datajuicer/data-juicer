import unittest

import dill

from data_juicer.core import NestedDataset
from data_juicer.ops.filter.text_length_filter import TextLengthFilter
from data_juicer.utils.fingerprint_utils import Hasher, generate_fingerprint
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
        self.assertEqual(Hasher.hash(op_a), Hasher.hash(op_b))

    def test_fingerprint_changes_when_data_params_change(self):
        """Different data-affecting params must produce different hash."""
        op_a = TextLengthFilter(min_len=10, max_len=100)
        op_b = TextLengthFilter(min_len=20, max_len=100)
        self.assertNotEqual(Hasher.hash(op_a), Hasher.hash(op_b))

    def test_fingerprint_stable_across_num_proc(self):
        """num_proc is not in _NON_FINGERPRINT_ATTRS, but it doesn't change
        between identical configs, so this is just a sanity check that
        Hasher.hash uses _fingerprint_bytes."""
        op_a = self._make_op(work_dir='/tmp/run_a')
        op_b = self._make_op(work_dir='/tmp/run_b')
        self.assertEqual(Hasher.hash(op_a), Hasher.hash(op_b))

    def test_end_to_end_generate_fingerprint(self):
        """generate_fingerprint(dataset, op.compute_stats) stable across
        work_dirs."""
        dataset = NestedDataset.from_list([
            {'text': 'hello world', 'stats': {}},
        ])
        op_a = self._make_op(work_dir='/tmp/run_a')
        op_b = self._make_op(work_dir='/tmp/run_b')
        fp_a = generate_fingerprint(dataset, op_a.compute_stats)
        fp_b = generate_fingerprint(dataset, op_b.compute_stats)
        self.assertEqual(fp_a, fp_b)

    def test_serialization_round_trip_preserves_all_attrs(self):
        """dill round-trip must preserve ALL attrs including work_dir,
        since __getstate__ is no longer overridden."""
        op = self._make_op(
            work_dir='/tmp/run_x',
            num_proc=16,
            skip_op_error=True,
        )
        restored = dill.loads(dill.dumps(op))

        # Data-affecting attributes preserved
        self.assertEqual(restored.min_len, 10)
        self.assertEqual(restored.max_len, 100)

        # Execution attrs also preserved (important for worker pickling)
        self.assertEqual(restored.work_dir, '/tmp/run_x')
        self.assertEqual(restored.num_proc, 16)
        self.assertTrue(restored.skip_op_error)


class WrappedFunctionFingerprintTest(DataJuicerTestCaseBase):
    """Tests that wrapped bound methods (via wrap_func_with_nested_access)
    produce stable fingerprints across work_dirs."""

    def test_wrapped_compute_stats_stable(self):
        from data_juicer.core.data.dj_dataset import wrap_func_with_nested_access

        op_a = TextLengthFilter(min_len=5, max_len=10000, work_dir='/tmp/a')
        op_b = TextLengthFilter(min_len=5, max_len=10000, work_dir='/tmp/b')
        wa = wrap_func_with_nested_access(op_a.compute_stats)
        wb = wrap_func_with_nested_access(op_b.compute_stats)
        self.assertEqual(Hasher.hash(wa), Hasher.hash(wb))

    def test_wrapped_differs_when_params_change(self):
        from data_juicer.core.data.dj_dataset import wrap_func_with_nested_access

        op_a = TextLengthFilter(min_len=5, max_len=10000, work_dir='/tmp/a')
        op_b = TextLengthFilter(min_len=50, max_len=10000, work_dir='/tmp/a')
        wa = wrap_func_with_nested_access(op_a.compute_stats)
        wb = wrap_func_with_nested_access(op_b.compute_stats)
        self.assertNotEqual(Hasher.hash(wa), Hasher.hash(wb))

    def test_multistep_pipeline_cache_hit(self):
        """Full pipeline with multiple OPs: second run with different
        work_dir must produce zero new cache files."""
        import glob
        import os

        from datasets import load_dataset, enable_caching

        from data_juicer.ops.filter.alphanumeric_filter import AlphanumericFilter
        from data_juicer.ops.filter.words_num_filter import WordsNumFilter
        from data_juicer.utils.constant import Fields

        enable_caching()
        ds = NestedDataset(load_dataset(
            'json',
            data_files='demos/data/demo-dataset.jsonl',
            split='train',
        ))
        if Fields.stats not in ds.features:
            ds = ds.map(lambda x: {Fields.stats: {}})
        cache_dir = os.path.dirname(ds.cache_files[0]['filename'])

        def run_pipeline(dataset, work_dir):
            ops = [
                TextLengthFilter(min_len=5, max_len=10000, work_dir=work_dir),
                WordsNumFilter(min_num=2, max_num=1000, work_dir=work_dir),
                AlphanumericFilter(min_ratio=0.0, max_ratio=1.0,
                                   work_dir=work_dir),
            ]
            cur = dataset
            for op in ops:
                cur = cur.map(op.compute_stats, num_proc=1)
                cur = cur.filter(op.process, num_proc=1)
            return cur

        run_pipeline(ds, '/tmp/pipeline_test_A')
        cache_after_a = set(glob.glob(os.path.join(cache_dir, '*.arrow')))

        run_pipeline(ds, '/tmp/pipeline_test_B')
        cache_after_b = set(glob.glob(os.path.join(cache_dir, '*.arrow')))

        new_files = cache_after_b - cache_after_a
        self.assertEqual(len(new_files), 0,
                         f'Pipeline B created {len(new_files)} new cache '
                         f'files; expected 0 (full cache hit)')


if __name__ == '__main__':
    unittest.main()

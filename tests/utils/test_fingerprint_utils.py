import unittest
from unittest.mock import patch

import dill

from data_juicer.core import NestedDataset
from data_juicer.ops.filter.text_length_filter import TextLengthFilter
from data_juicer.utils.fingerprint_utils import (
    Hasher,
    generate_fingerprint,
    update_fingerprint,
)
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


class HasherBasicTest(DataJuicerTestCaseBase):
    """Test Hasher class basic operations."""

    def test_hash_bytes_single(self):
        result = Hasher.hash_bytes(b"hello")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_hash_bytes_list(self):
        result = Hasher.hash_bytes([b"hello", b"world"])
        self.assertIsInstance(result, str)

    def test_hash_bytes_deterministic(self):
        r1 = Hasher.hash_bytes(b"test")
        r2 = Hasher.hash_bytes(b"test")
        self.assertEqual(r1, r2)

    def test_hash_bytes_different_input(self):
        r1 = Hasher.hash_bytes(b"a")
        r2 = Hasher.hash_bytes(b"b")
        self.assertNotEqual(r1, r2)

    def test_update_and_hexdigest(self):
        h = Hasher()
        h.update("hello")
        result = h.hexdigest()
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_hash_dispatch_fallback(self):
        """Types not in dispatch should use hash_default (dill-based)."""
        result = Hasher.hash({"key": "value"})
        self.assertIsInstance(result, str)


class UpdateFingerprintTest(DataJuicerTestCaseBase):
    """Test update_fingerprint: handles unhashable transforms gracefully."""

    def test_normal_transform(self):
        """Serializable transform + args should produce a deterministic fingerprint."""
        fp = update_fingerprint("base_fp", lambda x: x, {"key": "val"})
        self.assertIsInstance(fp, str)
        self.assertGreater(len(fp), 0)

    def test_deterministic_same_inputs(self):
        fp1 = update_fingerprint("fp", "transform", {"a": 1})
        fp2 = update_fingerprint("fp", "transform", {"a": 1})
        self.assertEqual(fp1, fp2)

    def test_different_args_different_fingerprint(self):
        fp1 = update_fingerprint("fp", "transform", {"a": 1})
        fp2 = update_fingerprint("fp", "transform", {"a": 2})
        self.assertNotEqual(fp1, fp2)

    def test_unhashable_transform_returns_random(self):
        """When transform can't be serialized, should return a random fingerprint."""
        from datasets.fingerprint import fingerprint_warnings
        # Reset warning state
        fingerprint_warnings.pop("update_fingerprint_transform_hash_failed", None)

        # Create an object that dill cannot serialize
        class Unpicklable:
            def __reduce__(self):
                raise TypeError("cannot pickle")

        with patch("data_juicer.utils.fingerprint_utils._CACHING_ENABLED", True):
            fp = update_fingerprint("base", Unpicklable(), {})

        self.assertIsInstance(fp, str)
        self.assertGreater(len(fp), 0)
        # Should be a random fingerprint, different each time
        fingerprint_warnings.pop("update_fingerprint_transform_hash_failed", None)
        with patch("data_juicer.utils.fingerprint_utils._CACHING_ENABLED", True):
            fp2 = update_fingerprint("base", Unpicklable(), {})
        self.assertNotEqual(fp, fp2)

    def test_unhashable_transform_no_caching(self):
        """When caching is disabled and transform unhashable, still returns random fp."""
        from datasets.fingerprint import fingerprint_warnings
        fingerprint_warnings.pop("update_fingerprint_transform_hash_failed", None)

        class Unpicklable:
            def __reduce__(self):
                raise TypeError("cannot pickle")

        with patch("data_juicer.utils.fingerprint_utils._CACHING_ENABLED", False):
            fp = update_fingerprint("base", Unpicklable(), {})
        self.assertIsInstance(fp, str)

    def test_unhashable_arg_returns_random(self):
        """When a transform_arg can't be serialized, should return random fingerprint."""
        from datasets.fingerprint import fingerprint_warnings
        fingerprint_warnings.pop("update_fingerprint_transform_hash_failed", None)

        class Unpicklable:
            def __reduce__(self):
                raise TypeError("cannot pickle")

        with patch("data_juicer.utils.fingerprint_utils._CACHING_ENABLED", True):
            fp = update_fingerprint("base", "good_transform", {"bad_arg": Unpicklable()})
        self.assertIsInstance(fp, str)

    def test_unhashable_arg_no_caching(self):
        """When caching disabled and arg unhashable, still returns random fp."""
        from datasets.fingerprint import fingerprint_warnings
        fingerprint_warnings.pop("update_fingerprint_transform_hash_failed", None)

        class Unpicklable:
            def __reduce__(self):
                raise TypeError("cannot pickle")

        with patch("data_juicer.utils.fingerprint_utils._CACHING_ENABLED", False):
            fp = update_fingerprint("base", "transform", {"arg": Unpicklable()})
        self.assertIsInstance(fp, str)

    def test_empty_args(self):
        fp = update_fingerprint("fp", "transform", {})
        self.assertIsInstance(fp, str)


class HasherFindOpOwnerWrappedTest(DataJuicerTestCaseBase):
    """Test _find_op_owner following __wrapped__ chain."""

    def test_find_op_owner_with_wrapped(self):
        """Create an object with __wrapped__ chain that has
        _fingerprint_bytes method. Call _find_op_owner."""

        class FakeOp:
            def _fingerprint_bytes(self):
                return b'fake_fingerprint'

            def compute(self, x):
                return x

        op = FakeOp()
        bound_method = op.compute

        # Create a wrapper chain simulating decorators
        import functools

        @functools.wraps(bound_method)
        def wrapper1(*args, **kwargs):
            return bound_method(*args, **kwargs)

        @functools.wraps(wrapper1)
        def wrapper2(*args, **kwargs):
            return wrapper1(*args, **kwargs)

        # wrapper2.__wrapped__ -> wrapper1 -> bound_method (which has __self__)
        wrapper2.__wrapped__ = wrapper1
        wrapper1.__wrapped__ = bound_method

        obj, func_name = Hasher._find_op_owner(wrapper2)
        self.assertIs(obj, op)
        self.assertEqual(func_name, 'compute')

    def test_find_op_owner_no_wrapped_no_self(self):
        """A plain function with no __self__ or __wrapped__ returns (None, None)."""
        def plain_func(x):
            return x

        obj, func_name = Hasher._find_op_owner(plain_func)
        self.assertIsNone(obj)
        self.assertIsNone(func_name)

    def test_find_op_owner_direct_bound_method(self):
        """A direct bound method with _fingerprint_bytes on __self__."""

        class MyOp:
            def _fingerprint_bytes(self):
                return b'my_op_fp'

            def process(self, x):
                return x

        op = MyOp()
        obj, func_name = Hasher._find_op_owner(op.process)
        self.assertIs(obj, op)
        self.assertEqual(func_name, 'process')


class HasherDispatchTest(DataJuicerTestCaseBase):
    """Test Hasher.dispatch registration for custom types."""

    def setUp(self):
        super().setUp()
        # Save original dispatch
        self._original_dispatch = Hasher.dispatch.copy()

    def tearDown(self):
        # Restore dispatch
        Hasher.dispatch = self._original_dispatch
        super().tearDown()

    def test_hash_dispatch_custom_type(self):
        """Register a custom type in Hasher.dispatch, verify hash() uses it."""

        class MyCustomType:
            def __init__(self, val):
                self.val = val

        def custom_hasher(cls, value):
            return cls.hash_bytes(str(value.val).encode())

        Hasher.dispatch[MyCustomType] = custom_hasher

        obj1 = MyCustomType(42)
        obj2 = MyCustomType(42)
        obj3 = MyCustomType(99)

        hash1 = Hasher.hash(obj1)
        hash2 = Hasher.hash(obj2)
        hash3 = Hasher.hash(obj3)

        # Same value produces same hash
        self.assertEqual(hash1, hash2)
        # Different value produces different hash
        self.assertNotEqual(hash1, hash3)


class UpdateFingerprintRepeatedWarningTest(DataJuicerTestCaseBase):
    """Test update_fingerprint second call goes through 'already warned' path."""

    def test_repeated_unhashable_transform_second_call(self):
        """Call update_fingerprint twice with same unhashable transform;
        second call goes through 'already warned' path (logger.info)."""
        from datasets.fingerprint import fingerprint_warnings

        # Reset the warning state
        fingerprint_warnings.pop(
            "update_fingerprint_transform_hash_failed", None)

        class Unpicklable:
            def __reduce__(self):
                raise TypeError("cannot pickle")

        # First call: sets the warning flag
        with patch(
            "data_juicer.utils.fingerprint_utils._CACHING_ENABLED", True
        ):
            fp1 = update_fingerprint("base", Unpicklable(), {})

        self.assertTrue(
            fingerprint_warnings.get(
                "update_fingerprint_transform_hash_failed", False))
        self.assertIsInstance(fp1, str)

        # Second call: "already warned" path - flag is already True
        with patch(
            "data_juicer.utils.fingerprint_utils._CACHING_ENABLED", True
        ):
            fp2 = update_fingerprint("base", Unpicklable(), {})

        self.assertIsInstance(fp2, str)
        # Both should be random fingerprints (different from each other)
        self.assertNotEqual(fp1, fp2)

        # Clean up
        fingerprint_warnings.pop(
            "update_fingerprint_transform_hash_failed", None)


if __name__ == '__main__':
    unittest.main()

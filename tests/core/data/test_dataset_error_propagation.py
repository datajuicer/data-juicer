import unittest
from unittest.mock import MagicMock, patch
from datasets import Dataset

from data_juicer.core.data import NestedDataset
from data_juicer.ops import Deduplicator
from data_juicer.utils.unittest_utils import (
    TEST_TAG,
    DataJuicerTestCaseBase,
)


class TestDjDatasetErrorPropagation(DataJuicerTestCaseBase):
    """Test that NestedDataset.process() propagates exceptions instead of
    calling exit(1), making it safe for library usage."""

    def setUp(self):
        super().setUp()
        self.data = [
            {'text': 'Hello', 'score': 1},
            {'text': 'World', 'score': 2},
        ]
        self.dataset = NestedDataset(Dataset.from_list(self.data))

    def test_process_raises_on_op_error(self):
        """When an operator raises an exception during process(),
        it should propagate as an exception rather than calling exit(1)."""
        failing_op = MagicMock()
        failing_op._name = 'test_failing_op'
        failing_op._op_cfg = {}
        failing_op.use_cuda.return_value = False
        failing_op.run.side_effect = RuntimeError('op failed')

        with self.assertRaises(RuntimeError) as ctx:
            self.dataset.process([failing_op])
        self.assertIn('op failed', str(ctx.exception))

    def test_process_does_not_call_exit(self):
        """Verify exit() is never called during error handling."""
        failing_op = MagicMock()
        failing_op._name = 'test_failing_op'
        failing_op._op_cfg = {}
        failing_op.use_cuda.return_value = False
        failing_op.run.side_effect = ValueError('bad value')

        with patch('builtins.exit') as mock_exit:
            with self.assertRaises(ValueError):
                self.dataset.process([failing_op])
            mock_exit.assert_not_called()


class TestRayDatasetErrorPropagation(DataJuicerTestCaseBase):
    """Test that RayDataset._run_single_op() propagates exceptions and
    that the runtime_env fallback in process() works correctly."""

    def setUp(self):
        super().setUp()

    def _make_failing_op(self, error=None, runtime_env=None):
        """Create a mock operator that passes isinstance(op, Deduplicator)
        and raises on run()."""
        op = MagicMock(spec=Deduplicator)
        op._name = 'test_failing_dedup'
        op.runtime_env = runtime_env
        op.num_cpus = None
        op.num_gpus = None
        op.memory = None
        op.run.side_effect = error or RuntimeError('dedup failed')
        return op

    @TEST_TAG('ray')
    def test_run_single_op_propagates_exception(self):
        """_run_single_op should propagate exceptions instead of exit(1)."""
        import ray
        from data_juicer.core.data.ray_dataset import RayDataset

        dataset = RayDataset(ray.data.from_items([{'text': 'hello'}]))
        op = self._make_failing_op()

        with self.assertRaises(RuntimeError) as ctx:
            dataset._run_single_op(op)
        self.assertIn('dedup failed', str(ctx.exception))

    @TEST_TAG('ray')
    def test_process_fallback_on_runtime_env_failure(self):
        """When an op with runtime_env fails, process() should retry
        without runtime_env (fallback logic at lines 194-207)."""
        import ray
        from data_juicer.core.data.ray_dataset import RayDataset

        dataset = RayDataset(ray.data.from_items([{'text': 'hello'}]),
                             auto_op_parallelism=False)
        op = self._make_failing_op(
            error=RuntimeError('env setup failed'),
            runtime_env={'pip': ['nonexistent-pkg']},
        )
        # Make the retry (with runtime_env=None) succeed
        op.run.side_effect = [
            RuntimeError('env setup failed'),  # first call fails
            ray.data.from_items([{'text': 'hello'}]),  # retry succeeds
        ]

        # Should not raise because the fallback succeeds
        result = dataset.process([op])
        self.assertIsNotNone(result)
        # Verify runtime_env was restored after fallback
        self.assertEqual(op.runtime_env, {'pip': ['nonexistent-pkg']})

    @TEST_TAG('ray')
    def test_process_raises_when_no_runtime_env_fallback(self):
        """When an op without runtime_env fails, process() should
        propagate the exception (no fallback available)."""
        import ray
        from data_juicer.core.data.ray_dataset import RayDataset

        dataset = RayDataset(ray.data.from_items([{'text': 'hello'}]),
                             auto_op_parallelism=False)
        op = self._make_failing_op(
            error=ValueError('processing error'),
            runtime_env=None,
        )

        with self.assertRaises(ValueError) as ctx:
            dataset.process([op])
        self.assertIn('processing error', str(ctx.exception))


if __name__ == '__main__':
    unittest.main()

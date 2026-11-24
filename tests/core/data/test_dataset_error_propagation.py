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


class _FailingRayDeduplicator(Deduplicator):
    """Exercise Ray orchestration with deterministic failures on each attempt."""

    _name = 'test_failing_dedup'
    _supported_exec_modes = ('ray',)

    def __init__(self, errors, runtime_env=None):
        super().__init__(runtime_env=runtime_env)
        self.errors = iter(errors)
        self.attempted_runtime_envs = []

    def run(self, dataset):
        self.attempted_runtime_envs.append(self.runtime_env)
        error = next(self.errors)
        if error is not None:
            raise error
        return dataset


class TestRayDatasetErrorPropagation(DataJuicerTestCaseBase):
    """Test that RayDataset._run_single_op() propagates exceptions and
    that the runtime_env fallback in process() works correctly."""

    @TEST_TAG('ray')
    def test_run_single_op_propagates_exception(self):
        """_run_single_op should propagate exceptions instead of exit(1)."""
        import ray
        from data_juicer.core.data.ray_dataset import RayDataset

        dataset = RayDataset(ray.data.from_items([{'text': 'hello'}]))
        error = RuntimeError('dedup failed')
        op = _FailingRayDeduplicator([error])

        with self.assertRaises(RuntimeError) as ctx:
            dataset._run_single_op(op)
        self.assertIs(ctx.exception, error)
        self.assertEqual(op.attempted_runtime_envs, [None])

    @TEST_TAG('ray')
    def test_process_fallback_on_runtime_env_failure(self):
        """When an op with runtime_env fails, process() should retry
        without runtime_env and restore it after the retry."""
        import ray
        from data_juicer.core.data.ray_dataset import RayDataset

        dataset = RayDataset(ray.data.from_items([{'text': 'hello'}]),
                             auto_op_parallelism=False)
        runtime_env = {'pip': ['nonexistent-pkg']}
        op = _FailingRayDeduplicator(
            [RuntimeError('env setup failed'), None],
            runtime_env=runtime_env,
        )

        result = dataset.process([op])
        self.assertEqual(result.data.take_all(), [{'text': 'hello'}])
        self.assertEqual(op.attempted_runtime_envs, [runtime_env, None])
        self.assertIs(op.runtime_env, runtime_env)

    @TEST_TAG('ray')
    def test_process_restores_runtime_env_when_fallback_fails(self):
        import ray
        from data_juicer.core.data.ray_dataset import RayDataset

        dataset = RayDataset(ray.data.from_items([{'text': 'hello'}]),
                             auto_op_parallelism=False)
        runtime_env = {'pip': ['nonexistent-pkg']}
        fallback_error = ValueError('fallback failed')
        op = _FailingRayDeduplicator(
            [RuntimeError('env setup failed'), fallback_error],
            runtime_env=runtime_env,
        )

        with self.assertRaises(ValueError) as ctx:
            dataset.process([op])
        self.assertIs(ctx.exception, fallback_error)
        self.assertEqual(op.attempted_runtime_envs, [runtime_env, None])
        self.assertIs(op.runtime_env, runtime_env)

    @TEST_TAG('ray')
    def test_process_raises_when_no_runtime_env_fallback(self):
        """When an op without runtime_env fails, process() should
        propagate the exception (no fallback available)."""
        import ray
        from data_juicer.core.data.ray_dataset import RayDataset

        dataset = RayDataset(ray.data.from_items([{'text': 'hello'}]),
                             auto_op_parallelism=False)
        error = ValueError('processing error')
        op = _FailingRayDeduplicator([error])

        with self.assertRaises(ValueError) as ctx:
            dataset.process([op])
        self.assertIs(ctx.exception, error)
        self.assertEqual(op.attempted_runtime_envs, [None])


if __name__ == '__main__':
    unittest.main()

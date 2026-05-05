import unittest
from unittest.mock import MagicMock, patch
from datasets import Dataset

from data_juicer.core.data import NestedDataset
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


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


if __name__ == '__main__':
    unittest.main()

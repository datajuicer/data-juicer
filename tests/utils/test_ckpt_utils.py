"""
Tests for checkpoint utilities.

Tests cover:
- CheckpointManager (original non-Ray checkpoint manager)
- RayCheckpointManager (Ray-based checkpoint manager)
- CheckpointStrategy enum
- All checkpoint strategies (EVERY_OP, EVERY_N_OPS, MANUAL, DISABLED)
- Operation grouping logic
- Checkpoint save/load with error conditions
- Edge cases (empty ops, corrupted files, etc.)
"""

import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from data_juicer.core.data import NestedDataset
from data_juicer.utils.ckpt_utils import (
    CheckpointManager,
    CheckpointStrategy,
    RayCheckpointManager,
)
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class CkptUtilsTest(DataJuicerTestCaseBase):
    """Tests for CheckpointManager (original non-Ray checkpoint manager)."""

    def setUp(self) -> None:
        super().setUp()
        self.temp_output_path = 'tmp/test_ckpt_utils/'

    def tearDown(self):
        if os.path.exists(self.temp_output_path):
            os.system(f'rm -rf {self.temp_output_path}')
        super().tearDown()

    def test_basic_func(self):
        ckpt_path = os.path.join(self.temp_output_path, 'ckpt_1')
        manager = CheckpointManager(ckpt_path, original_process_list=[
            {'test_op_1': {'test_key': 'test_value_1'}},
            {'test_op_2': {'test_key': 'test_value_2'}},
        ])
        self.assertEqual(manager.get_left_process_list(), [
            {'test_op_1': {'test_key': 'test_value_1'}},
            {'test_op_2': {'test_key': 'test_value_2'}},
        ])
        self.assertFalse(manager.ckpt_available)

        self.assertFalse(manager.check_ckpt())
        os.makedirs(ckpt_path, exist_ok=True)
        os.makedirs(os.path.join(ckpt_path, 'latest'), exist_ok=True)
        with open(os.path.join(ckpt_path, 'ckpt_op.json'), 'w') as fout:
            json.dump([
                {'test_op_1': {'test_key': 'test_value_1'}},
            ], fout)
        self.assertTrue(manager.check_ops_to_skip())

        manager = CheckpointManager(ckpt_path, original_process_list=[
            {'test_op_1': {'test_key': 'test_value_1'}},
            {'test_op_2': {'test_key': 'test_value_2'}},
        ])
        with open(os.path.join(ckpt_path, 'ckpt_op.json'), 'w') as fout:
            json.dump([
                {'test_op_1': {'test_key': 'test_value_1'}},
                {'test_op_2': {'test_key': 'test_value_2'}},
            ], fout)
        self.assertFalse(manager.check_ops_to_skip())

    def test_different_ops(self):
        ckpt_path = os.path.join(self.temp_output_path, 'ckpt_2')
        os.makedirs(ckpt_path, exist_ok=True)
        os.makedirs(os.path.join(ckpt_path, 'latest'), exist_ok=True)
        with open(os.path.join(ckpt_path, 'ckpt_op.json'), 'w') as fout:
            json.dump([
                {'test_op_2': {'test_key': 'test_value_2'}},
            ], fout)
        manager = CheckpointManager(ckpt_path, original_process_list=[
            {'test_op_1': {'test_key': 'test_value_1'}},
            {'test_op_2': {'test_key': 'test_value_2'}},
        ])
        self.assertFalse(manager.ckpt_available)

    def test_save_and_load_ckpt(self):
        ckpt_path = os.path.join(self.temp_output_path, 'ckpt_3')
        test_data = {
            'text': ['text1', 'text2', 'text3'],
        }
        dataset = NestedDataset.from_dict(test_data)
        manager = CheckpointManager(ckpt_path, original_process_list=[])
        self.assertFalse(os.path.exists(os.path.join(manager.ckpt_ds_dir, 'dataset_info.json')))
        manager.record({'test_op_1': {'test_key': 'test_value_1'}})
        manager.save_ckpt(dataset)
        self.assertTrue(os.path.exists(os.path.join(manager.ckpt_ds_dir, 'dataset_info.json')))
        self.assertTrue(os.path.exists(manager.ckpt_op_record))
        loaded_ckpt = manager.load_ckpt()
        self.assertDatasetEqual(dataset, loaded_ckpt)


class MockOperation:
    """Mock operation for testing."""

    def __init__(self, name: str):
        self._name = name


class RayCheckpointManagerTest(DataJuicerTestCaseBase):
    """Tests for RayCheckpointManager."""

    def setUp(self):
        super().setUp()
        self.tmp_dir = tempfile.mkdtemp(prefix='test_checkpoint_manager_')

    def tearDown(self):
        super().tearDown()
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

    # ==================== should_checkpoint() tests ====================

    def test_should_checkpoint_every_op_strategy(self):
        """Test EVERY_OP strategy checkpoints after every operation."""
        mgr = RayCheckpointManager(
            ckpt_dir=self.tmp_dir,
            checkpoint_enabled=True,
            checkpoint_strategy=CheckpointStrategy.EVERY_OP,
        )

        # Should checkpoint after every operation
        self.assertTrue(mgr.should_checkpoint(0, "op_a"))
        self.assertTrue(mgr.should_checkpoint(1, "op_b"))
        self.assertTrue(mgr.should_checkpoint(5, "op_c"))
        self.assertTrue(mgr.should_checkpoint(100, "op_d"))

    def test_should_checkpoint_every_n_ops_strategy(self):
        """Test EVERY_N_OPS strategy checkpoints every N operations."""
        mgr = RayCheckpointManager(
            ckpt_dir=self.tmp_dir,
            checkpoint_enabled=True,
            checkpoint_strategy=CheckpointStrategy.EVERY_N_OPS,
            checkpoint_n_ops=3,
        )

        # Should checkpoint at ops 2, 5, 8 (indices where (idx+1) % 3 == 0)
        self.assertFalse(mgr.should_checkpoint(0, "op_a"))  # 1 % 3 != 0
        self.assertFalse(mgr.should_checkpoint(1, "op_b"))  # 2 % 3 != 0
        self.assertTrue(mgr.should_checkpoint(2, "op_c"))   # 3 % 3 == 0
        self.assertFalse(mgr.should_checkpoint(3, "op_d"))  # 4 % 3 != 0
        self.assertFalse(mgr.should_checkpoint(4, "op_e"))  # 5 % 3 != 0
        self.assertTrue(mgr.should_checkpoint(5, "op_f"))   # 6 % 3 == 0

    def test_should_checkpoint_every_n_ops_with_n_equals_1(self):
        """Test EVERY_N_OPS with n=1 behaves like EVERY_OP."""
        mgr = RayCheckpointManager(
            ckpt_dir=self.tmp_dir,
            checkpoint_enabled=True,
            checkpoint_strategy=CheckpointStrategy.EVERY_N_OPS,
            checkpoint_n_ops=1,
        )

        # With n=1, should checkpoint every operation
        self.assertTrue(mgr.should_checkpoint(0, "op_a"))
        self.assertTrue(mgr.should_checkpoint(1, "op_b"))
        self.assertTrue(mgr.should_checkpoint(2, "op_c"))

    def test_should_checkpoint_every_n_ops_with_large_n(self):
        """Test EVERY_N_OPS with n larger than typical operation counts."""
        mgr = RayCheckpointManager(
            ckpt_dir=self.tmp_dir,
            checkpoint_enabled=True,
            checkpoint_strategy=CheckpointStrategy.EVERY_N_OPS,
            checkpoint_n_ops=100,
        )

        # Should only checkpoint at op 99 (index where (idx+1) % 100 == 0)
        for i in range(99):
            self.assertFalse(mgr.should_checkpoint(i, f"op_{i}"))
        self.assertTrue(mgr.should_checkpoint(99, "op_99"))

    def test_should_checkpoint_manual_strategy(self):
        """Test MANUAL strategy checkpoints only specified operations."""
        mgr = RayCheckpointManager(
            ckpt_dir=self.tmp_dir,
            checkpoint_enabled=True,
            checkpoint_strategy=CheckpointStrategy.MANUAL,
            checkpoint_op_names=["text_length_filter", "clean_links_mapper"],
        )

        # Should only checkpoint specified operations
        self.assertTrue(mgr.should_checkpoint(0, "text_length_filter"))
        self.assertTrue(mgr.should_checkpoint(1, "clean_links_mapper"))
        self.assertFalse(mgr.should_checkpoint(2, "whitespace_normalization_mapper"))
        self.assertFalse(mgr.should_checkpoint(3, "other_op"))

    def test_should_checkpoint_manual_strategy_empty_list(self):
        """Test MANUAL strategy with empty op_names list."""
        mgr = RayCheckpointManager(
            ckpt_dir=self.tmp_dir,
            checkpoint_enabled=True,
            checkpoint_strategy=CheckpointStrategy.MANUAL,
            checkpoint_op_names=[],
        )

        # Should never checkpoint with empty list
        self.assertFalse(mgr.should_checkpoint(0, "op_a"))
        self.assertFalse(mgr.should_checkpoint(1, "op_b"))

    def test_should_checkpoint_disabled_strategy(self):
        """Test DISABLED strategy never checkpoints."""
        mgr = RayCheckpointManager(
            ckpt_dir=self.tmp_dir,
            checkpoint_enabled=True,  # Even if enabled
            checkpoint_strategy=CheckpointStrategy.DISABLED,
        )

        # Should never checkpoint
        self.assertFalse(mgr.should_checkpoint(0, "op_a"))
        self.assertFalse(mgr.should_checkpoint(1, "op_b"))
        self.assertFalse(mgr.should_checkpoint(100, "op_c"))

        # Also verify checkpoint_enabled is set to False
        self.assertFalse(mgr.checkpoint_enabled)

    def test_should_checkpoint_when_disabled(self):
        """Test that disabled checkpointing never checkpoints regardless of strategy."""
        mgr = RayCheckpointManager(
            ckpt_dir=self.tmp_dir,
            checkpoint_enabled=False,
            checkpoint_strategy=CheckpointStrategy.EVERY_OP,
        )

        # Should never checkpoint when disabled
        self.assertFalse(mgr.should_checkpoint(0, "op_a"))
        self.assertFalse(mgr.should_checkpoint(1, "op_b"))

    # ==================== group_operations_for_checkpointing() tests ====================

    def test_group_operations_every_op(self):
        """Test operation grouping with EVERY_OP strategy."""
        mgr = RayCheckpointManager(
            ckpt_dir=self.tmp_dir,
            checkpoint_enabled=True,
            checkpoint_strategy=CheckpointStrategy.EVERY_OP,
        )

        ops = [MockOperation(f"op_{i}") for i in range(5)]
        groups = mgr.group_operations_for_checkpointing(ops)

        # Each operation should be its own group
        self.assertEqual(len(groups), 5)
        for i, (start_idx, end_idx, group_ops) in enumerate(groups):
            self.assertEqual(start_idx, i)
            self.assertEqual(end_idx, i + 1)
            self.assertEqual(len(group_ops), 1)
            self.assertEqual(group_ops[0]._name, f"op_{i}")

    def test_group_operations_every_n_ops(self):
        """Test operation grouping with EVERY_N_OPS strategy (n=2)."""
        mgr = RayCheckpointManager(
            ckpt_dir=self.tmp_dir,
            checkpoint_enabled=True,
            checkpoint_strategy=CheckpointStrategy.EVERY_N_OPS,
            checkpoint_n_ops=2,
        )

        ops = [MockOperation(f"op_{i}") for i in range(5)]
        groups = mgr.group_operations_for_checkpointing(ops)

        # Groups: [0,1], [2,3], [4]
        self.assertEqual(len(groups), 3)

        # First group: ops 0-1
        self.assertEqual(groups[0][0], 0)  # start_idx
        self.assertEqual(groups[0][1], 2)  # end_idx
        self.assertEqual(len(groups[0][2]), 2)

        # Second group: ops 2-3
        self.assertEqual(groups[1][0], 2)
        self.assertEqual(groups[1][1], 4)
        self.assertEqual(len(groups[1][2]), 2)

        # Third group: op 4 (remaining)
        self.assertEqual(groups[2][0], 4)
        self.assertEqual(groups[2][1], 5)
        self.assertEqual(len(groups[2][2]), 1)

    def test_group_operations_every_n_ops_exact_multiple(self):
        """Test grouping when op count is exact multiple of n."""
        mgr = RayCheckpointManager(
            ckpt_dir=self.tmp_dir,
            checkpoint_enabled=True,
            checkpoint_strategy=CheckpointStrategy.EVERY_N_OPS,
            checkpoint_n_ops=3,
        )

        ops = [MockOperation(f"op_{i}") for i in range(6)]
        groups = mgr.group_operations_for_checkpointing(ops)

        # Groups: [0,1,2], [3,4,5]
        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0][1] - groups[0][0], 3)
        self.assertEqual(groups[1][1] - groups[1][0], 3)

    def test_group_operations_manual_strategy(self):
        """Test operation grouping with MANUAL strategy."""
        mgr = RayCheckpointManager(
            ckpt_dir=self.tmp_dir,
            checkpoint_enabled=True,
            checkpoint_strategy=CheckpointStrategy.MANUAL,
            checkpoint_op_names=["op_1", "op_3"],
        )

        ops = [MockOperation(f"op_{i}") for i in range(5)]
        groups = mgr.group_operations_for_checkpointing(ops)

        # Groups: [0,1] (checkpoint at op_1), [2,3] (checkpoint at op_3), [4]
        self.assertEqual(len(groups), 3)

        # First group: ops 0-1 (checkpoint at op_1)
        self.assertEqual(groups[0][0], 0)
        self.assertEqual(groups[0][1], 2)

        # Second group: ops 2-3 (checkpoint at op_3)
        self.assertEqual(groups[1][0], 2)
        self.assertEqual(groups[1][1], 4)

        # Third group: op 4 (remaining, no checkpoint)
        self.assertEqual(groups[2][0], 4)
        self.assertEqual(groups[2][1], 5)

    def test_group_operations_disabled(self):
        """Test operation grouping with DISABLED strategy."""
        mgr = RayCheckpointManager(
            ckpt_dir=self.tmp_dir,
            checkpoint_enabled=False,
            checkpoint_strategy=CheckpointStrategy.DISABLED,
        )

        ops = [MockOperation(f"op_{i}") for i in range(5)]
        groups = mgr.group_operations_for_checkpointing(ops)

        # All operations in one group (no checkpoints)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0][0], 0)
        self.assertEqual(groups[0][1], 5)
        self.assertEqual(len(groups[0][2]), 5)

    def test_group_operations_empty_list(self):
        """Test operation grouping with empty operations list."""
        mgr = RayCheckpointManager(
            ckpt_dir=self.tmp_dir,
            checkpoint_enabled=True,
            checkpoint_strategy=CheckpointStrategy.EVERY_OP,
        )

        groups = mgr.group_operations_for_checkpointing([])

        # Should return empty list
        self.assertEqual(len(groups), 0)

    def test_group_operations_single_op(self):
        """Test operation grouping with single operation."""
        mgr = RayCheckpointManager(
            ckpt_dir=self.tmp_dir,
            checkpoint_enabled=True,
            checkpoint_strategy=CheckpointStrategy.EVERY_OP,
        )

        ops = [MockOperation("single_op")]
        groups = mgr.group_operations_for_checkpointing(ops)

        # Single group with single operation
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0][0], 0)
        self.assertEqual(groups[0][1], 1)

    # ==================== resolve_checkpoint_filename() tests ====================

    def test_resolve_checkpoint_filename_format(self):
        """Test checkpoint filename format."""
        mgr = RayCheckpointManager(ckpt_dir=self.tmp_dir)

        filename = mgr.resolve_checkpoint_filename(0, 0)
        self.assertEqual(filename, "checkpoint_op_0000_partition_0000.parquet")

        filename = mgr.resolve_checkpoint_filename(5, 3)
        self.assertEqual(filename, "checkpoint_op_0005_partition_0003.parquet")

        filename = mgr.resolve_checkpoint_filename(99, 15)
        self.assertEqual(filename, "checkpoint_op_0099_partition_0015.parquet")

    def test_resolve_checkpoint_filename_large_indices(self):
        """Test checkpoint filename with large indices."""
        mgr = RayCheckpointManager(ckpt_dir=self.tmp_dir)

        filename = mgr.resolve_checkpoint_filename(9999, 9999)
        self.assertEqual(filename, "checkpoint_op_9999_partition_9999.parquet")

    # ==================== find_latest_checkpoint() tests ====================

    def test_find_latest_checkpoint_no_checkpoints(self):
        """Test finding checkpoint when none exist."""
        mgr = RayCheckpointManager(ckpt_dir=self.tmp_dir)

        result = mgr.find_latest_checkpoint(partition_id=0)
        self.assertIsNone(result)

    def test_find_latest_checkpoint_single_checkpoint(self):
        """Test finding latest checkpoint with single checkpoint."""
        mgr = RayCheckpointManager(ckpt_dir=self.tmp_dir)

        # Create a mock checkpoint file
        checkpoint_file = os.path.join(self.tmp_dir, "checkpoint_op_0005_partition_0000.parquet")
        os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
        with open(checkpoint_file, 'w') as f:
            f.write("mock")

        result = mgr.find_latest_checkpoint(partition_id=0)

        self.assertIsNotNone(result)
        op_idx, op_name, checkpoint_path = result
        self.assertEqual(op_idx, 5)
        self.assertTrue(checkpoint_path.endswith(".parquet"))

    def test_find_latest_checkpoint_multiple_checkpoints(self):
        """Test finding latest checkpoint with multiple checkpoints."""
        mgr = RayCheckpointManager(ckpt_dir=self.tmp_dir)

        # Create multiple mock checkpoint files
        for op_idx in [2, 5, 8, 3]:
            checkpoint_file = os.path.join(
                self.tmp_dir,
                f"checkpoint_op_{op_idx:04d}_partition_0000.parquet"
            )
            with open(checkpoint_file, 'w') as f:
                f.write("mock")

        result = mgr.find_latest_checkpoint(partition_id=0)

        self.assertIsNotNone(result)
        op_idx, _, _ = result
        self.assertEqual(op_idx, 8)  # Should return highest op_idx

    def test_find_latest_checkpoint_different_partitions(self):
        """Test finding checkpoint respects partition_id."""
        mgr = RayCheckpointManager(ckpt_dir=self.tmp_dir)

        # Create checkpoints for different partitions
        for partition_id, op_idx in [(0, 5), (1, 8), (2, 3)]:
            checkpoint_file = os.path.join(
                self.tmp_dir,
                f"checkpoint_op_{op_idx:04d}_partition_{partition_id:04d}.parquet"
            )
            with open(checkpoint_file, 'w') as f:
                f.write("mock")

        # Check each partition finds its own checkpoint
        result_0 = mgr.find_latest_checkpoint(partition_id=0)
        result_1 = mgr.find_latest_checkpoint(partition_id=1)
        result_2 = mgr.find_latest_checkpoint(partition_id=2)

        self.assertEqual(result_0[0], 5)
        self.assertEqual(result_1[0], 8)
        self.assertEqual(result_2[0], 3)

    def test_find_latest_checkpoint_nonexistent_directory(self):
        """Test finding checkpoint when directory doesn't exist."""
        nonexistent_dir = os.path.join(self.tmp_dir, "nonexistent")
        mgr = RayCheckpointManager(ckpt_dir=nonexistent_dir)

        # Remove the directory that was created in __init__
        if os.path.exists(nonexistent_dir):
            os.rmdir(nonexistent_dir)

        result = mgr.find_latest_checkpoint(partition_id=0)
        self.assertIsNone(result)

    # ==================== load_checkpoint() tests ====================

    def test_load_checkpoint_nonexistent_file(self):
        """Test loading checkpoint that doesn't exist returns None."""
        mgr = RayCheckpointManager(ckpt_dir=self.tmp_dir)

        result = mgr.load_checkpoint(op_idx=0, partition_id=0)
        self.assertIsNone(result)

    def test_load_checkpoint_corrupted_file(self):
        """Test loading corrupted checkpoint file returns None gracefully."""
        mgr = RayCheckpointManager(ckpt_dir=self.tmp_dir)

        # Create a corrupted checkpoint file
        checkpoint_file = os.path.join(
            self.tmp_dir,
            "checkpoint_op_0000_partition_0000.parquet"
        )
        with open(checkpoint_file, 'w') as f:
            f.write("this is not valid parquet data")

        result = mgr.load_checkpoint(op_idx=0, partition_id=0)
        self.assertIsNone(result)

    # ==================== Initialization tests ====================

    def test_init_creates_checkpoint_directory(self):
        """Test that initialization creates the checkpoint directory."""
        new_dir = os.path.join(self.tmp_dir, "new_ckpt_dir")
        self.assertFalse(os.path.exists(new_dir))

        mgr = RayCheckpointManager(ckpt_dir=new_dir)

        self.assertTrue(os.path.exists(new_dir))

    def test_init_with_event_logger_none(self):
        """Test initialization without event logger."""
        mgr = RayCheckpointManager(
            ckpt_dir=self.tmp_dir,
            event_logger=None,
        )

        self.assertIsNone(mgr.event_logger)
        # Should still work for all operations
        self.assertTrue(mgr.should_checkpoint(0, "op"))

    def test_init_disabled_strategy_disables_checkpointing(self):
        """Test that DISABLED strategy sets checkpoint_enabled to False."""
        mgr = RayCheckpointManager(
            ckpt_dir=self.tmp_dir,
            checkpoint_enabled=True,  # Explicitly enabled
            checkpoint_strategy=CheckpointStrategy.DISABLED,
        )

        self.assertFalse(mgr.checkpoint_enabled)


class CheckpointStrategyEnumTest(DataJuicerTestCaseBase):
    """Tests for CheckpointStrategy enum."""

    def test_strategy_values(self):
        """Test that all strategies have correct string values."""
        self.assertEqual(CheckpointStrategy.EVERY_OP.value, "every_op")
        self.assertEqual(CheckpointStrategy.EVERY_N_OPS.value, "every_n_ops")
        self.assertEqual(CheckpointStrategy.MANUAL.value, "manual")
        self.assertEqual(CheckpointStrategy.DISABLED.value, "disabled")

    def test_strategy_from_string(self):
        """Test creating strategy from string value."""
        self.assertEqual(CheckpointStrategy("every_op"), CheckpointStrategy.EVERY_OP)
        self.assertEqual(CheckpointStrategy("every_n_ops"), CheckpointStrategy.EVERY_N_OPS)
        self.assertEqual(CheckpointStrategy("manual"), CheckpointStrategy.MANUAL)
        self.assertEqual(CheckpointStrategy("disabled"), CheckpointStrategy.DISABLED)

    def test_invalid_strategy_raises_error(self):
        """Test that invalid strategy string raises ValueError."""
        with self.assertRaises(ValueError):
            CheckpointStrategy("invalid_strategy")


if __name__ == '__main__':
    unittest.main()

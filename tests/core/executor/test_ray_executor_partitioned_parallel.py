import threading
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from data_juicer.core.executor.ray_executor_partitioned import PartitionedRayExecutor


class FakeData:
    def __init__(self, partition_ids):
        self.partition_ids = partition_ids

    def union(self, other):
        return FakeData(self.partition_ids + other.partition_ids)


class FakeRayDataset:
    def __init__(self, data, cfg=None):
        self.data = data


def test_partitions_are_processed_in_parallel_and_merged_in_order():
    """Partition jobs should overlap without changing their union order."""
    executor = PartitionedRayExecutor.__new__(PartitionedRayExecutor)
    executor.cfg = {}
    executor.max_concurrent_partitions = 3
    executor._log_event = Mock()
    partitions = [FakeData([i]) for i in range(3)]
    partitioning_info = SimpleNamespace(num_partitions=3, total_rows=3)
    executor._split_dataset_deterministic = Mock(return_value=(partitions, partitioning_info))

    start_barrier = threading.Barrier(len(partitions), timeout=2)
    partition_finished = [threading.Event() for _ in partitions]
    completion_order = []

    def process_with_checkpointing(dataset, partition_id, ops):
        start_barrier.wait()
        if partition_id < len(partitions) - 1:
            assert partition_finished[partition_id + 1].wait(timeout=2)
        completion_order.append(partition_id)
        partition_finished[partition_id].set()
        return dataset

    executor._process_with_checkpointing = process_with_checkpointing

    with patch(
        "data_juicer.core.executor.ray_executor_partitioned.RayDataset",
        FakeRayDataset,
    ):
        result = executor._process_with_simple_partitioning(FakeRayDataset(None), [])

    assert completion_order == [2, 1, 0]
    assert result.data.partition_ids == [0, 1, 2]


def test_partition_concurrency_is_bounded():
    executor = PartitionedRayExecutor.__new__(PartitionedRayExecutor)
    executor.cfg = {}
    executor.max_concurrent_partitions = 2
    executor._log_event = Mock()
    partitions = [FakeData([i]) for i in range(4)]
    partitioning_info = SimpleNamespace(num_partitions=4, total_rows=4)
    executor._split_dataset_deterministic = Mock(return_value=(partitions, partitioning_info))

    active_partitions = 0
    max_active_partitions = 0
    active_lock = threading.Lock()
    pair_barrier = threading.Barrier(executor.max_concurrent_partitions, timeout=2)

    def process_with_checkpointing(dataset, partition_id, ops):
        nonlocal active_partitions, max_active_partitions
        with active_lock:
            active_partitions += 1
            max_active_partitions = max(max_active_partitions, active_partitions)
        pair_barrier.wait()
        with active_lock:
            active_partitions -= 1
        return dataset

    executor._process_with_checkpointing = process_with_checkpointing

    with patch(
        "data_juicer.core.executor.ray_executor_partitioned.RayDataset",
        FakeRayDataset,
    ):
        result = executor._process_with_simple_partitioning(FakeRayDataset(None), [])

    assert max_active_partitions == executor.max_concurrent_partitions
    assert result.data.partition_ids == [0, 1, 2, 3]


def test_partition_failure_is_logged_and_propagated():
    executor = PartitionedRayExecutor.__new__(PartitionedRayExecutor)
    executor.cfg = {}
    executor.max_concurrent_partitions = 1
    executor._log_event = Mock()
    executor.log_partition_failed = Mock()
    partitions = [FakeData([0])]
    partitioning_info = SimpleNamespace(num_partitions=1, total_rows=1)
    executor._split_dataset_deterministic = Mock(return_value=(partitions, partitioning_info))
    executor._process_with_checkpointing = Mock(side_effect=RuntimeError("partition failed"))

    with (
        patch(
            "data_juicer.core.executor.ray_executor_partitioned.RayDataset",
            FakeRayDataset,
        ),
        pytest.raises(RuntimeError, match="partition failed"),
    ):
        executor._process_with_simple_partitioning(FakeRayDataset(None), [])

    executor.log_partition_failed.assert_called_once_with(0, "partition failed", retry_count=0)

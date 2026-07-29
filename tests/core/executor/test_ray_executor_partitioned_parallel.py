import threading
from types import SimpleNamespace
from unittest.mock import Mock, patch

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
    executor._log_event = Mock()
    partitions = [FakeData([i]) for i in range(3)]
    partitioning_info = SimpleNamespace(num_partitions=3, total_rows=3)
    executor._split_dataset_deterministic = Mock(return_value=(partitions, partitioning_info))

    start_barrier = threading.Barrier(len(partitions), timeout=2)

    def process_with_checkpointing(dataset, partition_id, ops):
        start_barrier.wait()
        return dataset

    executor._process_with_checkpointing = process_with_checkpointing

    with patch(
        "data_juicer.core.executor.ray_executor_partitioned.RayDataset",
        FakeRayDataset,
    ):
        result = executor._process_with_simple_partitioning(FakeRayDataset(None), [])

    assert result.data.partition_ids == [0, 1, 2]

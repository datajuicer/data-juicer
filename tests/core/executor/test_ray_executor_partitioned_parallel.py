from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from data_juicer.core.executor.event_logging_mixin import EventType
from data_juicer.core.executor.ray_executor_partitioned import PartitionedRayExecutor
from data_juicer.utils.ckpt_utils import CheckpointStrategy, RayCheckpointManager


class FakeData:
    def __init__(self, partition_ids):
        self.partition_ids = partition_ids

    def union(self, other):
        return FakeData(self.partition_ids + other.partition_ids)


class FakeRayDataset:
    def __init__(self, data, cfg=None):
        self.data = data


class FakeOp:
    def __init__(self, name, num_proc, auto=True):
        self._name = name
        self.num_proc = num_proc
        self._auto = auto

    def use_auto_proc(self):
        return self._auto


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
    event_types = [call.kwargs["event_type"] for call in executor._log_event.call_args_list]
    assert event_types.count(EventType.PARTITION_START) == len(partitions)
    assert event_types.count(EventType.PARTITION_COMPLETE) == len(partitions)


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
    event_types = [call.kwargs["event_type"] for call in executor._log_event.call_args_list]
    assert event_types == [EventType.PARTITION_START]


def test_auto_operator_parallelism_is_calculated_once():
    executor = PartitionedRayExecutor.__new__(PartitionedRayExecutor)
    executor.cfg = {"auto_op_parallelism": True}
    auto_op = FakeOp("auto", -1)
    explicit_op = FakeOp("explicit", 3, auto=False)

    def configure(ops):
        ops[0].num_proc = (2, 8)
        return ops

    with patch("data_juicer.utils.process_utils.calculate_ray_np", side_effect=configure) as calculate:
        executor._configure_auto_operator_parallelism([auto_op, explicit_op])

    calculate.assert_called_once_with([auto_op, explicit_op])
    assert executor._auto_parallel_op_ids == {id(auto_op)}
    assert auto_op.num_proc == (2, 8)
    assert explicit_op.num_proc == 3


@pytest.mark.parametrize(
    ("concurrency", "max_workers", "expected"),
    [
        (8, 4, 2),
        ((2, 8), 4, (1, 2)),
        ([1, 4], 2, (1, 2)),
        (None, 4, None),
        (4, 1, 4),
    ],
)
def test_auto_operator_concurrency_is_scaled_per_partition(concurrency, max_workers, expected):
    assert PartitionedRayExecutor._scale_concurrency_for_partitions(concurrency, max_workers) == expected


def test_auto_operator_parallelism_is_restored_after_partition_processing():
    executor = PartitionedRayExecutor.__new__(PartitionedRayExecutor)
    executor.cfg = {}
    executor.max_concurrent_partitions = 4
    executor._log_event = Mock()
    partitions = [FakeData([i]) for i in range(4)]
    partitioning_info = SimpleNamespace(num_partitions=4, total_rows=4)
    executor._split_dataset_deterministic = Mock(return_value=(partitions, partitioning_info))
    auto_op = FakeOp("auto", 8)
    explicit_op = FakeOp("explicit", 3, auto=False)
    executor._auto_parallel_op_ids = {id(auto_op)}
    observed_concurrency = []
    start_barrier = threading.Barrier(len(partitions), timeout=2)

    def process_with_checkpointing(dataset, partition_id, ops):
        assert dataset._auto_proc is False
        observed_concurrency.append((ops[0].num_proc, ops[1].num_proc))
        start_barrier.wait()
        return dataset

    executor._process_with_checkpointing = process_with_checkpointing

    with patch(
        "data_juicer.core.executor.ray_executor_partitioned.RayDataset",
        FakeRayDataset,
    ):
        result = executor._process_with_simple_partitioning(
            FakeRayDataset(None),
            [auto_op, explicit_op],
        )

    assert observed_concurrency == [(2, 3)] * len(partitions)
    assert auto_op.num_proc == 8
    assert explicit_op.num_proc == 3
    assert result.data.partition_ids == [0, 1, 2, 3]


def test_auto_operator_parallelism_is_restored_after_partition_failure():
    executor = PartitionedRayExecutor.__new__(PartitionedRayExecutor)
    executor.cfg = {}
    executor.max_concurrent_partitions = 2
    executor._log_event = Mock()
    executor.log_partition_failed = Mock()
    partitions = [FakeData([0]), FakeData([1])]
    partitioning_info = SimpleNamespace(num_partitions=2, total_rows=2)
    executor._split_dataset_deterministic = Mock(return_value=(partitions, partitioning_info))
    auto_op = FakeOp("auto", 4)
    executor._auto_parallel_op_ids = {id(auto_op)}
    executor._process_with_checkpointing = Mock(side_effect=RuntimeError("partition failed"))

    with (
        patch(
            "data_juicer.core.executor.ray_executor_partitioned.RayDataset",
            FakeRayDataset,
        ),
        pytest.raises(RuntimeError, match="partition failed"),
    ):
        executor._process_with_simple_partitioning(FakeRayDataset(None), [auto_op])

    assert auto_op.num_proc == 4


def test_checkpoint_paths_remain_partition_scoped_under_concurrency(tmp_path):
    manager = RayCheckpointManager(
        ckpt_dir=str(tmp_path),
        checkpoint_strategy=CheckpointStrategy.EVERY_OP,
    )
    write_barrier = threading.Barrier(4, timeout=2)

    class ConcurrentCheckpointData:
        def __init__(self, partition_id):
            self.partition_id = partition_id

        def write_parquet(self, checkpoint_path):
            write_barrier.wait()
            path = Path(checkpoint_path)
            path.mkdir(parents=True, exist_ok=True)
            (path / "partition.txt").write_text(str(self.partition_id))

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(
                manager.save_checkpoint,
                ConcurrentCheckpointData(partition_id),
                0,
                "mapper",
                partition_id,
            )
            for partition_id in range(4)
        ]
        checkpoint_paths = [future.result() for future in futures]

    assert len(set(checkpoint_paths)) == 4
    for partition_id, checkpoint_path in enumerate(checkpoint_paths):
        assert Path(checkpoint_path, "partition.txt").read_text() == str(partition_id)

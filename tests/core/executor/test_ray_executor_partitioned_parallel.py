import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from data_juicer.core.executor.dag_execution_mixin import DAGExecutionMixin
from data_juicer.core.executor.event_logging_mixin import EventType
from data_juicer.core.executor.pipeline_dag import DAGNodeStatus, PipelineDAG
from data_juicer.core.executor.ray_executor_partitioned import (
    _LOGICAL_PARTITION_COLUMN,
    PartitionedRayExecutor,
)
from data_juicer.utils.ckpt_utils import CheckpointStrategy, RayCheckpointManager
from data_juicer.utils.ray_cluster_utils import ClusterTopology


def _topology_from_resources(resources):
    """Build the cluster view the executor now resolves resources through."""
    return ClusterTopology(
        num_nodes=1,
        total_cpus=float(resources.get("CPU", 0)),
        total_gpus=float(resources.get("GPU", 0)),
        available_cpus=float(resources.get("CPU", 0)),
        available_gpus=float(resources.get("GPU", 0)),
    )


class FakeData:
    def __init__(self, partition_ids):
        self.partition_ids = partition_ids

    def union(self, other):
        return FakeData(self.partition_ids + other.partition_ids)


class FakeRayDataset:
    def __init__(self, data, cfg=None):
        self.data = data


class FakeOp:
    def __init__(
        self,
        name,
        num_proc,
        auto=True,
        actor=True,
        num_cpus=None,
        num_gpus=None,
        cuda=False,
    ):
        self._name = name
        self.num_proc = num_proc
        self._auto = auto
        self._actor = actor
        self.num_cpus = num_cpus
        self.num_gpus = num_gpus
        self._cuda = cuda
        self.batch_size = 1

    def use_auto_proc(self):
        return self._auto

    def use_ray_actor(self):
        return self._actor

    def use_cuda(self):
        return self._cuda


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


def test_concurrent_partitions_use_isolated_operator_instances():
    executor = PartitionedRayExecutor.__new__(PartitionedRayExecutor)
    executor.cfg = {}
    executor.max_concurrent_partitions = 3
    executor._log_event = Mock()
    partitions = [FakeData([i]) for i in range(3)]
    partitioning_info = SimpleNamespace(num_partitions=3, total_rows=3)
    executor._split_dataset_deterministic = Mock(return_value=(partitions, partitioning_info))
    original_op = FakeOp("mutable", 1)
    original_op.partition_marker = None

    start_barrier = threading.Barrier(len(partitions), timeout=2)
    observations = {}
    observations_lock = threading.Lock()

    def process_with_checkpointing(dataset, partition_id, ops):
        partition_op = ops[0]
        partition_op.partition_marker = partition_id
        start_barrier.wait()
        with observations_lock:
            observations[partition_id] = (id(partition_op), partition_op.partition_marker)
        return dataset

    executor._process_with_checkpointing = process_with_checkpointing

    with patch(
        "data_juicer.core.executor.ray_executor_partitioned.RayDataset",
        FakeRayDataset,
    ):
        executor._process_with_simple_partitioning(FakeRayDataset(None), [original_op])

    operator_ids = {operator_id for operator_id, _ in observations.values()}
    assert len(operator_ids) == len(partitions)
    assert id(original_op) not in operator_ids
    assert {partition_id: marker for partition_id, (_, marker) in observations.items()} == {
        partition_id: partition_id for partition_id in range(len(partitions))
    }
    assert original_op.partition_marker is None


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
        executor._configure_operator_parallelism([auto_op, explicit_op])

    calculate.assert_called_once_with([auto_op, explicit_op])
    assert executor._auto_parallel_op_ids == {id(auto_op)}
    assert executor._explicit_actor_op_ids == {id(explicit_op)}
    assert auto_op.num_proc == (2, 8)
    assert explicit_op.num_proc == 3


@pytest.mark.parametrize(
    ("resources", "ops", "expected"),
    [
        (
            {"CPU": 32, "GPU": 4},
            [FakeOp("gpu", 4, num_cpus=1, num_gpus=1, cuda=True)],
            4,
        ),
        (
            {"CPU": 32, "GPU": 4},
            [FakeOp("two-gpu", 2, num_cpus=1, num_gpus=2, cuda=True)],
            2,
        ),
        (
            {"CPU": 16},
            [FakeOp("cpu", None, actor=False, num_cpus=1)],
            4,
        ),
        (
            {"CPU": 16},
            [FakeOp("cpu-heavy", None, actor=False, num_cpus=8)],
            2,
        ),
        (
            {"CPU": 32, "GPU": 4},
            [
                FakeOp("gpu", 4, num_cpus=1, num_gpus=1, cuda=True),
                FakeOp("cpu-heavy", None, actor=False, num_cpus=16),
            ],
            2,
        ),
    ],
)
def test_auto_partition_concurrency_is_resource_aware(resources, ops, expected):
    executor = PartitionedRayExecutor.__new__(PartitionedRayExecutor)
    executor.max_concurrent_partitions = "auto"

    with patch(
        "data_juicer.utils.ray_cluster_utils.detect_cluster_topology",
        return_value=_topology_from_resources(resources),
    ):
        resolved = executor._resolve_max_concurrent_partitions(ops)

    assert resolved == expected
    assert executor.max_concurrent_partitions == expected


def test_explicit_partition_concurrency_overrides_auto_detection():
    executor = PartitionedRayExecutor.__new__(PartitionedRayExecutor)
    executor.max_concurrent_partitions = 3

    with patch("data_juicer.utils.ray_cluster_utils.detect_cluster_topology") as detect:
        resolved = executor._resolve_max_concurrent_partitions([])

    assert resolved == 3
    detect.assert_not_called()


def test_gpu_pipeline_uses_joint_safe_worker_limit():
    executor = PartitionedRayExecutor.__new__(PartitionedRayExecutor)
    executor.cfg = SimpleNamespace()
    executor.max_concurrent_partitions = "auto"
    executor.max_gpu_workers_per_device = 4
    ops = [
        FakeOp("sentiment", (10, 30), num_cpus=1, num_gpus=0.25, cuda=True),
        FakeOp("topic", (10, 30), num_cpus=1, num_gpus=0.25, cuda=True),
        FakeOp("clip", (10, 30), num_cpus=1, num_gpus=0.25, cuda=True),
    ]
    for op, memory_fraction in zip(ops, (0.01, 0.01, 0.01)):
        op.batch_size = 4
        op._gpu_memory_fraction = memory_fraction

    with patch(
        "data_juicer.utils.ray_cluster_utils.detect_cluster_topology",
        return_value=_topology_from_resources({"CPU": 32, "GPU": 2}),
    ):
        resolved = executor._resolve_max_concurrent_partitions(ops, total_samples=24)

    assert resolved == 2
    plan = executor.cfg._resolved_gpu_worker_plan
    assert plan["cpu_capacity"] == 8
    assert plan["gpu_capacity"] == 2
    assert plan["memory_capacity"] == 60
    assert plan["data_capacity"] == 6

    executor._auto_parallel_op_ids = {id(op) for op in ops}
    executor._cap_auto_gpu_operator_parallelism(ops, resolved, total_samples=24)
    assert [op.num_proc for op in ops] == [(2, 2), (2, 2), (2, 2)]


def test_throughput_aware_actor_plan_fills_gpus_without_partition_multiplier():
    executor = PartitionedRayExecutor.__new__(PartitionedRayExecutor)
    executor.cfg = SimpleNamespace()
    executor.max_gpu_workers_per_device = 4
    ops = [
        FakeOp("slow", 20, num_cpus=1, num_gpus=0.25, cuda=True),
        FakeOp("fast", 20, num_cpus=1, num_gpus=0.25, cuda=True),
        FakeOp("medium", 20, num_cpus=1, num_gpus=0.25, cuda=True),
    ]
    for op, throughput in zip(ops, (10, 40, 20)):
        op.batch_size = 10
        op._gpu_rows_per_second = throughput
        op._gpu_output_ratio = 1
        op._gpu_init_seconds = 1
        op._gpu_memory_fraction = 0.2
    executor._auto_parallel_op_ids = {id(op) for op in ops}
    executor._explicit_actor_op_ids = set()

    with patch(
        "data_juicer.utils.ray_cluster_utils.detect_cluster_topology",
        return_value=_topology_from_resources({"CPU": 16, "GPU": 2}),
    ):
        plan = executor._configure_throughput_aware_gpu_parallelism(ops, total_samples=1000)

    assert [op.num_proc for op in ops] == [5, 1, 2]
    assert sum(item["actors"] * item["num_gpus"] for item in plan["operators"]) == 2
    assert executor._throughput_planned_op_ids == {id(op) for op in ops}


def test_throughput_actor_plan_fails_before_oversubscribing_one_gpu():
    executor = PartitionedRayExecutor.__new__(PartitionedRayExecutor)
    executor.cfg = SimpleNamespace()
    executor.max_gpu_workers_per_device = 4
    ops = [
        FakeOp("first", 1, num_cpus=1, num_gpus=0.6, cuda=True),
        FakeOp("second", 1, num_cpus=1, num_gpus=0.6, cuda=True),
    ]
    for op in ops:
        op._gpu_rows_per_second = 10
        op._gpu_output_ratio = 1
        op._gpu_init_seconds = 1
        op._gpu_memory_fraction = 0.6
    executor._auto_parallel_op_ids = {id(op) for op in ops}
    executor._explicit_actor_op_ids = set()

    with (
        patch(
            "data_juicer.utils.ray_cluster_utils.detect_cluster_topology",
            return_value=_topology_from_resources({"CPU": 8, "GPU": 1}),
        ),
        pytest.raises(RuntimeError, match="minimum throughput-aware GPU actor plan"),
    ):
        executor._configure_throughput_aware_gpu_parallelism(ops, total_samples=100)


def test_execution_group_runs_once_and_keeps_partition_scoped_checkpoints():
    class Schema:
        names = [_LOGICAL_PARTITION_COLUMN, "value"]

    class GroupedData:
        def __init__(self, rows):
            self.rows = rows

        def count(self):
            return len(self.rows)

        def materialize(self):
            return self

        def schema(self):
            return Schema()

        def filter(self, function, fn_kwargs):
            return GroupedData([row for row in self.rows if function(row, **fn_kwargs)])

        def drop_columns(self, columns):
            return GroupedData([{key: value for key, value in row.items() if key not in columns} for row in self.rows])

    class GroupedRayDataset:
        process_calls = 0

        def __init__(self, data):
            self.data = data

        def process(self, ops):
            type(self).process_calls += 1
            return GroupedRayDataset(
                GroupedData([{**row, "value": row["value"] + 1} for row in self.data.rows])
            )

    saved = {}

    def save_checkpoint(dataset, op_idx, op_name, partition_id, cfg):
        saved[partition_id] = dataset.data.rows

    manager = SimpleNamespace(
        checkpoint_enabled=True,
        group_operations_for_checkpointing=lambda ops: [(0, 1, ops)],
        should_checkpoint=lambda op_idx, op_name: True,
        save_checkpoint=save_checkpoint,
    )
    executor = PartitionedRayExecutor.__new__(PartitionedRayExecutor)
    executor.cfg = {}
    executor.pipeline_dag = None
    executor.ckpt_manager = manager
    executor._wrap_with_precomputed_parallelism = lambda data: GroupedRayDataset(data)
    data = GroupedData(
        [
            {_LOGICAL_PARTITION_COLUMN: 0, "value": 10},
            {_LOGICAL_PARTITION_COLUMN: 1, "value": 20},
        ]
    )

    result = executor._process_execution_group_with_checkpointing(
        GroupedRayDataset(data),
        [0, 1],
        [SimpleNamespace(_name="gpu_mapper")],
    )

    assert GroupedRayDataset.process_calls == 1
    assert saved == {0: [{"value": 11}], 1: [{"value": 21}]}
    assert result.data.rows == [{"value": 11}, {"value": 21}]


def test_auto_partition_concurrency_falls_back_to_one_when_ray_inspection_fails():
    executor = PartitionedRayExecutor.__new__(PartitionedRayExecutor)
    executor.max_concurrent_partitions = "auto"
    fallback = ClusterTopology(
        num_nodes=1,
        total_cpus=0.0,
        total_gpus=0.0,
        available_cpus=0.0,
        available_gpus=0.0,
    )

    with patch(
        "data_juicer.utils.ray_cluster_utils.detect_cluster_topology",
        return_value=fallback,
    ):
        resolved = executor._resolve_max_concurrent_partitions([])

    assert resolved == 1
    assert executor.max_concurrent_partitions == 1


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


@pytest.mark.parametrize(
    ("concurrency", "max_workers", "expected"),
    [
        (8, 4, 2),
        (5, 4, 1),
        ((2, 8), 4, (1, 2)),
        ([1, 5], 4, (1, 1)),
        ((2, 8, 6), 4, (1, 2, 1)),
    ],
)
def test_explicit_actor_concurrency_is_safely_scaled_per_partition(concurrency, max_workers, expected):
    assert (
        PartitionedRayExecutor._scale_concurrency_for_partitions(
            concurrency,
            max_workers,
            round_up=False,
        )
        == expected
    )


def test_operator_parallelism_is_restored_after_partition_processing():
    executor = PartitionedRayExecutor.__new__(PartitionedRayExecutor)
    executor.cfg = {}
    executor.max_concurrent_partitions = 4
    executor._log_event = Mock()
    partitions = [FakeData([i]) for i in range(4)]
    partitioning_info = SimpleNamespace(num_partitions=4, total_rows=4)
    executor._split_dataset_deterministic = Mock(return_value=(partitions, partitioning_info))
    auto_op = FakeOp("auto", 8)
    explicit_op = FakeOp("explicit", 4, auto=False)
    executor._auto_parallel_op_ids = {id(auto_op)}
    executor._explicit_actor_op_ids = {id(explicit_op)}
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

    assert observed_concurrency == [(2, 1)] * len(partitions)
    assert auto_op.num_proc == 8
    assert explicit_op.num_proc == 4
    assert result.data.partition_ids == [0, 1, 2, 3]


def test_operator_parallelism_is_restored_after_partition_failure():
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


def test_partition_template_stays_scaled_after_sibling_failure():
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

    start_barrier = threading.Barrier(2, timeout=2)
    failure_announced = threading.Event()
    sibling_observation = []

    def process_with_checkpointing(dataset, partition_id, ops):
        start_barrier.wait()
        if partition_id == 0:
            failure_announced.set()
            raise RuntimeError("partition failed")

        assert failure_announced.wait(timeout=2)
        sibling_observation.append((ops[0].num_proc, auto_op.num_proc))
        return dataset

    executor._process_with_checkpointing = process_with_checkpointing

    with (
        patch(
            "data_juicer.core.executor.ray_executor_partitioned.RayDataset",
            FakeRayDataset,
        ),
        pytest.raises(RuntimeError, match="partition failed"),
    ):
        executor._process_with_simple_partitioning(FakeRayDataset(None), [auto_op])

    # The running partition keeps its scaled private plan while the shared
    # operator has already been restored for later convergence stages.
    assert sibling_observation == [(2, 4)]
    assert auto_op.num_proc == 4


def test_explicit_actor_budget_limits_partition_concurrency():
    executor = PartitionedRayExecutor.__new__(PartitionedRayExecutor)
    executor.cfg = {}
    executor.max_concurrent_partitions = 4
    executor._log_event = Mock()
    partitions = [FakeData([i]) for i in range(4)]
    partitioning_info = SimpleNamespace(num_partitions=4, total_rows=4)
    executor._split_dataset_deterministic = Mock(return_value=(partitions, partitioning_info))
    explicit_op = FakeOp("explicit", 2, auto=False)
    executor._auto_parallel_op_ids = set()
    executor._explicit_actor_op_ids = {id(explicit_op)}

    active_partitions = 0
    max_active_partitions = 0
    active_lock = threading.Lock()
    pair_barrier = threading.Barrier(2, timeout=2)
    observed_concurrency = []

    def process_with_checkpointing(dataset, partition_id, ops):
        nonlocal active_partitions, max_active_partitions
        with active_lock:
            active_partitions += 1
            max_active_partitions = max(max_active_partitions, active_partitions)
            observed_concurrency.append(ops[0].num_proc)
        pair_barrier.wait()
        with active_lock:
            active_partitions -= 1
        return dataset

    executor._process_with_checkpointing = process_with_checkpointing

    with patch(
        "data_juicer.core.executor.ray_executor_partitioned.RayDataset",
        FakeRayDataset,
    ):
        result = executor._process_with_simple_partitioning(
            FakeRayDataset(None),
            [explicit_op],
        )

    assert max_active_partitions == 2
    assert observed_concurrency == [1] * len(partitions)
    assert explicit_op.num_proc == 2
    assert result.data.partition_ids == [0, 1, 2, 3]


def test_explicit_task_concurrency_is_not_partition_scaled():
    executor = PartitionedRayExecutor.__new__(PartitionedRayExecutor)
    executor.cfg = {"auto_op_parallelism": False}
    explicit_actor = FakeOp("actor", 4, auto=False)
    explicit_task = FakeOp("task", 4, auto=False, actor=False)

    executor._configure_operator_parallelism([explicit_actor, explicit_task])

    assert executor._auto_parallel_op_ids == set()
    assert executor._explicit_actor_op_ids == {id(explicit_actor)}
    assert executor._limit_partition_workers_for_explicit_actors([explicit_task], 8) == 8
    assert executor._scale_operator_parallelism([explicit_task], 8) == []
    assert explicit_task.num_proc == 4


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


def test_concurrent_dag_monitoring_tracks_current_node_per_partition(tmp_path):
    executor = PartitionedRayExecutor.__new__(PartitionedRayExecutor)
    DAGExecutionMixin.__init__(executor)
    executor.pipeline_dag = PipelineDAG(str(tmp_path))
    executor.log_dag_node_start = Mock()
    executor.log_dag_node_complete = Mock()

    num_partitions = 4
    node_ids = [f"op_001_fake_partition_{partition_id}" for partition_id in range(num_partitions)]
    executor.pipeline_dag.nodes = {
        node_id: {
            "node_id": node_id,
            "operation_name": "fake",
            "node_type": "partition_operation",
            "partition_id": partition_id,
            "execution_order": 1,
            "dependencies": [],
            "status": DAGNodeStatus.PENDING.value,
            "start_time": None,
            "end_time": None,
            "actual_duration": None,
            "error_message": None,
        }
        for partition_id, node_id in enumerate(node_ids)
    }

    started_barrier = threading.Barrier(num_partitions, timeout=2)
    observed_barrier = threading.Barrier(num_partitions, timeout=2)
    observed_current_nodes = {}
    active_snapshot = {}

    def monitor_partition(partition_id):
        node_id = node_ids[partition_id]
        executor._mark_dag_node_started(node_id)
        started_barrier.wait()
        observed_current_nodes[partition_id] = executor.current_dag_node
        if partition_id == 0:
            with executor._dag_state_lock:
                active_snapshot.update(executor.current_dag_nodes)
        observed_barrier.wait()
        executor._mark_dag_node_completed(node_id, duration=partition_id + 0.5)

    with ThreadPoolExecutor(max_workers=num_partitions) as pool:
        futures = [pool.submit(monitor_partition, partition_id) for partition_id in range(num_partitions)]
        for future in futures:
            future.result()

    assert observed_current_nodes == {partition_id: node_id for partition_id, node_id in enumerate(node_ids)}
    assert active_snapshot == {partition_id: node_id for partition_id, node_id in enumerate(node_ids)}
    assert executor.current_dag_nodes == {}
    for partition_id, node_id in enumerate(node_ids):
        node = executor.pipeline_dag.nodes[node_id]
        assert node["status"] == DAGNodeStatus.COMPLETED.value
        assert node["start_time"] is not None
        assert node["actual_duration"] == partition_id + 0.5

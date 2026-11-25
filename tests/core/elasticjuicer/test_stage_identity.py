from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

from data_juicer.core.elasticjuicer.stage_identity import (
    assign_stage_identities,
    stamped_stage_identity,
)
from data_juicer.core.executor.ray_executor_partitioned import PartitionedRayExecutor


def make_op(args=(), **kwargs):
    return SimpleNamespace(_name="same_mapper", _init_args=args, _init_kwargs=kwargs)


def test_repeated_ops_are_distinct_and_restarts_are_stable():
    first = [make_op(model="a"), make_op(model="a"), make_op(model="b")]
    second = [make_op(model="a"), make_op(model="a"), make_op(model="b")]
    assert assign_stage_identities(first) == assign_stage_identities(second)
    assert len({stamped_stage_identity(op) for op in first}) == 3


def test_constructor_arguments_and_recipe_order_affect_identity():
    assert assign_stage_identities([make_op(("a",))]) != assign_stage_identities([make_op(("b",))])
    assert assign_stage_identities([make_op(model="a"), make_op(model="b")]) != assign_stage_identities(
        [make_op(model="b"), make_op(model="a")]
    )


def test_per_run_work_directory_does_not_change_identity():
    assert assign_stage_identities([make_op(work_dir="/run/a")]) == assign_stage_identities(
        [make_op(work_dir="/run/b")]
    )


def test_resource_injection_and_partition_cloning_keep_stamped_identity():
    ops = [make_op(model="a"), make_op(model="a")]
    assign_stage_identities(ops)
    identity = stamped_stage_identity(ops[1])
    ops[1]._init_kwargs.update(memory=1.5, num_gpus=0.2)
    resumed_suffix = deepcopy(ops[1:])
    assert stamped_stage_identity(resumed_suffix[0]) == identity


def test_partitioned_executor_stamps_the_full_prepared_recipe():
    executor = PartitionedRayExecutor.__new__(PartitionedRayExecutor)
    executor.cfg = SimpleNamespace(process=[], op_fusion=False)
    ops = [make_op(), make_op()]
    with patch("data_juicer.core.executor.ray_executor_partitioned.load_ops", return_value=ops):
        assert executor._prepare_operators() is ops
    assert len(executor.cfg._resolved_stage_identities["stages"]) == 2
    assert stamped_stage_identity(ops[0]) != stamped_stage_identity(ops[1])


def test_plan_keeps_identities_and_initialization_for_explicit_and_auto_gpu_stages():
    ops = [make_op(), make_op()]
    for op, actors, init_seconds in zip(ops, (1, None), (30, 0.01)):
        op.accelerator = "cuda"
        op.num_proc = actors
        op.num_gpus = op._gpu_memory_fraction = 0.2
        op.num_cpus = op.batch_size = 1
        op._gpu_rows_per_second = 100
        op._gpu_output_ratio = 1
        op._gpu_init_seconds = init_seconds
        op.use_cuda = lambda: True
        op.use_ray_actor = lambda: True
    assign_stage_identities(ops)
    executor = PartitionedRayExecutor.__new__(PartitionedRayExecutor)
    executor.cfg = SimpleNamespace()
    executor._auto_parallel_op_ids = {id(ops[1])}
    executor.max_gpu_workers_per_device = 5
    executor.execution_group_size = "auto"
    executor.max_initialization_overhead_ratio = 0.1
    topology = SimpleNamespace(total_cpus=3, total_gpus=1, gpu_devices_per_node=(1,))
    with patch("data_juicer.utils.ray_cluster_utils.detect_cluster_topology", return_value=topology):
        plan = executor._configure_throughput_aware_gpu_parallelism(ops, total_samples=800)
    assert [stage["stage_id"] for stage in plan["operators"]] == [stamped_stage_identity(op) for op in ops]
    assert len({stage["stage_id"] for stage in plan["operators"]}) == 2
    assert ops[0].num_proc == 1
    assert executor._resolve_execution_group_size(SimpleNamespace(num_partitions=8, total_rows=800), ops) == 8

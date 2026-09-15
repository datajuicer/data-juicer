"""Resource and sampling regressions in the partitioned GPU planner."""

from types import SimpleNamespace
from unittest.mock import patch

from data_juicer.core.executor import gpu_memory_probe as probe
from data_juicer.core.executor.ray_executor_partitioned import PartitionedRayExecutor
from data_juicer.ops import Filter, Mapper
from data_juicer.utils.constant import Fields


def make_op(name, *, cuda=True, actors=None, gpus=0.2, cpus=1):
    return SimpleNamespace(
        _name=name,
        accelerator="cuda" if cuda else "cpu",
        num_proc=actors,
        num_gpus=gpus if cuda else 0,
        num_cpus=cpus,
        batch_size=1,
        _gpu_rows_per_second=10,
        _gpu_memory_fraction=gpus if cuda else 0,
        _gpu_output_ratio=1,
        _gpu_init_seconds=1,
        use_cuda=lambda: cuda,
        use_ray_actor=lambda: True,
    )


def plan(auto, fixed, *, cpus, gpus):
    executor = PartitionedRayExecutor.__new__(PartitionedRayExecutor)
    executor.cfg = SimpleNamespace()
    executor._auto_parallel_op_ids = {id(op) for op in auto}
    executor._explicit_actor_op_ids = {id(op) for op in fixed}
    executor.max_gpu_workers_per_device = 5
    topology = SimpleNamespace(total_cpus=cpus, total_gpus=gpus)
    with patch("data_juicer.utils.ray_cluster_utils.detect_cluster_topology", return_value=topology):
        return executor._configure_throughput_aware_gpu_parallelism(fixed + auto, total_samples=1000)


def test_planner_reserves_explicit_cpu_actors():
    cpu = make_op("fixed_cpu", cuda=False, actors=6)
    gpu = make_op("auto_gpu")
    result = plan([gpu], [cpu], cpus=8, gpus=1)
    actual_cpu = cpu.num_proc * cpu.num_cpus + gpu.num_proc * gpu.num_cpus
    assert actual_cpu <= 8, {"actual_cpu": actual_cpu, "reported_cpu": result["cpu_used"]}


def test_explicit_two_gpu_actor_fits_four_gpu_cluster():
    fixed = make_op("explicit_two_gpu", actors=1, gpus=2)
    auto = make_op("auto_single_gpu")
    result = plan([auto], [fixed], cpus=32, gpus=4)
    assert result is not None


def test_multi_gpu_actor_cannot_span_nodes():
    specs = [{"num_gpus": 2, "memory_fraction": 2, "max_per_device": 5}]
    assert not PartitionedRayExecutor._gpu_actor_requests_fit(specs, [1], 2, (1, 1))
    assert PartitionedRayExecutor._gpu_actor_requests_fit(specs, [1], 2, (2, 0))


def test_other_automatic_cpu_actor_pools_also_reserve_capacity():
    cpu = make_op("automatic_cpu", cuda=False, actors=6)
    gpu = make_op("automatic_gpu")
    plan([cpu, gpu], [], cpus=8, gpus=1)
    assert cpu.num_proc == 6
    assert cpu.num_proc + gpu.num_proc <= 8


class SingleRowMapper(Mapper):
    _name = "review_single_row"
    _input_columns = ("id",)
    _output_columns = ("processed",)

    def process_single(self, sample):
        return {**sample, "processed": True}


def test_direct_probe_replays_every_row_for_single_row_mapper():
    op = SingleRowMapper(batch_size=4, accelerator="cpu")
    rows = [{"id": index} for index in range(4)]
    result = probe._run_probe_op_rows(probe._op_spec(op), rows)
    assert [row["id"] for row in result] == [0, 1, 2, 3]


class CachedStatsFilter(Filter):
    _name = "review_cached_stats"
    _batched_op = True
    computed = 0

    def compute_stats_batched(self, samples):
        for stats in samples[Fields.stats]:
            if "score" not in stats:
                type(self).computed += 1
                stats["score"] = 1
        return samples

    def process_batched(self, samples):
        return [True] * len(samples[Fields.stats])


def test_steady_profile_recomputes_stats_after_warmup():
    CachedStatsFilter.computed = 0
    op = CachedStatsFilter(batch_size=1, accelerator="cpu")
    rows = [{"id": 0, Fields.stats: {"upstream_score": 0}}]
    probe._profile_probe_target(probe._op_spec(op), rows, warmup_batches=1, steady_batches=3)
    assert CachedStatsFilter.computed == 4, {"computed": CachedStatsFilter.computed, "mutated_input": rows}


class LazyLoadingMapper(Mapper):
    _name = "review_lazy_loading"
    _batched_op = True
    clock = 0.0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        type(self).clock += 0.01
        self.loaded = False

    def process_batched(self, samples):
        if not self.loaded:
            type(self).clock += 30
            self.loaded = True
        type(self).clock += 0.1
        return samples


def test_auto_group_accounts_for_first_call_model_loading():
    op = LazyLoadingMapper(batch_size=1, accelerator="cpu")
    LazyLoadingMapper.clock = 0
    with patch.object(probe.time, "monotonic", side_effect=lambda: LazyLoadingMapper.clock):
        _, profile = probe._profile_probe_target(probe._op_spec(op), [{"id": 1}], 1, 3)
    executor = PartitionedRayExecutor.__new__(PartitionedRayExecutor)
    executor.execution_group_size = "auto"
    executor.max_initialization_overhead_ratio = 0.1
    executor._resolved_throughput_actor_plan = {
        "operators": [
            {
                "initialization_seconds": profile["initialization"],
                "source_equivalent_rows_per_second": profile["steady_rows_per_second"],
            }
        ]
    }
    size = executor._resolve_execution_group_size(SimpleNamespace(num_partitions=8, total_rows=80), [op])
    assert size == 8, {"group_size": size, "profile": profile}


def test_gpu_actor_placement_uses_best_fit_for_fractional_requests():
    specs = [
        {"num_gpus": fraction, "memory_fraction": fraction, "max_per_device": 5}
        for fraction in (0.7, 0.6, 0.3, 0.2, 0.2)
    ]
    assert PartitionedRayExecutor._gpu_actor_requests_fit(specs, [1] * len(specs), 2)


def test_auto_group_accounts_for_profiled_explicit_gpu_actor_initialization():
    explicit = make_op("explicit", actors=1)
    automatic = make_op("automatic", actors=None)
    explicit._gpu_init_seconds = 30
    automatic._gpu_init_seconds = 0.01
    executor = PartitionedRayExecutor.__new__(PartitionedRayExecutor)
    executor.cfg = SimpleNamespace()
    executor.execution_group_size = "auto"
    executor.max_initialization_overhead_ratio = 0.1
    executor.max_gpu_workers_per_device = 5
    executor._auto_parallel_op_ids = {id(automatic)}
    executor._explicit_actor_op_ids = {id(explicit)}
    topology = SimpleNamespace(total_cpus=8, total_gpus=1)
    with patch("data_juicer.utils.ray_cluster_utils.detect_cluster_topology", return_value=topology):
        plan = executor._configure_throughput_aware_gpu_parallelism([explicit, automatic], total_samples=80)

    size = executor._resolve_execution_group_size(
        SimpleNamespace(num_partitions=8, total_rows=80),
        [explicit, automatic],
    )
    assert explicit.num_proc == 1
    assert [stage["name"] for stage in plan["operators"]] == ["explicit", "automatic"]
    assert size == 8


def test_planner_does_not_scale_non_bottleneck_when_bottleneck_cannot_grow():
    slow = make_op("slow", gpus=0.6)
    fast = make_op("fast", gpus=0.2)
    slow._gpu_rows_per_second = 1
    fast._gpu_rows_per_second = 100

    plan([slow, fast], [], cpus=8, gpus=1)

    assert slow.num_proc == 1
    assert fast.num_proc == 1

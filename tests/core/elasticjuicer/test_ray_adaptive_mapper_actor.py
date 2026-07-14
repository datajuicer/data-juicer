from contextlib import nullcontext

import pyarrow
import pytest

from data_juicer.core.elasticjuicer.ray_adaptive_mapper import RayAdaptiveMapperActor
from data_juicer.ops.base_op import Mapper


class ThresholdMapper:
    def __init__(self, oom_above, increment=1):
        self.oom_above = oom_above
        self.increment = increment

    def process(self, table):
        if table.num_rows > self.oom_above:
            raise RuntimeError("CUDA out of memory")
        values = [value + self.increment for value in table["value"].to_pylist()]
        return pyarrow.table({"value": values})


class NoopSampler:
    def __init__(self, sample_interval_sec=0.01):
        self.sample_interval_sec = sample_interval_sec

    def measure(self, batch_size):
        return nullcontext()


class SkippingThresholdMapper(Mapper):
    _batched_op = True

    def __init__(self, oom_above, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.oom_above = oom_above

    def process_batched(self, samples):
        if len(samples["value"]) > self.oom_above:
            raise RuntimeError("CUDA out of memory")
        return {"value": [value + 1 for value in samples["value"]]}


class SkippingBrokenMapper(Mapper):
    _batched_op = True

    def process_batched(self, samples):
        raise ValueError("invalid sample batch")


def test_actor_retries_losslessly_and_keeps_one_controller():
    actor = RayAdaptiveMapperActor(
        ThresholdMapper,
        operator_args=(8,),
        operator_kwargs={"increment": 3},
        initial_batch_size=32,
        max_batch_size=32,
        sampler_factory=NoopSampler,
    )
    controller = actor.controller

    first = actor(pyarrow.table({"value": list(range(100))}))
    first_state = actor.controller.state
    second = actor(pyarrow.table({"value": list(range(12))}))

    assert first["value"].to_pylist() == [value + 3 for value in range(100)]
    assert second["value"].to_pylist() == [value + 3 for value in range(12)]
    assert actor.controller is controller
    assert first_state.oom_events > 0
    assert actor.controller.state.success_events > first_state.success_events


def test_actor_reconstructs_the_operator_only_once():
    constructions = []

    class CountingMapper:
        def __init__(self):
            constructions.append(self)

        def process(self, table):
            return table

    actor = RayAdaptiveMapperActor(
        CountingMapper,
        initial_batch_size=4,
        max_batch_size=4,
        sampler_factory=NoopSampler,
    )

    actor(pyarrow.table({"value": [1, 2, 3, 4]}))
    actor(pyarrow.table({"value": [5, 6]}))

    assert constructions == [actor.operator]


def test_actor_sees_oom_hidden_by_mapper_skip_op_error_wrapper():
    actor = RayAdaptiveMapperActor(
        SkippingThresholdMapper,
        operator_args=(8,),
        operator_kwargs={"skip_op_error": True},
        initial_batch_size=32,
        max_batch_size=32,
        sampler_factory=NoopSampler,
    )

    result = actor(pyarrow.table({"value": list(range(40))}))

    assert result["value"] == [value + 1 for value in range(40)]
    assert actor.controller.state.oom_events > 0


def test_actor_preserves_skip_op_error_for_non_oom_failures():
    actor = RayAdaptiveMapperActor(
        SkippingBrokenMapper,
        operator_kwargs={"skip_op_error": True},
        initial_batch_size=8,
        max_batch_size=8,
        sampler_factory=NoopSampler,
    )

    result = actor(pyarrow.table({"value": list(range(8))}))

    assert result["value"] == []
    assert result["__dj__stats__"] == []
    assert result["__dj__source_file__"] == []
    assert actor.controller.state.oom_events == 0


def test_real_ray_actor_e2e_when_ray_is_available():
    import ray

    if not callable(getattr(getattr(ray, "data", None), "from_items", None)):
        pytest.skip("real Ray Data is not available in this test environment")

    started_here = not ray.is_initialized()
    if started_here:
        ray.init(num_cpus=1, include_dashboard=False)
    try:
        compute = ray.data.ActorPoolStrategy(size=1)
        source = ray.data.from_items([{"value": value} for value in range(100)])
        result = source.map_batches(
            RayAdaptiveMapperActor,
            fn_constructor_kwargs={
                "operator_class": ThresholdMapper,
                "operator_args": (8,),
                "operator_kwargs": {"increment": 3},
                "initial_batch_size": 32,
                "max_batch_size": 32,
            },
            batch_size=32,
            batch_format="pyarrow",
            compute=compute,
        )

        assert [row["value"] for row in result.take_all()] == [value + 3 for value in range(100)]
    finally:
        if started_here:
            ray.shutdown()

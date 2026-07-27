from contextlib import nullcontext
from types import SimpleNamespace

import pyarrow
import pytest

from data_juicer.core.elasticjuicer.actor_resource_sampler import ActorResourceSnapshot
from data_juicer.core.elasticjuicer.control_service import (
    ActorControlPoller,
    ControlService,
)
from data_juicer.core.elasticjuicer.quota import QuotaEnvelope, current_time_ms
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


class EmittingSampler(NoopSampler):
    def measure(self, batch_size):
        class Measurement:
            snapshot = None

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                self.snapshot = ActorResourceSnapshot(
                    timestamp=123.0,
                    process_id=42,
                    batch_size=batch_size,
                    rss_start_mb=100.0,
                    rss_end_mb=110.0,
                    rss_peak_mb=120.0,
                    rss_delta_mb=10.0,
                    latency_ms=250.0,
                    throughput=16.0,
                    cuda=None,
                    succeeded=exc_type is None,
                    error_type=None if exc_type is None else exc_type.__name__,
                )
                return False

        return Measurement()


class RemoteMethod:
    def __init__(self, function):
        self.function = function

    def remote(self, *args):
        return self.function(*args)


class DirectControlHandle:
    def __init__(self, service):
        self.register = RemoteMethod(service.register)
        self.get_latest = RemoteMethod(service.get_latest)


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


def test_actor_reports_sampler_snapshots_without_waiting_for_sink():
    events = []

    class RemoteRecord:
        def remote(self, event):
            events.append(event)
            return object()

    actor = RayAdaptiveMapperActor(
        ThresholdMapper,
        operator_args=(8,),
        initial_batch_size=16,
        max_batch_size=16,
        sampler_factory=EmittingSampler,
        metrics_sink=SimpleNamespace(record=RemoteRecord()),
        job_id="job-a",
        op_name="threshold_mapper",
        actor_id="actor-a",
    )

    result = actor(pyarrow.table({"value": list(range(20))}))

    assert result["value"].to_pylist() == [value + 1 for value in range(20)]
    assert len(events) > 1
    assert events[0].job_id == "job-a"
    assert events[0].actor_id == "actor-a"
    assert events[0].actor_incarnation_id == actor.actor_incarnation_id
    assert events[0].schema_version == 1
    assert events[0].source == "actor_resource_sampler"
    assert events[0].minimum_confidence == 1.0
    assert events[0].op_name == "threshold_mapper"
    assert events[0].snapshot.batch_size == 16
    assert events[0].snapshot.succeeded is False
    assert events[0].control.local_oom_upper_bound == 16
    assert events[0].control.current_batch_size == 8
    assert events[-1].snapshot.succeeded is True
    assert events[-1].control.local_success_lower_bound is not None
    metrics_state = actor.get_metrics_state()
    assert metrics_state["enabled"] is True
    assert metrics_state["submitted_events"] == len(events)
    assert metrics_state["pending_events"] == len(events)
    assert metrics_state["pending_events"] <= metrics_state["max_in_flight"]


def test_actor_reports_disabled_metrics_state_without_sink():
    actor = RayAdaptiveMapperActor(
        ThresholdMapper,
        operator_args=(8,),
        initial_batch_size=8,
        max_batch_size=8,
        sampler_factory=NoopSampler,
    )

    assert actor.get_metrics_state() == {
        "enabled": False,
        "submitted_events": 0,
        "dropped_events": 0,
        "pending_events": 0,
        "max_in_flight": 0,
        "last_sequence": 0,
    }


def test_actor_receives_service_quota_through_background_cache():
    import time

    service = ControlService("job-a")
    actor = RayAdaptiveMapperActor(
        ThresholdMapper,
        operator_args=(32,),
        initial_batch_size=32,
        max_batch_size=32,
        sampler_factory=NoopSampler,
        job_id="job-a",
        actor_id="actor-a",
        actor_incarnation_id="inc-a",
        control_service=DirectControlHandle(service),
        control_poll_interval_sec=0.001,
        control_poller_factory=lambda **kwargs: ActorControlPoller(
            get_fn=lambda value: value,
            wait_fn=lambda refs, **wait_kwargs: (refs, []),
            **kwargs,
        ),
    )
    try:
        now_ms = current_time_ms()
        service.publish_quota(
            QuotaEnvelope(
                "job-a",
                "actor-a",
                "inc-a",
                revision=1,
                issued_at_ms=now_ms,
                expires_at_ms=now_ms + 10_000,
                max_batch_size=7,
                reason="test",
            )
        )
        actor(pyarrow.table({"value": [0]}))
        time.sleep(0.002)
        result = actor(pyarrow.table({"value": list(range(20))}))
        assert result["value"].to_pylist() == [value + 1 for value in range(20)]
        assert actor.get_quota_state().hard_limit == 7
        assert actor.get_quota_state().last_revision == 1
    finally:
        actor.close()


def test_actor_applies_lower_quota_at_next_micro_slice_within_one_outer_batch():
    seen_batch_sizes = []

    class RecordingMapper:
        def process(self, table):
            seen_batch_sizes.append(table.num_rows)
            return table

    class SliceBoundaryPoller:
        def __init__(self, registration):
            now_ms = current_time_ms()
            self.quota = QuotaEnvelope(
                job_id=registration.job_id,
                actor_id=registration.actor_id,
                actor_incarnation_id=registration.actor_incarnation_id,
                revision=1,
                issued_at_ms=now_ms,
                expires_at_ms=now_ms + 60_000,
                max_batch_size=3,
                reason="mid_outer_batch_test",
            )
            self.boundaries = 0
            self.delivered = False

        def start(self):
            return None

        def poll_once(self):
            self.boundaries += 1

        def take_pending(self):
            if self.boundaries >= 2 and not self.delivered:
                self.delivered = True
                return self.quota
            return None

        def snapshot(self):
            return {"boundaries": self.boundaries}

        def close(self):
            return None

    pollers = []

    def create_poller(control_handle, registration, poll_interval_sec):
        poller = SliceBoundaryPoller(registration)
        pollers.append(poller)
        return poller

    actor = RayAdaptiveMapperActor(
        RecordingMapper,
        initial_batch_size=8,
        max_batch_size=8,
        sampler_factory=NoopSampler,
        job_id="job-a",
        actor_id="actor-a",
        actor_incarnation_id="inc-a",
        control_service=object(),
        control_poller_factory=create_poller,
    )

    result = actor(pyarrow.table({"value": list(range(20))}))

    assert result["value"].to_pylist() == list(range(20))
    assert seen_batch_sizes == [8, 3, 3, 3, 3]
    assert pollers[0].boundaries == 5
    assert actor.get_quota_state().hard_limit == 3


def test_control_service_failure_does_not_interrupt_actor_data_path():
    class BrokenMethod:
        def remote(self, *args):
            raise RuntimeError("service unavailable")

    broken = type("BrokenHandle", (), {"register": BrokenMethod(), "get_latest": BrokenMethod()})()
    actor = RayAdaptiveMapperActor(
        ThresholdMapper,
        operator_args=(8,),
        initial_batch_size=16,
        max_batch_size=16,
        sampler_factory=NoopSampler,
        job_id="job-a",
        control_service=broken,
        control_poll_interval_sec=0.001,
        control_poller_factory=lambda **kwargs: ActorControlPoller(
            get_fn=lambda value: value,
            wait_fn=lambda refs, **wait_kwargs: (refs, []),
            **kwargs,
        ),
    )
    try:
        result = actor(pyarrow.table({"value": list(range(20))}))
        assert result["value"].to_pylist() == [value + 1 for value in range(20)]
    finally:
        actor.close()


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

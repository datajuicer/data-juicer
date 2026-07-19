from types import SimpleNamespace

import pytest

from data_juicer.core.elasticjuicer.actor_resource_sampler import ActorResourceSnapshot
from data_juicer.core.elasticjuicer.async_metrics_sink import (
    ActorMetricsEvent,
    AsyncMetricsReporter,
    AsyncMetricsSink,
    create_ray_metrics_sink,
)


def _snapshot(batch_size=4, succeeded=True, error_type=None):
    return ActorResourceSnapshot(
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
        succeeded=succeeded,
        error_type=error_type,
    )


def _event(job_id, sequence, actor_id="actor-a"):
    return ActorMetricsEvent(
        job_id=job_id,
        actor_id=actor_id,
        op_name="mapper",
        sequence=sequence,
        snapshot=_snapshot(),
    )


def test_sink_is_job_scoped_and_rejects_cross_job_events():
    first = AsyncMetricsSink(job_id="job-a", max_events=4)
    second = AsyncMetricsSink(job_id="job-b", max_events=4)

    first.record(_event("job-a", 1))
    second.record(_event("job-b", 1))

    assert [event.job_id for event in first.snapshot()["events"]] == ["job-a"]
    assert [event.job_id for event in second.snapshot()["events"]] == ["job-b"]
    with pytest.raises(ValueError, match="job_id"):
        first.record(_event("job-b", 2))


def test_sink_history_is_bounded_and_counts_evictions():
    sink = AsyncMetricsSink(job_id="job-a", max_events=2)

    for sequence in range(1, 5):
        sink.record(_event("job-a", sequence))

    snapshot = sink.snapshot()
    assert [event.sequence for event in snapshot["events"]] == [3, 4]
    assert snapshot["received_events"] == 4
    assert snapshot["dropped_events"] == 2


def test_reporter_uses_bounded_fire_and_forget_window():
    calls = []

    class RemoteRecord:
        def remote(self, event):
            calls.append(event)
            return object()

    reporter = AsyncMetricsReporter(
        sink_handle=SimpleNamespace(record=RemoteRecord()),
        job_id="job-a",
        actor_id="actor-a",
        op_name="mapper",
        max_in_flight=2,
        wait_fn=lambda refs, **kwargs: ([], refs),
    )

    assert reporter.report(_snapshot()) is True
    assert reporter.report(_snapshot()) is True
    assert reporter.report(_snapshot()) is False

    assert len(calls) == 2
    assert calls[0].sequence == 1
    assert calls[0].snapshot.batch_size == 4
    assert reporter.snapshot() == {
        "submitted_events": 2,
        "dropped_events": 1,
        "pending_events": 2,
        "max_in_flight": 2,
        "last_sequence": 3,
    }


def test_reporter_reclaims_completed_references_before_submitting():
    calls = []

    class RemoteRecord:
        def remote(self, event):
            reference = object()
            calls.append(reference)
            return reference

    reporter = AsyncMetricsReporter(
        sink_handle=SimpleNamespace(record=RemoteRecord()),
        job_id="job-a",
        actor_id="actor-a",
        op_name="mapper",
        max_in_flight=1,
        wait_fn=lambda refs, **kwargs: (refs, []),
    )

    assert reporter.report(_snapshot()) is True
    assert reporter.report(_snapshot()) is True
    assert len(calls) == 2
    assert reporter.dropped_events == 0


def test_reporter_requires_a_remote_sink_handle():
    reporter = AsyncMetricsReporter(
        sink_handle=SimpleNamespace(record=lambda event: None),
        job_id="job-a",
        actor_id="actor-a",
        op_name="mapper",
    )

    with pytest.raises(TypeError, match="remote"):
        reporter.report(_snapshot())


def test_ray_factory_creates_an_unnamed_explicit_handle():
    calls = []

    class RemoteClass:
        def remote(self, *args, **kwargs):
            calls.append((args, kwargs))
            return "sink-handle"

    class FakeRay:
        def remote(self, **options):
            assert options == {"num_cpus": 0}

            def decorate(target):
                assert target is AsyncMetricsSink
                return RemoteClass()

            return decorate

    handle = create_ray_metrics_sink("job-a", max_events=17, ray_module=FakeRay())

    assert handle == "sink-handle"
    assert calls == [((), {"job_id": "job-a", "max_events": 17})]


def test_real_ray_sink_e2e_when_ray_is_available():
    import ray

    if not callable(getattr(ray, "get", None)):
        pytest.skip("real Ray is not available in this test environment")

    started_here = not ray.is_initialized()
    if started_here:
        ray.init(num_cpus=1, include_dashboard=False)
    try:
        sink = create_ray_metrics_sink("job-a")
        reporter = AsyncMetricsReporter(
            sink_handle=sink,
            job_id="job-a",
            actor_id="actor-a",
            op_name="mapper",
        )
        reporter.report(_snapshot())

        snapshot = ray.get(sink.snapshot.remote())
        assert snapshot["received_events"] == 1
        assert snapshot["events"][0].job_id == "job-a"
    finally:
        if started_here:
            ray.shutdown()

from dataclasses import replace

import pytest

from data_juicer.core.elasticjuicer.actor_resource_sampler import ActorResourceSnapshot
from data_juicer.core.elasticjuicer.async_metrics_sink import (
    ActorControlMetrics,
    ActorMetricsEvent,
    AsyncMetricsSink,
)
from data_juicer.core.elasticjuicer.captain import (
    ActorObservation,
    CaptainDecisionCore,
    CaptainLifecycle,
    CaptainRuntime,
    StageQuotaCoordinator,
    StageSnapshot,
    create_captain_lifecycle,
)
from data_juicer.core.elasticjuicer.control_service import ControlService
from data_juicer.core.elasticjuicer.quota import ActorRegistration, current_time_ms


def _observation(actor_id="actor-a", sequence=1, observed=1_000, **overrides):
    values = {
        "job_id": "job-a",
        "stage_id": "stage-a",
        "op_name": "mapper",
        "actor_id": actor_id,
        "actor_incarnation_id": f"{actor_id}-inc",
        "sequence": sequence,
        "observed_at_ms": observed,
        "quota_revision": 0,
        "current_batch_size": 16,
        "hard_limit": 32,
        "static_min_batch_size": 2,
        "static_max_batch_size": 32,
        "local_success_lower_bound": 16,
        "local_oom_upper_bound": None,
        "rss_peak_mb": 500.0,
        "cuda_reserved_mb": None,
        "cuda_peak_allocated_mb": None,
        "throughput": 100.0,
        "latency_ms": 160.0,
        "confidence": 1.0,
    }
    values.update(overrides)
    return ActorObservation(**values)


def _core(**kwargs):
    values = {
        "metrics_ttl_ms": 100,
        "min_decision_interval_ms": 10,
        "recovery_observations": 3,
        "process_memory_high_mb": 900,
        "process_memory_low_mb": 600,
    }
    values.update(kwargs)
    return CaptainDecisionCore(**values)


def test_stable_capacity_produces_no_unnecessary_decision():
    core = _core()
    observation = _observation(hard_limit=32, current_batch_size=32)

    assert core.observe(observation, now_ms=1_010) is None


def test_memory_poor_actor_shrinks_without_affecting_healthy_actor():
    core = _core()
    poor = _observation(actor_id="poor", rss_peak_mb=1_200)
    healthy = _observation(actor_id="healthy", rss_peak_mb=500)

    decision = core.observe(poor, now_ms=1_010)
    assert decision.actor_id == "poor"
    assert decision.max_batch_size == 8
    assert decision.reason == "memory_high_watermark"
    assert core.observe(healthy, now_ms=1_010) is None


def test_all_actors_can_shrink_independently():
    core = _core()
    decisions = [
        core.observe(_observation(actor_id=f"actor-{index}", rss_peak_mb=1_100), now_ms=1_010) for index in range(3)
    ]

    assert [decision.max_batch_size for decision in decisions] == [8, 8, 8]
    assert len({decision.actor_incarnation_id for decision in decisions}) == 3


def test_cuda_pressure_is_evaluated_separately_and_missing_cuda_cannot_recover():
    core = _core(
        recovery_observations=1,
        process_memory_high_mb=None,
        process_memory_low_mb=None,
        cuda_memory_high_mb=900,
        cuda_memory_low_mb=600,
    )
    pressured = _observation(cuda_reserved_mb=1_200, cuda_peak_allocated_mb=1_000)
    decision = core.observe(pressured, now_ms=1_010)

    assert decision.reason == "memory_high_watermark"
    assert decision.max_batch_size == 8

    missing = _observation(
        actor_id="missing-cuda",
        hard_limit=8,
        current_batch_size=8,
        cuda_reserved_mb=None,
        cuda_peak_allocated_mb=None,
    )
    assert core.observe(missing, now_ms=1_010) is None


def test_capacity_recovery_requires_stable_fresh_trace_and_is_rate_limited():
    core = _core()
    base = _observation(hard_limit=8, current_batch_size=8, rss_peak_mb=400)

    assert core.observe(base, now_ms=1_000) is None
    assert core.observe(replace(base, sequence=2, observed_at_ms=1_010), now_ms=1_010) is None
    decision = core.observe(replace(base, sequence=3, observed_at_ms=1_020), now_ms=1_020)

    assert decision.max_batch_size == 10
    assert decision.reason == "stable_low_pressure_recovery"
    assert decision.capacity_recovery_hint is False
    assert core.observe(replace(base, sequence=4, observed_at_ms=1_021), now_ms=1_021) is None


def test_stale_missing_low_confidence_loss_and_out_of_order_never_expand():
    core = _core(recovery_observations=1)
    base = _observation(hard_limit=8, current_batch_size=8, rss_peak_mb=400)

    assert core.observe(base, now_ms=1_200) is None
    assert core.observe(replace(base, sequence=2, confidence=0.2), now_ms=1_001) is None
    assert core.observe(replace(base, sequence=3, telemetry_loss=True), now_ms=1_001) is None
    assert core.observe(replace(base, sequence=2), now_ms=1_001) is None
    assert core.observe(replace(base, sequence=4, rss_peak_mb=None), now_ms=1_001) is None


def test_sequence_gap_is_unknown_and_requires_a_new_stable_recovery_streak():
    core = _core(recovery_observations=2)
    base = _observation(hard_limit=8, current_batch_size=8, rss_peak_mb=400)

    assert core.observe(base, now_ms=1_000) is None
    assert core.observe(replace(base, sequence=3, observed_at_ms=1_010), now_ms=1_010) is None
    assert core.observe(replace(base, sequence=4, observed_at_ms=1_020), now_ms=1_020) is None
    decision = core.observe(replace(base, sequence=5, observed_at_ms=1_030), now_ms=1_030)
    assert decision.reason == "stable_low_pressure_recovery"


def test_first_observation_with_noninitial_sequence_is_a_loss_gap():
    core = _core(recovery_observations=1)
    base = _observation(sequence=7, hard_limit=8, current_batch_size=8, rss_peak_mb=400)

    assert core.observe(base, now_ms=1_000) is None
    decision = core.observe(replace(base, sequence=8, observed_at_ms=1_010), now_ms=1_010)
    assert decision.reason == "stable_low_pressure_recovery"


def test_local_oom_bound_wins_over_remote_cap_and_recovery_only_emits_hint():
    core = _core(recovery_observations=2)
    bounded = _observation(local_oom_upper_bound=10, hard_limit=32, current_batch_size=8)
    shrink = core.observe(bounded, now_ms=1_001)

    assert shrink.max_batch_size == 9
    assert shrink.reason == "local_oom_bound"

    low = replace(bounded, hard_limit=9, rss_peak_mb=400, sequence=2, observed_at_ms=1_020)
    assert core.observe(low, now_ms=1_020) is None
    recovery = core.observe(replace(low, sequence=3, observed_at_ms=1_030), now_ms=1_030)
    assert recovery.max_batch_size > 9
    assert recovery.capacity_recovery_hint is True


def test_revision_is_monotonic_after_captain_restart_seed():
    core = _core()
    core.seed_revision("job-a", "actor-a", "actor-a-inc", 41)
    decision = core.observe(_observation(rss_peak_mb=1_200), now_ms=1_001)
    assert decision.revision == 42


def _registration_for(observation):
    return ActorRegistration(
        observation.job_id,
        observation.stage_id,
        observation.op_name,
        observation.actor_id,
        observation.actor_incarnation_id,
        observation.static_min_batch_size,
        observation.static_max_batch_size,
    )


def _stage_snapshot(*observations, registrations=None, now_ms=1_010):
    registrations = registrations or tuple(_registration_for(observation) for observation in observations)
    return StageSnapshot(
        job_id="job-a",
        stage_id="stage-a",
        captured_at_ms=now_ms,
        registrations=tuple(registrations),
        observations=tuple(observations),
    )


def test_stage_skew_shrinks_hot_actor_and_holds_healthy_actor_recovery():
    coordinator = StageQuotaCoordinator(_core(recovery_observations=1))
    hot = _observation(actor_id="hot", rss_peak_mb=1_200, hard_limit=16)
    healthy = _observation(actor_id="healthy", rss_peak_mb=400, hard_limit=8, current_batch_size=8)

    decisions = coordinator.decide(_stage_snapshot(hot, healthy))

    assert [(decision.actor_id, decision.max_batch_size) for decision in decisions] == [("hot", 8)]


def test_stage_partial_or_stale_metrics_block_all_recovery_until_complete_fresh_window():
    coordinator = StageQuotaCoordinator(_core(recovery_observations=1))
    actor_a = _observation(actor_id="actor-a", hard_limit=8, current_batch_size=8, rss_peak_mb=400)
    actor_b = _observation(actor_id="actor-b", hard_limit=8, current_batch_size=8, rss_peak_mb=400)
    registrations = (_registration_for(actor_a), _registration_for(actor_b))

    assert coordinator.decide(_stage_snapshot(actor_a, registrations=registrations)) == []
    stale_b = replace(actor_b, observed_at_ms=800)
    assert coordinator.decide(_stage_snapshot(replace(actor_a, sequence=2), stale_b, registrations=registrations)) == []

    decisions = coordinator.decide(
        _stage_snapshot(
            replace(actor_a, sequence=3, observed_at_ms=1_020),
            replace(actor_b, sequence=2, observed_at_ms=1_020),
            registrations=registrations,
            now_ms=1_020,
        )
    )
    assert {(decision.actor_id, decision.max_batch_size) for decision in decisions} == {
        ("actor-a", 10),
        ("actor-b", 10),
    }


def test_stage_sequence_gap_on_one_actor_holds_other_actor_recovery():
    coordinator = StageQuotaCoordinator(_core(recovery_observations=1))
    actor_a = _observation(actor_id="actor-a", hard_limit=8, current_batch_size=8, rss_peak_mb=400)
    actor_b = _observation(actor_id="actor-b", hard_limit=8, current_batch_size=8, rss_peak_mb=400)

    assert coordinator.decide(_stage_snapshot(actor_a, actor_b))
    next_a = replace(actor_a, sequence=3, observed_at_ms=1_020)
    next_b = replace(actor_b, sequence=2, observed_at_ms=1_020)

    assert coordinator.decide(_stage_snapshot(next_a, next_b, now_ms=1_020)) == []


def test_stage_actor_restart_ignores_old_incarnation_and_waits_for_new_metrics():
    coordinator = StageQuotaCoordinator(_core(recovery_observations=1))
    old = _observation(actor_id="actor-a", hard_limit=8, current_batch_size=8, rss_peak_mb=400)
    new = replace(old, actor_incarnation_id="actor-a-new")
    active_registration = _registration_for(new)

    assert coordinator.decide(_stage_snapshot(old, registrations=(active_registration,))) == []
    decisions = coordinator.decide(_stage_snapshot(new, registrations=(active_registration,)))
    assert len(decisions) == 1
    assert decisions[0].actor_incarnation_id == "actor-a-new"


def test_stage_recovery_is_fair_across_heterogeneous_actor_bounds():
    coordinator = StageQuotaCoordinator(_core(recovery_observations=1))
    small = _observation(actor_id="small", hard_limit=8, current_batch_size=8, rss_peak_mb=400)
    large = _observation(
        actor_id="large",
        hard_limit=16,
        current_batch_size=16,
        static_max_batch_size=64,
        rss_peak_mb=400,
    )

    decisions = coordinator.decide(_stage_snapshot(small, large))

    assert {(decision.actor_id, decision.max_batch_size) for decision in decisions} == {
        ("small", 10),
        ("large", 20),
    }


class _RemoteMethod:
    def __init__(self, function):
        self.function = function

    def remote(self, *args):
        return self.function(*args)


class _Handle:
    def __init__(self, target):
        for method in ("snapshot", "publish_quota", "record"):
            if hasattr(target, method):
                setattr(self, method, _RemoteMethod(getattr(target, method)))


def _event(registration, sequence=1, rss_peak=1_200, quota_revision=0, observed_at_ms=1_000):
    snapshot = ActorResourceSnapshot(
        timestamp=observed_at_ms / 1000,
        process_id=42,
        batch_size=16,
        rss_start_mb=400,
        rss_end_mb=500,
        rss_peak_mb=rss_peak,
        rss_delta_mb=100,
        latency_ms=10,
        throughput=1_600,
        cuda=None,
        succeeded=True,
        error_type=None,
    )
    control = ActorControlMetrics(quota_revision, 16, 32, 2, 32, 16, None)
    return ActorMetricsEvent(
        job_id=registration.job_id,
        actor_id=registration.actor_id,
        actor_incarnation_id=registration.actor_incarnation_id,
        stage_id=registration.stage_id,
        op_name=registration.op_name,
        sequence=sequence,
        observed_at_ms=observed_at_ms,
        emitted_at_ms=observed_at_ms + 1,
        source=snapshot.source,
        snapshot=snapshot,
        control=control,
    )


def test_runtime_bootstraps_dispatches_auditable_quota_and_deduplicates_events():
    registration = ActorRegistration("job-a", "stage-a", "mapper", "actor-a", "inc-a", 2, 32)
    control = ControlService("job-a")
    control.register(registration)
    sink = AsyncMetricsSink("job-a")
    now_ms = current_time_ms()
    sink.record(_event(registration, observed_at_ms=now_ms))
    runtime = CaptainRuntime(_core(), _Handle(sink), _Handle(control), get_fn=lambda value: value)

    decisions = runtime.poll_once(now_ms=now_ms + 10)
    assert len(decisions) == 1
    assert decisions[0].reason == "memory_high_watermark"
    delivered = control.get_latest("job-a", "actor-a", "inc-a")
    assert delivered.max_batch_size == 8
    assert delivered.reason == "memory_high_watermark"
    assert runtime.poll_once(now_ms=now_ms + 11) == []


def test_runtime_holds_pending_quota_while_delivery_is_unavailable():
    registration = ActorRegistration("job-a", "stage-a", "mapper", "actor-a", "inc-a", 2, 32)
    control = ControlService("job-a")
    control.register(registration)
    sink = AsyncMetricsSink("job-a")
    now_ms = current_time_ms()
    sink.record(_event(registration, observed_at_ms=now_ms))
    handle = _Handle(control)

    def unavailable(*args):
        raise RuntimeError("control unavailable")

    handle.publish_quota = _RemoteMethod(unavailable)
    runtime = CaptainRuntime(_core(), _Handle(sink), handle, get_fn=lambda value: value)

    assert len(runtime.poll_once(now_ms=now_ms + 10)) == 1
    assert runtime.snapshot()["pending_deliveries"] == 1
    assert control.snapshot()["latest_quotas"] == []


def test_runtime_expires_failed_delivery_then_publishes_newer_revision_after_recovery():
    registration = ActorRegistration("job-a", "stage-a", "mapper", "actor-a", "inc-a", 2, 32)
    control = ControlService("job-a")
    control.register(registration)
    sink = AsyncMetricsSink("job-a")
    now_ms = current_time_ms()
    sink.record(_event(registration, observed_at_ms=now_ms))
    handle = _Handle(control)
    working_publish = handle.publish_quota
    handle.publish_quota = _RemoteMethod(lambda *args: (_ for _ in ()).throw(RuntimeError("unavailable")))
    runtime = CaptainRuntime(
        _core(),
        _Handle(sink),
        handle,
        quota_ttl_ms=10,
        retry_backoff_sec=0,
        get_fn=lambda value: value,
    )

    first = runtime.poll_once(now_ms=now_ms)
    assert first[0].revision == 1
    assert runtime.snapshot()["pending_deliveries"] == 1

    handle.publish_quota = working_publish
    sink.record(_event(registration, sequence=2, observed_at_ms=now_ms + 20))
    second = runtime.poll_once(now_ms=now_ms + 20)
    assert second[0].revision == 2
    assert runtime.snapshot()["pending_deliveries"] == 0
    assert control.get_latest("job-a", "actor-a", "inc-a").revision == 2


def test_runtime_refuses_to_decide_if_revision_bootstrap_is_unavailable():
    class BrokenSnapshot:
        def remote(self):
            raise RuntimeError("control unavailable")

    sink = AsyncMetricsSink("job-a")
    runtime = CaptainRuntime(
        _core(), _Handle(sink), type("Broken", (), {"snapshot": BrokenSnapshot()})(), get_fn=lambda x: x
    )
    assert runtime.poll_once(now_ms=1_000) == []
    assert runtime.snapshot()["bootstrapped"] is False


def test_invalid_watermark_order_is_rejected():
    with pytest.raises(ValueError, match="watermark"):
        _core(process_memory_low_mb=900, process_memory_high_mb=800)


def test_runtime_rpc_timeout_enters_backoff_before_retrying():
    control = ControlService("job-a")
    sink = AsyncMetricsSink("job-a")
    clock = [10.0]
    calls = []

    def resolve(value, timeout=None):
        calls.append(timeout)
        if len(calls) == 1:
            raise TimeoutError("stalled actor")
        return value

    runtime = CaptainRuntime(
        _core(),
        _Handle(sink),
        _Handle(control),
        rpc_timeout_sec=0.25,
        retry_backoff_sec=2.0,
        get_fn=resolve,
        clock=lambda: clock[0],
    )

    assert runtime.poll_once(now_ms=1_000) == []
    assert runtime.snapshot()["backing_off"] is True
    assert runtime.poll_once(now_ms=1_001) == []
    assert calls == [0.25]
    clock[0] = 12.0
    assert runtime.poll_once(now_ms=1_002) == []
    assert calls == [0.25, 0.25, 0.25]
    assert runtime.snapshot()["rpc_failures"] == 1


def test_captain_lifecycle_starts_polls_survives_error_and_stops():
    import time

    control = ControlService("job-a")
    sink = AsyncMetricsSink("job-a")
    runtime = CaptainRuntime(_core(), _Handle(sink), _Handle(control), get_fn=lambda value: value)
    original_poll = runtime.poll_once
    calls = []

    def flaky_poll():
        calls.append(None)
        if len(calls) == 1:
            raise RuntimeError("transient")
        return original_poll()

    runtime.poll_once = flaky_poll
    lifecycle = CaptainLifecycle(runtime, poll_interval_sec=0.001)
    lifecycle.start()
    deadline = time.monotonic() + 1
    while len(calls) < 2 and time.monotonic() < deadline:
        time.sleep(0.001)
    lifecycle.close()

    state = lifecycle.snapshot()
    assert len(calls) >= 2
    assert state["running"] is False
    assert state["loop_errors"] == 1
    assert state["polls"] >= 1


def test_product_lifecycle_factory_requires_complete_watermarks_and_applies_rpc_config():
    sink = _Handle(AsyncMetricsSink("job-a"))
    control = _Handle(ControlService("job-a"))
    with pytest.raises(ValueError, match="watermark pair"):
        create_captain_lifecycle(sink, control, {})
    with pytest.raises(ValueError, match="configured together"):
        create_captain_lifecycle(
            sink,
            control,
            {"elastic_juicer_captain_process_memory_high_mb": 1_000},
        )

    lifecycle = create_captain_lifecycle(
        sink,
        control,
        {
            "elastic_juicer_captain_process_memory_high_mb": 1_000,
            "elastic_juicer_captain_process_memory_low_mb": 600,
            "elastic_juicer_captain_rpc_timeout_sec": 0.25,
            "elastic_juicer_captain_retry_backoff_sec": 0.75,
            "elastic_juicer_captain_poll_interval_sec": 0.2,
        },
    )
    assert lifecycle.poll_interval_sec == 0.2
    assert lifecycle.runtime.rpc_timeout_sec == 0.25
    assert lifecycle.runtime.retry_backoff_sec == 0.75

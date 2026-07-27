import time

import pytest

from data_juicer.core.elasticjuicer.control_service import (
    ActorControlPoller,
    ControlService,
    StageProfile,
    create_ray_control_service,
)
from data_juicer.core.elasticjuicer.quota import ActorRegistration, QuotaEnvelope


def _registration(actor_id="actor-a", incarnation="inc-a", job_id="job-a"):
    return ActorRegistration(job_id, "stage-a", "mapper", actor_id, incarnation, 2, 32)


def _quota(registration, revision=1, maximum=8, issued=100, expires=200, **kwargs):
    return QuotaEnvelope(
        job_id=registration.job_id,
        actor_id=registration.actor_id,
        actor_incarnation_id=registration.actor_incarnation_id,
        revision=revision,
        issued_at_ms=issued,
        expires_at_ms=expires,
        max_batch_size=maximum,
        reason="test",
        **kwargs,
    )


def test_service_is_job_scoped_and_registration_is_idempotent():
    service = ControlService("job-a")
    registration = _registration()

    assert service.register(registration) == registration
    assert service.register(registration) == registration
    with pytest.raises(ValueError, match="job_id"):
        service.register(_registration(job_id="job-b"))

    snapshot = service.snapshot()
    assert snapshot["registrations"] == [registration]
    assert snapshot["registration_events"] == 2


def test_service_rejects_wrong_actor_stale_expired_and_below_min_atomically():
    service = ControlService("job-a")
    registration = service.register(_registration())

    with pytest.raises(ValueError, match="not registered"):
        service.publish_quota(_quota(_registration(actor_id="actor-b")), now_ms=150)
    with pytest.raises(ValueError, match="minimum"):
        service.publish_quota(_quota(registration, maximum=1), now_ms=150)

    assert service.publish_quota(_quota(registration, revision=2), now_ms=150).accepted
    assert service.publish_quota(_quota(registration, revision=1), now_ms=150).reason == "stale_revision"
    assert service.publish_quota(_quota(registration, revision=3), now_ms=250).reason == "expired"
    assert service.get_latest("job-a", "actor-a", "inc-a", after_revision=0).revision == 2


def test_actor_reconstruction_uses_a_new_revision_namespace():
    service = ControlService("job-a")
    old = service.register(_registration(incarnation="old"))
    new = service.register(_registration(incarnation="new"))

    assert service.publish_quota(_quota(old, revision=9), now_ms=150).accepted
    assert service.publish_quota(_quota(new, revision=1), now_ms=150).accepted
    assert service.get_latest("job-a", "actor-a", "old").revision == 9
    assert service.get_latest("job-a", "actor-a", "new").revision == 1
    assert service.snapshot()["active_registrations"] == [new]
    service.register(old)
    assert service.snapshot()["active_registrations"] == [new]


class _RemoteMethod:
    def __init__(self, function):
        self.function = function

    def remote(self, *args):
        return self.function(*args)


class _DirectHandle:
    def __init__(self, service):
        self.register = _RemoteMethod(service.register)
        self.get_latest = _RemoteMethod(service.get_latest)


def test_nonblocking_poller_registers_and_caches_at_boundaries():
    service = ControlService("job-a")
    registration = _registration()
    poller = ActorControlPoller(
        _DirectHandle(service),
        registration,
        poll_interval_sec=0.001,
        get_fn=lambda value: value,
        wait_fn=lambda refs, **kwargs: (refs, []),
    )
    poller.start()
    try:
        poller.poll_once()
        assert poller.snapshot()["registered"] is True
        service.publish_quota(_quota(registration), now_ms=150)
        deadline = time.monotonic() + 1
        while poller.snapshot()["pending_revision"] != 1 and time.monotonic() < deadline:
            poller.poll_once()
            time.sleep(0.001)

        assert poller.take_pending().revision == 1
        assert poller.take_pending() is None
        assert poller.snapshot()["poll_errors"] == 0
    finally:
        poller.close()


def test_poller_failure_is_bounded_and_non_blocking():
    class Broken:
        def remote(self, *args):
            raise RuntimeError("service unavailable")

    handle = type("Handle", (), {"register": Broken(), "get_latest": Broken()})()
    poller = ActorControlPoller(
        handle,
        _registration(),
        poll_interval_sec=0.001,
        get_fn=lambda value: value,
        wait_fn=lambda refs, **kwargs: (refs, []),
    )
    poller.start()
    try:
        poller.poll_once()
        assert poller.take_pending() is None
        assert poller.snapshot()["poll_errors"] > 0
    finally:
        poller.close()


def test_ray_factory_returns_an_explicit_unnamed_handle():
    calls = []

    class RemoteClass:
        def remote(self, *args, **kwargs):
            calls.append((args, kwargs))
            return "control-handle"

    class FakeRay:
        def remote(self, **options):
            assert options == {"num_cpus": 0}

            def decorate(target):
                assert target is ControlService
                return RemoteClass()

            return decorate

    assert create_ray_control_service("job-a", ray_module=FakeRay()) == "control-handle"
    assert calls == [((), {"job_id": "job-a", "lease_ttl_ms": 60_000, "profile_ttl_ms": 1_800_000})]


def _profile(stage_id="stage-000000:mapper", safe=8, oom=16, job_id="job-a", observed=100, **kwargs):
    return StageProfile(
        job_id=job_id,
        stage_id=stage_id,
        op_name="mapper",
        safe_batch_size=safe,
        oom_upper_bound=oom,
        observed_at_ms=observed,
        **kwargs,
    )


def test_stage_profile_requires_identity_and_at_least_one_learned_bound():
    with pytest.raises(ValueError, match="stage_id"):
        _profile(stage_id="")
    with pytest.raises(ValueError, match="at least one learned bound"):
        _profile(safe=None, oom=None)
    with pytest.raises(ValueError, match="safe_batch_size"):
        _profile(safe=0)
    with pytest.raises(ValueError, match="oom_upper_bound"):
        _profile(oom=0)


def test_stage_profile_report_is_job_scoped_validated_and_readable():
    service = ControlService("job-a")

    with pytest.raises(TypeError, match="StageProfile"):
        service.report_stage_profile(object())
    with pytest.raises(ValueError, match="job_id"):
        service.report_stage_profile(_profile(job_id="job-b"))
    with pytest.raises(ValueError, match="job_id"):
        service.get_stage_profile("job-b", "stage-000000:mapper")

    assert service.get_stage_profile("job-a", "stage-000000:mapper", now_ms=150) is None
    profile = _profile()
    assert service.report_stage_profile(profile) == profile
    assert service.get_stage_profile("job-a", "stage-000000:mapper", now_ms=150) == profile
    assert service.snapshot()["stage_profiles"] == [profile]


def test_stage_profile_read_requires_matching_fingerprint_and_resource_class():
    service = ControlService("job-a")
    profile = _profile(op_fingerprint="fp-a", resource_class="cuda:h20")
    service.report_stage_profile(profile)

    read = service.get_stage_profile(
        "job-a", profile.stage_id, op_fingerprint="fp-a", resource_class="cuda:h20", now_ms=150
    )
    assert read == profile
    # A changed operator config or device class never inherits stale bounds.
    assert (
        service.get_stage_profile(
            "job-a", profile.stage_id, op_fingerprint="fp-b", resource_class="cuda:h20", now_ms=150
        )
        is None
    )
    assert (
        service.get_stage_profile("job-a", profile.stage_id, op_fingerprint="fp-a", now_ms=150) is None
    )


def test_stage_profile_expires_after_ttl_and_is_removed_on_read():
    service = ControlService("job-a", profile_ttl_ms=1_000)
    service.report_stage_profile(_profile(observed=100))

    assert service.get_stage_profile("job-a", "stage-000000:mapper", now_ms=1_100) is not None
    assert service.get_stage_profile("job-a", "stage-000000:mapper", now_ms=1_101) is None
    # The expired entry is gone and the read is counted for diagnostics.
    assert service.snapshot()["stage_profiles"] == []
    assert service.snapshot()["expired_profile_reads"] == 1


def test_stage_profile_store_capacity_evicts_the_oldest_entry():
    service = ControlService("job-a", max_stage_profiles=2)
    service.report_stage_profile(_profile(stage_id="stage-old", observed=100))
    service.report_stage_profile(_profile(stage_id="stage-mid", observed=200))
    service.report_stage_profile(_profile(stage_id="stage-new", observed=300))

    assert service.get_stage_profile("job-a", "stage-old", now_ms=350) is None
    assert service.get_stage_profile("job-a", "stage-mid", now_ms=350) is not None
    assert service.get_stage_profile("job-a", "stage-new", now_ms=350) is not None


def test_stage_profile_merge_keeps_tightest_oom_bound_and_best_safe_size():
    service = ControlService("job-a")
    service.report_stage_profile(_profile(safe=4, oom=16, observed=100))

    merged = service.report_stage_profile(_profile(safe=8, oom=32, observed=200))
    assert merged.safe_batch_size == 8
    assert merged.oom_upper_bound == 16
    assert merged.observed_at_ms == 200

    # A tighter OOM bound also caps the proven safe size below the bound.
    merged = service.report_stage_profile(_profile(safe=None, oom=6, observed=300))
    assert merged.oom_upper_bound == 6
    assert merged.safe_batch_size == 5


def test_registration_leases_expire_renew_on_poll_and_support_deregister():
    service = ControlService("job-a", lease_ttl_ms=1_000)
    registration = service.register(_registration(), now_ms=100)

    assert service.snapshot(now_ms=1_100)["active_registrations"] == [registration]
    # A zombie that stops polling drops out of stage coordination after TTL.
    assert service.snapshot(now_ms=1_101)["active_registrations"] == []

    # Any quota poll doubles as a lease heartbeat.
    service.get_latest("job-a", "actor-a", "inc-a", now_ms=2_000)
    assert service.snapshot(now_ms=3_000)["active_registrations"] == [registration]

    # Explicit deregister retires the incarnation immediately.
    assert service.deregister("job-a", "actor-a", "inc-a") is True
    assert service.deregister("job-a", "actor-a", "inc-a") is False
    with pytest.raises(ValueError, match="job_id"):
        service.deregister("job-b", "actor-a", "inc-a")
    snapshot = service.snapshot(now_ms=2_001)
    assert snapshot["active_registrations"] == []
    assert snapshot["deregistrations"] == 1

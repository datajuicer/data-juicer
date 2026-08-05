import inspect
from contextlib import nullcontext

import pyarrow
import pytest

from data_juicer.core.elasticjuicer.quota import QuotaEnvelope, current_time_ms
from data_juicer.core.elasticjuicer.ray_adaptive_mapper import RayAdaptiveMapperActor


class RecordingMapper:
    def __init__(self):
        self.seen_batch_sizes = []

    def process(self, table):
        self.seen_batch_sizes.append(table.num_rows)
        return table


class NoopSampler:
    def __init__(self, sample_interval_sec=0.01):
        self.sample_interval_sec = sample_interval_sec

    def measure(self, batch_size):
        return nullcontext()


def _actor(initial=32, minimum=1, maximum=32, **kwargs):
    return RayAdaptiveMapperActor(
        RecordingMapper,
        initial_batch_size=initial,
        min_batch_size=minimum,
        max_batch_size=maximum,
        sampler_factory=NoopSampler,
        job_id="job-a",
        actor_id="actor-a",
        actor_incarnation_id="incarnation-a",
        **kwargs,
    )


def _quota(actor, revision=1, maximum=8, **overrides):
    now_ms = current_time_ms()
    values = {
        "job_id": actor.job_id,
        "actor_id": actor.actor_id,
        "actor_incarnation_id": actor.actor_incarnation_id,
        "revision": revision,
        "issued_at_ms": now_ms,
        "expires_at_ms": now_ms + 60_000,
        "max_batch_size": maximum,
        "reason": "test",
    }
    values.update(overrides)
    return QuotaEnvelope(**values)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("job_id", "", "job_id"),
        ("actor_id", 123, "actor_id"),
        ("actor_incarnation_id", "", "actor_incarnation_id"),
        ("revision", 0, "revision"),
        ("issued_at_ms", -1, "issued_at_ms"),
        ("max_batch_size", True, "max_batch_size"),
        ("schema_version", 2, "schema_version"),
    ],
)
def test_quota_contract_rejects_invalid_fields(field, value, message):
    actor = _actor()
    with pytest.raises((TypeError, ValueError), match=message):
        _quota(actor, **{field: value})


def test_quota_contract_requires_fresh_ttl_and_boolean_hint():
    actor = _actor()
    now_ms = current_time_ms()
    with pytest.raises(ValueError, match="later"):
        _quota(actor, issued_at_ms=now_ms, expires_at_ms=now_ms)
    with pytest.raises(TypeError, match="capacity_recovery_hint"):
        _quota(actor, capacity_recovery_hint=1)


def test_actor_applies_quota_as_an_exact_hard_upper_bound():
    actor = _actor()

    application = actor._apply_quota(_quota(actor, maximum=7))
    result = actor(pyarrow.table({"value": list(range(20))}))

    assert application.applied is True
    assert application.previous_hard_limit == 32
    assert application.effective_max_batch_size == 7
    assert actor.controller.state.hard_limit == 7
    assert max(actor.operator.seen_batch_sizes) == 7
    assert result["value"].to_pylist() == list(range(20))


def test_stale_duplicate_and_expired_revision_cannot_overwrite_newer_quota():
    actor = _actor()
    actor._apply_quota(_quota(actor, revision=2, maximum=8))

    stale = actor._apply_quota(_quota(actor, revision=1, maximum=20))
    duplicate = actor._apply_quota(_quota(actor, revision=2, maximum=24))
    now_ms = current_time_ms()
    expired = actor._apply_quota(
        _quota(actor, revision=3, maximum=30, issued_at_ms=now_ms - 20, expires_at_ms=now_ms - 10)
    )

    assert stale.reason == duplicate.reason == "stale_revision"
    assert expired.applied is False
    assert expired.reason == "expired"
    assert actor.controller.state.hard_limit == 8
    assert actor.get_quota_state().last_revision == 2


def test_quota_rejects_wrong_identity_without_consuming_revision():
    actor = _actor()

    with pytest.raises(ValueError, match="job_id"):
        actor._apply_quota(_quota(actor, job_id="job-b"))
    with pytest.raises(ValueError, match="actor_id"):
        actor._apply_quota(_quota(actor, actor_id="actor-b"))
    with pytest.raises(ValueError, match="actor_incarnation_id"):
        actor._apply_quota(_quota(actor, actor_incarnation_id="old-incarnation"))

    assert actor._apply_quota(_quota(actor)).applied is True
    assert actor.get_quota_state().last_revision == 1


def test_quota_below_actor_minimum_is_rejected_atomically():
    actor = _actor(initial=8, minimum=4, maximum=32)

    with pytest.raises(ValueError, match="minimum"):
        actor._apply_quota(_quota(actor, maximum=2))

    state = actor.get_quota_state()
    assert state.last_revision == 0
    assert state.hard_limit == 32
    assert state.current_batch_size == 8


def test_quota_above_static_maximum_is_safely_clamped():
    actor = _actor(maximum=32)
    application = actor._apply_quota(_quota(actor, maximum=64))

    assert application.requested_max_batch_size == 64
    assert application.effective_max_batch_size == 32
    assert actor.controller.state.hard_limit == 32


def test_relaxed_quota_and_recovery_hint_do_not_erase_local_oom_state():
    actor = _actor(oom_reprobe_successes=2, max_oom_reprobes=1)
    actor.controller.observe_oom(16)

    actor._apply_quota(_quota(actor, revision=1, maximum=6))
    application = actor._apply_quota(_quota(actor, revision=2, maximum=32, capacity_recovery_hint=True))

    assert application.recovery_hint_recorded is True
    assert actor.get_quota_state().local_oom_upper_bound == 16
    assert actor.controller.state.capacity_recovery_hint_pending is True
    assert actor.controller.next_batch_size(100) < 16

    for _ in range(2):
        actor.controller.observe_success(actor.controller.next_batch_size(100))
    assert actor.controller.state.oom_upper_bound is None


def test_recovery_hint_is_rejected_while_hard_cap_cannot_cross_oom_bound():
    actor = _actor(oom_reprobe_successes=1, max_oom_reprobes=1)
    actor.controller.observe_oom(16)

    application = actor._apply_quota(_quota(actor, revision=1, maximum=6, capacity_recovery_hint=True))

    assert application.applied is True
    assert application.recovery_hint_recorded is False
    assert actor.controller.state.capacity_recovery_hint_pending is False
    assert actor.controller.state.oom_upper_bound == 16


def test_remote_reset_authority_is_not_exposed_and_quota_path_has_no_ray_wait():
    assert not hasattr(RayAdaptiveMapperActor, "reset_oom_bound")
    assert not hasattr(RayAdaptiveMapperActor, "apply_quota")
    source = inspect.getsource(RayAdaptiveMapperActor._apply_quota)
    assert "ray.get" not in source
    assert "reset_oom_bound" not in source

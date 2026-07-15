import inspect
from contextlib import nullcontext

import pyarrow
import pytest

from data_juicer.core.elasticjuicer.quota import BatchSizeQuota
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


def _actor(initial=32, minimum=1, maximum=32):
    return RayAdaptiveMapperActor(
        RecordingMapper,
        initial_batch_size=initial,
        min_batch_size=minimum,
        max_batch_size=maximum,
        sampler_factory=NoopSampler,
        job_id="job-a",
        actor_id="actor-a",
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"job_id": "", "actor_id": "actor-a", "revision": 1, "max_batch_size": 8}, "job_id"),
        ({"job_id": 123, "actor_id": "actor-a", "revision": 1, "max_batch_size": 8}, "job_id"),
        ({"job_id": "job-a", "actor_id": "", "revision": 1, "max_batch_size": 8}, "actor_id"),
        ({"job_id": "job-a", "actor_id": 123, "revision": 1, "max_batch_size": 8}, "actor_id"),
        ({"job_id": "job-a", "actor_id": "actor-a", "revision": 0, "max_batch_size": 8}, "revision"),
        ({"job_id": "job-a", "actor_id": "actor-a", "revision": True, "max_batch_size": 8}, "revision"),
        ({"job_id": "job-a", "actor_id": "actor-a", "revision": 1, "max_batch_size": 0}, "max_batch_size"),
        ({"job_id": "job-a", "actor_id": "actor-a", "revision": 1, "max_batch_size": True}, "max_batch_size"),
    ],
)
def test_quota_contract_rejects_invalid_fields(kwargs, message):
    with pytest.raises(ValueError, match=message):
        BatchSizeQuota(**kwargs)


def test_actor_applies_quota_as_an_exact_hard_upper_bound():
    actor = _actor()

    application = actor.apply_quota(BatchSizeQuota("job-a", "actor-a", revision=1, max_batch_size=7))
    result = actor(pyarrow.table({"value": list(range(20))}))

    assert application.applied is True
    assert application.previous_hard_limit == 32
    assert application.effective_max_batch_size == 7
    assert actor.controller.state.hard_limit == 7
    assert max(actor.operator.seen_batch_sizes) == 7
    assert result["value"].to_pylist() == list(range(20))


def test_stale_or_duplicate_revision_cannot_overwrite_newer_quota():
    actor = _actor()
    actor.apply_quota(BatchSizeQuota("job-a", "actor-a", revision=2, max_batch_size=8))

    stale = actor.apply_quota(BatchSizeQuota("job-a", "actor-a", revision=1, max_batch_size=20))
    duplicate = actor.apply_quota(BatchSizeQuota("job-a", "actor-a", revision=2, max_batch_size=24))

    assert stale.applied is False
    assert stale.reason == "stale_revision"
    assert duplicate.applied is False
    assert duplicate.reason == "stale_revision"
    assert actor.controller.state.hard_limit == 8
    assert actor.get_quota_state().last_revision == 2


def test_quota_rejects_wrong_job_or_actor_without_consuming_revision():
    actor = _actor()

    with pytest.raises(ValueError, match="job_id"):
        actor.apply_quota(BatchSizeQuota("job-b", "actor-a", revision=9, max_batch_size=8))
    with pytest.raises(ValueError, match="actor_id"):
        actor.apply_quota(BatchSizeQuota("job-a", "actor-b", revision=9, max_batch_size=8))

    applied = actor.apply_quota(BatchSizeQuota("job-a", "actor-a", revision=1, max_batch_size=8))
    assert applied.applied is True
    assert actor.get_quota_state().last_revision == 1


def test_quota_below_actor_minimum_is_rejected_atomically():
    actor = _actor(initial=8, minimum=4, maximum=32)

    with pytest.raises(ValueError, match="minimum"):
        actor.apply_quota(BatchSizeQuota("job-a", "actor-a", revision=1, max_batch_size=2))

    state = actor.get_quota_state()
    assert state.last_revision == 0
    assert state.hard_limit == 32
    assert state.current_batch_size == 8


def test_quota_above_static_maximum_is_safely_clamped():
    actor = _actor(maximum=32)

    application = actor.apply_quota(BatchSizeQuota("job-a", "actor-a", revision=1, max_batch_size=64))

    assert application.requested_max_batch_size == 64
    assert application.effective_max_batch_size == 32
    assert actor.controller.state.hard_limit == 32


def test_relaxed_quota_does_not_erase_actor_local_oom_state():
    actor = _actor()
    actor.controller.observe_oom(16)
    oom_state = actor.controller.state

    actor.apply_quota(BatchSizeQuota("job-a", "actor-a", revision=1, max_batch_size=6))
    actor.apply_quota(BatchSizeQuota("job-a", "actor-a", revision=2, max_batch_size=32))
    state = actor.get_quota_state()

    assert state.local_oom_upper_bound == oom_state.oom_upper_bound == 16
    assert state.hard_limit == 32
    assert actor.controller.next_batch_size(100) < 16


def test_quota_path_contains_no_magic_blending_or_ray_dependency():
    source = inspect.getsource(RayAdaptiveMapperActor.apply_quota)

    assert "0.7" not in source
    assert "0.3" not in source
    assert "ray.get" not in source

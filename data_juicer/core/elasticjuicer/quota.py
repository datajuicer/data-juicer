"""Explicit driver-to-actor batch-size quota contract."""

from dataclasses import dataclass
from typing import Optional


def _require_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _require_nonempty_string(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


@dataclass(frozen=True)
class BatchSizeQuota:
    """A versioned hard upper bound addressed to one actor in one job."""

    job_id: str
    actor_id: str
    revision: int
    max_batch_size: int

    def __post_init__(self):
        _require_nonempty_string("job_id", self.job_id)
        _require_nonempty_string("actor_id", self.actor_id)
        _require_positive_int("revision", self.revision)
        _require_positive_int("max_batch_size", self.max_batch_size)


@dataclass(frozen=True)
class QuotaApplication:
    """Diagnostic result of applying or ignoring one quota message."""

    applied: bool
    revision: int
    requested_max_batch_size: int
    effective_max_batch_size: int
    previous_hard_limit: int
    current_batch_size: int
    reason: str


@dataclass(frozen=True)
class ActorQuotaState:
    """Driver-readable quota state without exposing the mutable controller."""

    job_id: Optional[str]
    actor_id: str
    last_revision: int
    min_batch_size: int
    static_max_batch_size: int
    hard_limit: int
    current_batch_size: int
    local_success_lower_bound: Optional[int]
    local_oom_upper_bound: Optional[int]


def apply_batch_size_quota(
    controller,
    quota: BatchSizeQuota,
    *,
    expected_job_id: str,
    expected_actor_id: str,
    last_revision: int,
) -> QuotaApplication:
    """Apply a quota as a hard cap without modifying local learned bounds."""

    if not isinstance(quota, BatchSizeQuota):
        raise TypeError("quota must be a BatchSizeQuota")
    if quota.job_id != expected_job_id:
        raise ValueError(f"quota job_id {quota.job_id!r} does not match actor job_id {expected_job_id!r}")
    if quota.actor_id != expected_actor_id:
        raise ValueError(f"quota actor_id {quota.actor_id!r} does not match actor_id {expected_actor_id!r}")

    before = controller.state
    if quota.revision <= last_revision:
        return QuotaApplication(
            applied=False,
            revision=quota.revision,
            requested_max_batch_size=quota.max_batch_size,
            effective_max_batch_size=before.hard_limit,
            previous_hard_limit=before.hard_limit,
            current_batch_size=before.current_batch_size,
            reason="stale_revision",
        )

    effective_limit = min(quota.max_batch_size, before.max_batch_size)
    if effective_limit < before.min_batch_size:
        raise ValueError(f"quota max_batch_size {quota.max_batch_size} is below actor minimum {before.min_batch_size}")

    current_batch_size = controller.set_hard_limit(effective_limit)
    return QuotaApplication(
        applied=True,
        revision=quota.revision,
        requested_max_batch_size=quota.max_batch_size,
        effective_max_batch_size=effective_limit,
        previous_hard_limit=before.hard_limit,
        current_batch_size=current_batch_size,
        reason="applied",
    )

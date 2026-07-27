"""Versioned driver-to-actor batch-size constraint contracts."""

import time
from dataclasses import dataclass
from typing import Optional

QUOTA_SCHEMA_VERSION = 1


def _require_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _require_nonnegative_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _require_nonempty_string(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def current_time_ms() -> int:
    return time.time_ns() // 1_000_000


@dataclass(frozen=True)
class ActorRegistration:
    """Immutable identity and bounds for one constructed actor process."""

    job_id: str
    stage_id: str
    op_name: str
    actor_id: str
    actor_incarnation_id: str
    static_min_batch_size: int
    static_max_batch_size: int
    partition_id: Optional[int] = None
    schema_version: int = QUOTA_SCHEMA_VERSION

    def __post_init__(self):
        _require_nonempty_string("job_id", self.job_id)
        _require_nonempty_string("stage_id", self.stage_id)
        _require_nonempty_string("op_name", self.op_name)
        _require_nonempty_string("actor_id", self.actor_id)
        _require_nonempty_string("actor_incarnation_id", self.actor_incarnation_id)
        _require_positive_int("static_min_batch_size", self.static_min_batch_size)
        _require_positive_int("static_max_batch_size", self.static_max_batch_size)
        if self.static_max_batch_size < self.static_min_batch_size:
            raise ValueError("static_max_batch_size must be >= static_min_batch_size")
        if self.partition_id is not None:
            _require_nonnegative_int("partition_id", self.partition_id)
        if self.schema_version != QUOTA_SCHEMA_VERSION:
            raise ValueError(f"unsupported registration schema_version: {self.schema_version}")


@dataclass(frozen=True)
class QuotaEnvelope:
    """A fresh, versioned hard upper bound addressed to one actor incarnation."""

    job_id: str
    actor_id: str
    actor_incarnation_id: str
    revision: int
    issued_at_ms: int
    expires_at_ms: int
    max_batch_size: int
    capacity_recovery_hint: bool = False
    reason: str = "unspecified"
    schema_version: int = QUOTA_SCHEMA_VERSION

    def __post_init__(self):
        _require_nonempty_string("job_id", self.job_id)
        _require_nonempty_string("actor_id", self.actor_id)
        _require_nonempty_string("actor_incarnation_id", self.actor_incarnation_id)
        _require_positive_int("revision", self.revision)
        _require_nonnegative_int("issued_at_ms", self.issued_at_ms)
        _require_nonnegative_int("expires_at_ms", self.expires_at_ms)
        _require_positive_int("max_batch_size", self.max_batch_size)
        if self.expires_at_ms <= self.issued_at_ms:
            raise ValueError("expires_at_ms must be later than issued_at_ms")
        if not isinstance(self.capacity_recovery_hint, bool):
            raise TypeError("capacity_recovery_hint must be a boolean")
        _require_nonempty_string("reason", self.reason)
        if self.schema_version != QUOTA_SCHEMA_VERSION:
            raise ValueError(f"unsupported quota schema_version: {self.schema_version}")

    def is_expired(self, now_ms: Optional[int] = None) -> bool:
        resolved_now = current_time_ms() if now_ms is None else now_ms
        _require_nonnegative_int("now_ms", resolved_now)
        return resolved_now >= self.expires_at_ms


# Compatibility name for callers of the EJ-9 contract. The fields and
# semantics are intentionally the EJ-9b envelope; direct reset is gone.
BatchSizeQuota = QuotaEnvelope


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
    recovery_hint_recorded: bool = False


@dataclass(frozen=True)
class ActorQuotaState:
    """Driver-readable quota state without exposing the mutable controller."""

    job_id: Optional[str]
    actor_id: str
    actor_incarnation_id: str
    last_revision: int
    min_batch_size: int
    static_max_batch_size: int
    hard_limit: int
    current_batch_size: int
    local_success_lower_bound: Optional[int]
    local_oom_upper_bound: Optional[int]


def apply_batch_size_quota(
    controller,
    quota: QuotaEnvelope,
    *,
    expected_job_id: str,
    expected_actor_id: str,
    expected_actor_incarnation_id: str,
    last_revision: int,
    now_ms: Optional[int] = None,
) -> QuotaApplication:
    """Apply a fresh hard cap without deleting actor-local learned bounds."""

    if not isinstance(quota, QuotaEnvelope):
        raise TypeError("quota must be a QuotaEnvelope")
    if quota.job_id != expected_job_id:
        raise ValueError(f"quota job_id {quota.job_id!r} does not match actor job_id {expected_job_id!r}")
    if quota.actor_id != expected_actor_id:
        raise ValueError(f"quota actor_id {quota.actor_id!r} does not match actor_id {expected_actor_id!r}")
    if quota.actor_incarnation_id != expected_actor_incarnation_id:
        raise ValueError(
            "quota actor_incarnation_id "
            f"{quota.actor_incarnation_id!r} does not match actor incarnation {expected_actor_incarnation_id!r}"
        )

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
    resolved_now = current_time_ms() if now_ms is None else now_ms
    if quota.is_expired(resolved_now):
        return QuotaApplication(
            applied=False,
            revision=quota.revision,
            requested_max_batch_size=quota.max_batch_size,
            effective_max_batch_size=before.hard_limit,
            previous_hard_limit=before.hard_limit,
            current_batch_size=before.current_batch_size,
            reason="expired",
        )

    effective_limit = min(quota.max_batch_size, before.max_batch_size)
    if effective_limit < before.min_batch_size:
        raise ValueError(f"quota max_batch_size {quota.max_batch_size} is below actor minimum {before.min_batch_size}")

    current_batch_size = controller.set_hard_limit(effective_limit)
    hint_recorded = False
    if quota.capacity_recovery_hint:
        hint_recorded = controller.record_capacity_recovery_hint(quota.expires_at_ms, now_ms=resolved_now)
    return QuotaApplication(
        applied=True,
        revision=quota.revision,
        requested_max_batch_size=quota.max_batch_size,
        effective_max_batch_size=effective_limit,
        previous_hard_limit=before.hard_limit,
        current_batch_size=current_batch_size,
        reason="applied_with_recovery_hint" if quota.capacity_recovery_hint else "applied",
        recovery_hint_recorded=hint_recorded,
    )

"""Job-scoped quota delivery without Ray Data ActorPool private APIs."""

import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

from .quota import ActorRegistration, QuotaEnvelope, current_time_ms

_ActorKey = Tuple[str, str]

STAGE_PROFILE_SCHEMA_VERSION = 1
_DEFAULT_RESOURCE_CLASS = "default"


@dataclass(frozen=True)
class QuotaPublishResult:
    accepted: bool
    revision: int
    reason: str


@dataclass(frozen=True)
class StageProfile:
    """Advisory cross-incarnation learning summary for one topology stage.

    A profile is written by actor incarnations and read once by later
    incarnations of the same stable ``stage_id`` (for example a new partition's
    ActorPool under ``ray_partitioned``). It is a seed, never a command: the
    reader adopts it only as its initial prior and the actor-local controller
    remains the sole authority afterwards.

    Profiles are keyed by ``(stage_id, op_fingerprint, resource_class)`` so a
    changed operator configuration or a different device class never inherits
    stale bounds, and carry a schema version so incompatible writers are
    rejected instead of silently merged.
    """

    job_id: str
    stage_id: str
    op_name: str
    safe_batch_size: Optional[int]
    oom_upper_bound: Optional[int]
    observed_at_ms: int
    op_fingerprint: str = ""
    resource_class: str = _DEFAULT_RESOURCE_CLASS
    partition_id: Optional[int] = None
    schema_version: int = STAGE_PROFILE_SCHEMA_VERSION

    def __post_init__(self):
        for name in ("job_id", "stage_id", "op_name"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(self.op_fingerprint, str):
            raise ValueError("op_fingerprint must be a string")
        if not isinstance(self.resource_class, str) or not self.resource_class:
            raise ValueError("resource_class must be a non-empty string")
        if self.safe_batch_size is None and self.oom_upper_bound is None:
            raise ValueError("a stage profile must carry at least one learned bound")
        if self.safe_batch_size is not None and self.safe_batch_size < 1:
            raise ValueError("safe_batch_size must be positive")
        if self.oom_upper_bound is not None and self.oom_upper_bound < 1:
            raise ValueError("oom_upper_bound must be positive")
        if self.observed_at_ms < 0:
            raise ValueError("observed_at_ms must be non-negative")
        if self.partition_id is not None and (
            isinstance(self.partition_id, bool) or not isinstance(self.partition_id, int) or self.partition_id < 0
        ):
            raise ValueError("partition_id must be a non-negative integer or None")
        if self.schema_version != STAGE_PROFILE_SCHEMA_VERSION:
            raise ValueError(f"unsupported stage profile schema_version: {self.schema_version}")

    @property
    def store_key(self) -> Tuple[str, str, str]:
        return (self.stage_id, self.op_fingerprint, self.resource_class)


class ControlService:
    """Store registrations, leases, quotas, and stage profiles for one job."""

    def __init__(
        self,
        job_id: str,
        lease_ttl_ms: int = 60_000,
        profile_ttl_ms: int = 1_800_000,
        max_stage_profiles: int = 256,
    ):
        if not isinstance(job_id, str) or not job_id.strip():
            raise ValueError("job_id must be a non-empty string")
        if lease_ttl_ms < 1:
            raise ValueError("lease_ttl_ms must be positive")
        if profile_ttl_ms < 1:
            raise ValueError("profile_ttl_ms must be positive")
        if max_stage_profiles < 1:
            raise ValueError("max_stage_profiles must be at least 1")
        self.job_id = job_id
        self.lease_ttl_ms = lease_ttl_ms
        self.profile_ttl_ms = profile_ttl_ms
        self.max_stage_profiles = max_stage_profiles
        self._registrations: Dict[_ActorKey, ActorRegistration] = {}
        self._active_incarnations: Dict[str, str] = {}
        self._latest_quotas: Dict[_ActorKey, QuotaEnvelope] = {}
        self._leases: Dict[_ActorKey, int] = {}
        self._stage_profiles: Dict[Tuple[str, str, str], StageProfile] = {}
        self._registration_events = 0
        self._accepted_quotas = 0
        self._rejected_quotas = 0
        self._deregistrations = 0
        self._expired_profile_reads = 0

    @staticmethod
    def _key(actor_id: str, actor_incarnation_id: str) -> _ActorKey:
        return actor_id, actor_incarnation_id

    def _renew_lease(self, key: _ActorKey, now_ms: Optional[int]) -> None:
        self._leases[key] = current_time_ms() if now_ms is None else now_ms

    def _lease_alive(self, key: _ActorKey, now_ms: int) -> bool:
        renewed_at = self._leases.get(key)
        return renewed_at is not None and now_ms - renewed_at <= self.lease_ttl_ms

    def register(self, registration: ActorRegistration, now_ms: Optional[int] = None) -> ActorRegistration:
        if not isinstance(registration, ActorRegistration):
            raise TypeError("registration must be an ActorRegistration")
        if registration.job_id != self.job_id:
            raise ValueError(
                f"registration job_id {registration.job_id!r} does not match service job_id {self.job_id!r}"
            )
        key = self._key(registration.actor_id, registration.actor_incarnation_id)
        existing = self._registrations.get(key)
        if existing is not None and existing != registration:
            raise ValueError("actor incarnation is already registered with different immutable metadata")
        is_new_incarnation = existing is None
        self._registrations[key] = registration
        self._renew_lease(key, now_ms)
        active_incarnation = self._active_incarnations.get(registration.actor_id)
        if active_incarnation is None or (
            is_new_incarnation and active_incarnation != registration.actor_incarnation_id
        ):
            self._active_incarnations[registration.actor_id] = registration.actor_incarnation_id
        self._registration_events += 1
        return registration

    def deregister(self, job_id: str, actor_id: str, actor_incarnation_id: str) -> bool:
        """Explicitly retire one actor incarnation from quota coordination."""

        if job_id != self.job_id:
            raise ValueError(f"deregister job_id {job_id!r} does not match service job_id {self.job_id!r}")
        key = self._key(actor_id, actor_incarnation_id)
        removed = self._leases.pop(key, None) is not None
        if self._active_incarnations.get(actor_id) == actor_incarnation_id:
            self._active_incarnations.pop(actor_id, None)
        if removed:
            self._deregistrations += 1
        return removed

    def publish_quota(self, quota: QuotaEnvelope, now_ms: Optional[int] = None) -> QuotaPublishResult:
        if not isinstance(quota, QuotaEnvelope):
            raise TypeError("quota must be a QuotaEnvelope")
        if quota.job_id != self.job_id:
            raise ValueError(f"quota job_id {quota.job_id!r} does not match service job_id {self.job_id!r}")
        key = self._key(quota.actor_id, quota.actor_incarnation_id)
        registration = self._registrations.get(key)
        if registration is None:
            raise ValueError("quota target actor incarnation is not registered")
        if quota.max_batch_size < registration.static_min_batch_size:
            raise ValueError(
                f"quota max_batch_size {quota.max_batch_size} is below actor minimum "
                f"{registration.static_min_batch_size}"
            )
        if quota.is_expired(now_ms):
            self._rejected_quotas += 1
            return QuotaPublishResult(False, quota.revision, "expired")
        current = self._latest_quotas.get(key)
        if current is not None and quota.revision <= current.revision:
            self._rejected_quotas += 1
            return QuotaPublishResult(False, quota.revision, "stale_revision")
        self._latest_quotas[key] = quota
        self._accepted_quotas += 1
        return QuotaPublishResult(True, quota.revision, "accepted")

    def get_latest(
        self,
        job_id: str,
        actor_id: str,
        actor_incarnation_id: str,
        after_revision: int = 0,
        now_ms: Optional[int] = None,
    ) -> Optional[QuotaEnvelope]:
        if job_id != self.job_id:
            raise ValueError(f"poll job_id {job_id!r} does not match service job_id {self.job_id!r}")
        if isinstance(after_revision, bool) or not isinstance(after_revision, int) or after_revision < 0:
            raise ValueError("after_revision must be a non-negative integer")
        key = self._key(actor_id, actor_incarnation_id)
        if key not in self._registrations:
            raise ValueError("polling actor incarnation is not registered")
        # Every poll doubles as a lease heartbeat so live actors never expire.
        self._renew_lease(key, now_ms)
        quota = self._latest_quotas.get(key)
        if quota is None or quota.revision <= after_revision:
            return None
        return quota

    def report_stage_profile(self, profile: StageProfile) -> StageProfile:
        """Merge one incarnation's learning into the job-scoped stage profile."""

        if not isinstance(profile, StageProfile):
            raise TypeError("profile must be a StageProfile")
        if profile.job_id != self.job_id:
            raise ValueError(f"profile job_id {profile.job_id!r} does not match service job_id {self.job_id!r}")
        if profile.schema_version != STAGE_PROFILE_SCHEMA_VERSION:
            # Re-check after Ray deserialization: pickle does not rerun
            # __post_init__, so incompatible writers must be rejected here.
            raise ValueError(f"unsupported stage profile schema_version: {profile.schema_version}")
        key = profile.store_key
        merged = self._merge_stage_profile(self._stage_profiles.get(key), profile)
        self._stage_profiles[key] = merged
        if len(self._stage_profiles) > self.max_stage_profiles:
            oldest_key = min(self._stage_profiles, key=lambda k: self._stage_profiles[k].observed_at_ms)
            del self._stage_profiles[oldest_key]
        return merged

    @staticmethod
    def _merge_stage_profile(current: Optional[StageProfile], update: StageProfile) -> StageProfile:
        """Keep the tightest OOM evidence and the best proven success size."""

        if current is None:
            return update
        oom_bounds = [bound for bound in (current.oom_upper_bound, update.oom_upper_bound) if bound is not None]
        oom_upper_bound = min(oom_bounds) if oom_bounds else None
        safe_sizes = [size for size in (current.safe_batch_size, update.safe_batch_size) if size is not None]
        safe_batch_size = max(safe_sizes) if safe_sizes else None
        if safe_batch_size is not None and oom_upper_bound is not None:
            safe_batch_size = min(safe_batch_size, max(1, oom_upper_bound - 1))
        return StageProfile(
            job_id=update.job_id,
            stage_id=update.stage_id,
            op_name=update.op_name,
            safe_batch_size=safe_batch_size,
            oom_upper_bound=oom_upper_bound,
            observed_at_ms=max(current.observed_at_ms, update.observed_at_ms),
            op_fingerprint=update.op_fingerprint,
            resource_class=update.resource_class,
            partition_id=update.partition_id,
        )

    def get_stage_profile(
        self,
        job_id: str,
        stage_id: str,
        op_fingerprint: str = "",
        resource_class: str = _DEFAULT_RESOURCE_CLASS,
        now_ms: Optional[int] = None,
    ) -> Optional[StageProfile]:
        if job_id != self.job_id:
            raise ValueError(f"profile job_id {job_id!r} does not match service job_id {self.job_id!r}")
        key = (stage_id, op_fingerprint, resource_class)
        profile = self._stage_profiles.get(key)
        if profile is None:
            return None
        now = current_time_ms() if now_ms is None else now_ms
        if now - profile.observed_at_ms > self.profile_ttl_ms:
            del self._stage_profiles[key]
            self._expired_profile_reads += 1
            return None
        return profile

    def snapshot(self, now_ms: Optional[int] = None):
        now = current_time_ms() if now_ms is None else now_ms
        active_registrations = [
            registration
            for (actor_id, incarnation_id), registration in self._registrations.items()
            if self._active_incarnations.get(actor_id) == incarnation_id
            and self._lease_alive((actor_id, incarnation_id), now)
        ]
        return {
            "job_id": self.job_id,
            "registrations": list(self._registrations.values()),
            "active_registrations": active_registrations,
            "latest_quotas": list(self._latest_quotas.values()),
            "stage_profiles": list(self._stage_profiles.values()),
            "registration_events": self._registration_events,
            "accepted_quotas": self._accepted_quotas,
            "rejected_quotas": self._rejected_quotas,
            "deregistrations": self._deregistrations,
            "expired_profile_reads": self._expired_profile_reads,
            "lease_ttl_ms": self.lease_ttl_ms,
        }


class ActorControlPoller:
    """Non-blocking control state pump advanced only at batch boundaries."""

    def __init__(
        self,
        control_handle,
        registration: ActorRegistration,
        poll_interval_sec: float = 0.1,
        get_fn: Optional[Callable] = None,
        wait_fn: Optional[Callable] = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        if control_handle is None:
            raise ValueError("control_handle must not be None")
        if poll_interval_sec <= 0:
            raise ValueError("poll_interval_sec must be positive")
        self.control_handle = control_handle
        self.registration = registration
        self.poll_interval_sec = poll_interval_sec
        self._get_fn = get_fn
        self._wait_fn = wait_fn
        self._clock = clock
        self._pending_quota: Optional[QuotaEnvelope] = None
        self._last_seen_revision = 0
        self._registered = False
        self._poll_errors = 0
        self._last_error: Optional[str] = None
        self._registration_reference = None
        self._poll_reference = None
        self._next_request_at = 0.0

    def start(self) -> None:
        """Submit registration without waiting for the service response."""

        if self._registration_reference is None and not self._registered:
            self._submit_registration()

    def close(self) -> None:
        """Drop local references; there is no actor-lifetime worker thread."""

        self._registration_reference = None
        self._poll_reference = None
        self._pending_quota = None

    def poll_once(self) -> None:
        """Advance ready requests without waiting or blocking the data path."""

        now = self._clock()
        if self._registration_reference is not None and self._is_ready(self._registration_reference):
            reference = self._registration_reference
            self._registration_reference = None
            try:
                self._resolve(reference)
            except Exception as error:
                self._record_error(error)
            else:
                self._registered = True
                self._last_error = None

        if self._poll_reference is not None and self._is_ready(self._poll_reference):
            reference = self._poll_reference
            self._poll_reference = None
            try:
                quota = self._resolve(reference)
            except Exception as error:
                self._registered = False
                self._record_error(error)
            else:
                if quota is not None and quota.revision > self._last_seen_revision:
                    self._pending_quota = quota
                    self._last_seen_revision = quota.revision
                self._last_error = None
            self._next_request_at = now + self.poll_interval_sec

        if now < self._next_request_at:
            return
        if not self._registered and self._registration_reference is None:
            self._submit_registration()
        elif self._registered and self._poll_reference is None:
            try:
                self._poll_reference = self._remote(
                    self.control_handle.get_latest,
                    self.registration.job_id,
                    self.registration.actor_id,
                    self.registration.actor_incarnation_id,
                    self._last_seen_revision,
                )
            except Exception as error:
                self._registered = False
                self._record_error(error)
            self._next_request_at = now + self.poll_interval_sec

    def take_pending(self) -> Optional[QuotaEnvelope]:
        quota = self._pending_quota
        self._pending_quota = None
        return quota

    def snapshot(self):
        pending_revision = None if self._pending_quota is None else self._pending_quota.revision
        return {
            "enabled": True,
            "registered": self._registered,
            "last_seen_revision": self._last_seen_revision,
            "pending_revision": pending_revision,
            "poll_errors": self._poll_errors,
            "last_error": self._last_error,
            "registration_in_flight": self._registration_reference is not None,
            "poll_in_flight": self._poll_reference is not None,
        }

    def _resolve(self, reference):
        if self._get_fn is not None:
            return self._get_fn(reference)
        import ray

        return ray.get(reference)

    def _is_ready(self, reference) -> bool:
        wait_fn = self._wait_fn
        if wait_fn is None:
            import ray

            wait_fn = ray.wait
        ready, _ = wait_fn([reference], num_returns=1, timeout=0)
        return bool(ready)

    @staticmethod
    def _remote(method, *args):
        remote = getattr(method, "remote", None)
        if not callable(remote):
            raise TypeError("control service methods must expose .remote")
        return remote(*args)

    def _submit_registration(self) -> None:
        try:
            self._registration_reference = self._remote(self.control_handle.register, self.registration)
        except Exception as error:
            self._record_error(error)
            self._next_request_at = self._clock() + self.poll_interval_sec

    def _record_error(self, error: Exception) -> None:
        self._poll_errors += 1
        self._last_error = f"{type(error).__name__}: {error}"


def create_ray_control_service(
    job_id: str,
    lease_ttl_ms: int = 60_000,
    profile_ttl_ms: int = 1_800_000,
    ray_module=None,
):
    """Create an unnamed explicit Ray actor handle for one job."""

    if ray_module is None:
        import ray as ray_module

    remote_class = ray_module.remote(num_cpus=0)(ControlService)
    return remote_class.remote(job_id=job_id, lease_ttl_ms=lease_ttl_ms, profile_ttl_ms=profile_ttl_ms)

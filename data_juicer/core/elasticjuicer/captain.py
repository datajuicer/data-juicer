"""Deterministic single-stage Captain-lite decision and delivery runtime."""

import math
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .async_metrics_sink import ActorMetricsEvent
from .quota import (
    QUOTA_SCHEMA_VERSION,
    ActorRegistration,
    QuotaEnvelope,
    current_time_ms,
)

CAPTAIN_OBSERVATION_SCHEMA_VERSION = 1
_ActorKey = Tuple[str, str, str]


@dataclass(frozen=True)
class ActorObservation:
    job_id: str
    stage_id: str
    op_name: str
    actor_id: str
    actor_incarnation_id: str
    sequence: int
    observed_at_ms: int
    quota_revision: int
    current_batch_size: int
    hard_limit: int
    static_min_batch_size: int
    static_max_batch_size: int
    local_success_lower_bound: Optional[int]
    local_oom_upper_bound: Optional[int]
    rss_peak_mb: Optional[float]
    cuda_reserved_mb: Optional[float]
    cuda_peak_allocated_mb: Optional[float]
    throughput: float
    latency_ms: float
    confidence: float
    telemetry_loss: bool = False
    schema_version: int = CAPTAIN_OBSERVATION_SCHEMA_VERSION

    def __post_init__(self):
        for name in ("job_id", "stage_id", "op_name", "actor_id", "actor_incarnation_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty string")
        if self.schema_version != CAPTAIN_OBSERVATION_SCHEMA_VERSION:
            raise ValueError(f"unsupported observation schema_version: {self.schema_version}")
        if self.sequence < 1 or self.observed_at_ms < 0 or self.quota_revision < 0:
            raise ValueError("sequence must be positive and timestamps/revisions non-negative")
        if self.static_min_batch_size < 1 or self.static_max_batch_size < self.static_min_batch_size:
            raise ValueError("invalid static batch bounds")
        if not self.static_min_batch_size <= self.current_batch_size <= self.static_max_batch_size:
            raise ValueError("current_batch_size must be within static bounds")
        if not self.static_min_batch_size <= self.hard_limit <= self.static_max_batch_size:
            raise ValueError("hard_limit must be within static bounds")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be in [0, 1]")

    @classmethod
    def from_metrics_event(cls, event: ActorMetricsEvent, telemetry_loss: bool = False):
        if event.control is None:
            raise ValueError("metrics event has no actor control state")
        cuda = event.snapshot.cuda
        return cls(
            job_id=event.job_id,
            stage_id=event.stage_id,
            op_name=event.op_name,
            actor_id=event.actor_id,
            actor_incarnation_id=event.actor_incarnation_id,
            sequence=event.sequence,
            observed_at_ms=event.observed_at_ms,
            quota_revision=event.control.quota_revision,
            current_batch_size=event.control.current_batch_size,
            hard_limit=event.control.hard_limit,
            static_min_batch_size=event.control.static_min_batch_size,
            static_max_batch_size=event.control.static_max_batch_size,
            local_success_lower_bound=event.control.local_success_lower_bound,
            local_oom_upper_bound=event.control.local_oom_upper_bound,
            rss_peak_mb=event.snapshot.rss_peak_mb,
            cuda_reserved_mb=None if cuda is None else cuda.reserved_mb,
            cuda_peak_allocated_mb=None if cuda is None else cuda.peak_allocated_mb,
            throughput=event.snapshot.throughput,
            latency_ms=event.snapshot.latency_ms,
            confidence=event.minimum_confidence,
            telemetry_loss=telemetry_loss,
        )


@dataclass(frozen=True)
class QuotaDecision:
    job_id: str
    stage_id: str
    actor_id: str
    actor_incarnation_id: str
    revision: int
    max_batch_size: int
    capacity_recovery_hint: bool
    reason: str
    evidence_observed_at_ms: int
    evidence_sequence: int

    def to_envelope(self, issued_at_ms: int, ttl_ms: int) -> QuotaEnvelope:
        if ttl_ms < 1:
            raise ValueError("ttl_ms must be positive")
        return QuotaEnvelope(
            schema_version=QUOTA_SCHEMA_VERSION,
            job_id=self.job_id,
            actor_id=self.actor_id,
            actor_incarnation_id=self.actor_incarnation_id,
            revision=self.revision,
            issued_at_ms=issued_at_ms,
            expires_at_ms=issued_at_ms + ttl_ms,
            max_batch_size=self.max_batch_size,
            capacity_recovery_hint=self.capacity_recovery_hint,
            reason=self.reason,
        )


@dataclass
class _DecisionState:
    last_sequence: int = 0
    last_revision: int = 0
    last_decision_at_ms: Optional[int] = None
    recovery_streak: int = 0


class PerActorQuotaPolicy:
    """Pure Python safety-first policy for one actor within a stage cycle."""

    def __init__(
        self,
        metrics_ttl_ms: int = 5_000,
        min_confidence: float = 0.8,
        min_decision_interval_ms: int = 1_000,
        shrink_factor: float = 0.5,
        growth_factor: float = 1.25,
        recovery_observations: int = 3,
        process_memory_high_mb: Optional[float] = None,
        process_memory_low_mb: Optional[float] = None,
        cuda_memory_high_mb: Optional[float] = None,
        cuda_memory_low_mb: Optional[float] = None,
    ):
        if metrics_ttl_ms < 1 or min_decision_interval_ms < 0 or recovery_observations < 1:
            raise ValueError("invalid Captain timing configuration")
        if not 0 <= min_confidence <= 1:
            raise ValueError("min_confidence must be in [0, 1]")
        if not 0 < shrink_factor < 1 or growth_factor <= 1:
            raise ValueError("invalid Captain growth/shrink factors")
        if process_memory_high_mb is not None and process_memory_low_mb is not None:
            if process_memory_low_mb >= process_memory_high_mb:
                raise ValueError("process memory low watermark must be below high watermark")
        if cuda_memory_high_mb is not None and cuda_memory_low_mb is not None:
            if cuda_memory_low_mb >= cuda_memory_high_mb:
                raise ValueError("CUDA memory low watermark must be below high watermark")
        self.metrics_ttl_ms = metrics_ttl_ms
        self.min_confidence = min_confidence
        self.min_decision_interval_ms = min_decision_interval_ms
        self.shrink_factor = shrink_factor
        self.growth_factor = growth_factor
        self.recovery_observations = recovery_observations
        self.process_memory_high_mb = process_memory_high_mb
        self.process_memory_low_mb = process_memory_low_mb
        self.cuda_memory_high_mb = cuda_memory_high_mb
        self.cuda_memory_low_mb = cuda_memory_low_mb
        self._states: Dict[_ActorKey, _DecisionState] = {}

    @staticmethod
    def _key(observation: ActorObservation) -> _ActorKey:
        return observation.job_id, observation.actor_id, observation.actor_incarnation_id

    def seed_revision(self, job_id: str, actor_id: str, actor_incarnation_id: str, revision: int) -> None:
        if revision < 0:
            raise ValueError("revision must be non-negative")
        key = (job_id, actor_id, actor_incarnation_id)
        state = self._states.setdefault(key, _DecisionState())
        state.last_revision = max(state.last_revision, revision)

    def observe(
        self,
        observation: ActorObservation,
        now_ms: int,
        *,
        allow_recovery: bool = True,
    ) -> Optional[QuotaDecision]:
        if not isinstance(observation, ActorObservation):
            raise TypeError("observation must be an ActorObservation")
        if now_ms < 0:
            raise ValueError("now_ms must be non-negative")
        key = self._key(observation)
        state = self._states.setdefault(key, _DecisionState())
        state.last_revision = max(state.last_revision, observation.quota_revision)
        if observation.sequence <= state.last_sequence:
            return None
        sequence_gap = observation.sequence != state.last_sequence + 1
        state.last_sequence = observation.sequence

        reliable = self.is_reliable(observation, now_ms) and not sequence_gap

        target = observation.hard_limit
        reason = None
        recovery_hint = False

        if observation.local_oom_upper_bound is not None:
            safe_oom_cap = max(observation.static_min_batch_size, observation.local_oom_upper_bound - 1)
            if safe_oom_cap < target:
                target = safe_oom_cap
                reason = "local_oom_bound"

        if reliable and self.is_high_pressure(observation):
            pressure_cap = max(
                observation.static_min_batch_size,
                int(math.floor(observation.current_batch_size * self.shrink_factor)),
            )
            if pressure_cap < target:
                target = pressure_cap
                reason = "memory_high_watermark"
            state.recovery_streak = 0
        elif reliable and allow_recovery and self.is_low_pressure(observation):
            state.recovery_streak += 1
        else:
            state.recovery_streak = 0

        urgent_shrink = target < observation.hard_limit
        interval_ready = (
            state.last_decision_at_ms is None or now_ms - state.last_decision_at_ms >= self.min_decision_interval_ms
        )
        if urgent_shrink and (reason == "local_oom_bound" or interval_ready):
            state.recovery_streak = 0
            return self._decision(observation, state, target, reason or "safety_shrink", False, now_ms)

        if (
            reliable
            and interval_ready
            and state.recovery_streak >= self.recovery_observations
            and observation.hard_limit < observation.static_max_batch_size
        ):
            target = min(
                observation.static_max_batch_size,
                max(observation.hard_limit + 1, int(math.ceil(observation.hard_limit * self.growth_factor))),
            )
            recovery_hint = observation.local_oom_upper_bound is not None
            state.recovery_streak = 0
            return self._decision(observation, state, target, "stable_low_pressure_recovery", recovery_hint, now_ms)

        return None

    def _decision(
        self,
        observation: ActorObservation,
        state: _DecisionState,
        target: int,
        reason: str,
        recovery_hint: bool,
        now_ms: int,
    ) -> QuotaDecision:
        target = max(observation.static_min_batch_size, min(target, observation.static_max_batch_size))
        state.last_revision += 1
        state.last_decision_at_ms = now_ms
        return QuotaDecision(
            job_id=observation.job_id,
            stage_id=observation.stage_id,
            actor_id=observation.actor_id,
            actor_incarnation_id=observation.actor_incarnation_id,
            revision=state.last_revision,
            max_batch_size=target,
            capacity_recovery_hint=recovery_hint,
            reason=reason,
            evidence_observed_at_ms=observation.observed_at_ms,
            evidence_sequence=observation.sequence,
        )

    def is_reliable(self, observation: ActorObservation, now_ms: int) -> bool:
        age_ms = now_ms - observation.observed_at_ms
        return (
            0 <= age_ms <= self.metrics_ttl_ms
            and observation.confidence >= self.min_confidence
            and not observation.telemetry_loss
        )

    def has_sequence_gap(self, observation: ActorObservation) -> bool:
        state = self._states.get(self._key(observation))
        last_sequence = 0 if state is None else state.last_sequence
        return observation.sequence != last_sequence + 1

    def is_high_pressure(self, observation: ActorObservation) -> bool:
        process_high = (
            self.process_memory_high_mb is not None
            and observation.rss_peak_mb is not None
            and observation.rss_peak_mb >= self.process_memory_high_mb
        )
        cuda_values = [
            value for value in (observation.cuda_reserved_mb, observation.cuda_peak_allocated_mb) if value is not None
        ]
        cuda_high = (
            self.cuda_memory_high_mb is not None and bool(cuda_values) and max(cuda_values) >= self.cuda_memory_high_mb
        )
        return process_high or cuda_high

    def is_low_pressure(self, observation: ActorObservation) -> bool:
        checks = []
        if self.process_memory_low_mb is not None:
            if observation.rss_peak_mb is None:
                return False
            checks.append(observation.rss_peak_mb <= self.process_memory_low_mb)
        if self.cuda_memory_low_mb is not None:
            cuda_values = [
                value
                for value in (observation.cuda_reserved_mb, observation.cuda_peak_allocated_mb)
                if value is not None
            ]
            if not cuda_values:
                return False
            checks.append(max(cuda_values) <= self.cuda_memory_low_mb)
        return bool(checks) and all(checks)


@dataclass(frozen=True)
class StageSnapshot:
    """One decision-cycle view of the active actors in a topology stage."""

    job_id: str
    stage_id: str
    captured_at_ms: int
    registrations: Tuple[ActorRegistration, ...]
    observations: Tuple[ActorObservation, ...]

    def __post_init__(self):
        if not self.job_id or not self.stage_id:
            raise ValueError("stage snapshot identity must not be empty")
        if self.captured_at_ms < 0:
            raise ValueError("captured_at_ms must be non-negative")
        identities = set()
        for registration in self.registrations:
            if registration.job_id != self.job_id or registration.stage_id != self.stage_id:
                raise ValueError("registration does not belong to stage snapshot")
            identity = (registration.actor_id, registration.actor_incarnation_id)
            if identity in identities:
                raise ValueError("stage snapshot contains a duplicate actor incarnation")
            identities.add(identity)
        for observation in self.observations:
            if observation.job_id != self.job_id or observation.stage_id != self.stage_id:
                raise ValueError("observation does not belong to stage snapshot")


class StageQuotaCoordinator:
    """Coordinate per-actor caps from one complete stage decision window.

    Safety shrink remains actor-local. Recovery/expansion is allowed only when
    every active actor incarnation has a fresh, reliable, low-pressure
    observation in the same stage snapshot. This is the MVP fairness rule for
    heterogeneous actors: partial telemetry cannot make the observed subset
    consume more of the stage's capacity.
    """

    def __init__(self, actor_policy: PerActorQuotaPolicy):
        if not isinstance(actor_policy, PerActorQuotaPolicy):
            raise TypeError("actor_policy must be a PerActorQuotaPolicy")
        self.actor_policy = actor_policy

    def decide(self, snapshot: StageSnapshot) -> List[QuotaDecision]:
        active = {
            (registration.actor_id, registration.actor_incarnation_id): registration
            for registration in snapshot.registrations
        }
        latest = {}
        for observation in snapshot.observations:
            identity = (observation.actor_id, observation.actor_incarnation_id)
            if identity not in active:
                continue
            previous = latest.get(identity)
            if previous is None or observation.sequence > previous.sequence:
                latest[identity] = observation

        stage_can_recover = bool(active) and len(latest) == len(active)
        if stage_can_recover:
            stage_can_recover = all(
                self.actor_policy.is_reliable(observation, snapshot.captured_at_ms)
                and not self.actor_policy.has_sequence_gap(observation)
                and self.actor_policy.is_low_pressure(observation)
                for observation in latest.values()
            )

        decisions = []
        for identity in sorted(active):
            observation = latest.get(identity)
            if observation is None:
                continue
            decision = self.actor_policy.observe(
                observation,
                snapshot.captured_at_ms,
                allow_recovery=stage_can_recover,
            )
            if decision is not None:
                decisions.append(decision)
        return decisions


# Compatibility name retained while callers migrate to the explicit layering.
CaptainDecisionCore = PerActorQuotaPolicy


class CaptainRuntime:
    """Driver-side stage coordinator with bounded RPC and delivery retry."""

    def __init__(
        self,
        decision_core: PerActorQuotaPolicy,
        metrics_sink_handle,
        control_service_handle,
        quota_ttl_ms: int = 10_000,
        rpc_timeout_sec: float = 5.0,
        retry_backoff_sec: float = 0.1,
        get_fn=None,
        clock=time.monotonic,
    ):
        if quota_ttl_ms < 1 or rpc_timeout_sec <= 0 or retry_backoff_sec < 0:
            raise ValueError("Captain TTL/timeout must be positive and backoff non-negative")
        if not isinstance(decision_core, PerActorQuotaPolicy):
            raise TypeError("decision_core must be a PerActorQuotaPolicy")
        self.decision_core = decision_core
        self.stage_coordinator = StageQuotaCoordinator(decision_core)
        self.metrics_sink_handle = metrics_sink_handle
        self.control_service_handle = control_service_handle
        self.quota_ttl_ms = quota_ttl_ms
        self.rpc_timeout_sec = rpc_timeout_sec
        self.retry_backoff_sec = retry_backoff_sec
        self._get_fn = get_fn
        self._clock = clock
        self._bootstrapped = False
        self._last_sink_drops = 0
        self._pending: Dict[_ActorKey, QuotaEnvelope] = {}
        self._next_control_attempt_at = 0.0
        self._next_sink_attempt_at = 0.0
        self._rpc_failures = 0
        self._last_control_error: Optional[str] = None
        self._last_sink_error: Optional[str] = None

    def _resolve(self, reference):
        if self._get_fn is not None:
            try:
                return self._get_fn(reference, timeout=self.rpc_timeout_sec)
            except TypeError:
                return self._get_fn(reference)
        import ray

        return ray.get(reference, timeout=self.rpc_timeout_sec)

    @staticmethod
    def _remote(method, *args):
        remote = getattr(method, "remote", None)
        if not callable(remote):
            raise TypeError("runtime handles must expose .remote methods")
        return remote(*args)

    def poll_once(self, now_ms: Optional[int] = None) -> List[QuotaDecision]:
        resolved_now = current_time_ms() if now_ms is None else now_ms
        control_snapshot = self._fetch_control_snapshot()
        if control_snapshot is None:
            return []
        self._bootstrap_revisions(control_snapshot)
        self._retry_pending(resolved_now)
        if self._clock() < self._next_sink_attempt_at:
            return []
        try:
            sink_snapshot = self._resolve(self._remote(self.metrics_sink_handle.snapshot))
        except Exception as error:
            self._record_rpc_failure(error, "sink")
            return []
        self._record_rpc_success("sink")

        dropped = int(sink_snapshot.get("dropped_events", 0))
        transport_loss = dropped > self._last_sink_drops
        self._last_sink_drops = max(self._last_sink_drops, dropped)
        latest_observations = {}
        for event in sink_snapshot.get("events", []):
            try:
                observation = ActorObservation.from_metrics_event(event, telemetry_loss=transport_loss)
            except (TypeError, ValueError):
                continue
            key = (observation.job_id, observation.actor_id, observation.actor_incarnation_id)
            if key in self._pending:
                continue
            previous = latest_observations.get(key)
            if previous is None or observation.sequence > previous.sequence:
                latest_observations[key] = observation

        registrations = control_snapshot.get("active_registrations")
        if registrations is None:
            registrations = control_snapshot.get("registrations", [])
        stage_registrations = {}
        for registration in registrations:
            stage_key = (registration.job_id, registration.stage_id)
            stage_registrations.setdefault(stage_key, []).append(registration)

        decisions = []
        for (job_id, stage_id), members in sorted(stage_registrations.items()):
            member_keys = {(member.job_id, member.actor_id, member.actor_incarnation_id) for member in members}
            observations = tuple(observation for key, observation in latest_observations.items() if key in member_keys)
            stage_snapshot = StageSnapshot(
                job_id=job_id,
                stage_id=stage_id,
                captured_at_ms=resolved_now,
                registrations=tuple(members),
                observations=observations,
            )
            decisions.extend(self.stage_coordinator.decide(stage_snapshot))

        for decision in decisions:
            key = (decision.job_id, decision.actor_id, decision.actor_incarnation_id)
            envelope = decision.to_envelope(resolved_now, self.quota_ttl_ms)
            if not self._publish(envelope):
                self._pending[key] = envelope
        return decisions

    def _fetch_control_snapshot(self):
        if self._clock() < self._next_control_attempt_at:
            return None
        try:
            snapshot = self._resolve(self._remote(self.control_service_handle.snapshot))
        except Exception as error:
            self._record_rpc_failure(error, "control")
            return None
        self._record_rpc_success("control")
        return snapshot

    def _bootstrap_revisions(self, snapshot) -> None:
        for quota in snapshot.get("latest_quotas", []):
            self.decision_core.seed_revision(
                quota.job_id,
                quota.actor_id,
                quota.actor_incarnation_id,
                quota.revision,
            )
        self._bootstrapped = True

    def _publish(self, envelope: QuotaEnvelope) -> bool:
        if self._clock() < self._next_control_attempt_at:
            return False
        try:
            result = self._resolve(self._remote(self.control_service_handle.publish_quota, envelope))
        except Exception as error:
            self._record_rpc_failure(error, "control")
            return False
        self._record_rpc_success("control")
        return bool(result.accepted or result.reason == "stale_revision")

    def _record_rpc_failure(self, error: Exception, channel: str) -> None:
        self._rpc_failures += 1
        message = f"{type(error).__name__}: {error}"
        next_attempt = self._clock() + self.retry_backoff_sec
        if channel == "control":
            self._last_control_error = message
            self._next_control_attempt_at = next_attempt
        else:
            self._last_sink_error = message
            self._next_sink_attempt_at = next_attempt

    def _record_rpc_success(self, channel: str) -> None:
        if channel == "control":
            self._last_control_error = None
            self._next_control_attempt_at = 0.0
        else:
            self._last_sink_error = None
            self._next_sink_attempt_at = 0.0

    def _retry_pending(self, now_ms: int) -> None:
        for key, envelope in list(self._pending.items()):
            if envelope.is_expired(now_ms):
                self._pending.pop(key, None)
                continue
            if self._publish(envelope):
                self._pending.pop(key, None)

    def snapshot(self):
        return {
            "bootstrapped": self._bootstrapped,
            "pending_deliveries": len(self._pending),
            "last_sink_drops": self._last_sink_drops,
            "rpc_failures": self._rpc_failures,
            "last_rpc_error": self._last_control_error or self._last_sink_error,
            "last_control_error": self._last_control_error,
            "last_sink_error": self._last_sink_error,
            "backing_off": self._clock() < max(self._next_control_attempt_at, self._next_sink_attempt_at),
        }


class CaptainLifecycle:
    """Own the product driver polling thread for one job-scoped Captain."""

    def __init__(self, runtime: CaptainRuntime, poll_interval_sec: float = 0.5):
        if not isinstance(runtime, CaptainRuntime):
            raise TypeError("runtime must be a CaptainRuntime")
        if poll_interval_sec <= 0:
            raise ValueError("poll_interval_sec must be positive")
        self.runtime = runtime
        self.poll_interval_sec = poll_interval_sec
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._polls = 0
        self._loop_errors = 0
        self._last_error: Optional[str] = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="elasticjuicer-captain",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(1.0, self.poll_interval_sec * 2, self.runtime.rpc_timeout_sec + 1.0))

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.runtime.poll_once()
            except Exception as error:
                self._loop_errors += 1
                self._last_error = f"{type(error).__name__}: {error}"
            else:
                self._polls += 1
                self._last_error = None
            self._stop_event.wait(self.poll_interval_sec)

    def snapshot(self):
        return {
            "running": self._thread is not None and self._thread.is_alive(),
            "polls": self._polls,
            "loop_errors": self._loop_errors,
            "last_error": self._last_error,
            "runtime": self.runtime.snapshot(),
        }


def create_captain_lifecycle(metrics_sink_handle, control_service_handle, cfg) -> CaptainLifecycle:
    """Build the explicitly enabled product Captain from user configuration."""

    def value(name, default=None):
        getter = getattr(cfg, "get", None)
        if callable(getter):
            return getter(name, default)
        return getattr(cfg, name, default)

    process_high = value("elastic_juicer_captain_process_memory_high_mb")
    process_low = value("elastic_juicer_captain_process_memory_low_mb")
    cuda_high = value("elastic_juicer_captain_cuda_memory_high_mb")
    cuda_low = value("elastic_juicer_captain_cuda_memory_low_mb")
    if (process_high is None) != (process_low is None):
        raise ValueError("Captain process memory high/low watermarks must be configured together")
    if (cuda_high is None) != (cuda_low is None):
        raise ValueError("Captain CUDA memory high/low watermarks must be configured together")
    if process_high is None and cuda_high is None:
        raise ValueError("Captain requires a process or CUDA memory watermark pair")

    policy = PerActorQuotaPolicy(
        metrics_ttl_ms=int(value("elastic_juicer_captain_metrics_ttl_ms", 5_000)),
        min_confidence=float(value("elastic_juicer_captain_min_confidence", 0.8)),
        min_decision_interval_ms=int(value("elastic_juicer_captain_min_decision_interval_ms", 1_000)),
        recovery_observations=int(value("elastic_juicer_captain_recovery_observations", 3)),
        process_memory_high_mb=None if process_high is None else float(process_high),
        process_memory_low_mb=None if process_low is None else float(process_low),
        cuda_memory_high_mb=None if cuda_high is None else float(cuda_high),
        cuda_memory_low_mb=None if cuda_low is None else float(cuda_low),
    )
    runtime = CaptainRuntime(
        policy,
        metrics_sink_handle,
        control_service_handle,
        quota_ttl_ms=int(value("elastic_juicer_captain_quota_ttl_ms", 10_000)),
        rpc_timeout_sec=float(value("elastic_juicer_captain_rpc_timeout_sec", 5.0)),
        retry_backoff_sec=float(value("elastic_juicer_captain_retry_backoff_sec", 0.1)),
    )
    return CaptainLifecycle(
        runtime,
        poll_interval_sec=float(value("elastic_juicer_captain_poll_interval_sec", 0.5)),
    )

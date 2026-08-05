"""Job-scoped, fire-and-forget runtime metrics transport for Ray actors."""

import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Optional

from .actor_resource_sampler import ActorResourceSnapshot

METRICS_EVENT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ActorControlMetrics:
    """Actor-local control state observed with a resource snapshot."""

    quota_revision: int
    current_batch_size: int
    hard_limit: int
    static_min_batch_size: int
    static_max_batch_size: int
    local_success_lower_bound: Optional[int]
    local_oom_upper_bound: Optional[int]

    def __post_init__(self):
        if self.quota_revision < 0:
            raise ValueError("quota_revision must be non-negative")
        if self.static_min_batch_size < 1:
            raise ValueError("static_min_batch_size must be positive")
        if self.static_max_batch_size < self.static_min_batch_size:
            raise ValueError("static_max_batch_size must be >= static_min_batch_size")
        if not self.static_min_batch_size <= self.current_batch_size <= self.static_max_batch_size:
            raise ValueError("current_batch_size must be within static bounds")
        if not self.static_min_batch_size <= self.hard_limit <= self.static_max_batch_size:
            raise ValueError("hard_limit must be within static bounds")


@dataclass(frozen=True)
class ActorMetricsEvent:
    """One ordered actor-local resource observation."""

    job_id: str
    actor_id: str
    actor_incarnation_id: str
    stage_id: str
    op_name: str
    sequence: int
    observed_at_ms: int
    emitted_at_ms: int
    source: str
    snapshot: ActorResourceSnapshot
    control: Optional[ActorControlMetrics] = None
    partition_id: Optional[int] = None
    schema_version: int = METRICS_EVENT_SCHEMA_VERSION

    def __post_init__(self):
        if not self.job_id:
            raise ValueError("job_id must not be empty")
        if not self.actor_id:
            raise ValueError("actor_id must not be empty")
        if not self.actor_incarnation_id:
            raise ValueError("actor_incarnation_id must not be empty")
        if not self.stage_id:
            raise ValueError("stage_id must not be empty")
        if not self.op_name:
            raise ValueError("op_name must not be empty")
        if self.sequence < 1:
            raise ValueError("sequence must be at least 1")
        if self.observed_at_ms < 0:
            raise ValueError("observed_at_ms must be non-negative")
        if self.emitted_at_ms < 0:
            raise ValueError("emitted_at_ms must be non-negative")
        if not self.source:
            raise ValueError("source must not be empty")
        if not isinstance(self.snapshot, ActorResourceSnapshot):
            raise TypeError("snapshot must be an ActorResourceSnapshot")
        if self.control is not None and not isinstance(self.control, ActorControlMetrics):
            raise TypeError("control must be ActorControlMetrics or None")
        if self.partition_id is not None and (
            isinstance(self.partition_id, bool) or not isinstance(self.partition_id, int) or self.partition_id < 0
        ):
            raise ValueError("partition_id must be a non-negative integer or None")
        if self.schema_version != METRICS_EVENT_SCHEMA_VERSION:
            raise ValueError(f"unsupported metrics schema_version: {self.schema_version}")

    def is_fresh(self, now_ms: int, ttl_ms: int) -> bool:
        if now_ms < 0:
            raise ValueError("now_ms must be non-negative")
        if ttl_ms < 1:
            raise ValueError("ttl_ms must be positive")
        age_ms = now_ms - self.observed_at_ms
        return 0 <= age_ms <= ttl_ms

    @property
    def minimum_confidence(self) -> float:
        confidences = [self.snapshot.process_confidence, self.snapshot.rss_peak_confidence]
        if self.snapshot.cuda is not None:
            confidences.append(self.snapshot.cuda.confidence)
        return min(confidences)


class AsyncMetricsSink:
    """Bounded Ray-actor-friendly buffer isolated to one Data-Juicer job."""

    def __init__(self, job_id: str, max_events: int = 2048):
        if not job_id:
            raise ValueError("job_id must not be empty")
        if max_events < 1:
            raise ValueError("max_events must be at least 1")
        self.job_id = job_id
        self.max_events = max_events
        self._events: Deque[ActorMetricsEvent] = deque(maxlen=max_events)
        self._received_events = 0
        self._dropped_events = 0

    def record(self, event: ActorMetricsEvent) -> None:
        if not isinstance(event, ActorMetricsEvent):
            raise TypeError("event must be an ActorMetricsEvent")
        if event.job_id != self.job_id:
            raise ValueError(f"event job_id {event.job_id!r} does not match sink job_id {self.job_id!r}")
        if len(self._events) == self.max_events:
            self._dropped_events += 1
        self._events.append(event)
        self._received_events += 1

    def snapshot(self):
        """Return a driver-readable copy without exposing the mutable buffer."""

        return {
            "job_id": self.job_id,
            "max_events": self.max_events,
            "received_events": self._received_events,
            "dropped_events": self._dropped_events,
            "events": list(self._events),
        }


class AsyncMetricsReporter:
    """Send metrics without allowing producer-side Ray tasks to grow unbounded."""

    def __init__(
        self,
        sink_handle,
        job_id: str,
        actor_id: str,
        actor_incarnation_id: str,
        stage_id: str,
        op_name: str,
        max_in_flight: int = 64,
        wait_fn=None,
        control_state_provider: Optional[Callable[[], ActorControlMetrics]] = None,
        partition_id: Optional[int] = None,
    ):
        if sink_handle is None:
            raise ValueError("sink_handle must not be None")
        if not job_id:
            raise ValueError("job_id must not be empty")
        if not actor_id:
            raise ValueError("actor_id must not be empty")
        if not op_name:
            raise ValueError("op_name must not be empty")
        if max_in_flight < 1:
            raise ValueError("max_in_flight must be at least 1")
        self.sink_handle = sink_handle
        self.job_id = job_id
        self.actor_id = actor_id
        if not actor_incarnation_id:
            raise ValueError("actor_incarnation_id must not be empty")
        self.actor_incarnation_id = actor_incarnation_id
        if not stage_id:
            raise ValueError("stage_id must not be empty")
        self.stage_id = stage_id
        self.op_name = op_name
        self.max_in_flight = max_in_flight
        if partition_id is not None and (
            isinstance(partition_id, bool) or not isinstance(partition_id, int) or partition_id < 0
        ):
            raise ValueError("partition_id must be a non-negative integer or None")
        self.partition_id = partition_id
        self._sequence = 0
        self._in_flight = deque()
        self._wait_fn = wait_fn
        self._control_state_provider = control_state_provider
        self.submitted_events = 0
        self.dropped_events = 0

    def report(self, snapshot: ActorResourceSnapshot) -> bool:
        """Submit one event, or drop it when the bounded producer window is full."""

        self._sequence += 1
        self._refresh_in_flight()
        if len(self._in_flight) >= self.max_in_flight:
            self.dropped_events += 1
            return False
        event = ActorMetricsEvent(
            job_id=self.job_id,
            actor_id=self.actor_id,
            actor_incarnation_id=self.actor_incarnation_id,
            stage_id=self.stage_id,
            op_name=self.op_name,
            sequence=self._sequence,
            observed_at_ms=int(snapshot.timestamp * 1000),
            emitted_at_ms=time.time_ns() // 1_000_000,
            source=snapshot.source,
            snapshot=snapshot,
            control=None if self._control_state_provider is None else self._control_state_provider(),
            partition_id=self.partition_id,
        )
        record_remote = getattr(getattr(self.sink_handle, "record", None), "remote", None)
        if not callable(record_remote):
            raise TypeError("sink_handle.record.remote must be callable")
        self._in_flight.append(record_remote(event))
        self.submitted_events += 1
        return True

    def snapshot(self):
        """Return producer pressure counters without waiting for the Sink."""

        self._refresh_in_flight()
        return {
            "submitted_events": self.submitted_events,
            "dropped_events": self.dropped_events,
            "pending_events": len(self._in_flight),
            "max_in_flight": self.max_in_flight,
            "last_sequence": self._sequence,
        }

    def _refresh_in_flight(self) -> None:
        if not self._in_flight:
            return
        wait_fn = self._wait_fn
        if wait_fn is None:
            try:
                import ray

                if not ray.is_initialized():
                    return
                wait_fn = ray.wait
            except (ImportError, AttributeError):
                return
        references = list(self._in_flight)
        try:
            _, pending = wait_fn(references, num_returns=len(references), timeout=0)
        except (TypeError, ValueError):
            # Non-Ray test doubles cannot be polled. Retaining them is the safe
            # behavior because the producer window will still remain bounded.
            return
        self._in_flight = deque(pending)


def create_ray_metrics_sink(job_id: str, max_events: int = 2048, ray_module=None):
    """Create an unnamed sink actor and return its explicit handle."""

    if ray_module is None:
        import ray as ray_module

    remote_class = ray_module.remote(num_cpus=0)(AsyncMetricsSink)
    return remote_class.remote(job_id=job_id, max_events=max_events)

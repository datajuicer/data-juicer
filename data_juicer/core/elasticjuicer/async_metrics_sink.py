"""Job-scoped, fire-and-forget runtime metrics transport for Ray actors."""

from collections import deque
from dataclasses import dataclass
from typing import Deque

from .actor_resource_sampler import ActorResourceSnapshot


@dataclass(frozen=True)
class ActorMetricsEvent:
    """One ordered actor-local resource observation."""

    job_id: str
    actor_id: str
    op_name: str
    sequence: int
    snapshot: ActorResourceSnapshot

    def __post_init__(self):
        if not self.job_id:
            raise ValueError("job_id must not be empty")
        if not self.actor_id:
            raise ValueError("actor_id must not be empty")
        if not self.op_name:
            raise ValueError("op_name must not be empty")
        if self.sequence < 1:
            raise ValueError("sequence must be at least 1")
        if not isinstance(self.snapshot, ActorResourceSnapshot):
            raise TypeError("snapshot must be an ActorResourceSnapshot")


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
    """Send metrics to a Ray actor handle without synchronizing on results."""

    def __init__(self, sink_handle, job_id: str, actor_id: str, op_name: str):
        if sink_handle is None:
            raise ValueError("sink_handle must not be None")
        if not job_id:
            raise ValueError("job_id must not be empty")
        if not actor_id:
            raise ValueError("actor_id must not be empty")
        if not op_name:
            raise ValueError("op_name must not be empty")
        self.sink_handle = sink_handle
        self.job_id = job_id
        self.actor_id = actor_id
        self.op_name = op_name
        self._sequence = 0

    def report(self, snapshot: ActorResourceSnapshot) -> None:
        self._sequence += 1
        event = ActorMetricsEvent(
            job_id=self.job_id,
            actor_id=self.actor_id,
            op_name=self.op_name,
            sequence=self._sequence,
            snapshot=snapshot,
        )
        record_remote = getattr(getattr(self.sink_handle, "record", None), "remote", None)
        if not callable(record_remote):
            raise TypeError("sink_handle.record.remote must be callable")
        record_remote(event)


def create_ray_metrics_sink(job_id: str, max_events: int = 2048, ray_module=None):
    """Create an unnamed sink actor and return its explicit handle."""

    if ray_module is None:
        import ray as ray_module

    remote_class = ray_module.remote(num_cpus=0)(AsyncMetricsSink)
    return remote_class.remote(job_id=job_id, max_events=max_events)

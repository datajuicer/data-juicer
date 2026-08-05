"""Dependency-free profiling metric contracts.

Utilization values use percentages in ``[0, 100]`` and memory values use MB.
The scope describes the owner of the observation; it must not be inferred from
the metric name.  In particular, the legacy Adapter probe is system scoped.
"""

from bisect import bisect_left, insort
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from math import ceil, floor, isfinite
from typing import Any, Deque, Dict, Iterable, Optional

RESOURCE_METRIC_UNITS = {
    "timestamp": "seconds_since_epoch",
    "cpu_percent": "percent",
    "memory_mb": "megabytes",
    "gpu_memory_mb": "megabytes",
    "gpu_utilization": "percent",
    "latency_ms": "milliseconds",
    "throughput": "samples_per_second",
}


class MetricScope(str, Enum):
    """Owner represented by a resource observation."""

    PROCESS = "process"
    SYSTEM = "system"
    DEVICE = "device"


@dataclass(frozen=True)
class ResourceSnapshot:
    """One operator execution observation with explicit metric semantics."""

    timestamp: float
    batch_size: int
    cpu_percent: float
    memory_mb: float
    gpu_memory_mb: Optional[float] = None
    gpu_utilization: Optional[float] = None
    latency_ms: float = 0.0
    throughput: float = 0.0
    source: str = "resource_monitor"
    scope: MetricScope = MetricScope.PROCESS
    confidence: float = 1.0

    def __post_init__(self):
        if isinstance(self.scope, str):
            try:
                object.__setattr__(self, "scope", MetricScope(self.scope))
            except ValueError as error:
                raise ValueError(f"scope is unsupported: {self.scope}") from error
        if not self.source:
            raise ValueError("source must not be empty")
        if not isfinite(self.timestamp) or self.timestamp < 0:
            raise ValueError("timestamp must be a finite non-negative value")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self._validate_percentage("cpu_percent", self.cpu_percent)
        self._validate_non_negative("memory_mb", self.memory_mb)
        self._validate_non_negative("gpu_memory_mb", self.gpu_memory_mb)
        self._validate_percentage("gpu_utilization", self.gpu_utilization)
        self._validate_non_negative("latency_ms", self.latency_ms)
        self._validate_non_negative("throughput", self.throughput)
        if not isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be in [0, 1]")

    @staticmethod
    def _validate_non_negative(name: str, value: Optional[float]):
        if value is not None and (not isfinite(value) or value < 0):
            raise ValueError(f"{name} must be a finite non-negative value")

    @staticmethod
    def _validate_percentage(name: str, value: Optional[float]):
        if value is not None and (not isfinite(value) or not 0 <= value <= 100):
            raise ValueError(f"{name} must be in [0, 100]")

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "timestamp": self.timestamp,
            "batch_size": self.batch_size,
            "cpu_percent": self.cpu_percent,
            "memory_mb": self.memory_mb,
            "gpu_memory_mb": self.gpu_memory_mb,
            "gpu_utilization": self.gpu_utilization,
            "latency_ms": self.latency_ms,
            "throughput": self.throughput,
            "source": self.source,
            "scope": self.scope.value,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResourceSnapshot":
        """Build a snapshot from its persisted form."""

        return cls(**data)


@dataclass
class OpExecutionStats:
    """Incremental operator statistics with bounded raw sample history.

    Totals, averages, and peaks cover the complete execution. Percentiles are
    diagnostic values over the retained history window.
    """

    op_name: str
    max_history: int = 256
    total_samples: int = 0
    total_batches: int = 0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    avg_throughput: float = 0.0
    avg_memory_mb: float = 0.0
    peak_memory_mb: float = 0.0
    avg_gpu_memory_mb: Optional[float] = None
    peak_gpu_memory_mb: Optional[float] = None
    snapshots: Deque[ResourceSnapshot] = field(init=False, repr=False)
    _throughput_samples: int = field(default=0, init=False, repr=False)
    _gpu_memory_samples: int = field(default=0, init=False, repr=False)
    _sorted_latencies: list = field(default_factory=list, init=False, repr=False)

    def __post_init__(self):
        if not self.op_name:
            raise ValueError("op_name must not be empty")
        if self.max_history < 1:
            raise ValueError("max_history must be at least 1")
        self.snapshots = deque(maxlen=self.max_history)

    def update(self, snapshot: ResourceSnapshot):
        """Update global aggregates and the bounded diagnostic window."""

        if len(self.snapshots) == self.max_history:
            evicted = self.snapshots[0]
            index = bisect_left(self._sorted_latencies, evicted.latency_ms)
            self._sorted_latencies.pop(index)
        self.snapshots.append(snapshot)
        insort(self._sorted_latencies, snapshot.latency_ms)

        self.total_samples += snapshot.batch_size
        self.total_batches += 1
        self.avg_latency_ms += (snapshot.latency_ms - self.avg_latency_ms) / self.total_batches
        self.avg_memory_mb += (snapshot.memory_mb - self.avg_memory_mb) / self.total_batches
        self.peak_memory_mb = max(self.peak_memory_mb, snapshot.memory_mb)

        if snapshot.throughput > 0:
            self._throughput_samples += 1
            self.avg_throughput += (snapshot.throughput - self.avg_throughput) / self._throughput_samples

        if snapshot.gpu_memory_mb is not None:
            self._gpu_memory_samples += 1
            if self.avg_gpu_memory_mb is None:
                self.avg_gpu_memory_mb = snapshot.gpu_memory_mb
            else:
                self.avg_gpu_memory_mb += (snapshot.gpu_memory_mb - self.avg_gpu_memory_mb) / self._gpu_memory_samples
            if self.peak_gpu_memory_mb is None:
                self.peak_gpu_memory_mb = snapshot.gpu_memory_mb
            else:
                self.peak_gpu_memory_mb = max(self.peak_gpu_memory_mb, snapshot.gpu_memory_mb)

        self.p95_latency_ms = self._percentile(0.95)
        self.p99_latency_ms = self._percentile(0.99)

    def _percentile(self, quantile: float) -> float:
        if not self._sorted_latencies:
            return 0.0
        rank = (len(self._sorted_latencies) - 1) * quantile
        lower = floor(rank)
        upper = ceil(rank)
        if lower == upper:
            return float(self._sorted_latencies[lower])
        fraction = rank - lower
        return float(self._sorted_latencies[lower] * (1 - fraction) + self._sorted_latencies[upper] * fraction)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation including aggregates."""

        return {
            "op_name": self.op_name,
            "max_history": self.max_history,
            "total_samples": self.total_samples,
            "total_batches": self.total_batches,
            "avg_latency_ms": self.avg_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "avg_throughput": self.avg_throughput,
            "avg_memory_mb": self.avg_memory_mb,
            "peak_memory_mb": self.peak_memory_mb,
            "avg_gpu_memory_mb": self.avg_gpu_memory_mb,
            "peak_gpu_memory_mb": self.peak_gpu_memory_mb,
            "throughput_samples": self._throughput_samples,
            "gpu_memory_samples": self._gpu_memory_samples,
            "snapshots": [snapshot.to_dict() for snapshot in self.snapshots],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OpExecutionStats":
        """Restore aggregates without counting retained snapshots twice."""

        stats = cls(op_name=data["op_name"], max_history=data.get("max_history", 256))
        snapshots: Iterable[Dict[str, Any]] = data.get("snapshots", [])
        for snapshot_data in snapshots:
            snapshot = ResourceSnapshot.from_dict(snapshot_data)
            stats.snapshots.append(snapshot)
            insort(stats._sorted_latencies, snapshot.latency_ms)

        stats.total_samples = int(data.get("total_samples", sum(item.batch_size for item in stats.snapshots)))
        stats.total_batches = int(data.get("total_batches", len(stats.snapshots)))
        stats.avg_latency_ms = float(data.get("avg_latency_ms", 0.0))
        stats.p95_latency_ms = float(data.get("p95_latency_ms", stats._percentile(0.95)))
        stats.p99_latency_ms = float(data.get("p99_latency_ms", stats._percentile(0.99)))
        stats.avg_throughput = float(data.get("avg_throughput", 0.0))
        stats.avg_memory_mb = float(data.get("avg_memory_mb", 0.0))
        stats.peak_memory_mb = float(data.get("peak_memory_mb", 0.0))
        stats.avg_gpu_memory_mb = data.get("avg_gpu_memory_mb")
        stats.peak_gpu_memory_mb = data.get("peak_gpu_memory_mb")
        stats._throughput_samples = int(data.get("throughput_samples", 0))
        stats._gpu_memory_samples = int(data.get("gpu_memory_samples", 0))
        return stats

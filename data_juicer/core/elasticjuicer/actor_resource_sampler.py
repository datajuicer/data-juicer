"""Actor-owned process and CUDA resource sampling for one batch."""

import threading
import time
from dataclasses import dataclass
from math import isfinite
from typing import Callable, Optional

import psutil

from .profiler.metrics import MetricScope

_BYTES_PER_MB = 1024 * 1024
_DEFAULT_CUDA_BACKEND = object()
ACTOR_RESOURCE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CudaDeviceMemory:
    """Allocator metrics for the CUDA device currently selected by the actor."""

    device_index: int
    allocated_mb: float
    reserved_mb: float
    peak_allocated_mb: float
    scope: MetricScope = MetricScope.DEVICE
    confidence: float = 1.0

    def __post_init__(self):
        if self.scope != MetricScope.DEVICE:
            raise ValueError("CUDA memory scope must be device")
        if not isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("CUDA confidence must be in [0, 1]")


@dataclass(frozen=True)
class ActorResourceSnapshot:
    """Resource outcome of one actor-local batch execution."""

    timestamp: float
    process_id: int
    batch_size: int
    rss_start_mb: float
    rss_end_mb: float
    rss_peak_mb: float
    rss_delta_mb: float
    latency_ms: float
    throughput: float
    cuda: Optional[CudaDeviceMemory]
    succeeded: bool
    error_type: Optional[str]
    source: str = "actor_resource_sampler"
    process_scope: MetricScope = MetricScope.PROCESS
    process_confidence: float = 1.0
    rss_peak_confidence: float = 1.0
    schema_version: int = ACTOR_RESOURCE_SCHEMA_VERSION

    def __post_init__(self):
        if self.schema_version != ACTOR_RESOURCE_SCHEMA_VERSION:
            raise ValueError(f"unsupported actor resource schema_version: {self.schema_version}")
        if self.process_scope != MetricScope.PROCESS:
            raise ValueError("actor RSS scope must be process")
        if not self.source:
            raise ValueError("source must not be empty")
        if not isfinite(self.timestamp) or self.timestamp < 0:
            raise ValueError("timestamp must be a finite non-negative value")
        if not isfinite(self.process_confidence) or not 0 <= self.process_confidence <= 1:
            raise ValueError("process_confidence must be in [0, 1]")
        if not isfinite(self.rss_peak_confidence) or not 0 <= self.rss_peak_confidence <= 1:
            raise ValueError("rss_peak_confidence must be in [0, 1]")


class TorchCudaBackend:
    """Small optional adapter around ``torch.cuda``."""

    def __init__(self):
        try:
            import torch
        except ImportError:
            self._cuda = None
        else:
            self._cuda = getattr(torch, "cuda", None)

    def is_available(self) -> bool:
        if self._cuda is None:
            return False
        try:
            return bool(self._cuda.is_available())
        except Exception:
            return False

    def current_device(self) -> int:
        return int(self._cuda.current_device())

    def reset_peak_memory_stats(self, device: int):
        self._cuda.reset_peak_memory_stats(device)

    def memory_allocated(self, device: int) -> int:
        return int(self._cuda.memory_allocated(device))

    def memory_reserved(self, device: int) -> int:
        return int(self._cuda.memory_reserved(device))

    def max_memory_allocated(self, device: int) -> int:
        return int(self._cuda.max_memory_allocated(device))


class ActorResourceSampler:
    """Measure resources owned by the current actor process and CUDA context."""

    def __init__(
        self,
        process=None,
        cuda_backend=_DEFAULT_CUDA_BACKEND,
        sample_interval_sec: float = 0.01,
        clock: Callable[[], float] = time.perf_counter,
        wall_clock: Callable[[], float] = time.time,
        snapshot_callback: Optional[Callable[[ActorResourceSnapshot], None]] = None,
    ):
        if sample_interval_sec <= 0:
            raise ValueError("sample_interval_sec must be positive")
        self.process = psutil.Process() if process is None else process
        self.cuda_backend = TorchCudaBackend() if cuda_backend is _DEFAULT_CUDA_BACKEND else cuda_backend
        self.sample_interval_sec = sample_interval_sec
        self._clock = clock
        self._wall_clock = wall_clock
        self._snapshot_lock = threading.Lock()
        self.last_snapshot: Optional[ActorResourceSnapshot] = None
        self._snapshot_callback = snapshot_callback
        self._active_lock = threading.Lock()
        self._active_measurement: Optional[BatchResourceMeasurement] = None
        self._poll_stop_event = threading.Event()
        self._poll_thread: Optional[threading.Thread] = None

    def measure(self, batch_size: int) -> "BatchResourceMeasurement":
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        return BatchResourceMeasurement(self, batch_size)

    def _record(self, snapshot: ActorResourceSnapshot):
        with self._snapshot_lock:
            self.last_snapshot = snapshot
        callback = self._snapshot_callback
        if callback is not None:
            try:
                callback(snapshot)
            except Exception as error:
                from loguru import logger

                logger.warning(f"Failed to report ElasticJuicer actor metrics: {error}")

    def set_snapshot_callback(self, callback: Optional[Callable[[ActorResourceSnapshot], None]]) -> None:
        self._snapshot_callback = callback

    def close(self) -> None:
        """Stop the actor-lifetime polling thread, if it was started."""

        self._poll_stop_event.set()
        thread = self._poll_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(1.0, self.sample_interval_sec * 2))

    def _activate(self, measurement: "BatchResourceMeasurement") -> None:
        with self._active_lock:
            if self._active_measurement is not None:
                raise RuntimeError("only one batch measurement may be active per sampler")
            self._active_measurement = measurement
            if self._poll_thread is None or not self._poll_thread.is_alive():
                self._poll_stop_event.clear()
                self._poll_thread = threading.Thread(
                    target=self._poll_rss,
                    name="elasticjuicer-rss-sampler",
                    daemon=True,
                )
                self._poll_thread.start()

    def _deactivate(self, measurement: "BatchResourceMeasurement") -> None:
        with self._active_lock:
            if self._active_measurement is measurement:
                self._active_measurement = None

    def _poll_rss(self) -> None:
        while not self._poll_stop_event.wait(self.sample_interval_sec):
            with self._active_lock:
                measurement = self._active_measurement
                if measurement is not None:
                    measurement.sample_now()


class BatchResourceMeasurement:
    """Context manager that owns the polling lifecycle for one batch."""

    def __init__(self, sampler: ActorResourceSampler, batch_size: int):
        self.sampler = sampler
        self.batch_size = batch_size
        self.snapshot: Optional[ActorResourceSnapshot] = None
        self._rss_lock = threading.Lock()
        self._start_rss_bytes = 0
        self._end_rss_bytes = 0
        self._peak_rss_bytes = 0
        self._start_time = 0.0
        self._cuda_device: Optional[int] = None
        self._entered = False
        self._rss_samples = 0

    def __enter__(self) -> "BatchResourceMeasurement":
        if self._entered:
            raise RuntimeError("a batch measurement cannot be reused")
        self._entered = True
        self._start_rss_bytes = self._read_rss_bytes()
        self._peak_rss_bytes = self._start_rss_bytes
        self._prepare_cuda_peak()
        self.sampler._activate(self)
        # Exclude sampler setup from the operator latency measurement.
        self._start_time = self.sampler._clock()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Capture the operator boundary before sampler teardown.
        end_time = self.sampler._clock()
        self.sampler._deactivate(self)
        self._end_rss_bytes = self._read_rss_bytes()
        self._record_peak(self._end_rss_bytes)

        latency_seconds = max(0.0, end_time - self._start_time)
        latency_ms = latency_seconds * 1000.0
        throughput = self.batch_size / latency_seconds if latency_seconds > 0 else 0.0
        self.snapshot = ActorResourceSnapshot(
            timestamp=self.sampler._wall_clock(),
            process_id=int(self.sampler.process.pid),
            batch_size=self.batch_size,
            rss_start_mb=self._to_mb(self._start_rss_bytes),
            rss_end_mb=self._to_mb(self._end_rss_bytes),
            rss_peak_mb=self._to_mb(self._peak_rss_bytes),
            rss_delta_mb=self._to_mb(self._end_rss_bytes - self._start_rss_bytes),
            latency_ms=latency_ms,
            throughput=throughput,
            cuda=self._read_cuda_memory(),
            succeeded=exc_type is None,
            error_type=None if exc_type is None else exc_type.__name__,
            rss_peak_confidence=self._rss_peak_confidence(latency_seconds),
        )
        self.sampler._record(self.snapshot)
        return False

    def sample_now(self) -> float:
        """Take an immediate RSS sample, useful for explicit checkpoints."""

        if not self._entered:
            raise RuntimeError("measurement has not started")
        rss_bytes = self._read_rss_bytes()
        self._record_peak(rss_bytes, sampled=True)
        return self._to_mb(rss_bytes)

    def _rss_peak_confidence(self, latency_seconds: float) -> float:
        if latency_seconds <= 0:
            return 0.25
        with self._rss_lock:
            sample_count = self._rss_samples
        coverage = sample_count * self.sampler.sample_interval_sec / latency_seconds
        return max(0.25, min(0.99, coverage))

    def _read_rss_bytes(self) -> int:
        return int(self.sampler.process.memory_info().rss)

    def _record_peak(self, rss_bytes: int, sampled: bool = False):
        with self._rss_lock:
            self._peak_rss_bytes = max(self._peak_rss_bytes, rss_bytes)
            if sampled:
                self._rss_samples += 1

    def _prepare_cuda_peak(self):
        backend = self.sampler.cuda_backend
        if backend is None or not backend.is_available():
            return
        try:
            self._cuda_device = int(backend.current_device())
            backend.reset_peak_memory_stats(self._cuda_device)
        except Exception:
            self._cuda_device = None

    def _read_cuda_memory(self) -> Optional[CudaDeviceMemory]:
        if self._cuda_device is None:
            return None
        backend = self.sampler.cuda_backend
        try:
            return CudaDeviceMemory(
                device_index=self._cuda_device,
                allocated_mb=self._to_mb(backend.memory_allocated(self._cuda_device)),
                reserved_mb=self._to_mb(backend.memory_reserved(self._cuda_device)),
                peak_allocated_mb=self._to_mb(backend.max_memory_allocated(self._cuda_device)),
            )
        except Exception:
            return None

    @staticmethod
    def _to_mb(value_bytes: int) -> float:
        return value_bytes / _BYTES_PER_MB

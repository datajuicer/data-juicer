"""
Resource Monitor - Thin Adapter over Ray Memory Infrastructure

Operator-level execution measurement (latency, throughput) is unique to
ElasticJuicer. System-level memory polling delegates to Ray event-based
memory monitor when available, falling back to psutil outside Ray.

Based on Pollux-style agent monitoring.
"""

import time
import psutil
import threading
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Tuple, Set
from collections import defaultdict
import numpy as np

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


@dataclass
class SampleFeatures:
    """Sample-level content features for cost model conditioning."""
    duration_s: float = 0.0          # Video/audio duration in seconds
    frame_count: int = 0             # Video frame count
    resolution_hw: Tuple[int, int] = (0, 0)  # (height, width)
    text_length: int = 0             # Text character/token length
    modality: Set[str] = field(default_factory=set)  # {'text','image','video','audio'}

    def to_vector(self) -> list:
        """Convert to numerical vector for model input."""
        import math
        return [
            math.log(self.duration_s + 1),
            math.log(self.frame_count + 1),
            math.log(self.resolution_hw[0] * self.resolution_hw[1] + 1),
            math.log(self.text_length + 1),
            1.0 if 'text' in self.modality else 0.0,
            1.0 if 'image' in self.modality else 0.0,
            1.0 if 'video' in self.modality else 0.0,
            1.0 if 'audio' in self.modality else 0.0,
        ]


@dataclass
class ResourceSnapshot:
    """Single measurement of resource usage"""
    timestamp: float
    batch_size: int
    # CPU metrics
    cpu_percent: float
    memory_mb: float
    # GPU metrics (if available)
    gpu_memory_mb: Optional[float] = None
    gpu_utilization: Optional[float] = None
    # Performance metrics
    latency_ms: float = 0.0
    throughput: float = 0.0  # samples/sec
    # Sample features (optional)
    sample_features: Optional['SampleFeatures'] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp,
            'batch_size': self.batch_size,
            'cpu_percent': self.cpu_percent,
            'memory_mb': self.memory_mb,
            'gpu_memory_mb': self.gpu_memory_mb,
            'gpu_utilization': self.gpu_utilization,
            'latency_ms': self.latency_ms,
            'throughput': self.throughput,
            'sample_features': self.sample_features,
        }


@dataclass
class OpExecutionStats:
    """Aggregated statistics for an operator"""
    op_name: str
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
    snapshots: List[ResourceSnapshot] = field(default_factory=list)
    
    def update(self, snapshot: ResourceSnapshot):
        """Update statistics with new snapshot"""
        self.snapshots.append(snapshot)
        self.total_samples += snapshot.batch_size
        self.total_batches += 1
        
        # Update averages
        latencies = [s.latency_ms for s in self.snapshots]
        self.avg_latency_ms = np.mean(latencies)
        self.p95_latency_ms = np.percentile(latencies, 95)
        self.p99_latency_ms = np.percentile(latencies, 99)
        
        throughputs = [s.throughput for s in self.snapshots if s.throughput > 0]
        if throughputs:
            self.avg_throughput = np.mean(throughputs)
        
        memories = [s.memory_mb for s in self.snapshots]
        self.avg_memory_mb = np.mean(memories)
        self.peak_memory_mb = max(memories)
        
        if snapshot.gpu_memory_mb is not None:
            gpu_mems = [s.gpu_memory_mb for s in self.snapshots if s.gpu_memory_mb is not None]
            if gpu_mems:
                self.avg_gpu_memory_mb = np.mean(gpu_mems)
                self.peak_gpu_memory_mb = max(gpu_mems)


class ResourceMonitor:
    """
    Lightweight resource monitor for operators.
    
    Inspired by PolluxAgent - measures resource-throughput curves in real-time.
    """

    # Only actually poll expensive metrics every Nth call
    _POLL_INTERVAL = 5

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.stats_by_op: Dict[str, OpExecutionStats] = defaultdict(OpExecutionStats)
        self._lock = threading.Lock()
        self.process = psutil.Process()

        # Throttle state for expensive polls
        self._poll_count = 0
        self._cached_cpu_percent = 0.0
        self._cached_gpu_stats = None

        # Cache library availability (avoid try/except on every poll)
        self._has_ray = False
        self._has_gputil = False
        self._ray_memory_monitor = None

        try:
            import ray
            if ray.is_initialized():
                self._has_ray = True
                try:
                    from ray._private.memory_monitor import MemoryMonitor
                    self._ray_memory_monitor = MemoryMonitor()
                except (ImportError, AttributeError):
                    pass
        except ImportError:
            pass

        try:
            import GPUtil  # noqa: F401
            self._has_gputil = True
        except ImportError:
            pass
        
    def measure_execution(self, op_name: str, batch_size: int, sample_features: Optional['SampleFeatures'] = None):
        """
        Context manager to measure operator execution.
        
        Usage:
            with monitor.measure_execution("my_filter", batch_size=100):
                # Process batch
                result = op.process(batch)
        """
        return ExecutionContext(self, op_name, batch_size, sample_features=sample_features)
    
    def record_snapshot(self, op_name: str, snapshot: ResourceSnapshot):
        """Record a resource snapshot for an operator"""
        if not self.enabled:
            return
            
        with self._lock:
            if op_name not in self.stats_by_op:
                self.stats_by_op[op_name] = OpExecutionStats(op_name=op_name)
            self.stats_by_op[op_name].update(snapshot)
    
    def get_stats(self, op_name: str) -> Optional[OpExecutionStats]:
        """Get statistics for a specific operator"""
        return self.stats_by_op.get(op_name)
    
    def get_all_stats(self) -> Dict[str, OpExecutionStats]:
        """Get statistics for all operators"""
        return dict(self.stats_by_op)
    
    def clear(self):
        """Clear all collected statistics"""
        with self._lock:
            self.stats_by_op.clear()

    def flush_to_store(self, store: 'ProfilingStore') -> int:
        """Transfer all per-op stats to a ProfilingStore.

        Each op's ``OpExecutionStats`` is written via
        ``store.update_execution_stats``.  The monitor's internal state is NOT
        cleared so subsequent calls are idempotent (the store simply overwrites
        the same keys with potentially newer data).

        Args:
            store: Target :class:`ProfilingStore` instance.

        Returns:
            Number of ops flushed.
        """
        with self._lock:
            snapshot = dict(self.stats_by_op)

        flushed = 0
        for op_name, stats in snapshot.items():
            store.update_execution_stats(op_name, stats)
            flushed += 1
        return flushed
    
    def _get_current_resources(self) -> Dict[str, Any]:
        """Get current resource usage.
        
        Delegates to Ray's memory infrastructure when running inside Ray,
        falling back to psutil for standalone execution.

        Expensive metrics (cpu_percent, GPUtil.getGPUs) are throttled and
        only refreshed every ``_POLL_INTERVAL`` calls; other calls reuse
        cached values. Memory (psutil) is cheap and polled every call.
        """
        self._poll_count += 1

        # Refresh expensive metrics on first call and every Nth call thereafter
        should_poll = (
            self._poll_count == 1
            or self._poll_count % self._POLL_INTERVAL == 1
        )
        if should_poll:
            self._cached_cpu_percent = self.process.cpu_percent()
            if self._has_gputil:
                import GPUtil
                try:
                    self._cached_gpu_stats = GPUtil.getGPUs()
                except Exception:
                    self._cached_gpu_stats = None

        cpu_percent = self._cached_cpu_percent
        memory_mb = None
        gpu_memory_mb = None
        gpu_utilization = None

        # Try Ray's memory monitor first (event-based, lower overhead)
        if self._has_ray and self._ray_memory_monitor is not None:
            try:
                system_memory = self._ray_memory_monitor.get_memory_usage()
                if system_memory is not None:
                    memory_mb = system_memory / (1024 * 1024)
            except Exception:
                pass

        # Fallback to psutil if Ray monitor didn't provide memory (cheap call)
        if memory_mb is None:
            memory_mb = self.process.memory_info().rss / (1024 * 1024)

        # Use cached GPU stats
        gpus = self._cached_gpu_stats
        if gpus:
            gpu = gpus[0]
            gpu_memory_mb = gpu.memoryUsed
            gpu_utilization = gpu.load * 100

        return {
            'cpu_percent': cpu_percent,
            'memory_mb': memory_mb,
            'gpu_memory_mb': gpu_memory_mb,
            'gpu_utilization': gpu_utilization,
        }


class ExecutionContext:
    """Context manager for measuring operator execution"""
    
    def __init__(self, monitor: ResourceMonitor, op_name: str, batch_size: int, sample_features: Optional['SampleFeatures'] = None):
        self.monitor = monitor
        self.op_name = op_name
        self.batch_size = batch_size
        self.sample_features = sample_features
        self.start_time = None
        
    def __enter__(self):
        if self.monitor.enabled:
            self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.monitor.enabled or self.start_time is None:
            return
            
        # Calculate latency
        end_time = time.time()
        latency_s = end_time - self.start_time
        latency_ms = latency_s * 1000
        
        # Calculate throughput
        throughput = self.batch_size / latency_s if latency_s > 0 else 0
        
        # Get resource usage
        resources = self.monitor._get_current_resources()
        
        # Create snapshot
        snapshot = ResourceSnapshot(
            timestamp=end_time,
            batch_size=self.batch_size,
            cpu_percent=resources['cpu_percent'],
            memory_mb=resources['memory_mb'],
            gpu_memory_mb=resources['gpu_memory_mb'],
            gpu_utilization=resources['gpu_utilization'],
            latency_ms=latency_ms,
            throughput=throughput,
            sample_features=self.sample_features,
        )
        
        # Record snapshot
        self.monitor.record_snapshot(self.op_name, snapshot)


class MonitoredOp:
    """
    Wrapper to inject monitoring into Data-Juicer operators.
    
    Usage:
        original_op = SomeFilter(**config)
        monitored_op = MonitoredOp(original_op, monitor)
    """
    
    def __init__(self, operator, monitor: ResourceMonitor):
        self.operator = operator
        self.monitor = monitor
        self.op_name = operator.__class__.__name__
        
    def __getattr__(self, name):
        """Delegate attribute access to wrapped operator"""
        return getattr(self.operator, name)
    
    def process(self, *args, **kwargs):
        """Wrap process method with monitoring"""
        # Estimate batch size
        batch_size = self._estimate_batch_size(args, kwargs)
        
        with self.monitor.measure_execution(self.op_name, batch_size):
            return self.operator.process(*args, **kwargs)
    
    def compute_stats(self, *args, **kwargs):
        """Wrap compute_stats method with monitoring (for filters)"""
        batch_size = self._estimate_batch_size(args, kwargs)
        
        with self.monitor.measure_execution(f"{self.op_name}_stats", batch_size):
            return self.operator.compute_stats(*args, **kwargs)
    
    def _estimate_batch_size(self, args, kwargs) -> int:
        """Estimate batch size from arguments"""
        # For single sample: return 1
        # For batched: try to extract from first argument (usually a dict/dataset)
        if args:
            sample = args[0]
            if isinstance(sample, dict):
                # Check if it's batched data
                for value in sample.values():
                    if isinstance(value, list):
                        return len(value)
        return 1

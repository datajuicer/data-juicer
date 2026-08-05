"""
Resource Monitor

Lightweight monitoring for Data-Juicer operators to collect:
- Batch size
- Resource usage (CPU, GPU memory, RAM)
- Processing latency
- Throughput

Based on Pollux-style agent monitoring.
"""

import threading
import time
from typing import Any, Dict, Optional

import psutil

from .metrics import OpExecutionStats, ResourceSnapshot

try:
    import GPUtil

    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


class ResourceMonitor:
    """
    Lightweight resource monitor for operators.

    Inspired by PolluxAgent - measures resource-throughput curves in real-time.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.stats_by_op: Dict[str, OpExecutionStats] = {}
        self._lock = threading.Lock()
        self.process = psutil.Process()

    def measure_execution(self, op_name: str, batch_size: int):
        """
        Context manager to measure operator execution.

        Usage:
            with monitor.measure_execution("my_filter", batch_size=100):
                # Process batch
                result = op.process(batch)
        """
        return ExecutionContext(self, op_name, batch_size)

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

    def _get_current_resources(self) -> Dict[str, Any]:
        """Get current resource usage"""
        # psutil's process percentage can exceed 100 on multicore systems.
        # Normalize it to the process share of total host CPU capacity so it
        # matches the schema's percentage range.
        logical_cpu_count = psutil.cpu_count(logical=True) or 1
        cpu_percent = min(100.0, self.process.cpu_percent() / logical_cpu_count)
        memory_mb = self.process.memory_info().rss / (1024 * 1024)

        gpu_memory_mb = None
        gpu_utilization = None

        if GPU_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    # Use first GPU for now
                    gpu = gpus[0]
                    gpu_memory_mb = gpu.memoryUsed
                    gpu_utilization = gpu.load * 100
            except Exception:
                pass

        return {
            "cpu_percent": cpu_percent,
            "memory_mb": memory_mb,
            "gpu_memory_mb": gpu_memory_mb,
            "gpu_utilization": gpu_utilization,
        }


class ExecutionContext:
    """Context manager for measuring operator execution"""

    def __init__(self, monitor: ResourceMonitor, op_name: str, batch_size: int):
        self.monitor = monitor
        self.op_name = op_name
        self.batch_size = batch_size
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
            cpu_percent=resources["cpu_percent"],
            memory_mb=resources["memory_mb"],
            gpu_memory_mb=resources["gpu_memory_mb"],
            gpu_utilization=resources["gpu_utilization"],
            latency_ms=latency_ms,
            throughput=throughput,
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

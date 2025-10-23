#!/usr/bin/env python3
"""
Metrics collection system for performance benchmarking.
"""

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict

import psutil
from loguru import logger


@dataclass
class BenchmarkMetrics:
    """Comprehensive performance metrics for a benchmark run."""

    # Timing metrics
    total_wall_time: float
    processing_time: float
    io_time: float
    overhead_time: float

    # Throughput metrics
    samples_per_second: float
    bytes_per_second: float
    operations_per_second: float

    # Resource metrics
    peak_memory_mb: float
    average_cpu_percent: float
    peak_cpu_percent: float

    # Quality metrics
    samples_processed: int
    samples_retained: int
    data_retention_rate: float

    # Configuration
    config_hash: str
    strategy_name: str

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def speedup_vs_baseline(self, baseline: "BenchmarkMetrics") -> float:
        """Calculate speedup vs baseline."""
        if baseline.total_wall_time == 0:
            return 0.0
        return baseline.total_wall_time / self.total_wall_time

    def memory_efficiency(self, baseline: "BenchmarkMetrics") -> float:
        """Calculate memory efficiency vs baseline."""
        if self.peak_memory_mb == 0:
            return 0.0
        return baseline.peak_memory_mb / self.peak_memory_mb

    def throughput_improvement(self, baseline: "BenchmarkMetrics") -> float:
        """Calculate throughput improvement vs baseline."""
        if baseline.samples_per_second == 0:
            return 0.0
        return self.samples_per_second / baseline.samples_per_second


class MetricsCollector:
    """Collects comprehensive performance metrics during benchmark runs."""

    def __init__(self):
        self.monitoring = False
        self.monitor_thread = None
        self.metrics_data = []
        self.start_time = None
        self.end_time = None

    def start_monitoring(self):
        """Start monitoring system resources."""
        self.monitoring = True
        self.metrics_data = []
        self.start_time = time.time()

        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_resources)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

        logger.debug("Started metrics monitoring")

    def stop_monitoring(self):
        """Stop monitoring and return collected metrics."""
        if not self.monitoring:
            return None

        self.monitoring = False
        self.end_time = time.time()

        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)

        logger.debug("Stopped metrics monitoring")
        return self._calculate_metrics()

    def _monitor_resources(self):
        """Monitor system resources in background thread."""
        process = psutil.Process()

        while self.monitoring:
            try:
                # Get current metrics
                cpu_percent = process.cpu_percent()
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024  # Convert to MB

                self.metrics_data.append({"timestamp": time.time(), "cpu_percent": cpu_percent, "memory_mb": memory_mb})

                time.sleep(0.1)  # Sample every 100ms

            except Exception as e:
                logger.warning(f"Error monitoring resources: {e}")
                break

    def _calculate_metrics(self) -> BenchmarkMetrics:
        """Calculate final metrics from collected data."""
        if not self.metrics_data:
            return BenchmarkMetrics(
                total_wall_time=0,
                processing_time=0,
                io_time=0,
                overhead_time=0,
                samples_per_second=0,
                bytes_per_second=0,
                operations_per_second=0,
                peak_memory_mb=0,
                average_cpu_percent=0,
                peak_cpu_percent=0,
                samples_processed=0,
                samples_retained=0,
                data_retention_rate=0,
                config_hash="",
                strategy_name="",
            )

        # Calculate timing metrics
        total_wall_time = self.end_time - self.start_time if self.end_time and self.start_time else 0

        # Calculate resource metrics
        memory_values = [d["memory_mb"] for d in self.metrics_data]
        cpu_values = [d["cpu_percent"] for d in self.metrics_data]

        peak_memory = max(memory_values) if memory_values else 0
        avg_cpu = sum(cpu_values) / len(cpu_values) if cpu_values else 0
        peak_cpu = max(cpu_values) if cpu_values else 0

        return BenchmarkMetrics(
            total_wall_time=total_wall_time,
            processing_time=total_wall_time,  # Simplified for now
            io_time=0,  # Would need more detailed tracking
            overhead_time=0,  # Would need more detailed tracking
            samples_per_second=0,  # Will be set by benchmark runner
            bytes_per_second=0,  # Will be set by benchmark runner
            operations_per_second=0,  # Will be set by benchmark runner
            peak_memory_mb=peak_memory,
            average_cpu_percent=avg_cpu,
            peak_cpu_percent=peak_cpu,
            samples_processed=0,  # Will be set by benchmark runner
            samples_retained=0,  # Will be set by benchmark runner
            data_retention_rate=0,  # Will be set by benchmark runner
            config_hash="",  # Will be set by benchmark runner
            strategy_name="",  # Will be set by benchmark runner
        )

    def get_system_info(self) -> Dict[str, Any]:
        """Get system information for context."""
        return {
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "disk_free_gb": psutil.disk_usage("/").free / (1024**3),
            "python_version": os.sys.version,
            "platform": os.name,
        }

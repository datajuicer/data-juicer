"""Optional process resource observations for disposable GPU probe workers.

Adapted from ElasticJuicer's actor_resource_sampler (4274a9d). Only RSS is
polled in the background. CUDA allocator counters are read on the caller
thread at phase boundaries, without initializing CUDA, resetting its peaks,
or adding synchronizations. They describe this process, not the whole GPU.
"""

import math
import os
import socket
import sys
import threading
import time
from contextlib import contextmanager, nullcontext

import psutil


def sample_probe_phase(sampler, phase, batch_index=None):
    return sampler.phase(phase, batch_index) if sampler is not None else nullcontext()


class ProbeResourceSampler:
    """Bounded, best-effort diagnostics; never a resource-planning authority.

    CUDA peaks are cumulative since worker startup, including construction.
    Phase wall time measures host execution, not synchronized GPU latency.
    RSS peaks are sampled lower bounds; child processes are not included.
    """

    def __init__(self, process=None, sample_interval_seconds=0.01, max_samples=256):
        if not math.isfinite(sample_interval_seconds) or sample_interval_seconds <= 0:
            raise ValueError("sample_interval_seconds must be finite and positive")
        if isinstance(max_samples, bool) or not isinstance(max_samples, int) or max_samples < 1:
            raise ValueError("max_samples must be a positive integer")
        self.sample_interval_seconds = sample_interval_seconds
        self.max_samples = max_samples
        self.samples = []
        self.dropped_samples = 0
        self.errors = {"process": 0, "cuda": 0, "polling": 0}
        self._lock = threading.Lock()
        try:
            self.process = process if process is not None else psutil.Process()
        except Exception:
            self.process = None
            self.errors["process"] += 1
        self.worker = {
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "ray_node_id": None,
            "ray_gpu_ids": [],
        }
        ray = sys.modules.get("ray")
        try:
            if ray is not None and ray.is_initialized():
                context = ray.get_runtime_context()
                self.worker["ray_node_id"] = str(context.get_node_id())
                self.worker["ray_gpu_ids"] = [str(value) for value in context.get_accelerator_ids().get("GPU", [])]
        except Exception:
            pass  # Direct non-Ray probes can still report process diagnostics.

    def _read_process(self, include_cpu=False):
        rss, cpu = None, None
        if self.process is None:
            return rss, cpu
        try:
            rss = int(self.process.memory_info().rss)
            if include_cpu:
                times = self.process.cpu_times()
                cpu = float(times.user + times.system)
                if not math.isfinite(cpu):
                    cpu = None
        except Exception:
            with self._lock:
                self.errors["process"] += 1
        return rss, cpu

    def _read_cuda(self):
        # Do not import torch or initialize a context merely to observe it.
        torch = sys.modules.get("torch")
        if torch is None:
            return None
        try:
            cuda = torch.cuda
            if not cuda.is_initialized():
                return None
            return {
                "device_index": 0,  # Probe workers reserve exactly one GPU.
                "allocated_bytes": int(cuda.memory_allocated(0)),
                "reserved_bytes": int(cuda.memory_reserved(0)),
                "peak_allocated_bytes": int(cuda.max_memory_allocated(0)),
                "peak_reserved_bytes": int(cuda.max_memory_reserved(0)),
            }
        except Exception:
            with self._lock:
                self.errors["cuda"] += 1
            return None

    @contextmanager
    def phase(self, name, batch_index=None):
        # Keep both storage and sampling work bounded for unusually long probes.
        if len(self.samples) >= self.max_samples:
            self.dropped_samples += 1
            yield
            return
        rss_start, cpu_start = self._read_process(include_cpu=True)
        cuda_start = self._read_cuda()
        peak = rss_start
        polls = 0
        stop = threading.Event()
        peak_lock = threading.Lock()

        def poll_rss():
            nonlocal peak, polls
            while not stop.wait(self.sample_interval_seconds):
                rss, _ = self._read_process()
                if rss is not None:
                    with peak_lock:
                        peak = max(peak, rss) if peak is not None else rss
                        polls += 1

        thread = threading.Thread(target=poll_rss, name="gpu-probe-rss-sampler", daemon=True)
        try:
            thread.start()
        except Exception:
            self.errors["polling"] += 1
            thread = None
        started_at = time.time()
        started = time.perf_counter()
        error_type = None
        try:
            yield
        except BaseException as error:
            error_type = type(error).__name__
            raise
        finally:
            # Capture the operator boundary before sampler teardown.
            seconds = max(0.0, time.perf_counter() - started)
            stop.set()
            if thread is not None:
                thread.join(timeout=1.0)
                if thread.is_alive():
                    self.errors["polling"] += 1
            rss_end, cpu_end = self._read_process(include_cpu=True)
            with peak_lock:
                if rss_end is not None:
                    peak = max(peak, rss_end) if peak is not None else rss_end
                rss_peak, rss_polls = peak, polls
            self.samples.append(
                {
                    "phase": name,
                    "batch_index": batch_index,
                    "started_at_unix_seconds": started_at,
                    "host_wall_seconds": seconds,
                    "cpu_seconds": (
                        max(0.0, cpu_end - cpu_start) if cpu_start is not None and cpu_end is not None else None
                    ),
                    "rss_start_bytes": rss_start,
                    "rss_end_bytes": rss_end,
                    "rss_sampled_peak_bytes": rss_peak,
                    "rss_poll_count": rss_polls,
                    "cuda_start": cuda_start,
                    "cuda_end": self._read_cuda(),
                    "succeeded": error_type is None,
                    "error_type": error_type,
                }
            )

    def snapshot(self):
        return {
            "schema_version": 1,
            "worker": dict(self.worker),
            "process_scope": "worker_process",
            "cuda_scope": "worker_process_allocator_on_device",
            "cuda_peak_scope": "since_worker_start",
            "latency_scope": "host_wall_without_added_cuda_synchronization",
            "rss_peak_scope": "sampled_lower_bound",
            "sample_interval_seconds": self.sample_interval_seconds,
            "samples": list(self.samples),
            "dropped_samples": self.dropped_samples,
            "sampling_errors": dict(self.errors),
        }

import inspect
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import data_juicer.core.elasticjuicer.actor_resource_sampler as sampler_module
from data_juicer.core.elasticjuicer.actor_resource_sampler import (
    ActorResourceSampler,
    TorchCudaBackend,
)

MB = 1024 * 1024


class FakeProcess:
    pid = 1234

    def __init__(self, rss_values_mb):
        self._rss_values = [int(value * MB) for value in rss_values_mb]

    def memory_info(self):
        if len(self._rss_values) > 1:
            rss = self._rss_values.pop(0)
        else:
            rss = self._rss_values[0]
        return SimpleNamespace(rss=rss)


class SequenceClock:
    def __init__(self, *values):
        self._values = iter(values)

    def __call__(self):
        return next(self._values)


class FakeCudaBackend:
    def __init__(self, device_index=3):
        self.device_index = device_index
        self.calls = []

    def is_available(self):
        return True

    def current_device(self):
        self.calls.append(("current_device", None))
        return self.device_index

    def reset_peak_memory_stats(self, device):
        self.calls.append(("reset_peak_memory_stats", device))

    def memory_allocated(self, device):
        self.calls.append(("memory_allocated", device))
        return 20 * MB

    def memory_reserved(self, device):
        self.calls.append(("memory_reserved", device))
        return 30 * MB

    def max_memory_allocated(self, device):
        self.calls.append(("max_memory_allocated", device))
        return 40 * MB


class FailingReadCudaBackend(FakeCudaBackend):
    def memory_allocated(self, device):
        raise RuntimeError("cuda metrics unavailable")


class SignalingProcess(FakeProcess):
    def __init__(self, rss_values_mb):
        super().__init__(rss_values_mb)
        self.polled = threading.Event()
        self.calls = 0

    def memory_info(self):
        self.calls += 1
        result = super().memory_info()
        if self.calls == 2:
            self.polled.set()
        return result


def test_cpu_batch_snapshot_uses_process_rss_peak_delta_latency_and_throughput():
    sampler = ActorResourceSampler(
        process=FakeProcess([100, 140, 120]),
        cuda_backend=None,
        clock=SequenceClock(10.0, 10.25),
        wall_clock=lambda: 123.0,
        sample_interval_sec=60.0,
    )

    with sampler.measure(batch_size=4) as measurement:
        measurement.sample_now()

    snapshot = measurement.snapshot
    assert snapshot.process_id == 1234
    assert snapshot.batch_size == 4
    assert snapshot.rss_start_mb == 100.0
    assert snapshot.rss_end_mb == 120.0
    assert snapshot.rss_peak_mb == 140.0
    assert snapshot.rss_delta_mb == 20.0
    assert snapshot.latency_ms == 250.0
    assert snapshot.throughput == 16.0
    assert snapshot.cuda is None
    assert snapshot.succeeded is True
    assert snapshot.error_type is None
    assert snapshot.timestamp == 123.0
    assert snapshot.rss_peak_confidence == 0.99


def test_default_sampler_binds_to_current_process():
    process = FakeProcess([100, 100])

    with patch.object(sampler_module.psutil, "Process", return_value=process) as process_factory:
        sampler = ActorResourceSampler(cuda_backend=None)

    process_factory.assert_called_once_with()
    assert sampler.process is process


def test_background_polling_captures_rss_peak_inside_batch():
    process = SignalingProcess([100, 180, 120])
    sampler = ActorResourceSampler(
        process=process,
        cuda_backend=None,
        sample_interval_sec=0.001,
    )

    with sampler.measure(batch_size=1) as measurement:
        assert process.polled.wait(timeout=1.0)

    assert measurement.snapshot.rss_start_mb == 100.0
    assert measurement.snapshot.rss_peak_mb == 180.0
    assert measurement.snapshot.rss_end_mb == 120.0
    sampler.close()


def test_sampler_reuses_one_polling_thread_across_batches():
    sampler = ActorResourceSampler(
        process=FakeProcess([100, 100, 100, 100]),
        cuda_backend=None,
        sample_interval_sec=60.0,
    )

    with sampler.measure(batch_size=1):
        pass
    first_thread = sampler._poll_thread
    with sampler.measure(batch_size=1):
        pass

    assert sampler._poll_thread is first_thread
    assert first_thread.is_alive()
    sampler.close()
    assert not first_thread.is_alive()


def test_exception_is_recorded_without_being_suppressed():
    sampler = ActorResourceSampler(
        process=FakeProcess([100, 110]),
        cuda_backend=None,
        clock=SequenceClock(1.0, 1.1),
        sample_interval_sec=60.0,
    )

    with pytest.raises(ValueError, match="broken mapper"):
        with sampler.measure(batch_size=2):
            raise ValueError("broken mapper")

    assert sampler.last_snapshot is not None
    assert sampler.last_snapshot.succeeded is False
    assert sampler.last_snapshot.error_type == "ValueError"
    assert sampler.last_snapshot.rss_peak_confidence == 0.25


def test_sampler_publishes_each_completed_snapshot_to_callback():
    snapshots = []
    sampler = ActorResourceSampler(
        process=FakeProcess([100, 110]),
        cuda_backend=None,
        clock=SequenceClock(1.0, 1.1),
        sample_interval_sec=60.0,
        snapshot_callback=snapshots.append,
    )

    with sampler.measure(batch_size=2) as measurement:
        pass

    assert snapshots == [measurement.snapshot]


def test_metrics_callback_failure_does_not_break_batch_execution():
    def broken_callback(snapshot):
        raise RuntimeError("sink unavailable")

    sampler = ActorResourceSampler(
        process=FakeProcess([100, 110]),
        cuda_backend=None,
        clock=SequenceClock(1.0, 1.1),
        sample_interval_sec=60.0,
        snapshot_callback=broken_callback,
    )

    with sampler.measure(batch_size=2) as measurement:
        pass

    assert measurement.snapshot.succeeded is True
    assert sampler.last_snapshot is measurement.snapshot


def test_cuda_metrics_use_current_assigned_device_not_device_zero():
    cuda = FakeCudaBackend(device_index=3)
    sampler = ActorResourceSampler(
        process=FakeProcess([100, 100]),
        cuda_backend=cuda,
        clock=SequenceClock(1.0, 1.5),
        sample_interval_sec=60.0,
    )

    with sampler.measure(batch_size=8) as measurement:
        pass

    assert measurement.snapshot.cuda.device_index == 3
    assert measurement.snapshot.cuda.allocated_mb == 20.0
    assert measurement.snapshot.cuda.reserved_mb == 30.0
    assert measurement.snapshot.cuda.peak_allocated_mb == 40.0
    device_calls = [device for name, device in cuda.calls if name != "current_device"]
    assert device_calls
    assert set(device_calls) == {3}


def test_cuda_metric_failure_does_not_mask_business_exception():
    cuda = FailingReadCudaBackend()
    sampler = ActorResourceSampler(
        process=FakeProcess([100, 100]),
        cuda_backend=cuda,
        sample_interval_sec=60.0,
    )

    with pytest.raises(ValueError, match="mapper failed"):
        with sampler.measure(batch_size=1):
            raise ValueError("mapper failed")

    assert sampler.last_snapshot.error_type == "ValueError"
    assert sampler.last_snapshot.cuda is None


def test_sampler_does_not_read_system_memory_or_profile_store():
    source = inspect.getsource(sampler_module)

    assert "virtual_memory" not in source
    assert "GPUtil" not in source
    assert "ProfilingStore" not in source


def test_sampler_rejects_non_positive_interval():
    with pytest.raises(ValueError, match="sample_interval_sec"):
        ActorResourceSampler(
            process=FakeProcess([100, 100]),
            cuda_backend=None,
            sample_interval_sec=0,
        )


def test_sampler_rejects_non_positive_batch_size():
    sampler = ActorResourceSampler(process=FakeProcess([100, 100]), cuda_backend=None)

    with pytest.raises(ValueError, match="batch_size"):
        sampler.measure(batch_size=0)


def test_zero_duration_has_zero_throughput():
    sampler = ActorResourceSampler(
        process=FakeProcess([100, 100]),
        cuda_backend=None,
        clock=SequenceClock(1.0, 1.0),
        sample_interval_sec=60.0,
    )

    with sampler.measure(batch_size=4) as measurement:
        pass

    assert measurement.snapshot.latency_ms == 0.0
    assert measurement.snapshot.throughput == 0.0


def _real_cuda_available():
    backend = TorchCudaBackend()
    return backend.is_available()


@pytest.mark.skipif(not _real_cuda_available(), reason="CUDA is not available")
def test_optional_real_cuda_sampler_uses_torch_current_device():
    backend = TorchCudaBackend()
    sampler = ActorResourceSampler(cuda_backend=backend)

    with sampler.measure(batch_size=1) as measurement:
        pass

    assert measurement.snapshot.cuda.device_index == backend.current_device()

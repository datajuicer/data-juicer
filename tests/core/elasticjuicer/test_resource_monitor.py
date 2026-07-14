import pytest
from unittest.mock import patch, MagicMock

from data_juicer.core.elasticjuicer.profiler.resource_monitor import (
    ResourceMonitor,
    ResourceSnapshot,
    ExecutionContext,
)


class FakeClock:
    def __init__(self, start=100.0):
        self.now = start

    def __call__(self):
        self.now += 0.5
        return self.now


def _fake_process_rss(mb=500, cpu_percent=25.0):
    mock_proc = MagicMock()
    mock_proc.cpu_percent.return_value = cpu_percent
    mock_proc.memory_info.return_value = MagicMock(rss=int(mb * 1024 * 1024))
    return mock_proc


def test_snapshot_records_batch_size_and_latency():
    monitor = ResourceMonitor(enabled=True)
    monitor.process = _fake_process_rss(500)

    with monitor.measure_execution("test_op", batch_size=10):
        pass

    stats = monitor.get_stats("test_op")
    assert stats is not None
    assert stats.total_batches == 1
    assert stats.total_samples == 10
    assert stats.snapshots[-1].batch_size == 10
    assert stats.snapshots[-1].latency_ms >= 0


def test_disabled_monitor_records_nothing():
    monitor = ResourceMonitor(enabled=False)
    monitor.process = _fake_process_rss()

    with monitor.measure_execution("test_op", batch_size=5):
        pass

    assert monitor.get_stats("test_op") is None


def test_multiple_batches_accumulate():
    monitor = ResourceMonitor(enabled=True)
    monitor.process = _fake_process_rss(500)

    for i in range(5):
        with monitor.measure_execution("op", batch_size=4):
            pass

    stats = monitor.get_stats("op")
    assert stats.total_batches == 5
    assert stats.total_samples == 20
    assert len(stats.snapshots) == 5


def test_record_snapshot_direct():
    monitor = ResourceMonitor(enabled=True)
    snap = ResourceSnapshot(
        timestamp=1.0,
        batch_size=8,
        cpu_percent=50.0,
        memory_mb=1024.0,
        latency_ms=100.0,
        throughput=80.0,
    )
    monitor.record_snapshot("direct_op", snap)

    stats = monitor.get_stats("direct_op")
    assert stats is not None
    assert stats.snapshots[0].memory_mb == 1024.0
    assert stats.peak_memory_mb == 1024.0


def test_clear_resets_all_stats():
    monitor = ResourceMonitor(enabled=True)
    monitor.process = _fake_process_rss()

    with monitor.measure_execution("op1", batch_size=1):
        pass
    with monitor.measure_execution("op2", batch_size=2):
        pass

    assert len(monitor.get_all_stats()) == 2
    monitor.clear()
    assert len(monitor.get_all_stats()) == 0


def test_throughput_calculation():
    monitor = ResourceMonitor(enabled=True)
    monitor.process = _fake_process_rss()

    with monitor.measure_execution("op", batch_size=100):
        pass

    stats = monitor.get_stats("op")
    snap = stats.snapshots[-1]
    if snap.latency_ms > 0:
        expected = 100 / (snap.latency_ms / 1000)
        assert abs(snap.throughput - expected) < 0.1


def test_get_current_resources_returns_expected_keys():
    monitor = ResourceMonitor(enabled=True)
    monitor.process = _fake_process_rss(256)

    resources = monitor._get_current_resources()
    assert "cpu_percent" in resources
    assert "memory_mb" in resources
    assert resources["memory_mb"] == pytest.approx(256, abs=1)


@patch("data_juicer.core.elasticjuicer.profiler.resource_monitor.psutil.cpu_count", return_value=4)
def test_process_cpu_is_normalized_to_total_host_capacity(_mock_cpu_count):
    monitor = ResourceMonitor(enabled=True)
    monitor.process = _fake_process_rss(cpu_percent=250.0)

    resources = monitor._get_current_resources()

    assert resources["cpu_percent"] == 62.5

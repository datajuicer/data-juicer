import pytest

from data_juicer.core.elasticjuicer.profiler.metrics import (
    RESOURCE_METRIC_UNITS,
    MetricScope,
    OpExecutionStats,
    ResourceSnapshot,
)


def _snapshot(index: int, **overrides) -> ResourceSnapshot:
    values = {
        "timestamp": float(index),
        "batch_size": index + 1,
        "cpu_percent": 10.0 + index,
        "memory_mb": 100.0 + index,
        "latency_ms": 20.0 + index,
        "throughput": 5.0 + index,
        "source": "unit_test",
        "scope": MetricScope.PROCESS,
        "confidence": 0.9,
    }
    values.update(overrides)
    return ResourceSnapshot(**values)


def test_snapshot_declares_units_scope_and_confidence():
    snapshot = _snapshot(0)

    assert RESOURCE_METRIC_UNITS == {
        "timestamp": "seconds_since_epoch",
        "cpu_percent": "percent",
        "memory_mb": "megabytes",
        "gpu_memory_mb": "megabytes",
        "gpu_utilization": "percent",
        "latency_ms": "milliseconds",
        "throughput": "samples_per_second",
    }
    assert snapshot.scope is MetricScope.PROCESS
    assert snapshot.confidence == 0.9
    assert snapshot.to_dict()["scope"] == "process"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("batch_size", 0),
        ("cpu_percent", 100.1),
        ("memory_mb", -1.0),
        ("gpu_utilization", -0.1),
        ("latency_ms", -1.0),
        ("throughput", -1.0),
        ("confidence", 1.1),
    ],
)
def test_snapshot_rejects_invalid_metric_values(field, value):
    with pytest.raises(ValueError, match=field):
        _snapshot(0, **{field: value})


def test_execution_stats_keep_bounded_history_and_global_totals():
    stats = OpExecutionStats(op_name="mapper", max_history=3)

    for index in range(5):
        stats.update(_snapshot(index))

    assert stats.total_batches == 5
    assert stats.total_samples == 15
    assert [snapshot.timestamp for snapshot in stats.snapshots] == [2.0, 3.0, 4.0]
    assert stats.avg_latency_ms == pytest.approx(22.0)
    assert stats.avg_throughput == pytest.approx(7.0)
    assert stats.avg_memory_mb == pytest.approx(102.0)
    assert stats.peak_memory_mb == 104.0


def test_execution_stats_update_running_averages_incrementally():
    stats = OpExecutionStats(op_name="mapper", max_history=2)

    stats.update(_snapshot(0, latency_ms=10.0, throughput=2.0, memory_mb=20.0))
    assert stats.avg_latency_ms == 10.0
    stats.update(_snapshot(1, latency_ms=20.0, throughput=4.0, memory_mb=40.0))
    assert stats.avg_latency_ms == 15.0
    stats.update(_snapshot(2, latency_ms=60.0, throughput=0.0, memory_mb=90.0))

    # Averages and peaks cover the whole execution, even after old raw samples
    # are evicted from the bounded diagnostic window.
    assert stats.avg_latency_ms == 30.0
    assert stats.avg_throughput == 3.0
    assert stats.avg_memory_mb == 50.0
    assert stats.peak_memory_mb == 90.0
    assert len(stats.snapshots) == 2

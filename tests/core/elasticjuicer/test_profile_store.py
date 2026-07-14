import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from data_juicer.core.elasticjuicer.profiler.metrics import (
    MetricScope,
    OpExecutionStats,
    ResourceSnapshot,
)
from data_juicer.core.elasticjuicer.profiler.profiling_store import (
    ProfilingStore,
    ResourceThroughputCurve,
)


def _make_snapshots(n=10, base_batch=10, base_mem=100):
    return [
        ResourceSnapshot(
            timestamp=float(i),
            batch_size=base_batch + i,
            cpu_percent=20.0,
            memory_mb=base_mem + i * 10,
            latency_ms=50.0 + i,
            throughput=(base_batch + i) / (50.0 + i) * 1000,
            source="unit_test",
            scope=MetricScope.PROCESS,
            confidence=0.95,
        )
        for i in range(n)
    ]


def test_create_store_in_temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ProfilingStore(storage_dir=tmpdir)
        assert Path(tmpdir).exists()


def test_update_and_retrieve_execution_stats():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ProfilingStore(storage_dir=tmpdir)
        stats = OpExecutionStats(op_name="filter_a")
        for snap in _make_snapshots(5):
            stats.update(snap)

        store.update_execution_stats("filter_a", stats)
        retrieved = store.get_execution_stats("filter_a")
        assert retrieved is not None
        assert retrieved.total_batches == 5


def test_save_and_load_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ProfilingStore(storage_dir=tmpdir)
        stats = OpExecutionStats(op_name="op_x")
        for snap in _make_snapshots(6):
            stats.update(snap)
        store.update_execution_stats("op_x", stats)
        store.save_all()

        store2 = ProfilingStore(storage_dir=tmpdir)
        loaded = store2.get_execution_stats("op_x")
        assert loaded is not None
        assert loaded.total_batches == 6
        assert loaded.snapshots[-1].scope is MetricScope.PROCESS
        assert loaded.snapshots[-1].confidence == 0.95
        payload = json.loads((Path(tmpdir) / "profiles.json").read_text())
        assert payload["schema_version"] == 1
        assert not (Path(tmpdir) / "profiles.tmp").exists()
        assert not (Path(tmpdir) / "execution_stats.pkl").exists()


def test_throughput_curve_fitted_with_enough_data():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ProfilingStore(storage_dir=tmpdir)
        stats = OpExecutionStats(op_name="op_y")
        for snap in _make_snapshots(10):
            stats.update(snap)

        store.update_execution_stats("op_y", stats)
        curve = store.get_throughput_curve("op_y")
        assert curve is not None
        assert curve.n_samples > 0


def test_throughput_curve_not_fitted_with_too_few():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ProfilingStore(storage_dir=tmpdir)
        stats = OpExecutionStats(op_name="op_z")
        for snap in _make_snapshots(2):
            stats.update(snap)

        store.update_execution_stats("op_z", stats)
        curve = store.get_throughput_curve("op_z")
        assert curve is None


def test_predict_memory_for_batch():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ProfilingStore(storage_dir=tmpdir)
        stats = OpExecutionStats(op_name="op_m")
        for snap in _make_snapshots(10, base_batch=5, base_mem=50):
            stats.update(snap)
        store.update_execution_stats("op_m", stats)

        predicted = store.predict_memory_for_batch("op_m", batch_size=20)
        assert predicted is not None
        assert predicted > 0


def test_predict_memory_returns_none_for_unknown_op():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ProfilingStore(storage_dir=tmpdir)
        assert store.predict_memory_for_batch("nonexistent", 10) is None


def test_get_safe_batch_size_conservative_for_unknown():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ProfilingStore(storage_dir=tmpdir)
        bs = store.get_safe_batch_size("unknown_op", available_memory_mb=5000)
        assert bs == 1


def test_legacy_pickle_is_never_loaded():
    with tempfile.TemporaryDirectory() as tmpdir:
        stats_file = Path(tmpdir) / "execution_stats.pkl"
        stats_file.write_bytes(b"cos\nsystem\n(S'echo unsafe'\ntR.")

        store = ProfilingStore(storage_dir=tmpdir)
        assert store.get_execution_stats("anything") is None


def test_corrupted_versioned_profile_is_reported(tmp_path):
    (tmp_path / "profiles.json").write_text("{not json")

    store = ProfilingStore(storage_dir=str(tmp_path))

    assert store.execution_stats == {}
    assert store.load_errors
    assert "profiles.json" in store.load_errors[0]


def test_unknown_profile_schema_is_rejected(tmp_path):
    (tmp_path / "profiles.json").write_text(json.dumps({"schema_version": 999}))

    store = ProfilingStore(storage_dir=str(tmp_path))

    assert store.execution_stats == {}
    assert "unsupported profile schema" in store.load_errors[0]


def test_store_has_no_pickle_migration_entrypoint(tmp_path):
    store = ProfilingStore(storage_dir=str(tmp_path))

    assert not hasattr(store, "migrate_legacy_pickle")


def test_roundtrip_preserves_totals_larger_than_bounded_history(tmp_path):
    stats = OpExecutionStats(op_name="bounded", max_history=2)
    for snapshot in _make_snapshots(5):
        stats.update(snapshot)

    store = ProfilingStore(storage_dir=str(tmp_path))
    store.update_execution_stats("bounded", stats)
    store.save_all()

    loaded = ProfilingStore(storage_dir=str(tmp_path)).get_execution_stats("bounded")
    assert loaded.total_batches == 5
    assert loaded.total_samples == 60
    assert len(loaded.snapshots) == 2
    assert loaded.avg_latency_ms == pytest.approx(stats.avg_latency_ms)


def test_failed_atomic_replace_keeps_previous_profile(tmp_path):
    store = ProfilingStore(storage_dir=str(tmp_path))
    stats = OpExecutionStats(op_name="stable")
    stats.update(_make_snapshots(1)[0])
    store.update_execution_stats("stable", stats)
    store.save_all()
    original = (tmp_path / "profiles.json").read_bytes()

    stats.update(_make_snapshots(1, base_batch=99)[0])
    with patch(
        "data_juicer.core.elasticjuicer.profiler.profiling_store.os.replace",
        side_effect=OSError("simulated replace failure"),
    ):
        with pytest.raises(OSError, match="simulated"):
            store.save_all()

    assert (tmp_path / "profiles.json").read_bytes() == original
    assert not list(tmp_path.glob(".*.tmp"))


def test_resource_throughput_curve_predict():
    curve = ResourceThroughputCurve(
        op_name="test",
        coefficients={"batch_coef": 2.0, "memory_coef": 0.1, "intercept": 10.0},
        model_type="linear",
    )
    t = curve.predict_throughput(batch_size=10, memory_mb=100)
    assert t == 2.0 * 10 + 0.1 * 100 + 10.0


def test_resource_throughput_curve_power_model():
    curve = ResourceThroughputCurve(
        op_name="test",
        coefficients={"scale": 5.0, "power": 0.8},
        model_type="power",
    )
    t = curve.predict_throughput(batch_size=10, memory_mb=0)
    assert abs(t - 5.0 * (10**0.8)) < 1e-6


def test_export_report(tmp_path):
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ProfilingStore(storage_dir=tmpdir)
        stats = OpExecutionStats(op_name="report_op")
        for snap in _make_snapshots(5):
            stats.update(snap)
        store.update_execution_stats("report_op", stats)

        report_path = str(tmp_path / "report.md")
        store.export_report(report_path)
        content = Path(report_path).read_text()
        assert "report_op" in content
        assert "ElasticJuicer" in content

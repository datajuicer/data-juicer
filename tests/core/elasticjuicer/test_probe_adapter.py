from unittest.mock import MagicMock

import pytest

from data_juicer.core.adapter import Adapter
from data_juicer.core.elasticjuicer.profiler.metrics import MetricScope
from data_juicer.core.elasticjuicer.profiler.probe_adapter import ProbeAdapter
from data_juicer.core.elasticjuicer.profiler.profiling_store import ProfilingStore


class FakeOperator:
    def __init__(self, name):
        self._name = name


@pytest.fixture
def probe_result():
    return {
        "time": 0.25,
        "speed": 32.0,
        "resource": [
            {
                "timestamp": 10.0,
                "CPU util.": 0.25,
                "Used mem.": 1000.0,
                "GPU used mem.": [100.0, 200.0],
                "GPU util.": [0.1, 0.4],
            },
            {
                "timestamp": 10.1,
                "CPU util.": 0.75,
                "Used mem.": 1200.0,
                "GPU used mem.": [150.0, 250.0],
                "GPU util.": [0.2, 0.6],
            },
        ],
    }


def test_probe_adapter_converts_system_probe_and_persists_it(tmp_path, probe_result):
    store = ProfilingStore(storage_dir=str(tmp_path))
    adapter = ProbeAdapter(store)

    adapter.stash([FakeOperator("image_mapper")], [probe_result], batch_size=8)

    loaded = ProfilingStore(storage_dir=str(tmp_path)).get_execution_stats("image_mapper")
    assert loaded is not None
    snapshot = loaded.snapshots[-1]
    assert snapshot.batch_size == 8
    assert snapshot.cpu_percent == 75.0
    assert snapshot.memory_mb == 1200.0
    assert snapshot.gpu_memory_mb == 250.0
    assert snapshot.gpu_utilization == 60.0
    assert snapshot.latency_ms == 250.0
    assert snapshot.throughput == 32.0
    assert snapshot.scope is MetricScope.SYSTEM
    assert snapshot.source == "adapter_probe"
    assert snapshot.confidence == 0.5


def test_probe_adapter_rejects_misaligned_operators_and_results(tmp_path, probe_result):
    adapter = ProbeAdapter(ProfilingStore(storage_dir=str(tmp_path)))

    with pytest.raises(ValueError, match="same length"):
        adapter.stash([FakeOperator("one"), FakeOperator("two")], [probe_result], batch_size=8)


def test_adapt_workloads_stashes_probe_results(probe_result):
    adapter = Adapter({"batch_size": 8})
    adapter.probe_adapter = MagicMock()
    adapter.probe_small_batch = MagicMock(return_value=([probe_result], 8))
    adapter.batch_size_strategy = MagicMock(return_value=[4])
    operators = [FakeOperator("image_mapper")]

    assert adapter.adapt_workloads([{}] * 8, operators) == [4]
    adapter.probe_adapter.stash.assert_called_once_with(operators, [probe_result], batch_size=8)


def test_adapt_workloads_persists_profile_under_job_work_dir(tmp_path, probe_result):
    adapter = Adapter({"batch_size": 8, "work_dir": str(tmp_path)})
    adapter.probe_small_batch = MagicMock(return_value=([probe_result], 8))
    adapter.batch_size_strategy = MagicMock(return_value=[4])

    adapter.adapt_workloads([{}] * 8, [FakeOperator("image_mapper")])

    profile_dir = tmp_path / "elastic_juicer_profiles"
    loaded = ProfilingStore(storage_dir=str(profile_dir)).get_execution_stats("image_mapper")
    assert loaded is not None
    assert loaded.snapshots[-1].scope is MetricScope.SYSTEM

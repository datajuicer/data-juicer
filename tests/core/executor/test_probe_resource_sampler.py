import json
import sys
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from data_juicer.core.executor import gpu_memory_probe as probe
from data_juicer.core.executor.probe_resource_sampler import ProbeResourceSampler
from tests.core.executor.test_gpu_memory_probe import (
    FakeOp,
    ProfilingProbeMapper,
    metrics,
)


class FakeProcess:
    def __init__(self):
        self.polled = threading.Event()
        self.cpu_reads = 0

    def memory_info(self):
        if threading.current_thread().name == "gpu-probe-rss-sampler":
            self.polled.set()
            return SimpleNamespace(rss=500)
        return SimpleNamespace(rss=100)

    def cpu_times(self):
        self.cpu_reads += 1
        return SimpleNamespace(user=self.cpu_reads, system=0.5)


def test_cpu_rss_peak_and_cuda_are_scoped_to_the_worker(monkeypatch):
    process = FakeProcess()
    calls = []
    initialized = False

    def read_counter(device):
        calls.append((threading.current_thread().ident, device))
        return 1024

    cuda = SimpleNamespace(
        is_initialized=lambda: initialized,
        memory_allocated=read_counter,
        memory_reserved=read_counter,
        max_memory_allocated=read_counter,
        max_memory_reserved=read_counter,
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    monkeypatch.setitem(
        sys.modules,
        "ray",
        SimpleNamespace(
            is_initialized=lambda: True,
            get_runtime_context=lambda: SimpleNamespace(
                get_node_id=lambda: "node-a", get_accelerator_ids=lambda: {"GPU": ["3"]}
            ),
        ),
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    sampler = ProbeResourceSampler(process=process, sample_interval_seconds=0.001)
    with sampler.phase("construction"):
        assert not calls  # Observing construction must not initialize CUDA.
        initialized = True
        assert process.polled.wait(timeout=2)
    report = sampler.snapshot()
    sample = report["samples"][0]
    assert sample["rss_sampled_peak_bytes"] == 500
    assert sample["rss_start_bytes"] == sample["rss_end_bytes"] == 100
    assert sample["cpu_seconds"] == 1
    assert sample["rss_poll_count"] >= 1
    assert sample["cuda_start"] is None
    assert sample["cuda_end"]["peak_allocated_bytes"] == 1024
    assert calls == [(threading.current_thread().ident, 0)] * 4
    assert report["worker"]["ray_node_id"] == "node-a"
    assert report["worker"]["ray_gpu_ids"] == ["3"]
    assert report["worker"]["cuda_visible_devices"] == "3"
    assert report["cuda_scope"] == "worker_process_allocator_on_device"
    assert report["cuda_peak_scope"] == "since_worker_start"
    assert not any(report["sampling_errors"].values())
    assert not any(thread.name == "gpu-probe-rss-sampler" for thread in threading.enumerate())
    json.dumps(report, allow_nan=False)


def test_sampling_failure_does_not_hide_operator_failure(monkeypatch):
    def unavailable(*args):
        raise RuntimeError("sampler unavailable")

    process = SimpleNamespace(memory_info=unavailable)
    cuda = SimpleNamespace(is_initialized=lambda: True, memory_allocated=unavailable)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    sampler = ProbeResourceSampler(process=process)
    operator_error = ValueError("invalid model input")
    with pytest.raises(ValueError) as raised:
        with sampler.phase("steady", 0):
            raise operator_error
    assert raised.value is operator_error
    report = sampler.snapshot()
    assert report["samples"][0]["error_type"] == "ValueError"
    assert report["samples"][0]["succeeded"] is False
    assert report["samples"][0]["rss_sampled_peak_bytes"] is None
    assert report["sampling_errors"]["process"] >= 2
    assert report["sampling_errors"]["cuda"] == 2


def test_samples_are_bounded_and_cpu_only_probes_do_not_import_torch(monkeypatch):
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    sampler = ProbeResourceSampler(process=FakeProcess(), max_samples=2)
    for index in range(5):
        with sampler.phase("steady", index):
            pass
    assert "torch" not in sys.modules
    report = sampler.snapshot()
    assert len(report["samples"]) == 2
    assert report["dropped_samples"] == 3
    assert all(sample["cuda_end"] is None for sample in report["samples"])


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_poll_interval_is_rejected(value):
    with pytest.raises(ValueError):
        ProbeResourceSampler(sample_interval_seconds=value)


@pytest.mark.parametrize("ordered", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_probe_paths_preserve_results_and_attach_optional_samples(ordered, enabled):
    op = ProfilingProbeMapper(batch_size=2)
    rows = [{"text": "a"}, {"text": "b"}]
    ProfilingProbeMapper.init_count = 0
    ProfilingProbeMapper.process_count = 0

    with patch.object(probe, "_measure_cuda_call", side_effect=lambda fn: (fn(), metrics())):
        if ordered:
            result = probe._run_probe_stage(type(op), (), op._init_kwargs, rows, True, 1, 3, resource_sampling=enabled)
            assert result["rows"] == rows
        else:
            result = probe._run_parallel_probe_job(
                {
                    "op_index": 0,
                    "target": probe._op_spec(op),
                    "dependencies": [],
                    "resource_sampling": enabled,
                },
                rows,
            )
            assert result["output_count"] == 2
    assert ProfilingProbeMapper.init_count == 1
    assert ProfilingProbeMapper.process_count == 4
    assert result["profile"]["output_ratio"] == 1
    assert result["metrics"]["measured_memory_mb"] == 100
    if enabled:
        samples = result["metrics"]["resource_samples"]["samples"]
        assert [sample["phase"] for sample in samples] == ["construction", "warmup", "steady", "steady", "steady"]
        assert [sample["batch_index"] for sample in samples] == [None, 0, 0, 1, 2]
        assert all(sample["succeeded"] for sample in samples)
        assert all(sample["rss_start_bytes"] > 0 for sample in samples)
    else:
        assert "resource_samples" not in result["metrics"]


def test_sampling_does_not_change_memory_plan_and_is_persisted(tmp_path):
    op = FakeOp("gpu", accelerator="cuda", batch_size=2)
    collector = ProbeResourceSampler(process=FakeProcess())
    sampled_metrics = {**metrics(), "resource_samples": collector.snapshot()}
    planner = probe.GPUMemoryProbe(str(tmp_path), resource_sampling=True)
    record = planner._record_from_metrics(0, op, sampled_metrics, 2, probe_mode="parallel", dependencies=[])
    original = planner._record_from_metrics(0, op, metrics(), 2, probe_mode="parallel", dependencies=[])
    assert {key: value for key, value in record.items() if key != "resource_samples"} == original
    assert record["resource_samples"] is not sampled_metrics["resource_samples"]
    planner._save_report([record])
    assert planner._load_report() == [record]
    assert probe.GPUMemoryProbe(str(tmp_path), resource_sampling=False)._load_report() == []


def test_enabling_sampling_invalidates_reports_without_diagnostics(tmp_path):
    disabled = probe.GPUMemoryProbe(str(tmp_path))
    disabled._save_report([{"legacy": True}])
    # Reports from PR1054 have no resource_sampling field; default-off can reuse them.
    report = json.loads((tmp_path / "gpu_probe_results.json").read_text())
    del report["resource_sampling"]
    (tmp_path / "gpu_probe_results.json").write_text(json.dumps(report))
    assert disabled._load_report() == [{"legacy": True}]
    assert probe.GPUMemoryProbe(str(tmp_path), resource_sampling=True)._load_report() == []


def test_resource_sampling_requires_boolean(tmp_path):
    with pytest.raises(ValueError, match="boolean"):
        probe.GPUMemoryProbe(str(tmp_path), resource_sampling="false")

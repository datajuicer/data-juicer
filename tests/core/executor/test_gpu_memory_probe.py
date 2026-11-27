import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from data_juicer.core.executor import gpu_memory_probe as gpu_probe
from data_juicer.core.executor.gpu_memory_probe import (
    GPUMemoryProbe,
    _measure_cuda_call,
    _op_spec,
    _profile_probe_target,
    _run_parallel_probe_job,
    _run_probe_op_rows,
    _run_probe_stage,
    calculate_resource_plan,
    fill_to_batch,
    needs_gpu_probe,
    plan_parallel_probe_jobs,
    probe_sample_count,
)
from data_juicer.core.executor.ray_executor_partitioned import PartitionedRayExecutor
from data_juicer.ops import Filter, Mapper, Pipeline
from data_juicer.utils.unittest_utils import TEST_TAG, DataJuicerTestCaseBase


class FakeDataset:
    def __init__(self, rows):
        self.rows = rows
        self.requested = []

    def get(self, count):
        self.requested.append(count)
        return self.rows[:count]


class FakeRayDataset(FakeDataset):
    """Expose the Ray dataset surface used by offset and shuffle sampling."""

    def __init__(self, rows, *, num_blocks=None):
        super().__init__(rows)
        self.seeds = []
        self.data = SimpleNamespace(
            randomize_block_order=self._randomize_block_order,
            iter_rows=lambda: iter(self.rows),
        )
        if num_blocks is not None:
            self.data.num_blocks = lambda: num_blocks

    def _randomize_block_order(self, *, seed=None):
        self.seeds.append(seed)
        reordered = list(reversed(self.rows))
        return SimpleNamespace(take=lambda count: reordered[:count])


class FakeOp:
    def __init__(self, name, *, accelerator="cpu", batch_size=1, memory=None, num_gpus=None):
        self._name = name
        self.accelerator = accelerator
        self.batch_size = batch_size
        self.memory = memory
        self.num_gpus = num_gpus
        self.num_cpus = None
        self.runtime_env = None
        self._init_args = ()
        self._init_kwargs = {"batch_size": batch_size, "accelerator": accelerator}
        self._op_cfg = {
            name: {
                "batch_size": batch_size,
                "accelerator": accelerator,
            }
        }


class FakeGPUPipeline(Pipeline):
    _name = "fake_gpu_pipeline"
    _accelerator = "cuda"

    def run(self, dataset):
        return dataset


class CapturingProbeOp:
    init_kwargs = None

    def __init__(self, **kwargs):
        type(self).init_kwargs = kwargs

    def run(self, dataset):
        return dataset


class ArrayProducingProbeOp(CapturingProbeOp):
    def run(self, dataset):
        return dataset.map(
            lambda batch: {
                "pixels": [np.full((2, 3, 3), index, dtype=np.uint8) for index in range(len(batch["text"]))]
            },
            batched=True,
        )


class ArrayConsumingProbeOp(CapturingProbeOp):
    def run(self, dataset):
        pixels = dataset[:]["pixels"]
        assert isinstance(pixels, list)
        assert isinstance(pixels[0], np.ndarray)
        assert pixels[0].shape == (2, 3, 3)
        return dataset


class DeclaredProbeMapper(Mapper):
    _batched_op = True

    def __init__(
        self,
        name="declared",
        *,
        accelerator="cpu",
        input_columns=(),
        output_columns=(),
        **kwargs,
    ):
        super().__init__(
            accelerator=accelerator,
            input_columns=input_columns,
            output_columns=output_columns,
            **kwargs,
        )
        self._name = name
        self._op_cfg = {
            name: {
                "accelerator": accelerator,
                "input_columns": list(input_columns),
                "output_columns": list(output_columns),
                "batch_size": self.batch_size,
            }
        }

    def process_batched(self, samples, rank=None):
        return samples


class ArrayBatchProducer(Mapper):
    _batched_op = True

    def process_batched(self, samples):
        samples["pixels"] = [np.full((2, 3, 3), index, dtype=np.uint8) for index in range(len(samples["text"]))]
        return samples


class ArrayBatchConsumer(Mapper):
    _batched_op = True

    def process_batched(self, samples, rank=None):
        assert isinstance(samples["pixels"], list)
        assert isinstance(samples["pixels"][0], np.ndarray)
        samples["shape"] = [tuple(value.shape) for value in samples["pixels"]]
        return samples


class ProfilingProbeMapper(Mapper):
    _batched_op = True
    init_count = 0
    process_count = 0

    def __init__(self, **kwargs):
        type(self).init_count += 1
        super().__init__(**kwargs)

    def process_batched(self, samples, rank=None):
        type(self).process_count += 1
        return samples


class FlakyBatchMapper(Mapper):
    """Fail on batches that contain an invalid row, like a real bad sample."""

    _batched_op = True

    def process_batched(self, samples, rank=None):
        if any(text == "bad" for text in samples["text"]):
            raise ValueError("invalid sample")
        return samples


class UndeclaredProbeMapper(Mapper):
    """A contract-free Mapper, so preflight keeps it on ordered replay."""

    _batched_op = True

    def process_batched(self, samples, rank=None):
        return samples


class UndeclaredProbeFilter(Filter):
    """A contract-free Filter that keeps nothing."""

    _batched_op = True

    def compute_stats_batched(self, samples, rank=None):
        return samples

    def process_batched(self, samples):
        return [False] * len(samples["text"])


def metrics(measured=100, total=1000, name="Fake GPU"):
    return {
        "gpu_name": name,
        "total_mb": total,
        "baseline_used_mb": 20,
        "peak_used_mb": 20 + measured,
        "torch_peak_reserved_mb": measured,
        "measured_memory_mb": measured,
    }


def test_target_selection_and_maximum_batch_size():
    cpu = FakeOp("cpu", batch_size=100)
    target_small = FakeOp("small", accelerator="cuda", batch_size=3)
    target_large = FakeOp("large", accelerator="cuda", batch_size=11)
    explicit_gpu = FakeOp("explicit", accelerator="cuda", batch_size=20, num_gpus=0.5)
    memory_gpu = FakeOp("memory", accelerator="cuda", batch_size=30, memory=2)

    assert needs_gpu_probe(target_small)
    assert not needs_gpu_probe(cpu)
    assert not needs_gpu_probe(explicit_gpu)
    assert not needs_gpu_probe(memory_gpu)
    assert probe_sample_count([cpu, target_small, target_large, explicit_gpu, memory_gpu]) == 11


def test_batch_fill_and_resource_rounding():
    assert fill_to_batch([{"id": 1}, {"id": 2}], 5) == [
        {"id": 1},
        {"id": 2},
        {"id": 1},
        {"id": 2},
        {"id": 1},
    ]
    assert fill_to_batch([{"id": 1}, {"id": 2}, {"id": 3}], 2) == [
        {"id": 1},
        {"id": 2},
    ]
    plan = calculate_resource_plan(100, 1000)
    assert plan == {
        "planned_memory_mb": 111,
        "memory": 111 / 1024,
        "memory_fraction": 0.12,
        "num_gpus": 0.2,
    }
    assert calculate_resource_plan(100, 1000, max_gpu_workers_per_device=10)["num_gpus"] == 0.12

    with pytest.raises(RuntimeError, match="fit on one GPU"):
        calculate_resource_plan(950, 1000)
    with pytest.raises(RuntimeError, match="invalid peak"):
        calculate_resource_plan(0, 1000)


def test_cuda_measurement_does_not_touch_context_until_target_returns():
    calls = []
    fake_cuda = SimpleNamespace(
        synchronize=lambda device: calls.append(("synchronize", device)),
        max_memory_reserved=lambda device: calls.append(("peak", device)) or 300 * 1024**2,
    )

    def snapshot():
        calls.append(("snapshot", 0))
        return {"used_mb": 250, "total_mb": 1000, "gpu_name": "Fake GPU"}

    def target():
        calls.append(("target", 0))
        return "result"

    with (
        patch.dict(sys.modules, {"torch": SimpleNamespace(cuda=fake_cuda)}),
        patch(
            "data_juicer.core.executor.gpu_memory_probe._cuda_memory_snapshot",
            side_effect=snapshot,
        ),
    ):
        result, measured = _measure_cuda_call(target)

    assert result == "result"
    assert calls == [("target", 0), ("synchronize", 0), ("snapshot", 0), ("peak", 0)]
    assert measured["baseline_used_mb"] == 0
    assert measured["peak_used_mb"] == 250
    assert measured["torch_peak_reserved_mb"] == 300
    assert measured["measured_memory_mb"] == 300
    assert measured["memory_measurement_mode"] == "post_call_device_usage_and_allocator_peak"


def test_probe_stage_disables_inner_multiprocessing_and_keeps_recipe_fault_tolerance():
    result = _run_probe_stage(
        CapturingProbeOp,
        (),
        {"num_proc": 8, "auto_op_parallelism": True, "skip_op_error": True},
        [{"text": "sample"}],
        False,
    )

    assert result["rows"] == [{"text": "sample"}]
    assert CapturingProbeOp.init_kwargs["num_proc"] is None
    assert CapturingProbeOp.init_kwargs["auto_op_parallelism"] is False
    assert CapturingProbeOp.init_kwargs["skip_op_error"] is True


def test_replay_skips_invalid_rows_when_the_recipe_asks_for_it():
    rows = [{"text": "good"}, {"text": "bad"}, {"text": "also good"}]
    tolerant = FlakyBatchMapper(batch_size=1, skip_op_error=True)
    strict = FlakyBatchMapper(batch_size=1, skip_op_error=False)

    assert _run_probe_op_rows(_op_spec(tolerant), rows) == [
        {"text": "good"},
        {"text": "also good"},
    ]
    with pytest.raises(ValueError, match="invalid sample"):
        _run_probe_op_rows(_op_spec(strict), rows)


def test_probe_stage_preserves_multidimensional_numpy_arrays_between_ops():
    produced = _run_probe_stage(
        ArrayProducingProbeOp,
        (),
        {},
        [{"text": "first"}, {"text": "second"}],
        False,
    )

    assert isinstance(produced["rows"][0]["pixels"], np.ndarray)
    assert produced["rows"][0]["pixels"].dtype == np.uint8
    consumed = _run_probe_stage(
        ArrayConsumingProbeOp,
        (),
        {},
        produced["rows"],
        False,
    )
    assert isinstance(consumed["rows"][0]["pixels"], np.ndarray)


def test_direct_batch_replay_preserves_numpy_arrays_without_arrow_round_trip():
    producer = ArrayBatchProducer(batch_size=2, input_columns=["text"], output_columns=["pixels"])
    consumer = ArrayBatchConsumer(
        batch_size=2,
        input_columns=["pixels"],
        output_columns=["shape"],
    )
    rows = [{"text": "first"}, {"text": "second"}]

    produced = _run_probe_op_rows(_op_spec(producer), rows)
    consumed = _run_probe_op_rows(_op_spec(consumer), produced)

    assert isinstance(produced[0]["pixels"], np.ndarray)
    assert produced[0]["pixels"].dtype == np.uint8
    assert consumed[0]["shape"] == (2, 3, 3)


def test_target_profile_separates_init_warmup_and_steady_batches():
    op = ProfilingProbeMapper(batch_size=2)
    spec = _op_spec(op)
    ProfilingProbeMapper.init_count = 0
    ProfilingProbeMapper.process_count = 0

    output, profile = _profile_probe_target(
        spec,
        [{"text": "a"}, {"text": "b"}],
        warmup_batches=1,
        steady_batches=3,
    )

    assert len(output) == 2
    assert ProfilingProbeMapper.init_count == 1
    assert ProfilingProbeMapper.process_count == 4
    assert profile["warmup_batches"] == 1
    assert profile["steady_batches"] == 3
    assert profile["steady_rows_per_second"] > 0
    assert profile["output_ratio"] == 1


def test_parallel_worker_replays_cpu_dependency_and_measures_one_target():
    producer = ArrayBatchProducer(
        batch_size=2,
        input_columns=["text"],
        output_columns=["pixels"],
    )
    consumer = ArrayBatchConsumer(
        accelerator="cuda",
        batch_size=3,
        input_columns=["pixels"],
        output_columns=["shape"],
    )
    job = {
        "op_index": 1,
        "dependencies": [_op_spec(producer)],
        "target": _op_spec(consumer),
    }

    marker = "DATA_JUICER_GPU_PREFLIGHT_OP"
    original_profile = _profile_probe_target

    def profile(*args, **kwargs):
        assert marker not in os.environ
        return original_profile(*args, **kwargs)

    def measure(function):
        assert os.environ[marker] == "custom_aesthetic_mapper"
        output = function()
        assert os.environ[marker] == "custom_aesthetic_mapper"
        return output, metrics(measured=123)

    with (
        patch.dict(os.environ, {marker: "custom_aesthetic_mapper"}),
        patch("data_juicer.core.executor.gpu_memory_probe._profile_probe_target", side_effect=profile),
        patch("data_juicer.core.executor.gpu_memory_probe._measure_cuda_call", side_effect=measure),
    ):
        result = _run_parallel_probe_job(job, [{"text": "a"}, {"text": "b"}])
        assert os.environ[marker] == "custom_aesthetic_mapper"

    assert result["op_index"] == 1
    assert result["sample_count"] == 3
    assert result["metrics"]["measured_memory_mb"] == 123
    assert result["timing_seconds"]["dependencies"]["ArrayBatchProducer"] >= 0
    assert result["timing_seconds"]["target"] >= 0
    assert result["timing_seconds"]["total"] >= result["timing_seconds"]["target"]


def test_dependency_planner_parallelizes_independent_gpu_mappers():
    bucket = DeclaredProbeMapper(
        name="bucket",
        input_columns=["images"],
        output_columns=["_bucket_img"],
    )
    first = DeclaredProbeMapper(
        name="first",
        accelerator="cuda",
        input_columns=["images", "_bucket_img"],
        output_columns=["__dj__meta__.first"],
    )
    second = DeclaredProbeMapper(
        name="second",
        accelerator="cuda",
        input_columns=["images", "_bucket_img"],
        output_columns=["__dj__meta__.second"],
    )

    jobs, fallback = plan_parallel_probe_jobs([bucket, first, second], {1: first, 2: second})

    assert fallback == {}
    assert [job["op_index"] for job in jobs] == [1, 2]
    assert [[op._name for op in job["dependencies"]] for job in jobs] == [["bucket"], ["bucket"]]


def test_dependency_planner_keeps_gpu_dependency_and_unknown_contract_ordered():
    producer = DeclaredProbeMapper(
        name="producer",
        accelerator="cuda",
        input_columns=["images"],
        output_columns=["embedding"],
    )
    consumer = DeclaredProbeMapper(
        name="consumer",
        accelerator="cuda",
        input_columns=["embedding"],
        output_columns=["score"],
    )
    jobs, fallback = plan_parallel_probe_jobs([producer, consumer], {0: producer, 1: consumer})

    assert [job["op_index"] for job in jobs] == [0]
    assert "depends on earlier GPU operator" in fallback[1]

    unknown = FakeOp("unknown")
    jobs, fallback = plan_parallel_probe_jobs([unknown, consumer], {1: consumer})
    assert jobs == []
    assert "missing input_columns/output_columns" in fallback[1]


def test_resolve_uses_parallel_runner_for_declared_independent_targets(tmp_path):
    bucket = DeclaredProbeMapper(
        name="bucket",
        batch_size=4,
        input_columns=["images"],
        output_columns=["_bucket_img"],
    )
    first = DeclaredProbeMapper(
        name="first",
        accelerator="cuda",
        batch_size=2,
        input_columns=["images", "_bucket_img"],
        output_columns=["__dj__meta__.first"],
    )
    second = DeclaredProbeMapper(
        name="second",
        accelerator="cuda",
        batch_size=4,
        input_columns=["images", "_bucket_img"],
        output_columns=["__dj__meta__.second"],
    )
    parallel_calls = []

    def parallel_runner(jobs, rows):
        parallel_calls.append(
            (
                [job["target"]._name for job in jobs],
                [[op._name for op in job["dependencies"]] for job in jobs],
                len(rows),
            )
        )
        return {
            1: {"sample_count": 2, "metrics": metrics(measured=100)},
            2: {"sample_count": 4, "metrics": metrics(measured=200)},
        }

    def ordered_runner(*args):
        raise AssertionError("fully declared independent targets should not use ordered replay")

    records = GPUMemoryProbe(
        str(tmp_path),
        stage_runner=ordered_runner,
        parallel_runner=parallel_runner,
    ).resolve(
        FakeDataset([{"images": [f"{index}.jpg"]} for index in range(8)]),
        [bucket, first, second],
    )

    assert parallel_calls == [(["first", "second"], [["bucket"], ["bucket"]], 4)]
    assert [record["probe_mode"] for record in records] == ["parallel", "parallel"]
    assert [record["sample_count"] for record in records] == [2, 4]
    assert first.num_gpus == 0.2
    assert second.num_gpus == 0.23


def test_gpu_dependent_target_falls_back_after_independent_parallel_probe(tmp_path):
    producer = DeclaredProbeMapper(
        name="producer",
        accelerator="cuda",
        batch_size=2,
        input_columns=["images"],
        output_columns=["embedding"],
    )
    consumer = DeclaredProbeMapper(
        name="consumer",
        accelerator="cuda",
        batch_size=3,
        input_columns=["embedding"],
        output_columns=["score"],
    )
    ordered_calls = []

    def parallel_runner(jobs, rows):
        assert [job["op_index"] for job in jobs] == [0]
        return {0: {"sample_count": 2, "metrics": metrics(measured=100)}}

    def ordered_runner(op, rows, measure_memory):
        ordered_calls.append((op._name, len(rows), measure_memory))
        result = {"rows": rows}
        if measure_memory:
            result["metrics"] = metrics(measured=200)
        return result

    records = GPUMemoryProbe(
        str(tmp_path),
        stage_runner=ordered_runner,
        parallel_runner=parallel_runner,
    ).resolve(FakeDataset([{"images": ["x.jpg"]} for _ in range(3)]), [producer, consumer])

    assert ordered_calls == [("producer", 3, False), ("consumer", 3, True)]
    assert [record["probe_mode"] for record in records] == ["parallel", "ordered"]


def test_parallel_probe_concurrency_serializes_by_default_and_bounds_an_explicit_cap(tmp_path):
    targets = [
        DeclaredProbeMapper(
            name=f"gpu_{index}",
            accelerator="cuda",
            input_columns=["images"],
            output_columns=[f"score_{index}"],
            num_cpus=4,
        )
        for index in range(10)
    ]
    jobs, fallback = plan_parallel_probe_jobs(targets, {index: op for index, op in enumerate(targets)})
    assert not fallback

    with patch(
        "data_juicer.core.executor.gpu_memory_probe.ray",
        SimpleNamespace(cluster_resources=lambda: {"GPU": 8, "CPU": 24}),
    ):
        automatic = GPUMemoryProbe(str(tmp_path))._parallel_probe_limit(jobs)
        explicit = GPUMemoryProbe(str(tmp_path), max_concurrent_probes=3)._parallel_probe_limit(jobs)
        over_provisioned = GPUMemoryProbe(str(tmp_path), max_concurrent_probes=20)._parallel_probe_limit(jobs)

    # Concurrent probes initialize several models at once, and storage bandwidth
    # is not a Ray resource, so 'auto' probes one target at a time.
    assert automatic == 1
    assert explicit == 3
    # Ten jobs and eight GPUs, but 24 CPUs only cover six four-CPU probes.
    assert over_provisioned == 6

    with patch(
        "data_juicer.core.executor.gpu_memory_probe.ray",
        SimpleNamespace(cluster_resources=lambda: {"GPU": 0, "CPU": 24}),
    ):
        with pytest.raises(RuntimeError, match="no GPU resources"):
            GPUMemoryProbe(str(tmp_path))._parallel_probe_limit(jobs)


def test_parallel_probe_timeout_names_stalled_operator_and_cancels_future(tmp_path):
    target = DeclaredProbeMapper(
        name="stalled_gpu",
        accelerator="cuda",
        input_columns=["images"],
        output_columns=["score"],
    )
    jobs, fallback = plan_parallel_probe_jobs([target], {0: target})
    assert not fallback

    future = object()
    cancelled = []
    remote_task = SimpleNamespace(options=lambda **kwargs: SimpleNamespace(remote=lambda *args: future))
    fake_ray = SimpleNamespace(
        cluster_resources=lambda: {"GPU": 1, "CPU": 1},
        remote=lambda **kwargs: lambda function: remote_task,
        wait=lambda refs, **kwargs: ([], refs),
        cancel=lambda ref, **kwargs: cancelled.append(ref),
    )

    with (
        patch("data_juicer.core.executor.gpu_memory_probe.ray", fake_ray),
        patch(
            "data_juicer.core.executor.gpu_memory_probe.time.monotonic",
            side_effect=[0.0, 0.0, 6.0],
        ),
        pytest.raises(TimeoutError, match=r"Op\[stalled_gpu\].*5s timeout"),
    ):
        GPUMemoryProbe(
            str(tmp_path),
            probe_timeout_seconds=5,
        )._run_parallel_jobs_with_ray(jobs, [{"images": ["x.jpg"]}])

    assert cancelled == [future]


def test_parallel_probe_failure_prevents_formal_run(tmp_path):
    target = DeclaredProbeMapper(
        name="gpu",
        accelerator="cuda",
        input_columns=["images"],
        output_columns=["score"],
    )

    def fail(*args):
        raise MemoryError("CUDA out of memory")

    with pytest.raises(RuntimeError, match="formal Ray experiment was not started") as error:
        GPUMemoryProbe(str(tmp_path), parallel_runner=fail).resolve(
            FakeDataset([{"images": ["x.jpg"]}]),
            [target],
        )

    assert isinstance(error.value.__cause__, MemoryError)


def test_measurement_from_a_fully_skipped_batch_is_rejected(tmp_path):
    target = DeclaredProbeMapper(
        name="gpu",
        accelerator="cuda",
        batch_size=2,
        input_columns=["images"],
        output_columns=["score"],
        skip_op_error=True,
    )

    def parallel_runner(jobs, rows):
        return {0: {"sample_count": 2, "output_count": 0, "metrics": metrics()}}

    with pytest.raises(RuntimeError, match="skip_op_error dropped the whole batch"):
        GPUMemoryProbe(str(tmp_path), parallel_runner=parallel_runner).resolve(
            FakeDataset([{"images": ["a.jpg"]}, {"images": ["b.jpg"]}]),
            [target],
        )


def test_empty_mapper_measurement_is_rejected_even_without_skip_op_error(tmp_path):
    """A Mapper owes one row per input row, so an empty batch is never measurable."""
    target = DeclaredProbeMapper(
        name="gpu",
        accelerator="cuda",
        batch_size=2,
        input_columns=["images"],
        output_columns=["score"],
        skip_op_error=False,
    )

    def parallel_runner(jobs, rows):
        return {0: {"sample_count": 2, "output_count": 0, "metrics": metrics()}}

    with pytest.raises(RuntimeError, match="should have raised"):
        GPUMemoryProbe(str(tmp_path), parallel_runner=parallel_runner).resolve(
            FakeDataset([{"images": ["a.jpg"]}, {"images": ["b.jpg"]}]),
            [target],
        )


def test_filter_target_that_keeps_no_samples_is_only_reported_as_worst_case(tmp_path):
    target = UndeclaredProbeFilter(accelerator="cuda", batch_size=2, skip_op_error=True)

    def run_stage(op, rows, measure_memory):
        return {"rows": [], "metrics": metrics()}

    with patch("data_juicer.core.executor.gpu_memory_probe.logger.warning") as warning:
        records = GPUMemoryProbe(str(tmp_path), stage_runner=run_stage).resolve(
            FakeDataset([{"text": "a"}, {"text": "b"}]),
            [target],
        )

    assert len(records) == 1
    assert target.num_gpus == 0.2
    assert "worst-case" in warning.call_args[0][0]


def test_ordered_replay_uses_front_sample_and_refills_after_filter(tmp_path):
    dataset = FakeDataset([{"id": index} for index in range(8)])
    cpu_filter = FakeOp("cpu_filter")
    first_gpu = FakeOp("first_gpu", accelerator="cuda", batch_size=2)
    second_gpu = FakeOp("second_gpu", accelerator="cuda", batch_size=4)
    calls = []

    def run_stage(op, rows, measure_memory):
        calls.append((op._name, len(rows), measure_memory))
        output_rows = rows[:1] if op is cpu_filter else rows
        result = {"rows": output_rows}
        if measure_memory:
            result["metrics"] = metrics(measured=100 if op is first_gpu else 200)
        return result

    records = GPUMemoryProbe(str(tmp_path), stage_runner=run_stage).resolve(
        dataset, [cpu_filter, first_gpu, second_gpu]
    )

    assert dataset.requested == [4]
    assert calls == [
        ("cpu_filter", 4, False),
        ("first_gpu", 2, True),
        ("first_gpu", 1, False),
        ("second_gpu", 4, True),
    ]
    assert [record["op_name"] for record in records] == ["first_gpu", "second_gpu"]
    assert first_gpu.memory == 111 / 1024
    assert first_gpu.num_gpus == 0.2
    assert second_gpu.memory == 221 / 1024
    assert second_gpu.num_gpus == 0.23
    assert first_gpu._gpu_memory_fraction == 0.12
    assert second_gpu._gpu_memory_fraction == 0.23
    assert first_gpu._init_kwargs["num_gpus"] == 0.2
    assert first_gpu._op_cfg["first_gpu"]["memory"] == 111 / 1024

    report = json.loads((tmp_path / "gpu_probe_results.json").read_text())
    assert report["version"] == 7
    assert report["observability_version"] == 5
    assert report["memory_headroom"] == 1.1
    assert report["max_gpu_workers_per_device"] == 5
    assert report["warmup_batches"] == 1
    assert report["steady_batches"] == 3
    assert report["sample_policy"] == {"offset": 0, "shuffle": False, "seed": None}
    assert report["sample_source"] == "the dataset head"
    assert report["operators"][1]["sample_count"] == 4


@pytest.mark.parametrize("first_batch,second_batch", [(1, 4), (2, 4), (4, 4), (8, 4)])
def test_ordered_measurement_preserves_full_upstream_lineage(tmp_path, first_batch, second_batch):
    rows = [{"id": index, "visits": []} for index in range(max(first_batch, second_batch))]
    first = FakeOp("first", accelerator="cuda", batch_size=first_batch)
    second = FakeOp("second", accelerator="cuda", batch_size=second_batch)
    calls = []

    def run_stage(op, inputs, measure):
        calls.append((op._name, [row["id"] for row in inputs], measure))
        if op is first:
            for row in inputs:
                row["visits"].append("first")
        else:
            assert [row["id"] for row in inputs] == list(range(second_batch))
            assert all(row["visits"] == ["first"] for row in inputs)
        return {"rows": inputs, "metrics": metrics(measured=100 if measure else 900)}

    records = GPUMemoryProbe(str(tmp_path), stage_runner=run_stage).resolve(FakeDataset(rows), [first, second])

    assert [record["sample_count"] for record in records] == [first_batch, second_batch]
    assert all(record["measured_memory_mb"] == 100 for record in records)
    assert all(row["visits"] == [] for row in rows), "Measured batches must not mutate the representative input"
    replays = [call for call in calls if not call[2]]
    assert len(replays) == int(first_batch < second_batch)
    if replays:
        assert replays[0][1] == list(range(second_batch))


def test_ordered_filter_can_keep_unmeasured_tail_for_downstream(tmp_path):
    first = UndeclaredProbeFilter(accelerator="cuda", batch_size=1)
    second = FakeOp("second", accelerator="cuda", batch_size=4)
    observed = []

    def run_stage(op, rows, measure):
        if op is first:
            # The measured head is empty, but the original tail survives.
            return {"rows": [row for row in rows if row["id"] >= 2], "metrics": metrics()}
        observed.append([row["id"] for row in rows])
        return {"rows": rows, "metrics": metrics()}

    GPUMemoryProbe(str(tmp_path), stage_runner=run_stage).resolve(
        FakeDataset([{"id": index} for index in range(4)]), [first, second]
    )
    assert observed == [[2, 3, 2, 3]]


def test_ordered_padding_is_not_propagated_as_real_upstream_rows(tmp_path):
    first = FakeOp("first", accelerator="cuda", batch_size=4)
    second = FakeOp("second", accelerator="cuda", batch_size=6)
    replay_sizes = []

    def run_stage(op, rows, measure):
        if op is first:
            if not measure:
                replay_sizes.append(len(rows))
            return {"rows": [{**row, "upstream_count": len(rows)} for row in rows], "metrics": metrics()}
        assert [row["id"] for row in rows] == [0, 1, 0, 1, 0, 1]
        assert all(row["upstream_count"] == 2 for row in rows)
        return {"rows": rows, "metrics": metrics()}

    GPUMemoryProbe(str(tmp_path), stage_runner=run_stage).resolve(FakeDataset([{"id": 0}, {"id": 1}]), [first, second])
    assert replay_sizes == [2]


def test_ordered_representative_replay_failure_aborts_before_saving_report(tmp_path):
    first = FakeOp("first", accelerator="cuda", batch_size=1)
    second = FakeOp("second", accelerator="cuda", batch_size=2)

    def run_stage(op, rows, measure):
        if not measure:
            raise ValueError("invalid tail image")
        return {"rows": rows, "metrics": metrics()}

    with pytest.raises(RuntimeError, match=r"representative rows.*Op\[first\]") as caught:
        GPUMemoryProbe(str(tmp_path), stage_runner=run_stage).resolve(
            FakeDataset([{"id": 0}, {"id": 1}]), [first, second]
        )
    assert isinstance(caught.value.__cause__, ValueError)
    assert not (tmp_path / "gpu_probe_results.json").exists()


def probe_sample_ids(tmp_path, dataset, **options):
    """Return the sample ids each probed stage received."""
    sampled = []

    def run_stage(op, rows, measure_memory):
        sampled.append([row["id"] for row in rows])
        return {"rows": rows, "metrics": metrics()}

    GPUMemoryProbe(str(tmp_path), stage_runner=run_stage, **options).resolve(
        dataset, [FakeOp("gpu", accelerator="cuda", batch_size=2)]
    )
    return sampled


def test_probe_sample_offset_skips_the_dataset_head(tmp_path):
    dataset = FakeDataset([{"id": index} for index in range(6)])

    assert probe_sample_ids(tmp_path, dataset, sample_offset=2) == [[2, 3]]
    assert dataset.requested == [4]


def test_probe_sample_offset_streams_past_the_head_of_a_ray_dataset(tmp_path):
    dataset = FakeRayDataset([{"id": index} for index in range(6)])

    assert probe_sample_ids(tmp_path, dataset, sample_offset=3) == [[3, 4]]
    assert dataset.requested == []


def test_probe_sample_offset_beyond_the_dataset_falls_back_to_the_head(tmp_path):
    dataset = FakeDataset([{"id": 0}, {"id": 1}])

    with patch("data_juicer.core.executor.gpu_memory_probe.logger.warning") as warning:
        assert probe_sample_ids(tmp_path, dataset, sample_offset=10) == [[0, 1]]

    assert dataset.requested == [12, 2]
    assert "row offset 10" in warning.call_args[0][0]


def test_probe_sample_shuffle_reads_a_randomized_block(tmp_path):
    dataset = FakeRayDataset([{"id": index} for index in range(6)])

    assert probe_sample_ids(tmp_path, dataset, sample_shuffle=True, sample_seed=7) == [[5, 4]]
    assert dataset.seeds == [7]
    assert dataset.requested == []


def test_probe_sample_shuffle_falls_back_to_the_head_without_block_support(tmp_path):
    dataset = FakeDataset([{"id": index} for index in range(6)])

    with patch("data_juicer.core.executor.gpu_memory_probe.logger.warning") as warning:
        assert probe_sample_ids(tmp_path, dataset, sample_shuffle=True) == [[0, 1]]

    assert dataset.requested == [2]
    assert "sampling the head instead" in warning.call_args[0][0]


def test_probe_sample_shuffle_reports_the_ignored_offset(tmp_path):
    dataset = FakeRayDataset([{"id": index} for index in range(6)])

    with patch("data_juicer.core.executor.gpu_memory_probe.logger.warning") as warning:
        assert probe_sample_ids(tmp_path, dataset, sample_shuffle=True, sample_offset=2) == [[5, 4]]

    assert "ignores gpu_probe_sample_offset=2" in warning.call_args[0][0]


def test_single_block_shuffle_hands_the_sample_back_to_the_offset(tmp_path):
    """Reordering one block reference moves nothing, so the offset must still apply."""
    dataset = FakeRayDataset([{"id": index} for index in range(6)], num_blocks=1)

    with patch("data_juicer.core.executor.gpu_memory_probe.logger.warning") as warning:
        assert probe_sample_ids(tmp_path, dataset, sample_shuffle=True, sample_offset=3) == [[3, 4]]

    assert dataset.seeds == []
    assert "single-block dataset" in warning.call_args[0][0]
    assert json.loads((tmp_path / "gpu_probe_results.json").read_text())["sample_source"] == "row offset 3"


def test_seedless_shuffle_report_is_observed_but_never_reused(tmp_path):
    """Its policy cannot distinguish two runs that sampled different rows."""
    dataset = FakeRayDataset([{"id": index} for index in range(6)])

    assert probe_sample_ids(tmp_path, dataset, sample_shuffle=True) == [[5, 4]]
    report = json.loads((tmp_path / "gpu_probe_results.json").read_text())
    assert report["sample_source"] == "randomized blocks (seed=None)"

    reprobed = probe_sample_ids(
        tmp_path,
        dataset,
        sample_shuffle=True,
        hardware_reader=lambda: {"gpu_name": "Fake GPU", "total_mb": 1000},
    )

    assert reprobed == [[5, 4]]


def test_seeded_shuffle_report_is_reused(tmp_path):
    dataset = FakeRayDataset([{"id": index} for index in range(6)])
    assert probe_sample_ids(tmp_path, dataset, sample_shuffle=True, sample_seed=7) == [[5, 4]]

    reprobed = probe_sample_ids(
        tmp_path,
        dataset,
        sample_shuffle=True,
        sample_seed=7,
        hardware_reader=lambda: {"gpu_name": "Fake GPU", "total_mb": 1000},
    )

    assert reprobed == []


def test_sampling_policy_change_invalidates_cached_plan(tmp_path):
    dataset = FakeDataset([{"id": index} for index in range(6)])
    assert probe_sample_ids(tmp_path, dataset) == [[0, 1]]

    reprobed = probe_sample_ids(
        tmp_path,
        dataset,
        sample_offset=2,
        hardware_reader=lambda: {"gpu_name": "Fake GPU", "total_mb": 1000},
    )

    assert reprobed == [[2, 3]]


def test_matching_report_is_reused_without_replaying_samples(tmp_path):
    first_dataset = FakeDataset([{"id": 1}, {"id": 2}])
    first_op = FakeOp("gpu", accelerator="cuda", batch_size=2)
    GPUMemoryProbe(str(tmp_path), stage_runner=lambda op, rows, measure: {"rows": rows, "metrics": metrics()}).resolve(
        first_dataset, [first_op]
    )

    fresh_op = FakeOp("gpu", accelerator="cuda", batch_size=2)
    fresh_dataset = FakeDataset([{"id": 99}])

    def must_not_run(*args):
        raise AssertionError("cached probe should not replay the recipe")

    records = GPUMemoryProbe(
        str(tmp_path),
        stage_runner=must_not_run,
        hardware_reader=lambda: {"gpu_name": "Fake GPU", "total_mb": 1000},
    ).resolve(fresh_dataset, [fresh_op])

    assert fresh_dataset.requested == []
    assert len(records) == 1
    assert fresh_op.num_gpus == 0.2
    assert fresh_op._gpu_memory_fraction == 0.12
    assert fresh_op.memory == 111 / 1024


def test_hardware_mismatch_reprobes(tmp_path):
    op = FakeOp("gpu", accelerator="cuda")
    GPUMemoryProbe(str(tmp_path), stage_runner=lambda op, rows, measure: {"rows": rows, "metrics": metrics()}).resolve(
        FakeDataset([{"id": 1}]), [op]
    )

    replayed = []

    def rerun(op, rows, measure):
        replayed.append(op._name)
        return {"rows": rows, "metrics": metrics(measured=50, total=2000, name="New GPU")}

    fresh_op = FakeOp("gpu", accelerator="cuda")
    GPUMemoryProbe(
        str(tmp_path),
        stage_runner=rerun,
        hardware_reader=lambda: {"gpu_name": "New GPU", "total_mb": 2000},
    ).resolve(FakeDataset([{"id": 1}]), [fresh_op])

    assert replayed == ["gpu"]
    assert fresh_op.num_gpus == 0.2
    assert fresh_op._gpu_memory_fraction == 0.03


def test_imported_report_reuses_measurements_across_job_work_directories(tmp_path):
    first_dir, second_dir = tmp_path / "job-one", tmp_path / "job-two"
    first = FakeOp("gpu", accelerator="cuda", batch_size=2)
    first._op_cfg["gpu"]["work_dir"] = str(first_dir)
    GPUMemoryProbe(str(first_dir), stage_runner=lambda op, rows, measure: {"rows": rows, "metrics": metrics()}).resolve(
        FakeDataset([{"id": 1}, {"id": 2}]), [first]
    )

    second_dir.mkdir()
    (second_dir / "gpu_probe_results.json").write_bytes((first_dir / "gpu_probe_results.json").read_bytes())
    fresh = FakeOp("gpu", accelerator="cuda", batch_size=2)
    fresh._op_cfg["gpu"]["work_dir"] = str(second_dir)
    dataset = FakeDataset([{"id": 1}, {"id": 2}])

    def must_not_probe(*args):
        raise AssertionError("An imported compatible report must not reload and probe the model")

    records = GPUMemoryProbe(
        str(second_dir),
        stage_runner=must_not_probe,
        hardware_reader=lambda: {"gpu_name": "Fake GPU", "total_mb": 1000},
    ).resolve(dataset, [fresh])
    assert dataset.requested == []
    assert len(records) == 1 and fresh.memory == 111 / 1024
    assert fresh._op_cfg["gpu"]["work_dir"] == str(second_dir)


@pytest.mark.parametrize(
    "key,before,after",
    [
        ("batch_size", 2, 4),
        ("model_path", "model-a", "model-b"),
        ("precision", "float32", "float16"),
        ("memory_budget_bytes", 1024, 2048),
    ],
)
def test_imported_report_still_reprobes_semantic_changes(tmp_path, key, before, after):
    first = FakeOp("gpu", accelerator="cuda", batch_size=2)
    first._op_cfg["gpu"].update(work_dir="old-job", **{key: before})
    GPUMemoryProbe(str(tmp_path), stage_runner=lambda op, rows, measure: {"rows": rows, "metrics": metrics()}).resolve(
        FakeDataset([{"id": 1}, {"id": 2}]), [first]
    )
    fresh = FakeOp("gpu", accelerator="cuda", batch_size=2)
    fresh._op_cfg["gpu"].update(work_dir="new-job", **{key: after})
    replayed = []

    def rerun(op, rows, measure):
        replayed.append(op._name)
        return {"rows": rows, "metrics": metrics()}

    GPUMemoryProbe(
        str(tmp_path),
        stage_runner=rerun,
        hardware_reader=lambda: {"gpu_name": "Fake GPU", "total_mb": 1000},
    ).resolve(FakeDataset([{"id": 1}, {"id": 2}]), [fresh])
    assert replayed == ["gpu"]


def test_old_profile_schema_is_reprobed(tmp_path):
    op = FakeOp("gpu", accelerator="cuda")

    def runner(op, rows, measure):
        return {"rows": rows, "metrics": metrics()}

    GPUMemoryProbe(str(tmp_path), stage_runner=runner).resolve(FakeDataset([{"id": 1}]), [op])
    path = tmp_path / "gpu_probe_results.json"
    report = json.loads(path.read_text())
    report["version"] = 6
    path.write_text(json.dumps(report))
    dataset = FakeDataset([{"id": 2}])
    GPUMemoryProbe(str(tmp_path), stage_runner=runner).resolve(dataset, [FakeOp("gpu", accelerator="cuda")])
    assert dataset.requested == [1]
    assert json.loads(path.read_text())["version"] == 7


def test_worker_cap_change_invalidates_cached_plan(tmp_path):
    op = FakeOp("gpu", accelerator="cuda")
    GPUMemoryProbe(
        str(tmp_path),
        max_gpu_workers_per_device=4,
        stage_runner=lambda op, rows, measure: {"rows": rows, "metrics": metrics()},
    ).resolve(FakeDataset([{"id": 1}]), [op])

    replayed = []

    def rerun(op, rows, measure):
        replayed.append(op._name)
        return {"rows": rows, "metrics": metrics()}

    fresh_op = FakeOp("gpu", accelerator="cuda")
    GPUMemoryProbe(
        str(tmp_path),
        max_gpu_workers_per_device=2,
        stage_runner=rerun,
        hardware_reader=lambda: {"gpu_name": "Fake GPU", "total_mb": 1000},
    ).resolve(FakeDataset([{"id": 1}]), [fresh_op])

    assert replayed == ["gpu"]
    assert fresh_op.num_gpus == 0.5


def test_explicit_resources_are_a_noop(tmp_path):
    op = FakeOp("gpu", accelerator="cuda", num_gpus=0.25)
    dataset = FakeDataset([{"id": 1}])
    assert GPUMemoryProbe(str(tmp_path), stage_runner=SimpleNamespace()).resolve(dataset, [op]) == []
    assert dataset.requested == []
    assert not (tmp_path / "gpu_probe_results.json").exists()


def test_explicit_gpu_fraction_is_preserved_while_throughput_is_profiled(tmp_path):
    op = DeclaredProbeMapper(
        name="configured",
        accelerator="cuda",
        num_gpus=0.5,
        input_columns=["text"],
        output_columns=["text"],
    )

    def parallel_runner(jobs, rows):
        return {
            0: {
                "sample_count": 1,
                "metrics": metrics(measured=100),
                "profile": {
                    "initialization": 2,
                    "warmup": 1,
                    "steady_total": 0.5,
                    "steady_batch_seconds": 0.5,
                    "steady_rows_per_second": 2,
                    "output_ratio": 1,
                    "warmup_batches": 1,
                    "steady_batches": 1,
                },
            }
        }

    records = GPUMemoryProbe(str(tmp_path), parallel_runner=parallel_runner).resolve(
        FakeDataset([{"text": "sample"}]),
        [op],
    )

    assert records[0]["resource_mode"] == "configured"
    assert op.num_gpus == 0.5
    assert op._gpu_memory_fraction == 0.12
    assert op._gpu_rows_per_second == 2


def test_explicit_gpu_fraction_below_measured_memory_is_raised_to_the_memory_share(tmp_path):
    op = DeclaredProbeMapper(
        name="undersized",
        accelerator="cuda",
        num_gpus=0.05,
        input_columns=["text"],
        output_columns=["text"],
    )

    def parallel_runner(jobs, rows):
        return {0: {"sample_count": 1, "metrics": metrics(measured=100)}}

    with patch("data_juicer.core.executor.gpu_memory_probe.logger.warning") as warning:
        records = GPUMemoryProbe(str(tmp_path), parallel_runner=parallel_runner).resolve(
            FakeDataset([{"text": "sample"}]),
            [op],
        )

    assert records[0]["resource_mode"] == "configured"
    assert records[0]["num_gpus"] == 0.12
    assert op.num_gpus == 0.12
    assert op._gpu_memory_fraction == 0.12
    assert op._op_cfg["undersized"]["num_gpus"] == 0.12
    assert "Raising the Ray request to 0.12" in warning.call_args[0][0]


def test_cached_record_below_measured_memory_share_is_reconciled_on_reuse(tmp_path):
    GPUMemoryProbe(str(tmp_path), stage_runner=lambda op, rows, measure: {"rows": rows, "metrics": metrics()}).resolve(
        FakeDataset([{"id": 1}]), [FakeOp("gpu", accelerator="cuda")]
    )

    report_path = tmp_path / "gpu_probe_results.json"
    report = json.loads(report_path.read_text())
    report["operators"][0]["num_gpus"] = 0.05
    report_path.write_text(json.dumps(report))

    def must_not_run(*args):
        raise AssertionError("cached probe should not replay the recipe")

    fresh_op = FakeOp("gpu", accelerator="cuda")
    GPUMemoryProbe(
        str(tmp_path),
        stage_runner=must_not_run,
        hardware_reader=lambda: {"gpu_name": "Fake GPU", "total_mb": 1000},
    ).resolve(FakeDataset([{"id": 99}]), [fresh_op])

    assert fresh_op.num_gpus == 0.12
    assert fresh_op._gpu_memory_fraction == 0.12
    assert fresh_op._init_kwargs["num_gpus"] == 0.12


def test_pipeline_target_and_pipeline_barrier_fail_before_formal_run(tmp_path):
    pipeline = FakeGPUPipeline()
    with pytest.raises(RuntimeError, match="does not support Pipeline"):
        GPUMemoryProbe(str(tmp_path)).resolve(FakeDataset([{"id": 1}]), [pipeline])

    cpu_pipeline = FakeGPUPipeline(accelerator="cpu", num_gpus=1)
    target = FakeOp("gpu", accelerator="cuda")
    with pytest.raises(RuntimeError, match="across Pipeline"):
        GPUMemoryProbe(str(tmp_path)).resolve(FakeDataset([{"id": 1}]), [cpu_pipeline, target])


def test_stage_failure_is_fail_fast(tmp_path):
    target = FakeOp("gpu", accelerator="cuda")

    def fail(*args):
        raise MemoryError("CUDA out of memory")

    with pytest.raises(RuntimeError, match="formal Ray experiment was not started") as error:
        GPUMemoryProbe(str(tmp_path), stage_runner=fail).resolve(FakeDataset([{"id": 1}]), [target])
    assert isinstance(error.value.__cause__, MemoryError)


def test_executor_plans_parallelism_after_preflight(tmp_path):
    executor = PartitionedRayExecutor.__new__(PartitionedRayExecutor)
    executor.partition_mode = "auto"
    executor.work_dir = str(tmp_path)
    target = FakeOp("gpu", accelerator="cuda")
    order = []

    def probe(_probe, dataset, ops):
        order.append("probe")
        ops[0].num_gpus = 0.25

    def parallelism(ops):
        assert ops[0].num_gpus == 0.25
        order.append("parallelism")

    def partitions(ops, total_samples=None):
        assert ops[0].num_gpus == 0.25
        order.append("partitions")

    executor._count_dataset_rows = lambda dataset: None
    executor._cap_auto_gpu_operator_parallelism = lambda ops, safe_limit, total_samples=None: None

    executor._configure_operator_parallelism = parallelism
    executor._resolve_max_concurrent_partitions = partitions
    with patch.object(GPUMemoryProbe, "resolve", probe):
        executor._configure_pre_partition_resources(FakeDataset([{"id": 1}]), [target])

    assert order == ["probe", "parallelism", "partitions"]


def test_executor_uses_fixed_resources_when_preflight_is_disabled(tmp_path):
    executor = PartitionedRayExecutor.__new__(PartitionedRayExecutor)
    executor.gpu_preflight_enabled = False
    executor.work_dir = str(tmp_path)
    target = FakeOp("gpu", accelerator="cuda", num_gpus=0.2, memory=4.0)
    order = []

    executor._configure_operator_parallelism = lambda ops: order.append("parallelism")
    executor._resolve_max_concurrent_partitions = lambda ops, total_samples=None: order.append("partitions")

    with patch.object(GPUMemoryProbe, "resolve", side_effect=AssertionError("preflight must not run")):
        executor._configure_pre_partition_resources(FakeDataset([{"id": 1}]), [target])

    assert order == ["parallelism", "partitions"]


class GPUMemoryProbeTest(DataJuicerTestCaseBase):
    """CPU-only regression coverage for the GPU preflight planner."""

    def _with_tmp_path(self, function, *args):
        with tempfile.TemporaryDirectory() as directory:
            return function(Path(directory), *args)

    @TEST_TAG("standalone")
    def test_positive_numbers(self):
        self.assertIsNone(gpu_probe._positive_number(True))
        self.assertIsNone(gpu_probe._positive_number("bad"))
        self.assertIsNone(gpu_probe._positive_number(0))
        self.assertEqual(gpu_probe._positive_number("2.5"), 2.5)

    @TEST_TAG("standalone")
    def test_batch_size_normalization(self):
        for value, expected in [(True, 1), (None, 1), ("bad", 1), (0, 1), ("3", 3)]:
            self.assertEqual(gpu_probe._normalized_batch_size(SimpleNamespace(batch_size=value)), expected)

    @TEST_TAG("standalone")
    def test_gpu_profile_target_selection(self):
        mapper = DeclaredProbeMapper(accelerator="cuda", batch_size=4)
        self.assertTrue(gpu_probe.needs_gpu_profile(mapper))
        mapper.num_gpus = 2
        self.assertFalse(gpu_probe.needs_gpu_profile(mapper))
        self.assertFalse(gpu_probe.needs_gpu_profile(FakeOp("plain", accelerator="cuda")))

    @TEST_TAG("standalone")
    def test_fill_rejects_empty_rows(self):
        with self.assertRaisesRegex(RuntimeError, "no samples left"):
            fill_to_batch([], 2)

    @TEST_TAG("standalone")
    def test_declared_columns(self):
        self.assertEqual(gpu_probe._declared_columns(SimpleNamespace(cols=" a "), "cols"), {"a"})
        self.assertEqual(gpu_probe._declared_columns(SimpleNamespace(cols=["a", " ", 2]), "cols"), {"a", "2"})
        self.assertIsNone(gpu_probe._declared_columns(SimpleNamespace(cols=1), "cols"))

    @TEST_TAG("standalone")
    def test_column_overlap(self):
        self.assertTrue(gpu_probe._column_paths_overlap("meta", "meta.score"))
        self.assertTrue(gpu_probe._column_paths_overlap("meta.*", "meta.score"))
        self.assertFalse(gpu_probe._columns_overlap({"left"}, {"right"}))

    @TEST_TAG("standalone")
    def test_dependency_closure_handles_cycles(self):
        self.assertEqual(gpu_probe._dependency_closure(2, {2: {1}, 1: {0}, 0: {1}}), {0, 1})

    @TEST_TAG("standalone")
    def test_planner_rejects_unsupported_ops(self):
        jobs, reasons = plan_parallel_probe_jobs([FakeOp("not_mapper")], {0: FakeOp("not_mapper")})
        self.assertEqual(jobs, [])
        self.assertIn("not a Mapper/Filter", reasons[0])

    @TEST_TAG("standalone")
    def test_planner_rejects_indexed_replay(self):
        indexed = DeclaredProbeMapper(name="indexed", accelerator="cuda", input_columns=["x"], output_columns=["y"])
        indexed.index_key = "id"
        self.assertIn("dataset-level replay", plan_parallel_probe_jobs([indexed], {0: indexed})[1][0])

    @TEST_TAG("standalone")
    def test_planner_rejects_incompatible_runtime_env(self):
        target = DeclaredProbeMapper(name="target", accelerator="cuda", input_columns=["x"], output_columns=["y"])
        ancestor = DeclaredProbeMapper(name="ancestor", input_columns=["x"], output_columns=["x"])
        ancestor.runtime_env = {"env": "a"}
        target.runtime_env = {"env": "b"}
        self.assertIn(
            "incompatible ancestor runtime_env", plan_parallel_probe_jobs([ancestor, target], {1: target})[1][1]
        )

    @TEST_TAG("standalone")
    def test_numeric_sequence_features(self):
        shaped = SimpleNamespace(shape=(2, 3), dtype="uint8")
        nested = SimpleNamespace(feature=SimpleNamespace(feature=SimpleNamespace(dtype="float32")))
        non_numeric = SimpleNamespace(feature=SimpleNamespace(dtype="string"))
        self.assertEqual(gpu_probe._numeric_sequence_info(shaped), (2, "uint8"))
        self.assertEqual(gpu_probe._numeric_sequence_info(nested), (2, "float32"))
        self.assertEqual(gpu_probe._numeric_sequence_info(non_numeric), (0, None))

    @TEST_TAG("standalone")
    def test_detect_array_columns(self):
        rows = [{"array": np.zeros((2, 2)), "list": [1, 2]}]
        self.assertEqual(gpu_probe._array_columns_from_rows(rows), {"array"})

    @TEST_TAG("standalone")
    def test_array_format_ignores_missing_columns(self):
        dataset = SimpleNamespace(column_names=["x"])
        self.assertIs(gpu_probe._dataset_with_array_format(dataset, ["missing"]), dataset)

    @TEST_TAG("standalone")
    def test_resource_plan_validation(self):
        with self.assertRaisesRegex(RuntimeError, "invalid GPU capacity"):
            calculate_resource_plan(1, 0)
        for invalid in (True, 0):
            with self.assertRaisesRegex(ValueError, "positive integer"):
                calculate_resource_plan(1, 100, invalid)
        self.assertEqual(calculate_resource_plan(1, 100000)["memory_fraction"], 0.01)

    @TEST_TAG("standalone")
    def test_scheduling_gpu_fraction(self):
        self.assertEqual(gpu_probe.scheduling_num_gpus("op", 0.5, 0.4), 0.5)
        with patch.object(gpu_probe.logger, "warning") as warning:
            self.assertEqual(gpu_probe.scheduling_num_gpus("op", 0.1, 1.2), 1.0)
            warning.assert_called_once()

    @TEST_TAG("standalone")
    def test_row_batch_round_trip(self):
        rows = [{"a": 1}, {"b": 2, "a": 3}]
        batch = gpu_probe._rows_to_batch(rows)
        self.assertEqual(batch, {"a": [1, 3], "b": [None, 2]})
        self.assertEqual(gpu_probe._batch_to_rows(batch), [{"a": 1, "b": None}, {"a": 3, "b": 2}])
        self.assertEqual(gpu_probe._batch_to_rows({}), [])

    @TEST_TAG("standalone")
    def test_batch_rejects_inconsistent_columns(self):
        with self.assertRaisesRegex(RuntimeError, "inconsistent"):
            gpu_probe._batch_to_rows({"a": [1], "b": [1, 2]})

    @TEST_TAG("standalone")
    def test_probe_execution_kwargs_preserve_input(self):
        original = {"num_proc": 8, "auto_op_parallelism": True, "skip_op_error": True}
        actual = gpu_probe._probe_execution_kwargs(original)
        self.assertEqual(actual, {"num_proc": None, "auto_op_parallelism": False, "skip_op_error": True})
        self.assertEqual(original["num_proc"], 8)

    @TEST_TAG("standalone")
    def test_construct_probe_op_from_spec(self):
        spec = _op_spec(ProfilingProbeMapper(batch_size=2))
        self.assertIs(spec["op_class"], ProfilingProbeMapper)
        self.assertEqual(spec["batch_size"], 2)
        constructed = gpu_probe._construct_probe_op(spec)
        self.assertIsInstance(constructed, ProfilingProbeMapper)

    @TEST_TAG("standalone")
    def test_cuda_memory_snapshot(self):
        fake_cuda = SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            mem_get_info=lambda device: (600 * 1024**2, 1000 * 1024**2),
            get_device_properties=lambda device: SimpleNamespace(name="Mock GPU"),
        )
        with patch.dict(sys.modules, {"torch": SimpleNamespace(cuda=fake_cuda)}):
            self.assertEqual(
                gpu_probe._cuda_memory_snapshot(), {"used_mb": 400, "total_mb": 1000, "gpu_name": "Mock GPU"}
            )
        bad_cuda = SimpleNamespace(is_available=lambda: False, device_count=lambda: 0)
        with patch.dict(sys.modules, {"torch": SimpleNamespace(cuda=bad_cuda)}):
            with self.assertRaisesRegex(RuntimeError, "exactly one"):
                gpu_probe._cuda_memory_snapshot()

    @TEST_TAG("standalone")
    def test_assignment_without_ray_context(self):
        fake_ray = SimpleNamespace(get_runtime_context=Mock(side_effect=RuntimeError("no ray")))
        with patch.object(gpu_probe, "ray", fake_ray), patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2"}):
            self.assertIn("unavailable", gpu_probe._gpu_assignment_context())

    @TEST_TAG("standalone")
    def test_probe_stage_profile_restores_environment(self):
        def fake_measure(function):
            return (function(), metrics(measured=80))

        marker = gpu_probe._PREFLIGHT_OP_ENV_VAR
        with (
            patch.object(gpu_probe, "_measure_cuda_call", side_effect=fake_measure),
            patch.dict(os.environ, {marker: "outside"}),
        ):
            result = _run_probe_stage(
                CapturingProbeOp,
                (),
                {},
                [{"text": "sample"}],
                True,
                warmup_batches=1,
                steady_batches=2,
                op_name="capture",
            )
            self.assertEqual(os.environ[marker], "outside")
        self.assertEqual(result["rows"], [{"text": "sample"}])
        self.assertEqual(result["profile"]["steady_batches"], 2)
        self.assertEqual(result["metrics"]["measured_memory_mb"], 80)

    @TEST_TAG("standalone")
    def test_parallel_worker_replays_dependency(self):
        with patch.object(gpu_probe, "_gpu_assignment_context", return_value="mocked assignment"):
            test_parallel_worker_replays_cpu_dependency_and_measures_one_target()

    @TEST_TAG("standalone")
    def test_parallel_worker_rejects_empty_dependency(self):
        job = {
            "op_index": 1,
            "dependencies": [{**_op_spec(ArrayBatchProducer(batch_size=1)), "op_name": "empty"}],
            "target": _op_spec(ArrayBatchConsumer(accelerator="cuda", batch_size=1)),
        }
        with (
            patch.object(gpu_probe, "_run_probe_op_rows", return_value=[]),
            patch.object(gpu_probe, "_gpu_assignment_context", return_value="mocked assignment"),
        ):
            with self.assertRaisesRegex(RuntimeError, "no samples left after dependency"):
                _run_parallel_probe_job(job, [{"text": "a"}])

    @TEST_TAG("standalone")
    def test_ordered_measurement_preserves_full_upstream_lineage(self):
        for first_batch, second_batch in ((1, 4), (2, 4), (4, 4), (8, 4)):
            with self.subTest(first_batch=first_batch, second_batch=second_batch):
                self._with_tmp_path(test_ordered_measurement_preserves_full_upstream_lineage, first_batch, second_batch)

    @TEST_TAG("standalone")
    def test_block_count_fallback(self):
        probe = GPUMemoryProbe("unused")
        data = SimpleNamespace(
            num_blocks=Mock(side_effect=RuntimeError("unknown")),
            _plan=SimpleNamespace(initial_num_blocks=lambda: 3),
        )
        self.assertEqual(probe._block_count(data), 3)
        self.assertIsNone(probe._block_count(SimpleNamespace(num_blocks=lambda: 0)))

    @TEST_TAG("standalone")
    def test_constructor_rejects_invalid_options(self):
        invalid_options = [
            {"max_gpu_workers_per_device": True},
            {"max_gpu_workers_per_device": 0},
            {"max_concurrent_probes": True},
            {"max_concurrent_probes": 0},
            {"probe_timeout_seconds": True},
            {"probe_timeout_seconds": 0},
            {"warmup_batches": True},
            {"warmup_batches": -1},
            {"steady_batches": True},
            {"steady_batches": 0},
            {"sample_offset": True},
            {"sample_offset": -1},
            {"sample_seed": True},
        ]
        for options in invalid_options:
            with self.subTest(options=options), self.assertRaises(ValueError):
                GPUMemoryProbe("unused", **options)

    @TEST_TAG("standalone")
    def test_ray_stage_options_and_arguments(self):
        op = FakeOp("cpu", batch_size=2)
        rows = [{"id": 1}]
        expected = {"rows": rows}
        remote = Mock()
        remote.options.return_value = remote
        remote.remote.return_value = "future"
        decorator = Mock(return_value=remote)
        fake_ray = Mock()
        fake_ray.remote.return_value = decorator
        fake_ray.get.return_value = expected
        with patch.object(gpu_probe, "ray", fake_ray):
            result = GPUMemoryProbe("unused")._run_stage_with_ray(op, rows, False)
        self.assertIs(result, expected)
        fake_ray.remote.assert_called_once_with(max_calls=1)
        decorator.assert_called_once_with(_run_probe_stage)
        remote.options.assert_called_once_with(num_cpus=1, num_gpus=0)
        remote.remote.assert_called_once_with(type(op), (), op._init_kwargs, rows, False, 0, 1, "cpu")
        fake_ray.get.assert_called_once_with("future")

    @TEST_TAG("standalone")
    def test_ray_hardware_reader_reserves_one_gpu(self):
        expected = {"gpu_name": "Mock GPU", "total_mb": 1000}
        remote = Mock()
        remote.options.return_value = remote
        remote.remote.return_value = "future"
        decorator = Mock(return_value=remote)
        fake_ray = Mock()
        fake_ray.remote.return_value = decorator
        fake_ray.get.return_value = expected
        with patch.object(gpu_probe, "ray", fake_ray):
            result = GPUMemoryProbe._read_hardware_with_ray()
        self.assertIs(result, expected)
        fake_ray.remote.assert_called_once_with(max_calls=1)
        decorator.assert_called_once_with(gpu_probe._read_gpu_hardware)
        remote.options.assert_called_once_with(num_cpus=0, num_gpus=1)
        remote.remote.assert_called_once_with()
        fake_ray.get.assert_called_once_with("future")

    @TEST_TAG("standalone")
    def test_target_selection_and_maximum_batch_size(self):
        test_target_selection_and_maximum_batch_size()

    @TEST_TAG("standalone")
    def test_batch_fill_and_resource_rounding(self):
        test_batch_fill_and_resource_rounding()

    @TEST_TAG("standalone")
    def test_dependency_planner_parallelizes_independent_gpu_mappers(self):
        test_dependency_planner_parallelizes_independent_gpu_mappers()

    @TEST_TAG("standalone")
    def test_dependency_planner_keeps_gpu_dependency_and_unknown_contract_ordered(self):
        test_dependency_planner_keeps_gpu_dependency_and_unknown_contract_ordered()

    @TEST_TAG("standalone")
    def test_probe_stage_preserves_multidimensional_numpy_arrays_between_ops(self):
        test_probe_stage_preserves_multidimensional_numpy_arrays_between_ops()

    @TEST_TAG("standalone")
    def test_direct_batch_replay_preserves_numpy_arrays_without_arrow_round_trip(self):
        test_direct_batch_replay_preserves_numpy_arrays_without_arrow_round_trip()

    @TEST_TAG("standalone")
    def test_probe_stage_disables_inner_multiprocessing_and_keeps_recipe_fault_tolerance(self):
        test_probe_stage_disables_inner_multiprocessing_and_keeps_recipe_fault_tolerance()

    @TEST_TAG("standalone")
    def test_replay_skips_invalid_rows_when_the_recipe_asks_for_it(self):
        test_replay_skips_invalid_rows_when_the_recipe_asks_for_it()

    @TEST_TAG("standalone")
    def test_target_profile_separates_init_warmup_and_steady_batches(self):
        test_target_profile_separates_init_warmup_and_steady_batches()

    @TEST_TAG("standalone")
    def test_cuda_measurement_does_not_touch_context_until_target_returns(self):
        test_cuda_measurement_does_not_touch_context_until_target_returns()

    @TEST_TAG("standalone")
    def test_resolve_uses_parallel_runner_for_declared_independent_targets(self):
        self._with_tmp_path(test_resolve_uses_parallel_runner_for_declared_independent_targets)

    @TEST_TAG("standalone")
    def test_gpu_dependent_target_falls_back_after_independent_parallel_probe(self):
        self._with_tmp_path(test_gpu_dependent_target_falls_back_after_independent_parallel_probe)

    @TEST_TAG("standalone")
    def test_parallel_probe_failure_prevents_formal_run(self):
        self._with_tmp_path(test_parallel_probe_failure_prevents_formal_run)

    @TEST_TAG("standalone")
    def test_measurement_from_a_fully_skipped_batch_is_rejected(self):
        self._with_tmp_path(test_measurement_from_a_fully_skipped_batch_is_rejected)

    @TEST_TAG("standalone")
    def test_empty_mapper_measurement_is_rejected_even_without_skip_op_error(self):
        self._with_tmp_path(test_empty_mapper_measurement_is_rejected_even_without_skip_op_error)

    @TEST_TAG("standalone")
    def test_filter_target_that_keeps_no_samples_is_only_reported_as_worst_case(self):
        self._with_tmp_path(test_filter_target_that_keeps_no_samples_is_only_reported_as_worst_case)

    @TEST_TAG("standalone")
    def test_ordered_replay_uses_front_sample_and_refills_after_filter(self):
        self._with_tmp_path(test_ordered_replay_uses_front_sample_and_refills_after_filter)

    @TEST_TAG("standalone")
    def test_ordered_filter_can_keep_unmeasured_tail_for_downstream(self):
        self._with_tmp_path(test_ordered_filter_can_keep_unmeasured_tail_for_downstream)

    @TEST_TAG("standalone")
    def test_ordered_padding_is_not_propagated_as_real_upstream_rows(self):
        self._with_tmp_path(test_ordered_padding_is_not_propagated_as_real_upstream_rows)

    @TEST_TAG("standalone")
    def test_ordered_representative_replay_failure_aborts_before_saving_report(self):
        self._with_tmp_path(test_ordered_representative_replay_failure_aborts_before_saving_report)

    @TEST_TAG("standalone")
    def test_stage_failure_is_fail_fast(self):
        self._with_tmp_path(test_stage_failure_is_fail_fast)

    @TEST_TAG("standalone")
    def test_probe_sample_offset_skips_the_dataset_head(self):
        self._with_tmp_path(test_probe_sample_offset_skips_the_dataset_head)

    @TEST_TAG("standalone")
    def test_probe_sample_offset_streams_past_the_head_of_a_ray_dataset(self):
        self._with_tmp_path(test_probe_sample_offset_streams_past_the_head_of_a_ray_dataset)

    @TEST_TAG("standalone")
    def test_probe_sample_offset_beyond_the_dataset_falls_back_to_the_head(self):
        self._with_tmp_path(test_probe_sample_offset_beyond_the_dataset_falls_back_to_the_head)

    @TEST_TAG("standalone")
    def test_probe_sample_shuffle_reads_a_randomized_block(self):
        self._with_tmp_path(test_probe_sample_shuffle_reads_a_randomized_block)

    @TEST_TAG("standalone")
    def test_probe_sample_shuffle_falls_back_to_the_head_without_block_support(self):
        self._with_tmp_path(test_probe_sample_shuffle_falls_back_to_the_head_without_block_support)

    @TEST_TAG("standalone")
    def test_probe_sample_shuffle_reports_the_ignored_offset(self):
        self._with_tmp_path(test_probe_sample_shuffle_reports_the_ignored_offset)

    @TEST_TAG("standalone")
    def test_single_block_shuffle_hands_the_sample_back_to_the_offset(self):
        self._with_tmp_path(test_single_block_shuffle_hands_the_sample_back_to_the_offset)

    @TEST_TAG("standalone")
    def test_seedless_shuffle_report_is_observed_but_never_reused(self):
        self._with_tmp_path(test_seedless_shuffle_report_is_observed_but_never_reused)

    @TEST_TAG("standalone")
    def test_seeded_shuffle_report_is_reused(self):
        self._with_tmp_path(test_seeded_shuffle_report_is_reused)

    @TEST_TAG("standalone")
    def test_sampling_policy_change_invalidates_cached_plan(self):
        self._with_tmp_path(test_sampling_policy_change_invalidates_cached_plan)

    @TEST_TAG("standalone")
    def test_matching_report_is_reused_without_replaying_samples(self):
        self._with_tmp_path(test_matching_report_is_reused_without_replaying_samples)

    @TEST_TAG("standalone")
    def test_hardware_mismatch_reprobes(self):
        self._with_tmp_path(test_hardware_mismatch_reprobes)

    @TEST_TAG("standalone")
    def test_old_profile_schema_is_reprobed(self):
        self._with_tmp_path(test_old_profile_schema_is_reprobed)

    @TEST_TAG("standalone")
    def test_worker_cap_change_invalidates_cached_plan(self):
        self._with_tmp_path(test_worker_cap_change_invalidates_cached_plan)

    @TEST_TAG("standalone")
    def test_cached_record_below_measured_memory_share_is_reconciled_on_reuse(self):
        self._with_tmp_path(test_cached_record_below_measured_memory_share_is_reconciled_on_reuse)

    @TEST_TAG("standalone")
    def test_explicit_resources_are_a_noop(self):
        self._with_tmp_path(test_explicit_resources_are_a_noop)

    @TEST_TAG("standalone")
    def test_explicit_gpu_fraction_is_preserved_while_throughput_is_profiled(self):
        self._with_tmp_path(test_explicit_gpu_fraction_is_preserved_while_throughput_is_profiled)

    @TEST_TAG("standalone")
    def test_explicit_gpu_fraction_below_measured_memory_is_raised_to_the_memory_share(self):
        self._with_tmp_path(test_explicit_gpu_fraction_below_measured_memory_is_raised_to_the_memory_share)

    @TEST_TAG("standalone")
    def test_pipeline_target_and_pipeline_barrier_fail_before_formal_run(self):
        self._with_tmp_path(test_pipeline_target_and_pipeline_barrier_fail_before_formal_run)

    @TEST_TAG("standalone")
    def test_executor_plans_parallelism_after_preflight(self):
        self._with_tmp_path(test_executor_plans_parallelism_after_preflight)

    @TEST_TAG("standalone")
    def test_executor_uses_fixed_resources_when_preflight_is_disabled(self):
        self._with_tmp_path(test_executor_uses_fixed_resources_when_preflight_is_disabled)

    @TEST_TAG("standalone")
    def test_parallel_probe_concurrency_serializes_by_default_and_bounds_an_explicit_cap(self):
        self._with_tmp_path(test_parallel_probe_concurrency_serializes_by_default_and_bounds_an_explicit_cap)

    @TEST_TAG("standalone")
    def test_parallel_probe_timeout_names_stalled_operator_and_cancels_future(self):
        self._with_tmp_path(test_parallel_probe_timeout_names_stalled_operator_and_cancels_future)


if __name__ == "__main__":
    unittest.main()

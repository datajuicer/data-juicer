"""Exercise optional probe sampling on real Ray-assigned GPUs.

Synthetic allocations and deterministic arithmetic validate resource scopes,
device assignment, report/cache wiring and output preservation. This is not
a model throughput benchmark. Select idle devices with CUDA_VISIBLE_DEVICES.
"""

import argparse
import gc
import json
import os
import statistics
import time
from pathlib import Path

import ray
import torch

from data_juicer.core.executor.gpu_memory_probe import GPUMemoryProbe, _run_probe_stage
from data_juicer.ops import Mapper

MIB = 1024**2


class AllocationProbeMapper(Mapper):
    _batched_op = True
    _accelerator = "cuda"

    def __init__(self, label="sample", resident_mib=128, transient_mib=256, declared=True, **kwargs):
        super().__init__(
            input_columns=["id"] if declared else None,
            output_columns=[label] if declared else None,
            **kwargs,
        )
        self._name = f"allocation_probe_{label}"
        self.label = label
        self.resident_mib = resident_mib
        self.transient_mib = transient_mib
        self._resident = None
        self._op_cfg = {
            self._name: {
                "batch_size": self.batch_size,
                "resident_mib": resident_mib,
                "transient_mib": transient_mib,
                "declared": declared,
            }
        }

    def process_batched(self, batch, rank=None):
        torch.set_num_threads(1)
        if self._resident is None:
            self._resident = torch.ones(self.resident_mib * MIB, dtype=torch.uint8, device="cuda")
        temporary = torch.empty(self.transient_mib * MIB, dtype=torch.uint8, device="cuda")
        temporary.fill_(3)
        host_temporary = bytearray(32 * MIB)
        ids = list(batch["id"])
        values = torch.tensor(ids, dtype=torch.float32, device="cuda").mul(2).add(1).cpu().tolist()
        torch.cuda.synchronize()
        assert values == [float(value * 2 + 1) for value in ids]
        assert temporary[0].item() == 3 and self._resident[0].item() == 1
        assert host_temporary[-1] == 0
        # Give RSS polling a known live allocation to observe.
        time.sleep(0.1)
        batch[self.label] = values
        return batch


class Rows:
    def get(self, count):
        return [{"id": value} for value in range(count)]


def make_ops(count, declared=True):
    return [
        AllocationProbeMapper(
            label=f"value_{index}",
            resident_mib=128 + index * 16,
            transient_mib=256 + index * 32,
            declared=declared,
            batch_size=8,
            num_cpus=1,
            accelerator="cuda",
            skip_op_error=False,
        )
        for index in range(count)
    ]


def validate_samples(records):
    devices, intervals = set(), []
    for index, record in enumerate(records):
        report = record["resource_samples"]
        worker = report["worker"]
        assert worker["ray_node_id"] and len(worker["ray_gpu_ids"]) == 1
        devices.add((worker["ray_node_id"], worker["ray_gpu_ids"][0]))
        assert report["cuda_scope"] == "worker_process_allocator_on_device"
        assert report["cuda_peak_scope"] == "since_worker_start"
        assert report["dropped_samples"] == 0
        assert not any(report["sampling_errors"].values()), report
        samples = report["samples"]
        assert [sample["phase"] for sample in samples] == ["construction", "warmup", "steady", "steady", "steady"]
        assert all(sample["succeeded"] and sample["rss_start_bytes"] > 0 for sample in samples)
        assert all(sample["cpu_seconds"] is not None for sample in samples)
        assert sum(sample["rss_poll_count"] for sample in samples) > 0
        end = samples[-1]["cuda_end"]
        assert end["allocated_bytes"] >= (128 + index * 16) * MIB
        assert end["peak_allocated_bytes"] >= (384 + index * 48) * MIB
        assert end["peak_reserved_bytes"] / MIB <= record["torch_peak_reserved_mb"]
        assert record["profile"]["output_ratio"] == 1
        assert record["sample_count"] == 8
        intervals.append(
            (
                samples[0]["started_at_unix_seconds"],
                samples[-1]["started_at_unix_seconds"] + samples[-1]["host_wall_seconds"],
            )
        )
    overlap = max(sum(start <= point < end for start, end in intervals) for point, _ in intervals)
    return {"devices": [list(device) for device in sorted(devices)], "max_overlapping_targets": overlap}


def check_ordered_output(enabled):
    result = _run_probe_stage(
        AllocationProbeMapper,
        (),
        {"label": "value", "batch_size": 8, "accelerator": "cuda", "skip_op_error": False},
        [{"id": value} for value in range(8)],
        True,
        1,
        3,
        resource_sampling=enabled,
    )
    assert result["rows"] == [{"id": value, "value": float(value * 2 + 1)} for value in range(8)]
    assert ("resource_samples" in result["metrics"]) is enabled
    return result


def check_physical_oom(enabled):
    from data_juicer.core.executor.gpu_memory_probe import _measure_cuda_call
    from data_juicer.core.executor.probe_resource_sampler import (
        ProbeResourceSampler,
        sample_probe_phase,
    )

    sampler = ProbeResourceSampler() if enabled else None
    request = torch.cuda.get_device_properties(0).total_memory * 2

    def target():
        with sample_probe_phase(sampler, "steady", 0):
            return torch.empty(request, dtype=torch.uint8, device="cuda")

    try:
        _measure_cuda_call(target)
    except torch.OutOfMemoryError as error:
        result = {"exception_type": type(error).__name__, "message": str(error), "request_bytes": request}
    else:
        raise AssertionError("Expected a physical allocation OOM")
    gc.collect()
    torch.cuda.empty_cache()
    assert torch.ones(1, device="cuda").sum().item() == 1
    result["context_usable_after_oom"] = True
    result["ray_gpu_ids"] = ray.get_runtime_context().get_accelerator_ids()["GPU"]
    if sampler is not None:
        result["resource_samples"] = sampler.snapshot()
        assert sampler.samples[-1]["succeeded"] is False
        assert sampler.samples[-1]["error_type"] == "OutOfMemoryError"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.gpus < 2 or args.rounds < 1:
        parser.error("--gpus must be at least 2 and --rounds at least 1")
    args.output.mkdir(parents=True, exist_ok=False)
    assert torch.cuda.is_available() and torch.cuda.device_count() >= args.gpus
    summary = {
        "torch": torch.__version__,
        "ray": ray.__version__,
        "cuda_api": torch.version.cuda,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_count": args.gpus,
        "gpu_names": [torch.cuda.get_device_name(index) for index in range(args.gpus)],
        "workload": "synthetic allocation, GPU arithmetic and a 0.1-second hold; not a throughput benchmark",
        "rounds": [],
    }
    ray.init(
        address="local",
        num_cpus=max(8, args.gpus * 2),
        num_gpus=args.gpus,
        include_dashboard=False,
        object_store_memory=256 * MIB,
    )
    try:
        for round_index in range(args.rounds):
            pair = {}
            # Counterbalance cold-start/order effects; timings remain descriptive.
            for enabled in [False, True] if round_index % 2 == 0 else [True, False]:
                work_dir = args.output / f"round-{round_index}-sampling-{enabled}"
                planner = GPUMemoryProbe(str(work_dir), max_concurrent_probes=args.gpus, resource_sampling=enabled)
                started = time.perf_counter()
                records = planner.resolve(Rows(), make_ops(args.gpus))
                assert len(records) == args.gpus
                assert all(record["probe_mode"] == "parallel" for record in records)
                result = {"seconds": time.perf_counter() - started}
                if enabled:
                    result.update(validate_samples(records))
                    assert len(result["devices"]) == args.gpus, result
                    assert result["max_overlapping_targets"] >= 2, result

                    def forbidden(*unused):
                        raise AssertionError("A compatible cached report must not execute the operator")

                    cached = GPUMemoryProbe(
                        str(work_dir), resource_sampling=True, stage_runner=forbidden, parallel_runner=forbidden
                    ).resolve(Rows(), make_ops(args.gpus))
                    assert cached == records
                    result["cache_reused"] = True
                else:
                    assert all("resource_samples" not in record for record in records)
                result["records"] = records
                pair[str(enabled)] = result
                print(json.dumps({"round": round_index, "sampling": enabled, "seconds": result["seconds"]}), flush=True)
            for original, sampled in zip(pair["False"]["records"], pair["True"]["records"]):
                assert original["num_gpus"] == sampled["num_gpus"]
                assert original["sample_count"] == sampled["sample_count"]
                assert original["profile"]["output_ratio"] == sampled["profile"]["output_ratio"] == 1
                assert original["torch_peak_reserved_mb"] == sampled["torch_peak_reserved_mb"]
            summary["rounds"].append(pair)

        ordered = ray.remote(num_cpus=1, num_gpus=1, max_calls=1)(check_ordered_output)
        summary["ordered"] = ray.get([ordered.remote(False), ordered.remote(True)])
        assert summary["ordered"][0]["rows"] == summary["ordered"][1]["rows"]
        oom = ray.remote(num_cpus=1, num_gpus=1, max_calls=1)(check_physical_oom)
        summary["physical_oom"] = ray.get([oom.remote(False), oom.remote(True)])
        summary["median_probe_seconds"] = {
            str(enabled): statistics.median(pair[str(enabled)]["seconds"] for pair in summary["rounds"])
            for enabled in (False, True)
        }
        summary["status"] = "passed"
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
        print(json.dumps({"status": "passed", "output": str(args.output)}), flush=True)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()

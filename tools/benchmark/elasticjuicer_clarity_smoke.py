"""Compare local GPU Clarity output with adaptive microbatches.

Requires the external Vgen source tree and trained Clarity weights. The OOM
in this check is injected before model inference; this is a correctness smoke
test, not a hardware-capacity or throughput benchmark.
"""

import argparse
import hashlib
import json
import os
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pyarrow as pa
import torch
from safetensors import safe_open

from data_juicer.core.elasticjuicer.ray_adaptive_mapper import RayAdaptiveMapperActor
from data_juicer.core.elasticjuicer.stage_identity import assign_stage_identities
from data_juicer.utils.constant import Fields


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vgen-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=8)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A working CUDA-compatible device is required")
    if args.num_samples < 4:
        raise ValueError("Use at least four samples to exercise adaptive slicing")
    os.environ["VGEN_ROOT"] = str(args.vgen_root)
    sys.path.insert(0, str(args.vgen_root / "wip-vgen-dj"))
    from custom_ops.custom_clarity_mapper import CustomClarityMapper

    # A missing trained regression head must not become a random-head test.
    with safe_open(args.model_path / "model.safetensors", framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
    required_head = {"new_query", "new_query_head.weight", "new_query_head.bias"}
    if not required_head <= keys:
        raise RuntimeError(f"Missing trained Clarity weights: {sorted(required_head - keys)}")

    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()][: args.num_samples]
    if len(rows) != args.num_samples:
        raise ValueError("The input does not contain enough distinct samples")
    tags = "__data_juicer_logical_partition_id__"
    for index, row in enumerate(rows):
        row[tags] = index // 2
    batch = pa.Table.from_pylist(rows)
    kwargs = dict(
        model_path=str(args.model_path),
        batch_size=args.num_samples,
        gpu_batch_size=args.num_samples,
        num_proc=1,
        num_gpus=1,
        num_cpus=2,
        ray_execution_mode="actor",
        adaptive_batching=True,
        skip_op_error=False,
    )
    op = CustomClarityMapper(**kwargs)
    manifest = assign_stage_identities([op])
    actor = RayAdaptiveMapperActor(
        type(op), op._init_args, op._init_kwargs, args.num_samples, manifest["stages"][0]["stage_id"]
    )
    baseline = []
    for offset in range(0, len(rows), 2):
        result = actor.op.process_batched(batch.slice(offset, 2).to_pydict())
        baseline.extend(meta["clarity_score"] for meta in result[Fields.meta])
    direct = actor.mapper.mapper
    injected = []

    def with_injected_oom(part):
        if part.num_rows > 2:
            injected.append(part.num_rows)
            raise torch.cuda.OutOfMemoryError("Injected CUDA out of memory for the adaptive retry smoke test")
        return direct(part)

    actor.mapper.mapper = with_injected_oom
    output = actor(batch)
    actual = [meta["clarity_score"] for meta in output[Fields.meta]]
    assert output["id"] == [row["id"] for row in rows]
    assert output[tags] == [row[tags] for row in rows]
    assert all(score is not None and np.isfinite(score) for score in baseline + actual)
    np.testing.assert_allclose(actual, baseline, rtol=2e-3, atol=2e-3)
    assert injected and actor.mapper.oom_retries == len(injected)
    torch.cuda.synchronize()
    report = {
        "passed": True,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "model_path": str(args.model_path),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "samples": len(rows),
        "baseline_scores": baseline,
        "adaptive_scores": actual,
        "max_absolute_difference": float(np.max(np.abs(np.asarray(actual) - baseline))),
        "oom_kind": "injected before inference; not a real capacity OOM",
        "failed_batch_sizes": injected,
        "controller": deepcopy(actor.controller.state.__dict__),
        "peak_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

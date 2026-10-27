"""Reproducible CPU -> GPU fork -> CPU join experiment on local Qwen images.

Six arms distinguish the existing shared-pool scheduler, grouped operators,
fixed slot ownership, global compute flow and actor-local batching. Default is a small
integration experiment, not a statistically conclusive speedup measurement.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="JSONL: id, image_path, image_sha256")
    parser.add_argument(
        "--qualiclip", type=Path, required=True, help="Trusted local QualiCLIP weights and clip/model.py"
    )
    parser.add_argument("--dinov2", type=Path, required=True, help="Local Hugging Face DINOv2-large directory")
    parser.add_argument(
        "--output", type=Path, required=True, help="New directory; existing results are never overwritten"
    )
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument(
        "--repeats", type=int, default=1, help="Fresh pools per arm in each block; use >=3 for timing studies"
    )
    parser.add_argument(
        "--passes", type=int, default=2, help="First pass includes cold execution, subsequent passes reuse pool"
    )
    parser.add_argument("--unit-rows", type=int, default=32)
    parser.add_argument("--batch-cap", type=int, default=32)
    parser.add_argument("--initial-batch", type=int, default=8)
    parser.add_argument("--object-budget-gib", type=int, default=64)
    parser.add_argument("--inflight", type=int, default=16)
    parser.add_argument("--workload", choices=("uniform", "phase-shift"), default="phase-shift")
    parser.add_argument("--light-side", type=int, default=448)
    parser.add_argument("--heavy-side", type=int, default=896)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("legacy-fixed", "grouped-fixed", "static-fixed", "static-adaptive", "flow-fixed", "flow-adaptive"),
        default=["legacy-fixed", "static-fixed", "flow-fixed", "static-adaptive", "flow-adaptive", "grouped-fixed"],
    )
    args = parser.parse_args()
    if any(
        v < 1
        for v in (
            args.rows,
            args.repeats,
            args.passes,
            args.unit_rows,
            args.batch_cap,
            args.initial_batch,
            args.object_budget_gib,
            args.inflight,
        )
    ):
        parser.error("counts and batch sizes must be positive")
    if args.gpus < 2 or args.initial_batch > args.batch_cap:
        parser.error("at least two GPUs and initial-batch <= batch-cap are required")
    if any(side < 14 or side % 14 for side in (args.light_side, args.heavy_side)):
        parser.error("DINO sides must be positive multiples of its 14-pixel patch size")
    if len(set(args.modes)) != len(args.modes):
        parser.error("duplicate modes")
    return args


def make_plan(args, max_batch_bytes):
    from data_juicer.core.elasticjuicer.dag_plan import (
        DAGNode,
        ExecutionGroup,
        ResidentDAGPlan,
        WorkerPool,
    )

    nodes = [
        DAGNode(
            "decode",
            {"ej_flow_decode": {}},
            reads=("image_path", "image_sha256"),
            writes=("pixels", "image_hash", "width", "height"),
            batch_size=32,
        )
    ]
    for stage, path in (("quality", args.qualiclip), ("embedding", args.dinov2)):
        nodes.append(
            DAGNode(
                stage,
                {"ej_flow_model": {"stage": stage, "model_path": str(path.resolve())}},
                parents=("decode",),
                reads=("pixels", "input_side"),
                writes=(stage,),
                batch_size=args.batch_cap,
                initial_batch_size=args.initial_batch,
            )
        )
    nodes.append(
        DAGNode(
            "pack",
            {"ej_flow_pack": {}},
            parents=("quality", "embedding"),
            reads=("quality", "embedding"),
            writes=("embedding_norm",),
            join_keys=("image_hash",),
        )
    )
    return ResidentDAGPlan(
        tuple(nodes),
        (
            ExecutionGroup("decode", ("decode",), "cpu"),
            ExecutionGroup("quality", ("quality",), "gpu"),
            ExecutionGroup("embedding", ("embedding",), "gpu"),
            ExecutionGroup("pack", ("pack",), "cpu"),
        ),
        (WorkerPool("cpu", 4, 2), WorkerPool("gpu", args.gpus, 4, 1)),
        "pack",
        operator_modules=("compute_flow_image_ops",),
        max_inflight_units=args.inflight,
        max_batch_bytes=max_batch_bytes,
        object_budget_bytes=args.object_budget_gib * 1024**3,
        output_columns=("image_hash", "width", "height", "input_side", "quality", "embedding", "embedding_norm"),
    )


def write_json(path, value):
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def mode_plan(plan, mode):
    if mode != "grouped-fixed":
        return plan
    from data_juicer.core.elasticjuicer.dag_plan import ExecutionGroup

    return replace(
        plan, groups=(plan.groups[0], ExecutionGroup("models", ("quality", "embedding"), "gpu", True), plan.groups[-1])
    )


def compare(actual, expected):
    import numpy as np

    if set(actual) != set(expected):
        raise AssertionError("output IDs differ")
    error = {"quality": 0.0, "embedding": 0.0, "embedding_norm": 0.0}
    for key, output in actual.items():
        for field, value in output.items():
            if field in error:
                a, b = np.asarray(value), np.asarray(expected[key][field])
                delta = float(np.max(np.abs(a - b)))
                error[field] = max(error[field], delta)
                if a.shape != b.shape or not np.isfinite(a).all() or not np.allclose(a, b, atol=1e-4, rtol=0):
                    raise AssertionError(f"output changed for {key}/{field}: abs={delta}")
            elif value != expected[key][field]:
                raise AssertionError(f"metadata changed for {key}/{field}")
    return error


def execution_evidence(root, gpu_ids):
    events = [json.loads(line) for p in root.glob("*/events.jsonl") for line in p.read_text().splitlines()]
    gpu_events = [e for e in events if e["worker_id"] in gpu_ids]
    identities, groups, pairs, batch_sizes = {}, {}, {}, {}
    for event in gpu_events:
        key = (event["worker_id"], event["node"])
        identities.setdefault(key, set()).add(event["operator_instance"])
        groups.setdefault(event["worker_id"], set()).add(event["group"])
        pairs.setdefault((event["run_id"], event["unit"]), {})[event["node"]] = event
        batch_sizes.setdefault(event["node"], set()).update(a["rows"] for a in event["attempts"] if a["success"])
    overlaps = sum(
        min(p["quality"]["ended"], p["embedding"]["ended"]) > max(p["quality"]["started"], p["embedding"]["started"])
        for p in pairs.values()
    )
    if not all(len(ids) == 1 for ids in identities.values()):
        raise AssertionError("resident operator instance changed")
    return dict(
        gpu_workers_executed=len(groups),
        actors_executing_both_groups=sum(len(g) == 2 for g in groups.values()),
        instances_reused=True,
        overlapping_forks=overlaps,
        total_forks=len(pairs),
        successful_batch_sizes={k: sorted(v) for k, v in batch_sizes.items()},
        oom_attempts=sum(a["oom"] for e in gpu_events for a in e["attempts"]),
    )


def main():
    args = arguments()
    product, adapters = Path(__file__).resolve().parents[2], Path(__file__).resolve().parent
    paths = os.pathsep.join((str(product), str(adapters)))
    sys.path[:0] = [str(product), str(adapters)]
    os.environ.update(
        PYTHONPATH=paths,
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
    )
    import numpy as np
    import ray
    from PIL import Image

    from data_juicer.core.elasticjuicer.compute_flow import ComputeFlowConfig
    from data_juicer.core.executor.resident_dag_executor import ResidentDAGExecutor

    args.output.mkdir(parents=True, exist_ok=False)
    source = []
    with args.input.open() as stream:
        for line in stream:
            if len(source) == args.rows:
                break
            source.append(json.loads(line))
    if len(source) != args.rows or len({r["id"] for r in source}) != args.rows:
        raise ValueError("insufficient rows or duplicate input IDs")
    decoded_estimates = []
    for i, row in enumerate(source):
        if hashlib.sha256(Path(row["image_path"]).read_bytes()).hexdigest() != row["image_sha256"]:
            raise ValueError("manifest image hash mismatch")
        row["input_side"] = (
            args.light_side if args.workload == "phase-shift" and i < len(source) // 2 else args.heavy_side
        )
        with Image.open(row["image_path"]) as source_image:
            # payload_bytes deliberately counts an owning ndarray's storage
            # conservatively twice. Include headers/metadata before alignment.
            decoded_estimates.append(source_image.width * source_image.height * 6 + 65536)
    peak_estimate = max(sum(decoded_estimates[i : i + args.unit_rows]) for i in range(0, len(source), args.unit_rows))
    quantum = 64 * 1024**2
    max_batch_bytes = ((peak_estimate + quantum - 1) // quantum) * quantum
    smi = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name,uuid,memory.used", "--format=csv,noheader"], text=True
    )
    occupied = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip()
    if occupied or len(smi.splitlines()) < args.gpus:
        raise RuntimeError("GPU processes already exist or insufficient GPUs; no other job was stopped")
    plan = make_plan(args, max_batch_bytes)
    compiled = plan.compile()
    plans = {mode: mode_plan(plan, mode) for mode in args.modes}
    for variant in plans.values():
        variant.compile()
    provenance = dict(
        command=sys.argv,
        product_commit=subprocess.check_output(["git", "-C", str(product), "rev-parse", "HEAD"], text=True).strip(),
        dirty=bool(subprocess.check_output(["git", "-C", str(product), "status", "--porcelain"], text=True).strip()),
        plan=asdict(plan),
        plan_fingerprint=plan.fingerprint(),
        physical_variants={mode: asdict(variant) for mode, variant in plans.items()},
        gpu=smi,
        ray=ray.__version__,
        unique_images=len({r["image_sha256"] for r in source}),
        rows=len(source),
        decoded_batch_estimate_bytes=peak_estimate,
        effective_inflight_units=compiled["capacity"],
        tolerance_absolute=1e-4,
        code_sha256={
            str(p.relative_to(product)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__),
                adapters / "compute_flow_image_ops.py",
                product / "data_juicer/core/elasticjuicer/compute_flow.py",
                product / "data_juicer/core/executor/resident_dag_executor.py",
                product / "data_juicer/core/elasticjuicer/dag_worker.py",
                product / "data_juicer/core/elasticjuicer/dag_plan.py",
            )
        },
        scope="single-node development ablation; no physical actor resizing or large-scale speedup claim",
    )
    write_json(args.output / "provenance.json", provenance)
    (args.output / "input.jsonl").write_text("".join(json.dumps(r) + "\n" for r in source))
    results, reference = [], None
    try:
        ray.init(
            address="local",
            num_cpus=4 * args.gpus + 8,
            num_gpus=args.gpus,
            include_dashboard=False,
            object_store_memory=plan.object_budget_bytes,
            log_to_driver=False,
            runtime_env={"env_vars": {"PYTHONPATH": paths}},
        )
        for repeat in range(args.repeats):
            # Rotate order between paired blocks, fixed before looking at results.
            modes = args.modes[repeat % len(args.modes) :] + args.modes[: repeat % len(args.modes)]
            for mode in modes:
                cell = args.output / f"b{repeat}-{mode}"
                write_json(
                    args.output / "status.json", dict(status="running", current=cell.name, completed=len(results))
                )
                print("START", cell.name, flush=True)
                config = (
                    None
                    if mode.startswith(("legacy", "grouped"))
                    else ComputeFlowConfig(mode="dynamic" if mode.startswith("flow") else "static")
                )
                started = time.perf_counter()
                with ResidentDAGExecutor(
                    plans[mode], adaptive=mode.endswith("adaptive"), compute_flow=config, artifact_dir=cell
                ) as executor:
                    gpu_ids = {w["worker_id"] for w in executor.worker_identities if w["device"]}
                    if len({w["device"]["uuid"] for w in executor.worker_identities if w["device"]}) != args.gpus:
                        raise AssertionError("GPU pool does not cover unique devices")
                    for pass_index in range(args.passes):
                        output, exported = {}, cell / f"outputs-{pass_index}"
                        exported.mkdir()
                        pass_start = time.perf_counter()
                        for result in executor.iter_batches(
                            {
                                key: [r[key] for r in source[i : i + args.unit_rows]]
                                for key in ("id", "image_path", "image_sha256", "input_side")
                            }
                            for i in range(0, len(source), args.unit_rows)
                        ):
                            columns = result.columns()
                            # Streaming, full-feature artifact export is inside the measured interval.
                            np.savez(
                                exported / f"{result.unit:06d}.npz", **{k: np.asarray(v) for k, v in columns.items()}
                            )
                            for i, identity in enumerate(columns["id"]):
                                if identity in output:
                                    raise AssertionError("duplicate terminal ID")
                                output[identity] = {k: v[i] for k, v in columns.items() if k != "id"}
                        elapsed = time.perf_counter() - pass_start
                        cold_seconds = time.perf_counter() - started if pass_index == 0 else None
                        if set(output) != {r["id"] for r in source}:
                            raise AssertionError("terminal IDs do not conserve input")
                        if reference is None:
                            reference = output
                        errors = compare(output, reference)
                        restored = {}
                        for artifact in exported.glob("*.npz"):
                            with np.load(artifact, allow_pickle=False) as saved:
                                for i, identity in enumerate(saved["id"].tolist()):
                                    if identity in restored:
                                        raise AssertionError("duplicate exported ID")
                                    restored[identity] = {k: saved[k][i].tolist() for k in saved.files if k != "id"}
                        export_errors = compare(restored, output)
                        entry = dict(
                            block=repeat,
                            mode=mode,
                            pass_index=pass_index,
                            seconds=elapsed,
                            cold_pool_plus_first_pass_seconds=cold_seconds,
                            summary=executor.last_summary,
                            output_errors=errors,
                            export_errors=export_errors,
                        )
                        results.append(entry)
                        write_json(args.output / "results.json", results)
                        print("PASS", mode, pass_index, round(elapsed, 3), flush=True)
                    evidence = execution_evidence(cell, gpu_ids)
                    if evidence["gpu_workers_executed"] != args.gpus:
                        raise AssertionError("not every GPU pool worker executed real model work")
                    if mode.startswith("flow") and not evidence["actors_executing_both_groups"]:
                        raise AssertionError("no actor changed its execution group")
                    if mode.startswith("static") and evidence["actors_executing_both_groups"]:
                        raise AssertionError("static actor ownership changed")
                    if not evidence["overlapping_forks"]:
                        raise AssertionError("GPU branches did not overlap on any work unit")
                    write_json(cell / "execution-evidence.json", evidence)
        write_json(args.output / "status.json", dict(status="complete", completed=len(results), current=None))
    except BaseException as error:
        write_json(args.output / "status.json", dict(status="failed", completed=len(results), error=repr(error)))
        raise
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()

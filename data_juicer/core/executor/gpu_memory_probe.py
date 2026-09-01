"""Small-sample GPU memory preflight for partitioned Ray execution."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from fnmatch import fnmatchcase
from itertools import cycle, islice
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from loguru import logger

from data_juicer.ops import Filter, Mapper, Pipeline
from data_juicer.utils.lazy_loader import LazyLoader

ray = LazyLoader("ray")

_REPORT_VERSION = 4
_OBSERVABILITY_VERSION = 3
_REPORT_NAME = "gpu_probe_results.json"
_MEMORY_HEADROOM = 1.10
_DEFAULT_MAX_GPU_WORKERS_PER_DEVICE = 5
_PROGRESS_INTERVAL_SECONDS = 30.0
_MIB = 1024**2
_PREFLIGHT_OP_ENV_VAR = "DATA_JUICER_GPU_PREFLIGHT_OP"


def _positive_number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def needs_gpu_probe(op) -> bool:
    """Return whether a CUDA op omitted both GPU scheduling hints."""
    return (
        getattr(op, "accelerator", None) == "cuda"
        and _positive_number(getattr(op, "num_gpus", None)) is None
        and _positive_number(getattr(op, "memory", None)) is None
    )


def needs_gpu_profile(op) -> bool:
    """Return whether a CUDA batch op can be timed by the preflight worker."""
    num_gpus = _positive_number(getattr(op, "num_gpus", None))
    return (
        getattr(op, "accelerator", None) == "cuda"
        and isinstance(op, (Mapper, Filter))
        and (num_gpus is None or num_gpus <= 1)
    )


def probe_sample_count(ops: Sequence) -> int:
    """Use one maximum configured GPU batch as the preflight sample size."""
    targets = [op for op in ops if needs_gpu_probe(op) or needs_gpu_profile(op)]
    return max((_normalized_batch_size(op) for op in targets), default=0)


def _normalized_batch_size(op) -> int:
    value = getattr(op, "batch_size", 1)
    if isinstance(value, bool):
        return 1
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return 1


def fill_to_batch(rows: Sequence[Dict], batch_size: int) -> List[Dict]:
    """Return exactly one full batch, cycling surviving rows if necessary."""
    rows = list(rows)
    if not rows:
        raise RuntimeError("GPU preflight has no samples left for the next GPU operator.")
    if len(rows) >= batch_size:
        return rows[:batch_size]
    return list(islice(cycle(rows), batch_size))


def _declared_columns(op, attribute: str) -> Optional[Set[str]]:
    """Normalize an optional operator data-flow declaration.

    ``None`` deliberately means unknown, while an empty collection is a valid
    declaration.  The distinction is what lets old operators fall back to the
    ordered probe without being optimistically treated as independent.
    """
    value = getattr(op, attribute, None)
    if value is None:
        return None
    if isinstance(value, str):
        value = [value]
    try:
        return {str(column).strip() for column in value if str(column).strip()}
    except TypeError:
        return None


def _column_paths_overlap(left: str, right: str) -> bool:
    """Return whether two top-level or nested column patterns can overlap."""
    if left == right:
        return True
    if left.startswith(right + ".") or right.startswith(left + "."):
        return True
    return fnmatchcase(left, right) or fnmatchcase(right, left)


def _columns_overlap(left: Set[str], right: Set[str]) -> bool:
    return any(_column_paths_overlap(a, b) for a in left for b in right)


def _dependency_closure(index: int, dependencies: Mapping[int, Set[int]]) -> Set[int]:
    closure: Set[int] = set()
    pending = list(dependencies.get(index, set()))
    while pending:
        dependency = pending.pop()
        if dependency in closure:
            continue
        closure.add(dependency)
        pending.extend(dependencies.get(dependency, set()))
    return closure


def plan_parallel_probe_jobs(
    ops: Sequence,
    pending_targets: Mapping[int, Any],
) -> Tuple[List[Dict[str, Any]], Dict[int, str]]:
    """Plan isolated probes whose target has only declared CPU ancestors.

    The planner is intentionally proof-based.  A missing read/write contract,
    a dataset-level operator, an index-generating operator, a different runtime
    environment, or a preceding GPU data dependency keeps that target on the
    existing ordered replay path.
    """
    reads = [_declared_columns(op, "_input_columns") for op in ops]
    writes = [_declared_columns(op, "_output_columns") for op in ops]
    dependencies: Dict[int, Set[int]] = {index: set() for index in range(len(ops))}

    for index, op in enumerate(ops):
        if reads[index] is None or writes[index] is None:
            continue
        for earlier in range(index):
            # Non-Mapper operators can change row membership/order, so all
            # later stages depend on them even when their columns are disjoint.
            if not isinstance(ops[earlier], Mapper):
                dependencies[index].add(earlier)
            if reads[earlier] is None or writes[earlier] is None:
                continue
            if _columns_overlap(writes[earlier], reads[index] | writes[index]):
                dependencies[index].add(earlier)

    jobs: List[Dict[str, Any]] = []
    fallback_reasons: Dict[int, str] = {}
    for index, target in pending_targets.items():
        if not isinstance(target, (Mapper, Filter)):
            fallback_reasons[index] = "target is not a Mapper/Filter"
            continue

        missing = [earlier for earlier in range(index + 1) if reads[earlier] is None or writes[earlier] is None]
        if missing:
            names = [getattr(ops[item], "_name", type(ops[item]).__name__) for item in missing]
            fallback_reasons[index] = "missing input_columns/output_columns: " + ", ".join(names)
            continue

        ancestors = sorted(_dependency_closure(index, dependencies))
        unsupported = [
            item
            for item in ancestors
            if not isinstance(ops[item], (Mapper, Filter)) or getattr(ops[item], "index_key", None) is not None
        ]
        if getattr(target, "index_key", None) is not None:
            unsupported.append(index)
        if unsupported:
            names = [getattr(ops[item], "_name", type(ops[item]).__name__) for item in unsupported]
            fallback_reasons[index] = "requires dataset-level replay: " + ", ".join(names)
            continue

        gpu_ancestors = [item for item in ancestors if getattr(ops[item], "accelerator", None) == "cuda"]
        if gpu_ancestors:
            names = [getattr(ops[item], "_name", type(ops[item]).__name__) for item in gpu_ancestors]
            fallback_reasons[index] = "depends on earlier GPU operator(s): " + ", ".join(names)
            continue

        target_runtime_env = getattr(target, "runtime_env", None)
        incompatible = [item for item in ancestors if getattr(ops[item], "runtime_env", None) != target_runtime_env]
        if incompatible:
            names = [getattr(ops[item], "_name", type(ops[item]).__name__) for item in incompatible]
            fallback_reasons[index] = "has incompatible ancestor runtime_env: " + ", ".join(names)
            continue

        jobs.append(
            {
                "op_index": index,
                "target": target,
                "dependencies": [ops[item] for item in ancestors],
                "dependency_indices": ancestors,
            }
        )

    return jobs, fallback_reasons


def _numeric_sequence_info(feature) -> tuple[int, Optional[str]]:
    """Return nesting depth and leaf dtype for a numeric sequence feature."""
    if hasattr(feature, "shape") and hasattr(feature, "dtype"):
        return len(feature.shape), str(feature.dtype)

    depth = 0
    while hasattr(feature, "feature"):
        depth += 1
        feature = feature.feature
    dtype = str(getattr(feature, "dtype", ""))
    if dtype == "bool" or dtype.startswith(("int", "uint", "float", "complex")):
        return depth, dtype
    return 0, None


def _array_columns_from_rows(rows: Sequence[Dict]) -> set[str]:
    """Find columns whose in-memory values are already NumPy arrays."""
    import numpy as np

    return {key for row in rows for key, value in row.items() if isinstance(value, np.ndarray)}


def _dataset_with_array_format(dataset, columns: Sequence[str]):
    """Restore selected Arrow sequence columns as NumPy arrays for an op."""
    columns = [column for column in columns if column in dataset.column_names]
    if not columns:
        return dataset

    import numpy as np

    dtypes = {column: _numeric_sequence_info(dataset.features[column])[1] for column in columns}

    def restore_arrays(batch):
        for column in columns:
            dtype = dtypes[column]
            batch[column] = [np.asarray(value, dtype=dtype) if value is not None else None for value in batch[column]]
        return batch

    # A custom transform deliberately returns list[np.ndarray]. The built-in
    # NumPy formatter instead returns an outer object ndarray for variable
    # image shapes, making common mapper code such as ``values or fallback``
    # fail with an ambiguous truth-value error.
    return dataset.with_transform(restore_arrays)


def _rows_with_numpy_arrays(dataset, inherited_columns: Sequence[str]) -> List[Dict]:
    """Materialize rows without losing multi-dimensional numeric arrays.

    ``Dataset.to_list()`` converts arrays emitted by an upstream mapper into
    Python lists.  That changes the contract for pipelines such as image
    bucket preprocessing, whose downstream GPU mapper calls NumPy attributes
    such as ``shape`` and ``size``.  Preserve inherited array columns and new
    multi-dimensional numeric sequence columns while leaving ordinary lists
    (for example ``images: [path]``) untouched.
    """
    rows = dataset.to_list()
    if not rows:
        return rows

    array_columns = set(inherited_columns)
    array_columns.update(
        column for column, feature in dataset.features.items() if _numeric_sequence_info(feature)[0] >= 2
    )
    formatted = _dataset_with_array_format(dataset, sorted(array_columns))
    active_columns = array_columns.intersection(dataset.column_names)
    for index, row in enumerate(rows):
        formatted_row = formatted[index]
        for column in active_columns:
            value = formatted_row[column]
            _, dtype = _numeric_sequence_info(dataset.features[column])
            # Restore the Arrow leaf dtype explicitly so a uint8 image cannot
            # be promoted to int64 during an array-formatting round trip.
            if dtype is not None and hasattr(value, "astype"):
                value = value.astype(dtype, copy=False)
            row[column] = value
    return rows


def calculate_resource_plan(
    measured_mb: float,
    total_mb: float,
    max_gpu_workers_per_device: int = _DEFAULT_MAX_GPU_WORKERS_PER_DEVICE,
) -> Dict[str, float]:
    """Convert a measured peak into memory and conservative Ray resources.

    ``memory_fraction`` records what the operator actually needs. ``num_gpus``
    is the Ray scheduling request and also reserves a compute slot, preventing
    a tiny model from creating an unbounded number of actors on one device.
    """
    if measured_mb <= 0:
        raise RuntimeError(f"GPU preflight measured an invalid peak ({measured_mb} MiB).")
    if total_mb <= 0:
        raise RuntimeError(f"GPU preflight found an invalid GPU capacity ({total_mb} MiB).")
    if isinstance(max_gpu_workers_per_device, bool) or max_gpu_workers_per_device < 1:
        raise ValueError("max_gpu_workers_per_device must be a positive integer")

    planned_mb = int(math.ceil(measured_mb * _MEMORY_HEADROOM))
    if planned_mb > total_mb:
        raise RuntimeError(
            f"GPU preflight requires {planned_mb} MiB after 10% headroom, "
            f"but the assigned GPU has only {total_mb:.0f} MiB. "
            "Automatic probing supports only operators that fit on one GPU; "
            "configure memory and num_gpus explicitly for multi-GPU operators."
        )

    memory_fraction = math.ceil(planned_mb / total_mb * 100) / 100
    memory_fraction = min(1.0, max(0.01, memory_fraction))
    compute_fraction = 1.0 / int(max_gpu_workers_per_device)
    num_gpus = min(1.0, max(memory_fraction, compute_fraction))
    return {
        "planned_memory_mb": planned_mb,
        "memory": planned_mb / 1024,
        "memory_fraction": memory_fraction,
        "num_gpus": num_gpus,
    }


def _config_hash(op) -> str:
    config = getattr(op, "_op_cfg", {getattr(op, "_name", type(op).__name__): {}})
    payload = json.dumps(config, sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cuda_memory_snapshot() -> Dict[str, float]:
    """Read process-visible GPU usage and capacity through PyTorch."""
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "GPU preflight expected exactly one Ray-assigned GPU, " f"but PyTorch sees {torch.cuda.device_count()}."
        )
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    properties = torch.cuda.get_device_properties(0)
    return {
        "used_mb": (total_bytes - free_bytes) / _MIB,
        "total_mb": total_bytes / _MIB,
        "gpu_name": properties.name,
    }


def _read_gpu_hardware() -> Dict[str, float]:
    return _cuda_memory_snapshot()


def _gpu_assignment_context() -> str:
    try:
        gpu_ids = ray.get_runtime_context().get_accelerator_ids().get("GPU", [])
    except Exception:
        gpu_ids = "unavailable"
    return f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}, Ray GPU IDs={gpu_ids}"


def _measure_cuda_call(function: Callable[[], Any]) -> Tuple[Any, Dict[str, float]]:
    """Run ``function`` without touching its CUDA context concurrently.

    Probe workers are disposable and reserve a full GPU, so their allocator
    history starts clean.  Let the operator initialize CUDA exactly as a
    formal Ray actor would, then combine PyTorch's allocator peak with the
    process-visible device usage after the call.  A background
    ``cudaMemGetInfo`` polling thread used to run during model construction;
    it could serialize hundreds of parameter transfers on the same CUDA
    context and make ``model.to(cuda)`` take minutes.

    The allocator peak captures transient PyTorch allocations.  The final
    device snapshot covers persistent allocations made by other runtimes such
    as Paddle.  Counting all final device usage is deliberately conservative
    if an unmanaged process is sharing the Ray-assigned GPU.
    """
    result = function()
    import torch

    torch.cuda.synchronize(0)
    final = _cuda_memory_snapshot()
    torch_peak_mb = torch.cuda.max_memory_reserved(0) / _MIB
    measured_mb = max(final["used_mb"], torch_peak_mb)
    return result, {
        **final,
        "baseline_used_mb": 0.0,
        "peak_used_mb": final["used_mb"],
        "torch_peak_reserved_mb": torch_peak_mb,
        "measured_memory_mb": measured_mb,
        "memory_measurement_mode": "post_call_device_usage_and_allocator_peak",
    }


def _rows_to_batch(rows: Sequence[Dict]) -> Dict[str, List[Any]]:
    columns: List[str] = []
    seen = set()
    for row in rows:
        for column in row:
            if column not in seen:
                seen.add(column)
                columns.append(column)
    return {column: [row.get(column) for row in rows] for column in columns}


def _batch_to_rows(batch: Mapping[str, Sequence[Any]]) -> List[Dict[str, Any]]:
    if not batch:
        return []
    lengths = {len(values) for values in batch.values()}
    if len(lengths) != 1:
        raise RuntimeError(f"GPU preflight operator returned inconsistent batch column lengths: {sorted(lengths)}")
    row_count = lengths.pop()
    return [{column: values[index] for column, values in batch.items()} for index in range(row_count)]


def _construct_probe_op(spec: Mapping[str, Any]):
    kwargs = dict(spec.get("init_kwargs") or {})
    # Keep all work in the disposable Ray worker.  Dataset-level
    # multiprocessing after CUDA initialization would fork an unsafe child.
    kwargs.update(num_proc=None, auto_op_parallelism=False, skip_op_error=False)
    return spec["op_class"](*(spec.get("init_args") or ()), **kwargs)


def _run_probe_op_rows(spec: Mapping[str, Any], rows: Sequence[Dict]) -> List[Dict]:
    """Execute one Mapper/Filter directly on batches without an Arrow round-trip."""
    op = _construct_probe_op(spec)
    return _run_constructed_probe_op_rows(op, rows)


def _run_constructed_probe_op_rows(op, rows: Sequence[Dict]) -> List[Dict]:
    """Run rows through an already initialized operator instance."""
    from data_juicer.ops.fused_batch_executor import execute_sequential_batch

    batch_size = _normalized_batch_size(op)
    output_rows: List[Dict] = []
    for offset in range(0, len(rows), batch_size):
        batch = _rows_to_batch(rows[offset : offset + batch_size])
        result = execute_sequential_batch(
            batch,
            [op],
            rank=0,
            owner_name="GPU memory preflight",
        )
        output_rows.extend(_batch_to_rows(result))
    return output_rows


def _profile_probe_target(
    spec: Mapping[str, Any],
    rows: Sequence[Dict],
    warmup_batches: int,
    steady_batches: int,
) -> Tuple[List[Dict], Dict[str, Any]]:
    """Measure model initialization separately from warmup and steady state."""
    target_name = str(spec.get("op_name") or "unknown")
    init_started = time.monotonic()
    logger.info(f"GPU preflight profile Op[{target_name}] constructing operator instance.")
    op = _construct_probe_op(spec)
    initialization = time.monotonic() - init_started
    logger.info(f"GPU preflight profile Op[{target_name}] constructed operator instance " f"in {initialization:.2f}s.")

    warmup_started = time.monotonic()
    for batch_index in range(warmup_batches):
        batch_started = time.monotonic()
        logger.info(
            f"GPU preflight profile Op[{target_name}] starting warmup batch " f"{batch_index + 1}/{warmup_batches}."
        )
        _run_constructed_probe_op_rows(op, rows)
        logger.info(
            f"GPU preflight profile Op[{target_name}] finished warmup batch "
            f"{batch_index + 1}/{warmup_batches} in "
            f"{time.monotonic() - batch_started:.2f}s."
        )
    warmup = time.monotonic() - warmup_started

    steady_durations: List[float] = []
    steady_outputs: List[int] = []
    output_rows: List[Dict] = []
    for batch_index in range(steady_batches):
        batch_started = time.monotonic()
        logger.info(
            f"GPU preflight profile Op[{target_name}] starting steady batch " f"{batch_index + 1}/{steady_batches}."
        )
        output_rows = _run_constructed_probe_op_rows(op, rows)
        batch_seconds = time.monotonic() - batch_started
        steady_durations.append(batch_seconds)
        steady_outputs.append(len(output_rows))
        logger.info(
            f"GPU preflight profile Op[{target_name}] finished steady batch "
            f"{batch_index + 1}/{steady_batches} in {batch_seconds:.2f}s; "
            f"output_rows={len(output_rows)}."
        )

    steady_total = sum(steady_durations)
    input_rows = len(rows) * steady_batches
    output_count = sum(steady_outputs)
    profile = {
        "initialization": initialization,
        "warmup": warmup,
        "steady_total": steady_total,
        "steady_batch_seconds": steady_total / steady_batches,
        "steady_rows_per_second": input_rows / steady_total if steady_total > 0 else 0.0,
        "output_ratio": output_count / input_rows if input_rows else 0.0,
        "warmup_batches": warmup_batches,
        "steady_batches": steady_batches,
    }
    return output_rows, profile


def _op_spec(op) -> Dict[str, Any]:
    return {
        "op_class": type(op),
        "init_args": getattr(op, "_init_args", ()),
        "init_kwargs": getattr(op, "_init_kwargs", {}),
        "op_name": getattr(op, "_name", None) or type(op).__name__,
        "batch_size": _normalized_batch_size(op),
    }


def _run_parallel_probe_job(job: Mapping[str, Any], source_rows: List[Dict]) -> Dict[str, Any]:
    """Replay declared CPU ancestors and measure one target in one worker."""
    target = job["target"]
    target_name = target["op_name"]
    op_index = int(job["op_index"])
    target_batch_size = int(target["batch_size"])
    started = time.monotonic()
    dependency_timings: Dict[str, float] = {}
    dependency_names = [dependency["op_name"] for dependency in job["dependencies"]]
    logger.info(
        "GPU preflight worker started Op[{}] at recipe index {}: source_rows={}, "
        "target_batch={}, dependencies={}, {}".format(
            target_name,
            op_index,
            len(source_rows),
            target_batch_size,
            dependency_names,
            _gpu_assignment_context(),
        )
    )

    try:
        # Pure Mapper ancestry preserves row count, so it only needs the target's
        # one batch. Keep the full sampled prefix when a Filter may remove rows.
        has_filter = any(issubclass(dependency["op_class"], Filter) for dependency in job["dependencies"])
        rows = list(source_rows if has_filter else source_rows[:target_batch_size])
        for dependency in job["dependencies"]:
            dependency_name = dependency["op_name"]
            dependency_started = time.monotonic()
            logger.info(
                f"GPU preflight worker Op[{target_name}] starting dependency "
                f"Op[{dependency_name}] with {len(rows)} row(s)."
            )
            rows = _run_probe_op_rows(dependency, rows)
            dependency_seconds = time.monotonic() - dependency_started
            dependency_timings[dependency_name] = dependency_seconds
            logger.info(
                f"GPU preflight worker Op[{target_name}] finished dependency "
                f"Op[{dependency_name}] in {dependency_seconds:.2f}s; "
                f"output_rows={len(rows)}."
            )
            if not rows:
                raise RuntimeError("GPU preflight has no samples left after dependency " f"Op[{dependency_name}].")

        target_rows = fill_to_batch(rows, target_batch_size)

        warmup_batches = max(0, int(job.get("warmup_batches", 1)))
        steady_batches = max(1, int(job.get("steady_batches", 3)))

        def run_target():
            # Keep operator execution identical to the formal Ray actor.  A
            # diagnostic marker used here in the past enabled extra CUDA
            # synchronizations in custom operators and could turn a normal
            # model transfer into a multi-minute preflight stall.
            previous_marker = os.environ.pop(_PREFLIGHT_OP_ENV_VAR, None)
            try:
                return _profile_probe_target(target, target_rows, warmup_batches, steady_batches)
            finally:
                if previous_marker is None:
                    os.environ.pop(_PREFLIGHT_OP_ENV_VAR, None)
                else:
                    os.environ[_PREFLIGHT_OP_ENV_VAR] = previous_marker

        target_started = time.monotonic()
        logger.info(
            f"GPU preflight worker Op[{target_name}] starting measured target " f"with {len(target_rows)} row(s)."
        )
        (output_rows, profile), metrics = _measure_cuda_call(run_target)
        target_seconds = time.monotonic() - target_started
        total_seconds = time.monotonic() - started
        logger.info(
            f"GPU preflight worker completed Op[{target_name}] at recipe index {op_index}: "
            f"target={target_seconds:.2f}s, total={total_seconds:.2f}s, "
            f"peak={metrics['measured_memory_mb']:.0f} MiB."
        )
        return {
            "op_index": op_index,
            "sample_count": len(target_rows),
            "output_count": len(output_rows),
            "metrics": metrics,
            "profile": profile,
            "timing_seconds": {
                "dependencies": dependency_timings,
                "target": target_seconds,
                "total": total_seconds,
            },
        }
    except Exception:
        logger.exception(
            f"GPU preflight worker failed Op[{target_name}] at recipe index {op_index} "
            f"after {time.monotonic() - started:.2f}s."
        )
        raise


def _run_probe_stage(
    op_class,
    init_args,
    init_kwargs,
    rows: List[Dict],
    measure_memory: bool,
    warmup_batches: int = 0,
    steady_batches: int = 1,
    op_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Run one recipe stage in a disposable Ray worker."""
    from datasets import Dataset, disable_caching

    disable_caching()
    kwargs = dict(init_kwargs or {})
    # Hugging Face Datasets treats ``num_proc=1`` as multiprocessing and
    # starts a forked child.  CUDA has already been initialized in this
    # disposable Ray worker so that model construction is included in the
    # peak; using a fork here both fails CUDA re-initialization and would hide
    # the child's allocator peak from this process.  ``None`` keeps the whole
    # probe stage in the Ray worker while still processing batches serially.
    kwargs.update(num_proc=None, auto_op_parallelism=False, skip_op_error=False)

    def run_stage_once():
        op = op_class(*(init_args or ()), **kwargs)
        array_columns = _array_columns_from_rows(rows)
        input_dataset = _dataset_with_array_format(Dataset.from_list(rows), array_columns)
        output = op.run(input_dataset)
        return _rows_with_numpy_arrays(output, array_columns)

    def profile_stage():
        target_name = str(op_name or getattr(op_class, "_name", None) or op_class.__name__)
        previous_marker = os.environ.pop(_PREFLIGHT_OP_ENV_VAR, None)
        try:
            init_started = time.monotonic()
            logger.info(f"GPU preflight profile Op[{target_name}] constructing operator instance.")
            op = op_class(*(init_args or ()), **kwargs)
            initialization = time.monotonic() - init_started
            logger.info(
                f"GPU preflight profile Op[{target_name}] constructed operator instance " f"in {initialization:.2f}s."
            )
            array_columns = _array_columns_from_rows(rows)

            def execute_once():
                input_dataset = _dataset_with_array_format(Dataset.from_list(rows), array_columns)
                return _rows_with_numpy_arrays(op.run(input_dataset), array_columns)

            warmup_started = time.monotonic()
            for batch_index in range(warmup_batches):
                batch_started = time.monotonic()
                logger.info(
                    f"GPU preflight profile Op[{target_name}] starting warmup batch "
                    f"{batch_index + 1}/{warmup_batches}."
                )
                execute_once()
                logger.info(
                    f"GPU preflight profile Op[{target_name}] finished warmup batch "
                    f"{batch_index + 1}/{warmup_batches} in "
                    f"{time.monotonic() - batch_started:.2f}s."
                )
            warmup = time.monotonic() - warmup_started

            durations = []
            output_counts = []
            output_rows = []
            repeats = max(1, steady_batches)
            for batch_index in range(repeats):
                batch_started = time.monotonic()
                logger.info(
                    f"GPU preflight profile Op[{target_name}] starting steady batch " f"{batch_index + 1}/{repeats}."
                )
                output_rows = execute_once()
                batch_seconds = time.monotonic() - batch_started
                durations.append(batch_seconds)
                output_counts.append(len(output_rows))
                logger.info(
                    f"GPU preflight profile Op[{target_name}] finished steady batch "
                    f"{batch_index + 1}/{repeats} in {batch_seconds:.2f}s; "
                    f"output_rows={len(output_rows)}."
                )
            steady_total = sum(durations)
            input_count = len(rows) * repeats
            return output_rows, {
                "initialization": initialization,
                "warmup": warmup,
                "steady_total": steady_total,
                "steady_batch_seconds": steady_total / repeats,
                "steady_rows_per_second": input_count / steady_total if steady_total > 0 else 0.0,
                "output_ratio": sum(output_counts) / input_count if input_count else 0.0,
                "warmup_batches": warmup_batches,
                "steady_batches": repeats,
            }
        finally:
            if previous_marker is None:
                os.environ.pop(_PREFLIGHT_OP_ENV_VAR, None)
            else:
                os.environ[_PREFLIGHT_OP_ENV_VAR] = previous_marker

    if measure_memory:
        (output_rows, profile), metrics = _measure_cuda_call(profile_stage)
    else:
        output_rows = run_stage_once()
        metrics = None
    result = {"rows": output_rows}
    if measure_memory:
        result["metrics"] = metrics
        result["profile"] = profile
    return result


class GPUMemoryProbe:
    """Resolve missing GPU resources before formal partitioned execution."""

    def __init__(
        self,
        work_dir: str,
        *,
        max_gpu_workers_per_device: int = _DEFAULT_MAX_GPU_WORKERS_PER_DEVICE,
        max_concurrent_probes: Optional[int] = None,
        probe_timeout_seconds: Optional[float] = None,
        warmup_batches: int = 1,
        steady_batches: int = 3,
        stage_runner: Optional[Callable] = None,
        parallel_runner: Optional[Callable] = None,
        hardware_reader: Optional[Callable] = None,
    ):
        if isinstance(max_gpu_workers_per_device, bool) or max_gpu_workers_per_device < 1:
            raise ValueError("max_gpu_workers_per_device must be a positive integer")
        if max_concurrent_probes is not None and (
            isinstance(max_concurrent_probes, bool) or int(max_concurrent_probes) < 1
        ):
            raise ValueError("max_concurrent_probes must be a positive integer or None")
        if probe_timeout_seconds is not None and (
            isinstance(probe_timeout_seconds, bool) or float(probe_timeout_seconds) <= 0
        ):
            raise ValueError("probe_timeout_seconds must be a positive number or None")
        if isinstance(warmup_batches, bool) or int(warmup_batches) < 0:
            raise ValueError("warmup_batches must be a non-negative integer")
        if isinstance(steady_batches, bool) or int(steady_batches) < 1:
            raise ValueError("steady_batches must be a positive integer")
        self.work_dir = work_dir
        self.report_path = os.path.join(work_dir, _REPORT_NAME)
        self.max_gpu_workers_per_device = int(max_gpu_workers_per_device)
        self.max_concurrent_probes = int(max_concurrent_probes) if max_concurrent_probes is not None else None
        self.probe_timeout_seconds = float(probe_timeout_seconds) if probe_timeout_seconds is not None else None
        self.warmup_batches = int(warmup_batches)
        self.steady_batches = int(steady_batches)
        self._stage_runner = stage_runner or self._run_stage_with_ray
        self._parallel_runner = parallel_runner or self._run_parallel_jobs_with_ray
        self._hardware_reader = hardware_reader or self._read_hardware_with_ray

    def resolve(self, dataset, ops: Sequence) -> List[Dict[str, Any]]:
        """Load reusable measurements, then probe all remaining target ops."""
        candidates = {index: op for index, op in enumerate(ops) if needs_gpu_probe(op) or needs_gpu_profile(op)}
        if not candidates:
            return []

        pipeline_targets = [op for op in candidates.values() if isinstance(op, Pipeline)]
        if pipeline_targets:
            names = ", ".join(getattr(op, "_name", type(op).__name__) for op in pipeline_targets)
            raise RuntimeError(
                f"GPU preflight does not support Pipeline operators ({names}). "
                "Configure their memory and num_gpus explicitly."
            )

        existing_records = self._load_report()
        reusable = self._select_reusable_records(candidates, existing_records)
        for index, record in reusable.items():
            self._apply_record(ops[index], record)
            logger.info(
                f"Reusing GPU preflight result for Op[{record['op_name']}]: "
                f"memory={record['memory']:.4f} GB, num_gpus={record['num_gpus']:.2f}"
            )

        pending = {index: op for index, op in candidates.items() if index not in reusable}
        if not pending:
            return [reusable[index] for index in sorted(reusable)]

        last_target = max(pending)
        barriers = [
            getattr(op, "_name", type(op).__name__) for op in ops[: last_target + 1] if isinstance(op, Pipeline)
        ]
        if barriers:
            raise RuntimeError(
                "GPU preflight cannot replay recipe stages across Pipeline operator(s): "
                f"{', '.join(barriers)}. Configure downstream GPU operators' memory "
                "and num_gpus explicitly."
            )

        sample_count = probe_sample_count(list(pending.values()))
        rows = list(dataset.get(sample_count))
        if not rows:
            raise RuntimeError("GPU preflight cannot run because the input dataset is empty.")

        logger.info(
            f"Running GPU memory preflight for {len(pending)} operator(s) " f"with the first {len(rows)} sample(s)."
        )
        new_records: Dict[int, Dict[str, Any]] = {}

        parallel_jobs, fallback_reasons = plan_parallel_probe_jobs(ops[: last_target + 1], pending)
        parallel_indices = {job["op_index"] for job in parallel_jobs}
        for index, reason in fallback_reasons.items():
            logger.info(
                "GPU preflight keeps Op[{}] at recipe index {} on ordered replay: {}".format(
                    getattr(ops[index], "_name", type(ops[index]).__name__), index, reason
                )
            )

        if parallel_jobs:
            names = [getattr(job["target"], "_name", type(job["target"]).__name__) for job in parallel_jobs]
            logger.info(
                f"GPU preflight is launching {len(parallel_jobs)} dependency-safe target(s) "
                f"in parallel with one full GPU each: {names}"
            )
            try:
                parallel_results = self._parallel_runner(parallel_jobs, rows)
            except Exception as error:
                raise RuntimeError(
                    "Parallel GPU preflight failed. The formal Ray experiment was not started."
                ) from error

            for job in parallel_jobs:
                index = job["op_index"]
                try:
                    result = parallel_results[index]
                    dependencies = [getattr(op, "_name", type(op).__name__) for op in job["dependencies"]]
                    record = self._record_from_metrics(
                        index,
                        ops[index],
                        result.get("metrics") or {},
                        int(result.get("sample_count", _normalized_batch_size(ops[index]))),
                        probe_mode="parallel",
                        dependencies=dependencies,
                        profile=result.get("profile"),
                    )
                    timing_seconds = result.get("timing_seconds")
                    if isinstance(timing_seconds, Mapping):
                        record["timing_seconds"] = dict(timing_seconds)
                except Exception as error:
                    name = getattr(ops[index], "_name", type(ops[index]).__name__)
                    raise RuntimeError(
                        f"GPU preflight failed while measuring Op[{name}] at recipe index {index}. "
                        "The formal Ray experiment was not started."
                    ) from error
                self._accept_record(ops[index], record)
                new_records[index] = record

        serial_pending = {index: op for index, op in pending.items() if index not in parallel_indices}
        if serial_pending:
            serial_rows = rows
            last_serial_target = max(serial_pending)
            for index, op in enumerate(ops[: last_serial_target + 1]):
                is_target = index in serial_pending
                stage_rows = fill_to_batch(serial_rows, _normalized_batch_size(op)) if is_target else serial_rows
                try:
                    result = self._stage_runner(op, stage_rows, is_target)
                except Exception as error:
                    name = getattr(op, "_name", type(op).__name__)
                    raise RuntimeError(
                        f"GPU preflight failed while replaying Op[{name}] at recipe index {index}. "
                        "The formal Ray experiment was not started."
                    ) from error
                serial_rows = list(result["rows"])

                if is_target:
                    record = self._record_from_metrics(
                        index,
                        op,
                        result.get("metrics") or {},
                        len(stage_rows),
                        probe_mode="ordered",
                        dependencies=[getattr(item, "_name", type(item).__name__) for item in ops[:index]],
                        profile=result.get("profile"),
                    )
                    self._accept_record(op, record)
                    new_records[index] = record

        merged = {**reusable, **new_records}
        self._save_report([merged[index] for index in sorted(merged)])
        return [merged[index] for index in sorted(merged)]

    def _record_from_metrics(
        self,
        index: int,
        op,
        metrics: Mapping[str, Any],
        sample_count: int,
        *,
        probe_mode: str,
        dependencies: Sequence[str],
        profile: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        measured_plan = calculate_resource_plan(
            float(metrics.get("measured_memory_mb", 0)),
            float(metrics.get("total_mb", 0)),
            self.max_gpu_workers_per_device,
        )
        configured_memory = _positive_number(getattr(op, "memory", None))
        configured_num_gpus = _positive_number(getattr(op, "num_gpus", None))
        plan = {
            **measured_plan,
            "memory": configured_memory or measured_plan["memory"],
            "num_gpus": configured_num_gpus or measured_plan["num_gpus"],
        }
        record = {
            "op_index": index,
            "op_name": getattr(op, "_name", type(op).__name__),
            "op_class": f"{type(op).__module__}.{type(op).__qualname__}",
            "config_hash": _config_hash(op),
            "batch_size": _normalized_batch_size(op),
            "sample_count": sample_count,
            "probe_mode": probe_mode,
            "replayed_dependencies": list(dependencies),
            "resource_mode": ("auto" if configured_memory is None and configured_num_gpus is None else "configured"),
            "gpu_name": metrics.get("gpu_name", "unknown"),
            "gpu_total_mb": float(metrics["total_mb"]),
            "baseline_used_mb": float(metrics.get("baseline_used_mb", 0)),
            "peak_used_mb": float(metrics.get("peak_used_mb", 0)),
            "torch_peak_reserved_mb": float(metrics.get("torch_peak_reserved_mb", 0)),
            "measured_memory_mb": float(metrics["measured_memory_mb"]),
            "memory_measurement_mode": metrics.get(
                "memory_measurement_mode",
                "legacy",
            ),
            **plan,
        }
        if isinstance(profile, Mapping):
            record["profile"] = {
                "initialization": float(profile.get("initialization", 0) or 0),
                "warmup": float(profile.get("warmup", 0) or 0),
                "steady_total": float(profile.get("steady_total", 0) or 0),
                "steady_batch_seconds": float(profile.get("steady_batch_seconds", 0) or 0),
                "steady_rows_per_second": float(profile.get("steady_rows_per_second", 0) or 0),
                "output_ratio": float(profile.get("output_ratio", 1) or 0),
                "warmup_batches": int(profile.get("warmup_batches", self.warmup_batches)),
                "steady_batches": int(profile.get("steady_batches", self.steady_batches)),
            }
        return record

    def _accept_record(self, op, record: Dict[str, Any]) -> None:
        self._apply_record(op, record)
        logger.info(
            f"GPU preflight resolved Op[{record['op_name']}] via {record['probe_mode']} probe: "
            f"peak={record['measured_memory_mb']:.0f} MiB, "
            f"planned={record['planned_memory_mb']} MiB, "
            f"memory_fraction={record['memory_fraction']:.2f}, "
            f"scheduling_num_gpus={record['num_gpus']:.2f}, "
            f"steady_throughput={float(record.get('profile', {}).get('steady_rows_per_second', 0)):.2f} rows/s"
        )

    @staticmethod
    def _job_num_cpus(job: Mapping[str, Any]) -> float:
        members = [*job["dependencies"], job["target"]]
        return max((_positive_number(getattr(op, "num_cpus", None)) or 1 for op in members), default=1)

    def _parallel_probe_limit(self, jobs: Sequence[Mapping[str, Any]]) -> int:
        resources = ray.cluster_resources()
        gpu_slots = int(math.floor(float(resources.get("GPU", 0) or 0)))
        if gpu_slots < 1:
            raise RuntimeError("GPU preflight found no GPU resources in the Ray cluster.")
        max_job_cpus = max((self._job_num_cpus(job) for job in jobs), default=1)
        cpu_slots = int(math.floor(float(resources.get("CPU", 0) or 0) / max_job_cpus))
        if cpu_slots < 1:
            raise RuntimeError(f"GPU preflight has no CPU slot for a probe requiring {max_job_cpus:g} CPU(s).")
        limit = min(len(jobs), gpu_slots, cpu_slots)
        # Auto mode fills dependency-safe GPU/CPU slots. Users can set an
        # explicit cap when storage or host-memory bandwidth cannot sustain
        # that many concurrent model initializations.
        if self.max_concurrent_probes is not None:
            limit = min(limit, self.max_concurrent_probes)
        return max(1, limit)

    def _run_parallel_jobs_with_ray(
        self,
        jobs: Sequence[Mapping[str, Any]],
        rows: List[Dict],
    ) -> Dict[int, Dict[str, Any]]:
        """Run dependency-safe target probes with bounded Ray concurrency."""
        remote_probe = ray.remote(max_calls=1)(_run_parallel_probe_job)
        limit = self._parallel_probe_limit(jobs)
        logger.info(
            f"GPU preflight parallel concurrency={limit} "
            f"(targets={len(jobs)}, cluster_gpus={ray.cluster_resources().get('GPU', 0)})"
        )

        queue = list(jobs)
        running: Dict[Any, Dict[str, Any]] = {}
        results: Dict[int, Dict[str, Any]] = {}

        def submit(job):
            target = job["target"]
            wire_job = {
                "op_index": job["op_index"],
                "target": _op_spec(target),
                "dependencies": [_op_spec(op) for op in job["dependencies"]],
                "warmup_batches": self.warmup_batches,
                "steady_batches": self.steady_batches,
            }
            options = {"num_cpus": self._job_num_cpus(job), "num_gpus": 1}
            runtime_env = getattr(target, "runtime_env", None)
            if runtime_env is not None:
                options["runtime_env"] = runtime_env
            future = remote_probe.options(**options).remote(wire_job, rows)
            running[future] = {"job": job, "submitted_at": time.monotonic()}
            logger.info(
                "GPU preflight submitted Op[{}] at recipe index {} "
                "(running={}, queued={}).".format(
                    getattr(target, "_name", type(target).__name__),
                    job["op_index"],
                    len(running),
                    len(queue),
                )
            )

        while queue and len(running) < limit:
            submit(queue.pop(0))

        try:
            last_progress_log = time.monotonic()
            while running:
                wait_seconds = _PROGRESS_INTERVAL_SECONDS
                if self.probe_timeout_seconds is not None:
                    wait_seconds = min(wait_seconds, self.probe_timeout_seconds)
                ready, _ = ray.wait(list(running), num_returns=1, timeout=wait_seconds)
                now = time.monotonic()

                if ready:
                    # Drain futures that became ready together. Otherwise a
                    # completed task could be mistaken for an overdue running
                    # task when several probes finish near the timeout.
                    remaining = [future for future in running if future not in ready]
                    if remaining:
                        additionally_ready, _ = ray.wait(
                            remaining,
                            num_returns=len(remaining),
                            timeout=0,
                        )
                        ready.extend(additionally_ready)

                    for future in ready:
                        state = running.pop(future)
                        job = state["job"]
                        target = job["target"]
                        target_name = getattr(target, "_name", type(target).__name__)
                        result = ray.get(future)
                        elapsed = now - state["submitted_at"]
                        results[int(job["op_index"])] = result
                        logger.info(
                            f"GPU preflight received completed Op[{target_name}] at recipe index "
                            f"{job['op_index']} after {elapsed:.2f}s "
                            f"(completed={len(results)}/{len(jobs)}, running={len(running)}, queued={len(queue)})."
                        )
                        if queue:
                            submit(queue.pop(0))
                    last_progress_log = now

                if self.probe_timeout_seconds is not None:
                    overdue = [
                        state for state in running.values() if now - state["submitted_at"] >= self.probe_timeout_seconds
                    ]
                    if overdue:
                        state = max(overdue, key=lambda item: now - item["submitted_at"])
                        job = state["job"]
                        target = job["target"]
                        target_name = getattr(target, "_name", type(target).__name__)
                        elapsed = now - state["submitted_at"]
                        raise TimeoutError(
                            f"GPU preflight Op[{target_name}] at recipe index {job['op_index']} "
                            f"exceeded its {self.probe_timeout_seconds:g}s timeout "
                            f"(elapsed={elapsed:.1f}s)."
                        )

                if now - last_progress_log >= _PROGRESS_INTERVAL_SECONDS:
                    active = [
                        "{}:{:.0f}s".format(
                            getattr(state["job"]["target"], "_name", type(state["job"]["target"]).__name__),
                            now - state["submitted_at"],
                        )
                        for state in running.values()
                    ]
                    logger.info(
                        f"GPU preflight progress: completed={len(results)}/{len(jobs)}, "
                        f"running={active}, queued={len(queue)}."
                    )
                    last_progress_log = now
        except Exception:
            for future in running:
                try:
                    ray.cancel(future, force=True)
                except Exception:
                    pass
            raise
        return results

    def _run_stage_with_ray(self, op, rows: List[Dict], measure_memory: bool) -> Dict[str, Any]:
        num_cpus = _positive_number(getattr(op, "num_cpus", None)) or 1
        if measure_memory:
            num_gpus = 1
        elif getattr(op, "accelerator", None) == "cuda":
            # Earlier configured CUDA stages must run in recipe order. Reserve
            # a full card when only a memory hint exists.
            num_gpus = _positive_number(getattr(op, "num_gpus", None)) or 1
        else:
            num_gpus = 0

        remote_stage = ray.remote(max_calls=1)(_run_probe_stage)
        options = {"num_cpus": num_cpus, "num_gpus": num_gpus}
        runtime_env = getattr(op, "runtime_env", None)
        if runtime_env is not None:
            options["runtime_env"] = runtime_env
        future = remote_stage.options(**options).remote(
            type(op),
            getattr(op, "_init_args", ()),
            getattr(op, "_init_kwargs", {}),
            rows,
            measure_memory,
            self.warmup_batches if measure_memory else 0,
            self.steady_batches if measure_memory else 1,
            getattr(op, "_name", None) or type(op).__name__,
        )
        return ray.get(future)

    @staticmethod
    def _read_hardware_with_ray() -> Dict[str, Any]:
        remote_reader = ray.remote(max_calls=1)(_read_gpu_hardware)
        return ray.get(remote_reader.options(num_cpus=0, num_gpus=1).remote())

    def _select_reusable_records(self, candidates, records) -> Dict[int, Dict[str, Any]]:
        if not records:
            return {}
        by_index = {record.get("op_index"): record for record in records}
        matches = {
            index: by_index[index]
            for index, op in candidates.items()
            if index in by_index
            and by_index[index].get("op_name") == getattr(op, "_name", type(op).__name__)
            and by_index[index].get("op_class") == f"{type(op).__module__}.{type(op).__qualname__}"
            and by_index[index].get("config_hash") == _config_hash(op)
        }
        if not matches:
            return {}

        try:
            hardware = self._hardware_reader()
        except Exception as error:
            logger.warning(f"Could not validate cached GPU preflight hardware; re-probing: {error}")
            return {}

        return {
            index: record
            for index, record in matches.items()
            if record.get("gpu_name") == hardware.get("gpu_name")
            and abs(float(record.get("gpu_total_mb", 0)) - float(hardware.get("total_mb", 0))) <= 16
        }

    @staticmethod
    def _apply_record(op, record: Mapping[str, Any]) -> None:
        op.memory = float(record["memory"])
        op.num_gpus = float(record["num_gpus"])
        op._gpu_memory_fraction = float(record.get("memory_fraction", op.num_gpus))
        op._planned_gpu_memory_mb = int(record.get("planned_memory_mb", 0))
        op._gpu_total_mb = float(record.get("gpu_total_mb", 0))
        profile = record.get("profile") or {}
        op._gpu_init_seconds = float(profile.get("initialization", 0) or 0)
        op._gpu_warmup_seconds = float(profile.get("warmup", 0) or 0)
        op._gpu_steady_batch_seconds = float(profile.get("steady_batch_seconds", 0) or 0)
        op._gpu_rows_per_second = float(profile.get("steady_rows_per_second", 0) or 0)
        op._gpu_output_ratio = float(profile.get("output_ratio", 1) or 0)
        init_kwargs = getattr(op, "_init_kwargs", None)
        if isinstance(init_kwargs, dict):
            init_kwargs.update(memory=op.memory, num_gpus=op.num_gpus)

        op_cfg = getattr(op, "_op_cfg", None)
        if isinstance(op_cfg, Mapping) and len(op_cfg) == 1:
            args = next(iter(op_cfg.values()))
            try:
                args["memory"] = op.memory
                args["num_gpus"] = op.num_gpus
            except TypeError:
                setattr(args, "memory", op.memory)
                setattr(args, "num_gpus", op.num_gpus)

    def _load_report(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.report_path):
            return []
        try:
            with open(self.report_path, "r", encoding="utf-8") as stream:
                report = json.load(stream)
            if (
                report.get("version") != _REPORT_VERSION
                or report.get("memory_headroom") != _MEMORY_HEADROOM
                or report.get("max_gpu_workers_per_device") != self.max_gpu_workers_per_device
                or report.get("warmup_batches") != self.warmup_batches
                or report.get("steady_batches") != self.steady_batches
            ):
                return []
            return list(report.get("operators", []))
        except Exception as error:
            logger.warning(f"Ignoring invalid GPU preflight report {self.report_path}: {error}")
            return []

    def _save_report(self, records: List[Dict[str, Any]]) -> None:
        os.makedirs(self.work_dir, exist_ok=True)
        report = {
            "version": _REPORT_VERSION,
            "observability_version": _OBSERVABILITY_VERSION,
            "memory_headroom": _MEMORY_HEADROOM,
            "max_gpu_workers_per_device": self.max_gpu_workers_per_device,
            "warmup_batches": self.warmup_batches,
            "steady_batches": self.steady_batches,
            "operators": records,
        }
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=self.work_dir, prefix=f".{_REPORT_NAME}.", delete=False
        ) as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True)
            temp_path = stream.name
        os.replace(temp_path, self.report_path)
        logger.info(f"Saved GPU preflight results to {self.report_path}")

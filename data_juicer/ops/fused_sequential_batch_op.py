"""
FusedSequentialBatchOp - run multiple batch-local ops in one stage.

This fused op reduces scheduler/stage overhead by executing a list of
batch-local sub-operators sequentially inside one dataset map stage. Each
sub-op receives the batch returned by the previous sub-op, so normal mapper
and filter chains remain possible, including filters that drop rows.
"""

import os
import time as _time
from collections import defaultdict
from typing import Any, Dict, List, Optional

from data_juicer.ops.base_op import (
    NON_STATS_FILTERS,
    OPERATORS,
    TAGGING_OPS,
    Filter,
    Mapper,
)
from data_juicer.utils.constant import Fields

OP_NAME = "fused_sequential_batch_op"


# Inner-op kwargs that belong to Ray scheduling, not to the model. These must
# be stripped before constructing sub-ops, otherwise the sub-op would try to
# claim its own Ray resources inside the fused stage.
_RAY_SCHED_KWARGS = (
    "num_gpus",
    "num_proc",
    "num_cpus",
    "memory",
    "runtime_env",
    "ray_execution_mode",
    "cpu_required",
    "gpu_required",
    "mem_required",
)


@OPERATORS.register_module(OP_NAME)
class FusedSequentialBatchOp(Mapper):
    """Run multiple batch-local mapper/filter ops sequentially in one stage.

    Supports two initialization modes:
      1. op_specs mode: list of {"class_name": str, "kwargs": dict}.
      2. fused_ops mode: list of pre-built op instances.

    This class intentionally does not fan out work across threads. Its primary
    purpose is reducing stage overhead while preserving normal sequential
    semantics.
    """

    _batched_op = True

    def __init__(
        self,
        op_specs: Optional[List[Dict[str, Any]]] = None,
        fused_ops: Optional[List[Any]] = None,
        group_name: str = "",
        cleanup_columns: Optional[List[str]] = None,
        *args,
        **kwargs,
    ):
        """
        Args:
            op_specs: sub-op specs. Ray scheduling kwargs are stripped before
                constructing sub-ops.
            fused_ops: already-instantiated batch-local ops.
            group_name: human-readable label used in logs.
            cleanup_columns: top-level columns to remove after all sub-ops
                finish.
        """
        super().__init__(*args, **kwargs)

        if fused_ops and op_specs:
            raise ValueError("FusedSequentialBatchOp: provide either fused_ops or op_specs, not both.")

        self._fused_ops_input = list(fused_ops) if fused_ops else None
        self.op_specs = list(op_specs or [])
        self.group_name = group_name or "fused"
        self.cleanup_columns = list(cleanup_columns) if cleanup_columns else []
        self._contains_tagging_ops = self._detect_tagging_ops()

        # Lazy-init in worker process; avoids loading models on the driver.
        self._ops: Optional[List[Any]] = None

    def _detect_tagging_ops(self) -> bool:
        if self._fused_ops_input:
            return any(
                op._name in TAGGING_OPS.modules or getattr(op, "_contains_tagging_ops", False)
                for op in self._fused_ops_input
            )
        return any((spec.get("class_name") or spec.get("name")) in TAGGING_OPS.modules for spec in self.op_specs)

    def _ensure_ops(self):
        if self._ops is not None:
            return

        from loguru import logger

        if self._fused_ops_input:
            self._ops = list(self._fused_ops_input)
            logger.info(
                f"[FusedSequentialBatchOp:{self.group_name}] using "
                f"{len(self._ops)} pre-built ops: "
                f"{[op._name for op in self._ops]}"
            )
            self._init_profiling_state()
            return

        ops: List[Any] = []
        for spec in self.op_specs:
            cls_name = spec.get("class_name") or spec.get("name")
            if not cls_name:
                raise ValueError(f"FusedSequentialBatchOp[{self.group_name}]: spec missing 'class_name': {spec}")

            sub_kwargs = dict(spec.get("kwargs") or {})
            for key in _RAY_SCHED_KWARGS:
                sub_kwargs.pop(key, None)

            op_cls = OPERATORS.modules.get(cls_name)
            if op_cls is None:
                raise ValueError(
                    f"FusedSequentialBatchOp[{self.group_name}]: op '{cls_name}' "
                    f"not found in OPERATORS registry. Available: "
                    f"{sorted(OPERATORS.modules)[:20]}..."
                )
            ops.append(op_cls(**sub_kwargs))

        self._ops = ops
        self._preload_models()
        self._init_profiling_state()

    def _preload_models(self):
        from loguru import logger

        loadable = [
            (idx, op) for idx, op in enumerate(self._ops) if hasattr(op, "_ensure_model") and callable(op._ensure_model)
        ]
        if not loadable:
            return

        for idx, op in loadable:
            logger.info(
                f"[FusedSequentialBatchOp:{self.group_name}] "
                f"pre-loading model {idx + 1}/{len(self._ops)}: {op._name}"
            )
            op._ensure_model()

    def _init_profiling_state(self):
        from loguru import logger

        self._prof_batch_count = 0
        self._prof_log_interval = 10
        self._prof_op_wall_ms: Dict[str, List[float]] = defaultdict(list)
        self._prof_total_rows = 0

        logger.info(
            f"[FusedSequentialBatchOp:{self.group_name}] initialised "
            f"{len(self._ops)} sub-ops (pid={os.getpid()}) "
            f"[per-op profiling enabled, interval={self._prof_log_interval}]"
        )

    def process_batched(self, samples, rank=None):
        """Run sub-ops sequentially, passing each returned batch onward."""
        self._ensure_ops()

        if not self._ops:
            return samples

        num_samples = self._batch_size(samples)
        if num_samples == 0:
            return samples

        batch_t0 = _time.perf_counter()
        op_timings: Dict[str, float] = {}

        for op in self._ops:
            op_t0 = _time.perf_counter()
            if isinstance(op, Filter):
                samples = self._run_filter_op(op, samples, rank=rank)
            else:
                samples = self._ensure_meta_if_needed(samples, op)
                samples = self._run_sub_op(op, samples, rank=rank)
            op_timings[op._name] = (_time.perf_counter() - op_t0) * 1000.0
            if self._batch_size(samples) == 0:
                break

        batch_wall_ms = (_time.perf_counter() - batch_t0) * 1000.0
        final_num_samples = self._batch_size(samples)

        self._prof_batch_count += 1
        self._prof_total_rows += final_num_samples
        for op_name, ms in op_timings.items():
            self._prof_op_wall_ms[op_name].append(ms)

        if self._prof_batch_count % self._prof_log_interval == 0:
            self._log_profiling_stats(batch_wall_ms, final_num_samples)

        for col in self.cleanup_columns:
            if col in samples:
                del samples[col]

        return samples

    def _run_sub_op(self, op, samples, rank=None):
        process_args = {"rank": rank} if op.use_cuda() else {}
        result = op.process(samples, **process_args)
        if result is None:
            raise ValueError(f"Sub-op [{op._name}] returned None inside FusedSequentialBatchOp.")
        if not isinstance(result, dict):
            raise ValueError(
                f"Sub-op [{op._name}] returned unsupported batch type "
                f"[{type(result).__name__}] inside FusedSequentialBatchOp."
            )
        return result

    def _run_filter_op(self, op, samples, rank=None):
        samples = self._ensure_meta_if_needed(samples, op)
        samples = self._ensure_stats_if_needed(samples, op)
        compute_args = {"rank": rank} if op.use_cuda() else {}
        result = op.compute_stats(samples, **compute_args)
        if result is None:
            raise ValueError(f"Filter sub-op [{op._name}] returned None from compute_stats.")
        if not isinstance(result, dict):
            raise ValueError(
                f"Filter sub-op [{op._name}] returned unsupported stats batch type "
                f"[{type(result).__name__}] inside FusedSequentialBatchOp."
            )

        keep_mask = list(op.process(result))
        return self._filter_batch(result, keep_mask, op)

    def _filter_batch(self, samples, keep_mask, op):
        num_samples = self._batch_size(samples)
        if len(keep_mask) != num_samples:
            raise ValueError(
                f"Filter sub-op [{op._name}] returned keep mask length "
                f"[{len(keep_mask)}], expected [{num_samples}] inside "
                f"FusedSequentialBatchOp."
            )
        kept_indices = [idx for idx, keep in enumerate(keep_mask) if keep]
        return {key: [values[idx] for idx in kept_indices] for key, values in samples.items()}

    def _batch_size(self, samples):
        if not samples:
            return 0
        first_key = next(iter(samples.keys()))
        return len(samples[first_key])

    def _ensure_meta_if_needed(self, samples, op):
        if not self._needs_meta(samples, op):
            return samples
        num_samples = self._batch_size(samples)
        if Fields.meta not in samples or samples[Fields.meta] is None or len(samples[Fields.meta]) == 0:
            samples[Fields.meta] = [{} for _ in range(num_samples)]
        elif len(samples[Fields.meta]) != num_samples:
            raise ValueError(
                f"Fields.meta length [{len(samples[Fields.meta])}] does not "
                f"match batch size [{num_samples}] before sub-op [{op._name}] "
                f"inside FusedSequentialBatchOp."
            )
        else:
            for idx in range(num_samples):
                if samples[Fields.meta][idx] is None:
                    samples[Fields.meta][idx] = {}
        return samples

    def _ensure_stats_if_needed(self, samples, op):
        if not self._needs_stats(samples, op):
            return samples
        num_samples = self._batch_size(samples)
        if Fields.stats not in samples or samples[Fields.stats] is None or len(samples[Fields.stats]) == 0:
            samples[Fields.stats] = [{} for _ in range(num_samples)]
        elif len(samples[Fields.stats]) != num_samples:
            raise ValueError(
                f"Fields.stats length [{len(samples[Fields.stats])}] does not "
                f"match batch size [{num_samples}] before sub-op [{op._name}] "
                f"inside FusedSequentialBatchOp."
            )
        else:
            for idx in range(num_samples):
                if samples[Fields.stats][idx] is None:
                    samples[Fields.stats][idx] = {}
        return samples

    def _needs_meta(self, samples, op):
        if Fields.meta in samples:
            return True
        if getattr(op, "_requires_meta", False):
            return True
        if op._name in TAGGING_OPS.modules:
            return True
        output_columns = getattr(op, "_output_columns", []) or []
        return any(str(col).startswith(Fields.meta) for col in output_columns)

    def _needs_stats(self, samples, op):
        if Fields.stats in samples:
            return True
        if isinstance(op, Filter) and op._name not in NON_STATS_FILTERS.modules:
            return True
        output_columns = getattr(op, "_output_columns", []) or []
        return any(str(col).startswith(Fields.stats) for col in output_columns)

    def _log_profiling_stats(self, last_batch_ms: float, last_batch_size: int):
        from loguru import logger

        n = self._prof_log_interval
        header = (
            f"[FusedSequentialBatchOp:{self.group_name}] "
            f"PROFILING batch#{self._prof_batch_count} "
            f"(last {n} batches, {self._prof_total_rows} total output rows, pid={os.getpid()})"
        )
        lines = [header]
        lines.append(f"  {'Op':<32} {'Mean ms':>9} {'Max ms':>9} {'Min ms':>9} {'ms/row':>9}")
        lines.append(f"  {'-' * 32} {'-' * 9} {'-' * 9} {'-' * 9} {'-' * 9}")

        op_stats = []
        for op_name, timings in self._prof_op_wall_ms.items():
            recent = timings[-n:]
            mean_ms = sum(recent) / len(recent)
            op_stats.append((op_name, mean_ms, max(recent), min(recent)))

        op_stats.sort(key=lambda x: x[1], reverse=True)
        for op_name, mean_ms, max_ms, min_ms in op_stats:
            ms_per_row = mean_ms / last_batch_size if last_batch_size else 0
            lines.append(f"  {op_name:<32} {mean_ms:>9.1f} {max_ms:>9.1f} {min_ms:>9.1f} {ms_per_row:>9.2f}")

        lines.append(f"  {'TOTAL (sequential wall)':<32} {last_batch_ms:>9.1f}")
        logger.info("\n".join(lines))

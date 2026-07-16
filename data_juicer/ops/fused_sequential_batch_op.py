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

from data_juicer.ops.base_op import OPERATORS, TAGGING_OPS, Mapper
from data_juicer.ops.fused_batch_executor import (
    execute_sequential_batch,
    get_batch_size,
)

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

        num_samples = get_batch_size(samples)
        if num_samples == 0:
            return samples

        batch_t0 = _time.perf_counter()
        op_timings: Dict[str, float] = {}

        def record_op_timing(op, wall_ms):
            op_timings[op._name] = wall_ms

        samples = execute_sequential_batch(
            samples,
            self._ops,
            rank=rank,
            owner_name=f"FusedSequentialBatchOp:{self.group_name}",
            cleanup_columns=self.cleanup_columns,
            on_op_complete=record_op_timing,
        )

        batch_wall_ms = (_time.perf_counter() - batch_t0) * 1000.0
        final_num_samples = get_batch_size(samples)

        self._prof_batch_count += 1
        self._prof_total_rows += final_num_samples
        for op_name, ms in op_timings.items():
            self._prof_op_wall_ms[op_name].append(ms)

        if self._prof_batch_count % self._prof_log_interval == 0:
            self._log_profiling_stats(batch_wall_ms, final_num_samples)

        return samples

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

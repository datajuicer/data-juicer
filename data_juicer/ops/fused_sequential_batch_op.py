"""
FusedSequentialBatchOp - run multiple batch-local ops in one stage.

This fused op reduces scheduler/stage overhead by executing a list of
batch-local sub-operators sequentially inside one dataset map stage. Each
sub-op receives the batch returned by the previous sub-op, so normal mapper
chains and dependency chains remain possible.

The legacy FusedParallelMapper name is kept as a compatibility alias, but its
runtime semantics are now sequential.
"""

import os
import time as _time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from data_juicer.ops.base_op import OPERATORS, TAGGING_OPS, Mapper
from data_juicer.utils.constant import Fields

OP_NAME = "fused_sequential_batch_op"
LEGACY_OP_NAME = "fused_parallel_mapper"


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
    """Run multiple batch-local mappers sequentially in one map stage.

    Supports two initialization modes:
      1. op_specs mode: list of {"class_name": str, "kwargs": dict}.
      2. fused_ops mode: list of pre-built Mapper instances.

    This class intentionally does not fan out work across threads. Its primary
    purpose is reducing stage overhead while preserving normal sequential
    semantics.
    """

    _accelerator = "cuda"
    _batched_op = True

    def __init__(
        self,
        op_specs: Optional[List[Dict[str, Any]]] = None,
        fused_ops: Optional[List[Mapper]] = None,
        max_workers: Optional[int] = None,
        group_name: str = "",
        cleanup_columns: Optional[List[str]] = None,
        use_per_op_streams: bool = False,
        parallel_model_loading: bool = False,
        *args,
        **kwargs,
    ):
        """
        Args:
            op_specs: sub-op specs. Ray scheduling kwargs are stripped before
                constructing sub-ops.
            fused_ops: already-instantiated Mapper objects.
            max_workers: retained for backward compatibility; only used as an
                upper bound for optional parallel model loading.
            group_name: human-readable label used in logs.
            cleanup_columns: top-level columns to remove after all sub-ops
                finish.
            use_per_op_streams: retained for backward compatibility; ignored by
                sequential execution.
            parallel_model_loading: if True, load sub-op models in parallel
                when they expose _ensure_model().
        """
        super().__init__(*args, **kwargs)

        if fused_ops and op_specs:
            raise ValueError("FusedSequentialBatchOp: provide either fused_ops or op_specs, not both.")

        self._fused_ops_input = list(fused_ops) if fused_ops else None
        self.op_specs = list(op_specs or [])
        self.max_workers = (
            int(max_workers)
            if max_workers
            else max(1, len(self._fused_ops_input) if self._fused_ops_input else len(self.op_specs))
        )
        self.group_name = group_name or "fused"
        self.cleanup_columns = list(cleanup_columns) if cleanup_columns else []
        self.use_per_op_streams = use_per_op_streams
        self.parallel_model_loading = parallel_model_loading

        # Lazy-init in worker process; avoids loading models on the driver.
        self._ops: Optional[List[Mapper]] = None

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

        ops: List[Mapper] = []
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

        if self.parallel_model_loading and len(loadable) > 1:
            start = _time.monotonic()
            max_workers = min(len(loadable), max(1, self.max_workers))
            logger.info(f"[FusedSequentialBatchOp:{self.group_name}] parallel-loading {len(loadable)} models...")

            def _load_one(idx_op):
                idx, op = idx_op
                logger.info(
                    f"[FusedSequentialBatchOp:{self.group_name}] "
                    f"  loading model {idx + 1}/{len(self._ops)}: {op._name}"
                )
                op._ensure_model()

            try:
                with ThreadPoolExecutor(
                    max_workers=max_workers, thread_name_prefix=f"model-load-{self.group_name}"
                ) as load_pool:
                    futures = [load_pool.submit(_load_one, item) for item in loadable]
                    for future in futures:
                        future.result()
                elapsed = _time.monotonic() - start
                logger.info(f"[FusedSequentialBatchOp:{self.group_name}] parallel model loading done in {elapsed:.1f}s")
                return
            except Exception as exc:
                logger.warning(
                    f"[FusedSequentialBatchOp:{self.group_name}] "
                    f"parallel loading hit error ({type(exc).__name__}: {exc}); "
                    f"falling back to sequential"
                )

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
            samples = self._ensure_meta_if_needed(samples, op)
            op_t0 = _time.perf_counter()
            samples = self._run_sub_op(op, samples, rank=rank)
            op_timings[op._name] = (_time.perf_counter() - op_t0) * 1000.0

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
        process_args = {"rank": rank} if op.accelerator == "cuda" else {}
        result = op.process(samples, **process_args)
        if result is None:
            raise ValueError(f"Sub-op [{op._name}] returned None inside FusedSequentialBatchOp.")
        if not isinstance(result, dict):
            raise ValueError(
                f"Sub-op [{op._name}] returned unsupported batch type "
                f"[{type(result).__name__}] inside FusedSequentialBatchOp."
            )
        return result

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

    def _needs_meta(self, samples, op):
        if Fields.meta in samples:
            return True
        if getattr(op, "_requires_meta", False):
            return True
        if op._name in TAGGING_OPS.modules:
            return True
        output_columns = getattr(op, "_output_columns", []) or []
        return any(str(col).startswith(Fields.meta) for col in output_columns)

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


@OPERATORS.register_module(LEGACY_OP_NAME)
class FusedParallelMapper(FusedSequentialBatchOp):
    """Backward-compatible alias for FusedSequentialBatchOp."""

    pass

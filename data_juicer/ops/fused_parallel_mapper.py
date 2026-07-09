"""
FusedParallelMapper — DAG fan-out/fan-in within a single Ray Data stage
========================================================================

Ray Data's ``map_batches`` only supports linear chains. This mapper emulates
a fan-out / fan-in DAG inside one stage by hosting N independent sub-operators
on the same Ray actor and dispatching each batch to all of them concurrently
via a ThreadPoolExecutor.

Why this works:
  * Independent operators (e.g. Rotation / QualiCLIP / Clarity) all read the
    same input columns and each WRITES TO DISTINCT KEYS inside ``__dj__meta__``.
    Concurrent dict-key assignment is atomic under CPython's GIL, so parallel
    writers do not race on per-sample meta dicts.
  * PyTorch releases the GIL during CUDA kernel launches and cuDNN ops,
    so threads provide real GPU parallelism — multiple models on the same
    GPU interleave their kernels on a single CUDA stream.
  * One actor → one Ray scheduling slot → one set of model loads. This
    eliminates the linear-chain bottleneck (each row had to walk through
    every stage in series) and removes inter-stage data transfer overhead.

Constraints:
  * All sub-ops must produce DISJOINT meta keys (or disjoint top-level
    columns). Independent operators naturally satisfy this since each
    produces its own signal.
  * Aggregate VRAM of all sub-op models must fit one GPU's allocation.

Usage (explicit op_specs):
    fused = FusedParallelMapper(
        op_specs=[
            {"class_name": "rotation_mapper",  "kwargs": {...}},
            {"class_name": "clarity_mapper",   "kwargs": {...}},
        ],
        batch_size=128,
        num_gpus=1.0,
        num_proc=8,
    )

Usage (auto-fusion with pre-built instances):
    fused = FusedParallelMapper(
        fused_ops=[op1, op2, op3],  # already-instantiated Mapper objects
        batch_size=128,
        num_gpus=1.0,
    )
"""

import os
import time as _time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import torch

from data_juicer.ops.base_op import OPERATORS, Mapper
from data_juicer.utils.constant import Fields

OP_NAME = "fused_parallel_mapper"


# Inner-op kwargs that belong to Ray scheduling, not to the model. These must
# be stripped before constructing sub-ops, otherwise the sub-op would try to
# claim its own Ray resources (which it cannot, since it lives inside an outer
# actor that already owns the slot).
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
class FusedParallelMapper(Mapper):
    """Run multiple independent mappers in parallel on the same batch.

    Each sub-operator gets the SAME shared ``samples`` dict and mutates it
    in place. Sub-operators MUST write disjoint keys (typically distinct
    keys inside ``__dj__meta__[i]``).

    Supports two initialization modes:
      1. **op_specs mode**: list of ``{"class_name": str, "kwargs": dict}``
         dicts. Ops are lazily instantiated on the Ray worker.
      2. **fused_ops mode**: list of pre-built Mapper instances (from
         auto-fusion in ``fuse_operators()``). Ops are used directly.
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
            op_specs: list of sub-op specs, each ``{"class_name": str, "kwargs": dict}``.
                ``class_name`` must be present in the OPERATORS registry on the
                Ray worker. ``kwargs`` are forwarded to the sub-op constructor;
                Ray scheduling kwargs (num_gpus / num_proc / ...) are stripped.
                Mutually exclusive with ``fused_ops``.
            fused_ops: list of already-instantiated Mapper objects (from
                auto-fusion). When provided, ``op_specs`` is ignored and these
                ops are used directly. Mutually exclusive with ``op_specs``.
            max_workers: ThreadPool size. Defaults to ``len(ops)``.
            group_name: optional human-readable label used in logs and thread names.
            cleanup_columns: list of top-level column names to delete from the
                output batch after all sub-ops finish. Use this to release large
                intermediate data (e.g. ``_bucket_img``) without a separate stage.
            use_per_op_streams: if True, each sub-op gets its own CUDA stream
                so that GPU kernels from different ops can interleave on idle SMs.
            parallel_model_loading: if True, load all sub-op models in parallel
                using a ThreadPoolExecutor.
        """
        super().__init__(*args, **kwargs)

        if fused_ops and op_specs:
            raise ValueError("FusedParallelMapper: provide either fused_ops or op_specs, not both.")

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

        # Lazy-init in worker process — avoids loading models on the driver.
        self._ops: Optional[List[Mapper]] = None
        self._pool: Optional[ThreadPoolExecutor] = None
        self._streams: Optional[List[torch.cuda.Stream]] = None

    # ------------------------------------------------------------------
    # Lazy initialization (runs once per Ray worker actor)
    # ------------------------------------------------------------------
    def _ensure_ops(self):
        if self._ops is not None:
            return

        from loguru import logger

        # Mode 2: pre-built instances from auto-fusion
        if self._fused_ops_input:
            self._ops = list(self._fused_ops_input)
            logger.info(
                f"[FusedParallelMapper:{self.group_name}] using "
                f"{len(self._ops)} pre-built ops: "
                f"{[op._name for op in self._ops]}"
            )
            self._init_pool_and_streams()
            return

        # Mode 1: lazy init from op_specs
        ops: List[Mapper] = []
        for spec in self.op_specs:
            cls_name = spec.get("class_name") or spec.get("name")
            if not cls_name:
                raise ValueError(f"FusedParallelMapper[{self.group_name}]: " f"spec missing 'class_name': {spec}")

            sub_kwargs = dict(spec.get("kwargs") or {})
            for k in _RAY_SCHED_KWARGS:
                sub_kwargs.pop(k, None)

            op_cls = OPERATORS.modules.get(cls_name)
            if op_cls is None:
                raise ValueError(
                    f"FusedParallelMapper[{self.group_name}]: op '{cls_name}' "
                    f"not found in OPERATORS registry. Available: "
                    f"{sorted(OPERATORS.modules)[:20]}..."
                )

            ops.append(op_cls(**sub_kwargs))

        self._ops = ops

        # Pre-warm: initialize all sub-op models before any concurrent
        # inference execution.
        loadable = [
            (i, op) for i, op in enumerate(self._ops) if hasattr(op, "_ensure_model") and callable(op._ensure_model)
        ]

        if self.parallel_model_loading and len(loadable) > 1:
            _t0 = _time.monotonic()
            logger.info(f"[FusedParallelMapper:{self.group_name}] " f"parallel-loading {len(loadable)} models...")

            def _load_one(idx_op):
                i, op = idx_op
                logger.info(
                    f"[FusedParallelMapper:{self.group_name}] " f"  loading model {i+1}/{len(self._ops)}: {op._name}"
                )
                op._ensure_model()

            try:
                with ThreadPoolExecutor(
                    max_workers=len(loadable),
                    thread_name_prefix=f"model-load-{self.group_name}",
                ) as load_pool:
                    futs = [load_pool.submit(_load_one, item) for item in loadable]
                    for fut in futs:
                        fut.result()  # re-raise on error

                _elapsed = _time.monotonic() - _t0
                logger.info(
                    f"[FusedParallelMapper:{self.group_name}] " f"parallel model loading done in {_elapsed:.1f}s"
                )
            except Exception as exc:
                logger.warning(
                    f"[FusedParallelMapper:{self.group_name}] "
                    f"parallel loading hit error "
                    f"({type(exc).__name__}: {exc}); falling back to sequential"
                )
                for i, op in loadable:
                    logger.info(
                        f"[FusedParallelMapper:{self.group_name}] "
                        f"  seq-loading model {i+1}/{len(self._ops)}: {op._name}"
                    )
                    op._ensure_model()
                _elapsed = _time.monotonic() - _t0
                logger.info(f"[FusedParallelMapper:{self.group_name}] " f"sequential fallback done in {_elapsed:.1f}s")
        else:
            for i, op in enumerate(self._ops):
                if hasattr(op, "_ensure_model") and callable(op._ensure_model):
                    logger.info(
                        f"[FusedParallelMapper:{self.group_name}] "
                        f"pre-loading model {i+1}/{len(self._ops)}: {op._name}"
                    )
                    op._ensure_model()

        self._init_pool_and_streams()

    def _init_pool_and_streams(self):
        """Initialize thread pool and optional per-op CUDA streams."""
        from loguru import logger

        self._pool = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix=f"fused-{self.group_name}",
        )

        if self.use_per_op_streams and torch.cuda.is_available():
            self._streams = [torch.cuda.Stream() for _ in range(len(self._ops))]
            logger.info(
                f"[FusedParallelMapper:{self.group_name}] created " f"{len(self._streams)} dedicated CUDA streams"
            )
        else:
            self._streams = None

        # Per-op profiling state
        self._prof_batch_count = 0
        self._prof_log_interval = 10
        self._prof_op_wall_ms: Dict[str, List[float]] = defaultdict(list)
        self._prof_total_rows = 0

        logger.info(
            f"[FusedParallelMapper:{self.group_name}] initialised "
            f"{len(self._ops)} sub-ops with {self.max_workers} threads "
            f"(per_op_streams={self._streams is not None}, pid={os.getpid()}) "
            f"[per-op profiling enabled, interval={self._prof_log_interval}]"
        )

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------
    def process_batched(self, samples, rank=None):
        """Fan-out the batch to all sub-ops, run concurrently, fan-in by reference."""
        self._ensure_ops()

        if not self._ops:
            return samples

        # Determine batch size from any column.
        first_key = next(iter(samples.keys()))
        num_samples = len(samples[first_key])
        if num_samples == 0:
            return samples

        # Pre-initialize Fields.meta so concurrent sub-ops do not race on
        # whole-list assignment.
        if Fields.meta not in samples or samples[Fields.meta] is None or len(samples[Fields.meta]) == 0:
            samples[Fields.meta] = [{} for _ in range(num_samples)]
        else:
            for i in range(num_samples):
                if samples[Fields.meta][i] is None:
                    samples[Fields.meta][i] = {}

        # Single-op fast path — skip thread overhead.
        if len(self._ops) == 1:
            self._ops[0].process(samples, rank=rank)
            return samples

        # Per-op profiling: wall-clock timing per thread
        batch_t0 = _time.perf_counter()

        # Fan-out. Each sub-op mutates ``samples`` in place.
        # Thread functions return (op_name, elapsed_ms) to avoid
        # concurrent writes to a shared dict.
        if self._streams is not None:

            def _run_on_stream_timed(op, stream):
                t0 = _time.perf_counter()
                with torch.cuda.stream(stream):
                    op.process(samples, rank=rank)
                stream.synchronize()
                elapsed_ms = (_time.perf_counter() - t0) * 1000.0
                return op._name, elapsed_ms

            futures = [
                self._pool.submit(_run_on_stream_timed, op, stream) for op, stream in zip(self._ops, self._streams)
            ]
        else:

            def _run_timed(op):
                t0 = _time.perf_counter()
                op.process(samples, rank=rank)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed_ms = (_time.perf_counter() - t0) * 1000.0
                return op._name, elapsed_ms

            futures = [self._pool.submit(_run_timed, op) for op in self._ops]

        # Fan-in: collect timing results from each thread.
        op_timings: Dict[str, float] = {}
        for f in futures:
            op_name, elapsed_ms = f.result()
            op_timings[op_name] = elapsed_ms

        if self._streams is not None:
            torch.cuda.synchronize()

        batch_wall_ms = (_time.perf_counter() - batch_t0) * 1000.0

        # Accumulate and log profiling stats
        self._prof_batch_count += 1
        self._prof_total_rows += num_samples
        for op_name, ms in op_timings.items():
            self._prof_op_wall_ms[op_name].append(ms)

        if self._prof_batch_count % self._prof_log_interval == 0:
            self._log_profiling_stats(batch_wall_ms, num_samples)

        # Cleanup: remove intermediate columns no longer needed downstream.
        for col in self.cleanup_columns:
            if col in samples:
                del samples[col]

        return samples

    def _log_profiling_stats(self, last_batch_ms: float, last_batch_size: int):
        """Log per-op timing summary for the last N batches."""
        from loguru import logger

        n = self._prof_log_interval
        header = (
            f"[FusedParallelMapper:{self.group_name}] "
            f"PROFILING batch#{self._prof_batch_count} "
            f"(last {n} batches, {self._prof_total_rows} total rows, pid={os.getpid()})"
        )
        lines = [header]
        lines.append(f"  {'Op':<32} {'Mean ms':>9} {'Max ms':>9} {'Min ms':>9} {'ms/row':>9}")
        lines.append(f"  {'-'*32} {'-'*9} {'-'*9} {'-'*9} {'-'*9}")

        op_stats = []
        for op_name, timings in self._prof_op_wall_ms.items():
            recent = timings[-n:]
            mean_ms = sum(recent) / len(recent)
            max_ms = max(recent)
            min_ms = min(recent)
            op_stats.append((op_name, mean_ms, max_ms, min_ms))

        op_stats.sort(key=lambda x: x[1], reverse=True)

        for op_name, mean_ms, max_ms, min_ms in op_stats:
            ms_per_row = mean_ms / last_batch_size if last_batch_size else 0
            lines.append(f"  {op_name:<32} {mean_ms:>9.1f} {max_ms:>9.1f} {min_ms:>9.1f} {ms_per_row:>9.2f}")

        lines.append(f"  {'TOTAL (wall, critical-path)':<32} {last_batch_ms:>9.1f}")
        logger.info("\n".join(lines))

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def __del__(self):
        try:
            if self._pool is not None:
                self._pool.shutdown(wait=False)
        except Exception:
            pass

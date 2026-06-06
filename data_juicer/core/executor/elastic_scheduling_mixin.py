"""
Elastic Scheduling Mixin for GPU-Adaptive Batch Processing

Provides reusable ElasticJuicer adaptive scheduling logic that can be mixed
into any Ray-based executor. Extracted from ElasticRayExecutor to allow
both ElasticRayExecutor and PartitionedRayExecutor to share the same
GPU-adaptive scheduling infrastructure.

The bi-level scheduling architecture:
- Tower (macro-scheduler): Global resource allocation and rebalancing
- Captains (micro-schedulers): Per-operator batch size control with PID
- MicroScheduler: Per-actor adaptive batch sizing within map_batches

Usage:
    class MyExecutor(ExecutorBase, ElasticSchedulingMixin, ...):
        def __init__(self, cfg):
            ...
            ElasticSchedulingMixin.__init__(self)
            ...
"""

import os
import time
from dataclasses import asdict
from functools import partial
from typing import Any, Dict, List, Optional

from loguru import logger

from data_juicer.ops import Deduplicator, Filter, Mapper, Pipeline
from data_juicer.ops.base_op import DEFAULT_BATCH_SIZE, TAGGING_OPS
from data_juicer.utils.constant import Fields
from data_juicer.utils.lazy_loader import LazyLoader

ray = LazyLoader("ray")
pyarrow = LazyLoader("pyarrow")


def filter_batch(batch, filter_func):
    """Filter batch using filter function."""
    import pyarrow

    mask = pyarrow.array(filter_func(batch.to_pydict()))
    return batch.filter(mask)


class ElasticSchedulingMixin:
    """
    Mixin providing ElasticJuicer adaptive GPU scheduling capabilities.

    Any executor that mixes this in gains:
    - Adaptive batch sizing for GPU operators via AdaptiveOperator/MicroScheduler
    - PID-controlled memory management to prevent OOM
    - Tower-based global resource allocation (optional rebalancing)
    - Per-stage metrics collection

    Requirements on the host class:
    - Must have ``self.cfg`` (Namespace with config, including elastic_juicer)
    - Must be running in a Ray environment
    """

    def _init_elastic_scheduling(self):
        """Initialize elastic scheduling state. Call from host __init__."""
        self._elastic = None  # ElasticJuicer facade instance
        self._scheduler_config = None
        self._elastic_started = False
        self._profiling_store = None  # ProfilingStore for prior loading + flush

    # ------------------------------------------------------------------
    # Config parsing
    # ------------------------------------------------------------------

    def _parse_elastic_juicer_config(self, has_gpu_ops: bool = True):
        """Parse elastic_juicer config section and create SchedulerConfig.

        Auto-selects appropriate preset when no explicit config is provided:
        - GPU pipelines default to 'gpu' preset
        - CPU-only pipelines default to 'cpu_optimized' preset

        Args:
            has_gpu_ops: Whether the pipeline contains GPU operators.
                Used to select the appropriate default preset.
        """
        from data_juicer.core.elasticjuicer.scheduler.scheduler_config import (
            SchedulerConfig,
        )

        elastic_cfg = getattr(self.cfg, "elastic_juicer", None)
        if elastic_cfg is None:
            elastic_cfg = {}
        elif hasattr(elastic_cfg, "__dict__"):
            # Convert Namespace to dict
            elastic_cfg = vars(elastic_cfg)

        default_preset = "gpu" if has_gpu_ops else "cpu_optimized"
        scheduler_preset = elastic_cfg.get("scheduler_preset", default_preset)
        config_path = elastic_cfg.get("config_path", None)
        rebalance_interval = elastic_cfg.get("rebalance_interval", 5.0)

        # Load or create SchedulerConfig
        if config_path and os.path.exists(config_path):
            self._scheduler_config = SchedulerConfig.from_yaml(config_path)
            logger.info(f"Loaded ElasticJuicer config from {config_path}")
        else:
            presets = {
                "conservative": SchedulerConfig.conservative,
                "gpu": SchedulerConfig.gpu,
                "gpu_optimized": SchedulerConfig.gpu_optimized,
                "cpu_optimized": SchedulerConfig.cpu_optimized,
                "aggressive": SchedulerConfig.aggressive,
                "memory_constrained": SchedulerConfig.memory_constrained,
            }
            factory = presets.get(scheduler_preset, SchedulerConfig.gpu)
            self._scheduler_config = factory()
            logger.info(
                f"Using ElasticJuicer '{scheduler_preset}' preset config"
            )

        # Override rebalance interval if specified
        self._scheduler_config.rebalance_interval_sec = rebalance_interval

        return elastic_cfg

    # ------------------------------------------------------------------
    # Component initialization and lifecycle
    # ------------------------------------------------------------------

    def _init_elasticjuicer_components(self, ops: List):
        """Initialize ElasticJuicer facade with all scheduling components.

        Creates stage configs from operators and registers them with the
        ElasticJuicer facade, which manages Tower, Captains, and Ray actors.

        When a ProfilingStore directory from a prior run exists the method
        loads per-op priors (peak memory, safe batch size) and uses them
        as ``initial_batch_size`` instead of the blind SchedulerConfig default.
        """
        from data_juicer.core.elasticjuicer import ElasticJuicer
        from data_juicer.core.elasticjuicer.profiler.profiling_store import (
            ProfilingStore,
        )

        # Initialize ProfilingStore (loads priors from previous runs if any)
        elastic_cfg = getattr(self.cfg, "elastic_juicer", None) or {}
        if hasattr(elastic_cfg, "__dict__"):
            elastic_cfg = vars(elastic_cfg)
        store_dir = elastic_cfg.get(
            "profiling_store_dir", "./elastic_juicer_profiles"
        )
        try:
            self._profiling_store = ProfilingStore(storage_dir=store_dir)
        except Exception as e:
            logger.warning("Failed to initialize ProfilingStore: %s", e)
            self._profiling_store = None

        # Build stage configs from operators (with prior loading)
        stage_configs = []
        for i, op in enumerate(ops):
            op_name = f"stage_{i}_{op._name}"

            # Try to load prior from a previous run
            prior_bs = None
            if self._profiling_store is not None:
                prior_stats = self._profiling_store.get_execution_stats(
                    op._name
                )
                if prior_stats and prior_stats.peak_memory_mb > 0:
                    prior_bs = self._profiling_store.get_safe_batch_size(
                        op._name,
                        available_memory_mb=(
                            self._scheduler_config.safety_buffer_mb * 4
                        ),
                    )
                    logger.info(
                        "Loaded prior for %s: peak_memory=%.1f MB, "
                        "recommended_bs=%d",
                        op._name,
                        prior_stats.peak_memory_mb,
                        prior_bs,
                    )

            # Priority: op.batch_size > prior_bs > scheduler default
            initial_bs = (
                getattr(op, "batch_size", None)
                or prior_bs
                or self._scheduler_config.initial_batch_size
            )

            stage_configs.append(
                {
                    "name": op_name,
                    "batch_size": initial_bs,
                    "num_gpus": op.num_gpus or 0,
                    "num_actors": (
                        op.runtime_np()
                        if hasattr(op, "runtime_np")
                        else (op.num_proc or 1)
                    ),
                }
            )

        # Create ElasticJuicer facade
        self._elastic = ElasticJuicer(
            config=self._scheduler_config,
            cluster_state=None,  # auto-detect
        )

        # Register all stages with the facade
        self._elastic.register_stages(stage_configs)

        logger.info(
            f"ElasticJuicer facade initialized with {len(ops)} stages"
        )

    def _start_elastic(self):
        """Start the ElasticJuicer facade (Tower + MetricsBridge + actors)."""
        if self._elastic is not None and not self._elastic_started:
            self._elastic.start()
            self._elastic_started = True

    def _stop_elastic(self):
        """Stop the ElasticJuicer facade, flush stats, and clean up."""
        if self._elastic is not None and self._elastic_started:
            # Flush runtime stats to ProfilingStore before stopping
            if self._profiling_store is not None:
                try:
                    captains = getattr(self._elastic, "captains", {})
                    total_flushed = 0
                    for name, captain in captains.items():
                        if hasattr(captain, "monitor"):
                            n = captain.monitor.flush_to_store(
                                self._profiling_store
                            )
                            total_flushed += n
                    if total_flushed > 0:
                        self._profiling_store.save_all()
                        logger.info(
                            "Flushed %d op stats to ProfilingStore on stop",
                            total_flushed,
                        )
                except Exception as e:
                    logger.warning(
                        "Failed to flush stats to ProfilingStore: %s", e
                    )

            self._elastic.stop()
            self._elastic_started = False

    # ------------------------------------------------------------------
    # GPU operator detection
    # ------------------------------------------------------------------

    @staticmethod
    def _has_gpu_ops(ops: List) -> bool:
        """Return True if any operator in *ops* requires GPU."""
        return any((getattr(op, "num_gpus", 0) or 0) > 0 for op in ops)

    def _has_elastic_config(self) -> bool:
        """Return True if elastic_juicer config is present and non-empty.

        Note: For ElasticRayExecutor, elastic scheduling auto-activates with
        sensible defaults even without explicit config. This method is still
        used by PartitionedRayExecutor to gate optional elastic activation.
        """
        elastic_cfg = getattr(self.cfg, "elastic_juicer", None)
        if elastic_cfg is None:
            return False
        if hasattr(elastic_cfg, "__dict__"):
            return bool(vars(elastic_cfg))
        if isinstance(elastic_cfg, dict):
            return bool(elastic_cfg)
        return False

    def _should_use_elastic(self, ops: List) -> bool:
        """Determine whether elastic scheduling should be activated.

        Returns True when GPU operators are present. For CPU-only pipelines,
        ElasticRayExecutor uses a lightweight adaptive batching path instead
        of the full Tower/Captain hierarchy.
        """
        return self._has_gpu_ops(ops)

    # ------------------------------------------------------------------
    # Per-operator elastic execution
    # ------------------------------------------------------------------

    def _run_single_op_elastic(
        self,
        ds,
        op,
        stage_index: int,
        cached_columns: set,
        scheduler_config_dict: Dict[str, Any],
    ):
        """
        Execute a single operator with ElasticJuicer adaptive scheduling.

        For GPU operators: use AdaptiveOperator wrapper with MicroScheduler
        For CPU operators: use standard execution
        """
        import pyarrow as pa
        from ray.data import ActorPoolStrategy

        stage_name = f"stage_{stage_index}_{op._name}"

        # Handle tagging ops - add meta column if needed
        if (
            op._name in TAGGING_OPS.modules
            and Fields.meta not in cached_columns
        ):

            def process_batch_arrow(table: pa.Table):
                new_column_data = [{} for _ in range(len(table))]
                return table.append_column(Fields.meta, [new_column_data])

            ds = ds.map_batches(
                process_batch_arrow,
                batch_format="pyarrow",
                batch_size=DEFAULT_BATCH_SIZE,
            )
            cached_columns.add(Fields.meta)

        batch_size = (
            getattr(op, "batch_size", 1) if op.is_batched_op() else 1
        )

        if isinstance(op, Mapper):
            if op.use_ray_actor():
                # GPU Mapper: use AdaptiveOperator wrapper
                ds = self._apply_adaptive_gpu_op(
                    ds, op, stage_name, batch_size, scheduler_config_dict
                )
            else:
                # CPU Mapper: standard execution
                from ray.data._internal.util import get_compute_strategy

                num_proc = (
                    op.num_proc if op.num_proc and op.num_proc > 0 else None
                )
                compute = get_compute_strategy(
                    op.process, concurrency=num_proc
                )
                map_batches_kwargs = dict(
                    batch_size=batch_size,
                    batch_format="pyarrow",
                    num_cpus=op.num_cpus,
                    num_gpus=op.num_gpus,
                    compute=compute,
                    runtime_env=op.runtime_env,
                )
                ds = ds.map_batches(op.process, **map_batches_kwargs)

        elif isinstance(op, Filter):
            # Ensure stats column exists
            if Fields.stats not in cached_columns:

                def process_batch_arrow(table: pa.Table):
                    new_column_data = [{} for _ in range(len(table))]
                    return table.append_column(
                        Fields.stats, [new_column_data]
                    )

                ds = ds.map_batches(
                    process_batch_arrow,
                    batch_format="pyarrow",
                    batch_size=DEFAULT_BATCH_SIZE,
                )
                cached_columns.add(Fields.stats)

            if op.use_ray_actor():
                # GPU Filter: use AdaptiveOperator wrapper
                ds = self._apply_adaptive_gpu_op(
                    ds, op, stage_name, batch_size, scheduler_config_dict
                )
            else:
                # CPU Filter: compute_stats then filter
                from ray.data._internal.util import get_compute_strategy

                num_proc = (
                    op.num_proc if op.num_proc and op.num_proc > 0 else None
                )
                compute = get_compute_strategy(
                    op.compute_stats, concurrency=num_proc
                )
                map_batches_kwargs = dict(
                    batch_size=batch_size,
                    batch_format="pyarrow",
                    num_cpus=op.num_cpus,
                    num_gpus=op.num_gpus,
                    compute=compute,
                    runtime_env=op.runtime_env,
                )
                ds = ds.map_batches(op.compute_stats, **map_batches_kwargs)

            # Apply filter (for both GPU and CPU filters)
            if op.is_batched_op():
                ds = ds.map_batches(
                    partial(filter_batch, filter_func=op.process),
                    batch_format="pyarrow",
                    zero_copy_batch=True,
                    batch_size=DEFAULT_BATCH_SIZE,
                    runtime_env=op.runtime_env,
                )
            else:
                ds = ds.filter(op.process, runtime_env=op.runtime_env)

        elif isinstance(op, (Deduplicator, Pipeline)):
            # Global ops: run directly
            ds = op.run(ds)

        else:
            logger.error(
                "ElasticSchedulingMixin only supports Filter, Mapper, "
                "Deduplicator and Pipeline OPs"
            )
            raise NotImplementedError

        return ds, cached_columns

    def _apply_adaptive_gpu_op(
        self,
        ds,
        op,
        stage_name: str,
        batch_size: int,
        scheduler_config_dict: Dict[str, Any],
    ):
        """Apply GPU operator with AdaptiveOperator wrapper."""
        from ray.data import ActorPoolStrategy

        from data_juicer.core.executor.elastic_ray_executor import (
            AdaptiveOperator,
        )

        # Repartition for GPU actors
        # Use op.num_proc directly (not runtime_np()) to respect the
        # user-configured actor count from the benchmark/config.
        # runtime_np() auto-calculates based on system resources and can
        # overshoot, requesting more actors than GPUs can support.
        num_actors = op.num_proc or 1
        override_num_blocks = getattr(op, "override_num_blocks", None)
        if override_num_blocks is not None:
            ds = ds.repartition(override_num_blocks)
        else:
            ds = ds.repartition(num_actors * 2)

        logger.info(
            f"[{stage_name}] Actors: {num_actors}, "
            f"GPU/actor: {op.num_gpus}, batch_size: {batch_size}"
        )

        # Extract operator config for AdaptiveOperator
        op_kwargs = {}
        if hasattr(op, "_op_cfg") and op._op_cfg:
            # _op_cfg is like {"op_name": {kwargs}}
            op_name, op_args = list(op._op_cfg.items())[0]
            op_kwargs = dict(op_args) if op_args else {}
        else:
            # Fallback: extract from _init_kwargs
            if hasattr(op, "_init_kwargs") and op._init_kwargs:
                op_kwargs = dict(op._init_kwargs)

        # Get adaptive config from facade (if available)
        adaptive_config = {}
        if self._elastic is not None:
            adaptive_config = self._elastic.get_adaptive_op_config(stage_name)
            # Update with auto-scaled batch_size
            adaptive_config["initial_batch_size"] = batch_size
        else:
            # Fallback if facade not initialized
            adaptive_config = {
                "stage_name": stage_name,
                "initial_batch_size": batch_size,
                "scheduler_config_dict": scheduler_config_dict,
            }

        # Use map_batches with AdaptiveOperator class
        # Forward custom_operator_paths so that fresh Ray worker actors can
        # re-register custom ops before the OPERATORS lookup in
        # AdaptiveOperator.__init__.
        custom_operator_paths = getattr(
            self.cfg, "custom_operator_paths", None
        )
        ds = ds.map_batches(
            AdaptiveOperator,
            fn_constructor_kwargs={
                "op_class_name": op._name,
                "op_kwargs": op_kwargs,
                "custom_operator_paths": custom_operator_paths,
                **adaptive_config,
            },
            batch_size=batch_size,
            num_cpus=op.num_cpus or 1,
            num_gpus=op.num_gpus or 1,
            compute=ActorPoolStrategy(size=num_actors),
            batch_format="pyarrow",
            runtime_env=op.runtime_env,
        )

        return ds

    # ------------------------------------------------------------------
    # Convenience: process a list of ops with elastic scheduling
    # ------------------------------------------------------------------

    def _process_ops_elastic(self, ds, ops: List):
        """
        Process a list of operators on a Ray dataset using elastic scheduling.

        This is the elastic equivalent of ``dataset.process(ops)``.
        Iterates over each op and dispatches to ``_run_single_op_elastic``.

        Returns:
            Materialized Ray dataset after all ops.
        """
        scheduler_config_dict = asdict(self._scheduler_config)
        cached_columns = set(ds.columns()) if ds.columns() else set()

        for i, op in enumerate(ops):
            try:
                ds, cached_columns = self._run_single_op_elastic(
                    ds, op, i, cached_columns, scheduler_config_dict
                )
            except Exception as e:
                logger.error(f"Error processing operator {op}: {e}")
                if op.runtime_env is not None:
                    logger.error(
                        "Trying to fallback to base runtime environment"
                    )
                    original_runtime_env = op.runtime_env
                    try:
                        op.runtime_env = None
                        ds, cached_columns = self._run_single_op_elastic(
                            ds, op, i, cached_columns, scheduler_config_dict
                        )
                    finally:
                        op.runtime_env = original_runtime_env
                else:
                    raise e

        # Materialize dataset
        logger.info("Materializing dataset to collect real metrics...")
        ds = ds.materialize()
        return ds

    # ------------------------------------------------------------------
    # Metrics logging
    # ------------------------------------------------------------------

    def _log_elasticjuicer_metrics(
        self, input_rows: int, output_rows: int, duration: float
    ):
        """Log ElasticJuicer metrics summary using facade."""
        throughput = input_rows / duration if duration > 0 else 0

        logger.info("ElasticJuicer Pipeline Summary:")
        logger.info(
            f"  Input rows: {input_rows}, Output rows: {output_rows}"
        )
        logger.info(
            f"  Duration: {duration:.1f}s, Throughput: {throughput:.2f} samples/sec"
        )

        if self._elastic is None:
            return

        # Get metrics from facade
        metrics_summary = self._elastic.get_metrics_summary()
        for stage_name, metrics in metrics_summary.items():
            logger.info(
                f"  {stage_name}: {metrics.get('total_samples', 0)} samples, "
                f"bs={metrics.get('min_bs', 0)}\u2192{metrics.get('max_bs', 0)}, "
                f"avg_latency={metrics.get('avg_latency_ms', 0):.1f}ms"
            )

        # Get Captain stats from facade
        captain_stats = self._elastic.get_captain_stats()
        if captain_stats:
            logger.info("  Captain Statistics:")
            for stage_name, stats in captain_stats.items():
                if stats:
                    logger.info(
                        f"    [{stage_name}]: "
                        f"throughput={stats.get('throughput', 0):.1f} sps, "
                        f"latency={stats.get('latency_ms', 0):.1f}ms, "
                        f"batch_size={stats.get('batch_size', 0)}, "
                        f"backpressure={stats.get('backpressure', False)}, "
                        f"oom_events={stats.get('oom_count', 0)}"
                    )

        # Get Tower stats from facade
        tower_stats = self._elastic.get_tower_stats()
        if tower_stats:
            logger.info(f"  Tower stats: {tower_stats}")

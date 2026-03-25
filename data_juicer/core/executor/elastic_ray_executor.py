"""
Elastic Ray Executor with ElasticJuicer Adaptive Scheduling

This module provides ElasticRayExecutor, which integrates ElasticJuicer's
adaptive scheduling (Tower macro-scheduler, Captain micro-scheduler,
MicroScheduler batch size control) into Data-Juicer's standard execution pipeline.

The bi-level scheduling architecture:
- Tower (macro-scheduler): Global resource allocation and rebalancing
- Captains (micro-schedulers): Per-operator batch size control with PID

Usage in YAML config:
    executor_type: elastic_ray

    elastic_juicer:
        scheduler_preset: gpu          # conservative/gpu/aggressive
        rebalance_interval: 5.0        # Tower rebalance interval in seconds
        enable_offline_tuning: false   # PBT offline tuning
        config_path: null              # path to pre-tuned SchedulerConfig YAML
"""

import os
import shutil
import time
from dataclasses import asdict
from functools import partial
from typing import Any, Dict, List, Optional

from jsonargparse import Namespace
from loguru import logger
from pydantic import PositiveInt

from data_juicer.core.data.dataset_builder import DatasetBuilder
from data_juicer.core.executor import ExecutorBase
from data_juicer.core.executor.dag_execution_mixin import DAGExecutionMixin
from data_juicer.core.executor.event_logging_mixin import EventLoggingMixin
from data_juicer.core.ray_exporter import RayExporter
from data_juicer.core.tracer.ray_tracer import RayTracer
from data_juicer.ops import Deduplicator, Filter, Mapper, OPEnvManager, Pipeline, load_ops
from data_juicer.ops.base_op import DEFAULT_BATCH_SIZE, TAGGING_OPS
from data_juicer.ops.op_fusion import fuse_operators
from data_juicer.utils.constant import Fields
from data_juicer.utils.lazy_loader import LazyLoader

ray = LazyLoader("ray")
pyarrow = LazyLoader("pyarrow")


class TempDirManager:
    """Context manager for temporary directory cleanup."""

    def __init__(self, tmp_dir):
        self.tmp_dir = tmp_dir

    def __enter__(self):
        os.makedirs(self.tmp_dir, exist_ok=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if os.path.exists(self.tmp_dir):
            logger.info(f"Removing tmp dir {self.tmp_dir} ...")
            shutil.rmtree(self.tmp_dir)


# Note: _get_preset_config, _get_cluster_state, and _scheduler_config_to_dict
# have been moved to the ElasticJuicer facade in elastic_juicer.py


def filter_batch(batch, filter_func):
    """Filter batch using filter function."""
    import pyarrow

    mask = pyarrow.array(filter_func(batch.to_pydict()))
    return batch.filter(mask)


# Note: _create_shared_quota_store and MetricsBridge have been moved to 
# the ElasticJuicer facade in elastic_juicer.py


class AdaptiveOperator:
    """
    Adaptive wrapper for any DJ operator with MicroScheduler for batch sizing.

    This wrapper is used with Ray Data's map_batches() and ActorPoolStrategy
    to provide adaptive batch sizing for GPU operators.
    
    The bi-level scheduling integration:
    - MicroScheduler: Local batch size control based on memory feedback
    - SharedQuotaStore: Reads Tower quotas for global coordination
    """

    def __init__(
        self,
        op_class_name: str,
        op_kwargs: Dict[str, Any],
        stage_name: str,
        initial_batch_size: int,
        scheduler_config_dict: Dict[str, Any],
    ):
        """
        Initialize the adaptive operator wrapper.

        Args:
            op_class_name: Name of the operator class in OPERATORS registry
            op_kwargs: Constructor kwargs for the operator
            stage_name: Name for this stage (used in metrics reporting)
            initial_batch_size: Initial batch size for MicroScheduler
            scheduler_config_dict: SchedulerConfig as dict for MicroScheduler init
        """
        # Import and instantiate the actual operator
        from data_juicer.ops import OPERATORS

        op_cls = OPERATORS.modules[op_class_name]
        self.op = op_cls(**op_kwargs)
        self.stage_name = stage_name
        self.initial_batch_size = initial_batch_size

        # Create MicroScheduler with config from dict
        from data_juicer.core.elasticjuicer.scheduler.micro_scheduler import MicroScheduler

        self.micro_scheduler = MicroScheduler(
            initial_batch_size=initial_batch_size,
            max_batch_size=scheduler_config_dict.get("max_batch_size", 1000),
            min_batch_size=scheduler_config_dict.get("min_batch_size", 1),
            target_memory_utilization=scheduler_config_dict.get("target_memory_utilization", 0.85),
            safety_buffer_mb=scheduler_config_dict.get("safety_buffer_mb", 1000.0),
            use_gpu=scheduler_config_dict.get("use_gpu_memory", False),
        )

        # Try to connect to PipelineMetricsCollector with retry logic
        self.metrics_collector = None
        import ray
        for attempt in range(3):
            try:
                self.metrics_collector = ray.get_actor("elastic_pipeline_metrics")
                logger.info(f"[{self.stage_name}] Connected to PipelineMetricsCollector")
                break
            except Exception as e:
                delay = 0.5 * (attempt + 1)  # 0.5s, 1.0s, 1.5s
                if attempt < 2:
                    time.sleep(delay)
                else:
                    logger.warning(f"[{self.stage_name}] Failed to connect to PipelineMetricsCollector after 3 attempts: {e}")

        # Connect to SharedQuotaStore for Tower quotas with retry logic
        self.quota_store = None
        for attempt in range(3):
            try:
                self.quota_store = ray.get_actor("elastic_quota_store")
                logger.info(f"[{self.stage_name}] Connected to SharedQuotaStore")
                break
            except Exception as e:
                delay = 0.5 * (attempt + 1)  # 0.5s, 1.0s, 1.5s
                if attempt < 2:
                    time.sleep(delay)
                else:
                    logger.warning(f"[{self.stage_name}] Failed to connect to SharedQuotaStore after 3 attempts: {e}")

        # Quota check state
        self._last_quota_check = 0
        self._quota_check_interval = 2.0  # seconds
        self._backpressure = False

        # Statistics
        self.batch_sizes_used: List[int] = []
        self.samples_processed = 0
        self.total_latency_ms = 0.0
        
        # Store last batch for _update_scheduler feature extraction
        self._last_batch = None

    def _check_quota(self):
        """Check if Tower has issued new quota for this stage."""
        import time

        now = time.time()
        if now - self._last_quota_check < self._quota_check_interval:
            return
        self._last_quota_check = now

        if self.quota_store is None:
            return

        try:
            import ray

            quota = ray.get(self.quota_store.get_quota.remote(self.stage_name))
            if quota:
                # Apply Tower's batch size recommendation
                # Blend Tower recommendation with MicroScheduler's local decision
                tower_bs = quota.get("batch_size", None)
                if tower_bs and tower_bs > 0:
                    current = self.micro_scheduler.controller.current_batch_size
                    # Only blend if Tower recommends increase; don't let Tower pull batch size below initial
                    if tower_bs >= current:
                        blended = int(0.7 * current + 0.3 * tower_bs)
                        new_bs = max(self.initial_batch_size, blended)
                    else:
                        # Tower recommends decrease - only allow if significantly lower (backpressure scenario)
                        if tower_bs < current * 0.5:
                            new_bs = max(self.initial_batch_size, int(0.7 * current + 0.3 * tower_bs))
                        else:
                            new_bs = current  # Keep current, ignore minor decreases
                    if new_bs != current:
                        logger.info(f"[{self.stage_name}] Batch size: {current} -> {new_bs} (tower_bs={tower_bs})")
                    self.micro_scheduler.controller.current_batch_size = new_bs

                # Apply backpressure from Tower
                self._backpressure = quota.get("backpressure", False)
                if self._backpressure:
                    logger.info(f"[{self.stage_name}] Backpressure activated")
        except Exception as e:
            logger.debug(f"[{self.stage_name}] Quota check failed: {e}")

    def _extract_features(self, batch):
        """Extract sample features from batch for MicroScheduler."""
        try:
            num_rows = batch.num_rows if hasattr(batch, 'num_rows') else len(batch)
            batch_memory_bytes = batch.nbytes if hasattr(batch, 'nbytes') else 0
            per_sample_mb = (batch_memory_bytes / max(num_rows, 1)) / (1024 * 1024)
            
            from data_juicer.core.elasticjuicer.scheduler.micro_scheduler import SampleFeatures
            return SampleFeatures(
                batch_size=num_rows,
                estimated_memory_mb=per_sample_mb * num_rows,
            )
        except Exception:
            return None

    def __call__(self, batch):
        """Process batch with adaptive sub-batching."""
        import time

        import pyarrow as pa

        # Store batch for _update_scheduler feature extraction
        self._last_batch = batch

        # Check for Tower quotas periodically
        self._check_quota()

        # If backpressure from Tower, add small delay to reduce pressure on downstream
        if self._backpressure:
            time.sleep(0.1)

        # Get actual memory usage once per __call__ invocation for metrics reporting
        try:
            import psutil
            memory_mb = psutil.virtual_memory().used / (1024 * 1024)
        except Exception:
            memory_mb = 0.0

        total_rows = batch.num_rows if hasattr(batch, "num_rows") else len(batch)

        if total_rows == 0:
            return batch

        # Get recommended batch size from MicroScheduler (uses PID feedback)
        recommended_bs = self.micro_scheduler.get_batch_size(sample_features=self._extract_features(batch))
        recommended_bs = max(1, recommended_bs)

        # If batch is smaller than recommended, process whole batch
        if total_rows <= recommended_bs:
            self.batch_sizes_used.append(total_rows)

            t0 = time.time()
            try:
                result = self.op(batch)
                elapsed_ms = (time.time() - t0) * 1000
                success = True
            except Exception as e:
                elapsed_ms = (time.time() - t0) * 1000
                self.micro_scheduler.report_oom(batch_size=total_rows, memory_mb=0.0)
                raise

            # Report metrics
            if self.metrics_collector:
                try:
                    import ray

                    ray.get(
                        self.metrics_collector.report.remote(
                            self.stage_name, total_rows, elapsed_ms, memory_mb
                        )
                    )
                except Exception:
                    pass

            # Update MicroScheduler
            self._update_scheduler()

            self.samples_processed += total_rows
            self.total_latency_ms += elapsed_ms

            return result

        # Process in sub-batches
        offset = 0
        results = []

        while offset < total_rows:
            recommended_bs = self.micro_scheduler.get_batch_size(sample_features=self._extract_features(batch))
            recommended_bs = max(1, min(recommended_bs, total_rows - offset))
            self.batch_sizes_used.append(recommended_bs)

            end = min(offset + recommended_bs, total_rows)
            sub_batch = batch.slice(offset, end - offset)

            t0 = time.time()
            try:
                sub_result = self.op(sub_batch)
                elapsed_ms = (time.time() - t0) * 1000
                success = True
            except Exception as e:
                elapsed_ms = (time.time() - t0) * 1000
                success = False
                # On error (e.g. OOM), reduce batch size and retry with smaller batch
                self.micro_scheduler.report_oom(batch_size=end - offset, memory_mb=0.0)
                if end - offset > 1:
                    recommended_bs = max(1, recommended_bs // 2)
                    continue
                else:
                    raise

            results.append(sub_result)

            # Report metrics
            if self.metrics_collector:
                try:
                    import ray

                    ray.get(
                        self.metrics_collector.report.remote(
                            self.stage_name, end - offset, elapsed_ms, memory_mb
                        )
                    )
                except Exception:
                    pass

            # Update MicroScheduler
            self._update_scheduler()

            self.samples_processed += end - offset
            self.total_latency_ms += elapsed_ms
            offset = end

        if len(results) == 1:
            return results[0]

        # Concatenate PyArrow tables
        if isinstance(results[0], pa.Table):
            return pa.concat_tables(results)

        return results[0]

    def _update_scheduler(self):
        """Update MicroScheduler with current memory state."""
        try:
            import psutil

            memory_mb = psutil.virtual_memory().used / (1024 * 1024)
            sample_features = self._extract_features(self._last_batch) if self._last_batch is not None else None
            self.micro_scheduler.update(actual_memory_used=memory_mb, sample_features=sample_features)
        except Exception:
            pass


# Note: _create_pipeline_metrics_collector has been moved to the ElasticJuicer facade
# AdaptiveOperator connects to the named actors by name


class ElasticRayExecutor(ExecutorBase, DAGExecutionMixin, EventLoggingMixin):
    """
    Ray executor with ElasticJuicer adaptive scheduling.

    Integrates Tower (macro-scheduler), Captain (per-stage micro-scheduler),
    and MicroScheduler (per-actor batch adaptation) into Data-Juicer's
    standard execution pipeline.

    Features:
    - Adaptive batch sizing for GPU operators via MicroScheduler
    - PID-controlled memory management to prevent OOM
    - Tower-based global resource allocation (optional rebalancing)
    - Per-stage metrics collection

    Usage in YAML config:
        executor_type: elastic_ray

        elastic_juicer:
            scheduler_preset: gpu          # conservative/gpu/aggressive
            rebalance_interval: 5.0        # Tower rebalance interval in seconds
            enable_offline_tuning: false   # PBT offline tuning
            config_path: null              # path to pre-tuned SchedulerConfig YAML
    """

    def __init__(self, cfg: Optional[Namespace] = None):
        """
        Initialization method.

        :param cfg: optional config dict.
        """
        super().__init__(cfg)

        self.executor_type = "elastic_ray"
        self.work_dir = self.cfg.work_dir

        # Initialize EventLoggingMixin for job management and event logging
        EventLoggingMixin.__init__(self, cfg)

        # Initialize DAGExecutionMixin for AST/DAG functionality
        DAGExecutionMixin.__init__(self)

        # init ray
        logger.info("Initializing Ray for ElasticRayExecutor ...")

        from data_juicer.utils.ray_utils import initialize_ray

        initialize_ray(cfg=cfg, force=True)

        self.tmp_dir = os.path.join(
            self.work_dir, ".tmp", ray.get_runtime_context().get_job_id()
        )

        # init dataset builder
        self.datasetbuilder = DatasetBuilder(self.cfg, executor_type="ray")

        logger.info("Preparing exporter...")
        # Prepare export extra args, including S3 credentials if export_path is S3
        export_extra_args = (
            dict(self.cfg.export_extra_args)
            if hasattr(self.cfg, "export_extra_args")
            else {}
        )

        # If export_path is S3, extract AWS credentials
        if self.cfg.export_path.startswith("s3://"):
            if (
                hasattr(self.cfg, "export_aws_credentials")
                and self.cfg.export_aws_credentials
            ):
                export_aws_creds = self.cfg.export_aws_credentials
                credential_fields = {
                    "aws_access_key_id",
                    "aws_secret_access_key",
                    "aws_session_token",
                    "aws_region",
                    "endpoint_url",
                }
                for field in credential_fields.intersection(export_aws_creds):
                    export_extra_args[field] = export_aws_creds[field]

        self.exporter = RayExporter(
            self.cfg.export_path,
            self.cfg.export_type,
            self.cfg.export_shard_size,
            keep_stats_in_res_ds=self.cfg.keep_stats_in_res_ds,
            keep_hashes_in_res_ds=self.cfg.keep_hashes_in_res_ds,
            **export_extra_args,
        )

        # setup tracer
        self.tracer = None
        self.open_tracer = self.cfg.open_tracer
        if self.open_tracer:
            logger.info("Preparing tracer...")
            self.tracer = RayTracer.remote(
                self.work_dir,
                self.cfg.op_list_to_trace,
                show_num=self.cfg.trace_num,
                trace_keys=self.cfg.trace_keys,
            )

        # setup OPEnvManager
        self.op_env_manager = None
        if self.cfg.min_common_dep_num_to_combine >= 0:
            logger.info("Preparing OPEnvManager...")
            self.op_env_manager = OPEnvManager(
                min_common_dep_num_to_combine=self.cfg.min_common_dep_num_to_combine,
                conflict_resolve_strategy=self.cfg.conflict_resolve_strategy,
            )

        # ElasticJuicer facade (initialized in run())
        self._elastic = None  # ElasticJuicer facade instance
        self._scheduler_config = None

    def _parse_elastic_juicer_config(self):
        """Parse elastic_juicer config section and create SchedulerConfig."""
        from data_juicer.core.elasticjuicer.scheduler.scheduler_config import SchedulerConfig
        
        elastic_cfg = getattr(self.cfg, "elastic_juicer", None)
        if elastic_cfg is None:
            elastic_cfg = {}
        elif hasattr(elastic_cfg, "__dict__"):
            # Convert Namespace to dict
            elastic_cfg = dict(elastic_cfg)

        scheduler_preset = elastic_cfg.get("scheduler_preset", "gpu")
        config_path = elastic_cfg.get("config_path", None)
        rebalance_interval = elastic_cfg.get("rebalance_interval", 5.0)

        # Load or create SchedulerConfig
        if config_path and os.path.exists(config_path):
            self._scheduler_config = SchedulerConfig.from_yaml(config_path)
            logger.info(f"Loaded ElasticJuicer config from {config_path}")
        else:
            # Use preset
            presets = {
                "conservative": SchedulerConfig.conservative,
                "gpu": SchedulerConfig.gpu,
                "aggressive": SchedulerConfig.aggressive,
            }
            factory = presets.get(scheduler_preset, SchedulerConfig.gpu)
            self._scheduler_config = factory()
            logger.info(f"Using ElasticJuicer {scheduler_preset} preset config")

        # Override rebalance interval if specified
        self._scheduler_config.rebalance_interval_sec = rebalance_interval

        return elastic_cfg

    def _init_elasticjuicer_components(self, ops: List):
        """Initialize ElasticJuicer facade with all scheduling components.
        
        Creates stage configs from operators and registers them with the
        ElasticJuicer facade, which manages Tower, Captains, and Ray actors.
        """
        from data_juicer.core.elasticjuicer import ElasticJuicer
        
        # Build stage configs from operators
        stage_configs = []
        for i, op in enumerate(ops):
            stage_configs.append({
                'name': f"stage_{i}_{op._name}",
                'batch_size': getattr(op, 'batch_size', self._scheduler_config.initial_batch_size),
                'num_gpus': op.num_gpus or 0,
                'num_actors': op.runtime_np() if hasattr(op, 'runtime_np') else (op.num_proc or 1),
            })
        
        # Create ElasticJuicer facade
        self._elastic = ElasticJuicer(
            config=self._scheduler_config,
            cluster_state=None,  # auto-detect
        )
        
        # Register all stages with the facade
        self._elastic.register_stages(stage_configs)
        
        logger.info(
            f"ElasticJuicer facade initialized with {len(ops)} stages")

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
        For CPU operators: use standard execution (same as RayDataset._run_single_op)
        """
        import pyarrow as pa
        from ray.data import ActorPoolStrategy

        stage_name = f"stage_{stage_index}_{op._name}"

        # Handle tagging ops - add meta column if needed
        if op._name in TAGGING_OPS.modules and Fields.meta not in cached_columns:

            def process_batch_arrow(table: pa.Table):
                new_column_data = [{} for _ in range(len(table))]
                return table.append_column(Fields.meta, [new_column_data])

            ds = ds.map_batches(
                process_batch_arrow, batch_format="pyarrow", batch_size=DEFAULT_BATCH_SIZE
            )
            cached_columns.add(Fields.meta)

        batch_size = getattr(op, "batch_size", 1) if op.is_batched_op() else 1

        if isinstance(op, Mapper):
            if op.use_ray_actor():
                # GPU Mapper: use AdaptiveOperator wrapper
                ds = self._apply_adaptive_gpu_op(
                    ds, op, stage_name, batch_size, scheduler_config_dict
                )
            else:
                # CPU Mapper: standard execution
                from ray.data._internal.util import get_compute_strategy

                num_proc = op.num_proc if op.num_proc and op.num_proc > 0 else None
                compute = get_compute_strategy(op.process, concurrency=num_proc)
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
                    return table.append_column(Fields.stats, [new_column_data])

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

                num_proc = op.num_proc if op.num_proc and op.num_proc > 0 else None
                compute = get_compute_strategy(op.compute_stats, concurrency=num_proc)
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
                "ElasticRayExecutor only supports Filter, Mapper, "
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

        # Repartition for GPU actors
        num_actors = op.runtime_np() if hasattr(op, "runtime_np") else (op.num_proc or 1)
        override_num_blocks = getattr(op, "override_num_blocks", None)
        if override_num_blocks is not None:
            ds = ds.repartition(override_num_blocks)
        else:
            ds = ds.repartition(num_actors * 2)

        # Auto-scale batch size based on available GPU memory
        if batch_size <= 4:  # Only auto-scale very conservative defaults
            try:
                import subprocess
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    free_mb = min(int(x.strip()) for x in result.stdout.strip().split('\n') if x.strip())
                    # Conservative: assume 1000MB per sample for video ops (frames are memory-hungry)
                    auto_bs = max(batch_size, min(16, free_mb // 1000))
                    if auto_bs > batch_size:
                        logger.info(f"[{stage_name}] Auto batch_size: {batch_size} -> {auto_bs} (free GPU mem: {free_mb}MB)")
                        batch_size = auto_bs
            except Exception:
                pass

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
            adaptive_config['initial_batch_size'] = batch_size
        else:
            # Fallback if facade not initialized
            adaptive_config = {
                'stage_name': stage_name,
                'initial_batch_size': batch_size,
                'scheduler_config_dict': scheduler_config_dict,
            }

        # Use map_batches with AdaptiveOperator class
        ds = ds.map_batches(
            AdaptiveOperator,
            fn_constructor_kwargs={
                "op_class_name": op._name,
                "op_kwargs": op_kwargs,
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

    def run(
        self,
        load_data_np: Optional[PositiveInt] = None,
        skip_export: bool = False,
        skip_return: bool = False,
    ):
        """
        Running the dataset process pipeline with ElasticJuicer adaptive scheduling.

        :param load_data_np: number of workers when loading the dataset.
        :param skip_export: whether to skip exporting results to disk
        :param skip_return: skip return for API called.
        :return: processed dataset.
        """
        # 1. Parse ElasticJuicer config
        elastic_cfg = self._parse_elastic_juicer_config()

        # 2. Load data
        logger.info("Loading dataset with ElasticRayExecutor...")
        dataset = self.datasetbuilder.load_dataset(num_proc=load_data_np)
        columns = dataset.data.columns()

        # 3. Extract processes
        logger.info("Preparing process operators...")
        ops = load_ops(self.cfg.process, self.op_env_manager)

        # Initialize DAG execution planning
        self._initialize_dag_execution(self.cfg, ops=ops)

        # Log job start with DAG context
        dataset_info = {}
        if hasattr(self.cfg, "dataset_path") and self.cfg.dataset_path:
            dataset_info["dataset_path"] = self.cfg.dataset_path
        if hasattr(self.cfg, "dataset") and self.cfg.dataset:
            dataset_info["dataset"] = self.cfg.dataset

        job_config = {
            **dataset_info,
            "work_dir": self.work_dir,
            "executor_type": self.executor_type,
            "dag_node_count": len(self.pipeline_dag.nodes) if self.pipeline_dag else 0,
            "dag_edge_count": len(self.pipeline_dag.edges) if self.pipeline_dag else 0,
            "parallel_groups_count": (
                len(self.pipeline_dag.parallel_groups) if self.pipeline_dag else 0
            ),
        }
        self.log_job_start(job_config, len(ops))

        if self.cfg.op_fusion:
            logger.info(
                f"Start OP fusion and reordering with strategy "
                f"[{self.cfg.fusion_strategy}]..."
            )
            ops = fuse_operators(ops)

        # 4. Detect whether pipeline has GPU operators
        gpu_op_count = sum(1 for op in ops if (getattr(op, 'num_gpus', 0) or 0) > 0)
        has_gpu_ops = gpu_op_count > 0
        logger.info(f'Pipeline analysis: {len(ops)} total operators, {gpu_op_count} GPU operators')

        # 5. Empty dataset guard
        input_rows = dataset.data.count()
        if input_rows == 0:
            logger.warning('Empty dataset — skipping processing.')
            if not skip_export:
                self.exporter.export(dataset)
            return dataset if not skip_return else None

        with TempDirManager(self.tmp_dir):
            tstart = time.time()
            start_time = time.time()

            # Pre-execute DAG monitoring
            if self.pipeline_dag:
                self._pre_execute_operations_with_dag_monitoring(ops)

            if not has_gpu_ops:
                # CPU-only fallback path: use standard RayDataset.process()
                logger.info(f'No GPU operators detected in pipeline ({len(ops)} CPU ops). '
                            f'Using standard execution path (ElasticJuicer disabled).')

                # Execute operations using standard dataset.process() like RayExecutor
                dataset = dataset.process(ops, tracer=self.tracer)

                # Force materialization to get real execution
                logger.info("Materializing dataset to collect real metrics...")
                dataset.data = dataset.data.materialize()

                ds = dataset.data
            else:
                # GPU path: use full ElasticJuicer adaptive scheduling
                logger.info(f'GPU operators detected. '
                            f'Using ElasticJuicer adaptive scheduling.')

                # Initialize ElasticJuicer facade for GPU path
                self._init_elasticjuicer_components(ops)

                # Prepare scheduler config dict for serialization
                scheduler_config_dict = asdict(self._scheduler_config)

                logger.info("Processing data with ElasticJuicer adaptive scheduling...")

                # Start ElasticJuicer facade (Tower + MetricsBridge + Ray actors)
                if self._elastic is not None:
                    self._elastic.start()

                try:
                    # Execute operations with adaptive scheduling
                    ds = dataset.data
                    cached_columns = set(ds.columns()) if ds.columns() else set()

                    for i, op in enumerate(ops):
                        try:
                            ds, cached_columns = self._run_single_op_elastic(
                                ds, op, i, cached_columns, scheduler_config_dict
                            )
                        except Exception as e:
                            logger.error(f"Error processing operator {op}: {e}")
                            if op.runtime_env is not None:
                                logger.error("Trying to fallback to base runtime environment")
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

                finally:
                    # Stop ElasticJuicer facade (handles all cleanup)
                    if self._elastic is not None:
                        self._elastic.stop()

                # Update dataset.data for GPU path
                dataset.data = ds

            # Get metrics after execution
            duration = time.time() - start_time
            output_rows = ds.count()

            # Post-execute DAG monitoring
            if self.pipeline_dag:
                metrics = {
                    "duration": duration,
                    "input_rows": input_rows,
                    "output_rows": output_rows,
                }
                self._post_execute_operations_with_dag_monitoring(ops, metrics=metrics)

            # Collect and log ElasticJuicer metrics (only for GPU path)
            if has_gpu_ops:
                self._log_elasticjuicer_metrics(input_rows, output_rows, duration)

            # 6. Data export
            if not skip_export:
                logger.info("Exporting dataset to disk...")
                self.exporter.export(ds, columns=columns)

            tend = time.time()
            logger.info(f"All Ops are done in {tend - tstart:.3f}s.")

        # Log job completion
        job_duration = time.time() - tstart
        self.log_job_complete(job_duration, self.cfg.export_path)

        # Finalize tracer
        if self.tracer:
            ray.get(self.tracer.finalize_traces.remote())

        if not skip_return:
            return dataset

    def _log_elasticjuicer_metrics(
        self, input_rows: int, output_rows: int, duration: float
    ):
        """Log ElasticJuicer metrics summary using facade."""
        throughput = input_rows / duration if duration > 0 else 0

        logger.info(f"ElasticJuicer Pipeline Summary:")
        logger.info(f"  Input rows: {input_rows}, Output rows: {output_rows}")
        logger.info(f"  Duration: {duration:.1f}s, Throughput: {throughput:.2f} samples/sec")

        if self._elastic is None:
            return

        # Get metrics from facade
        metrics_summary = self._elastic.get_metrics_summary()
        for stage_name, metrics in metrics_summary.items():
            logger.info(
                f"  {stage_name}: {metrics.get('total_samples', 0)} samples, "
                f"bs={metrics.get('min_bs', 0)}→{metrics.get('max_bs', 0)}, "
                f"avg_latency={metrics.get('avg_latency_ms', 0):.1f}ms"
            )

        # Get Captain stats from facade
        captain_stats = self._elastic.get_captain_stats()
        if captain_stats:
            logger.info(f"  Captain Statistics:")
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

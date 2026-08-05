"""
Simplified Partitioned Ray Executor for Large Dataset Processing

This module implements a streamlined partitioned execution strategy for Ray mode that:
2. Splits the dataset into manageable partitions using Ray's .split() method
3. Processes each partition independently with Ray tasks
4. Merges results back into a single dataset for export
5. Supports convergence points for global operations (like deduplicators)
"""

import copy
import hashlib
import json
import math
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from jsonargparse import Namespace
from loguru import logger
from pydantic import PositiveInt

from data_juicer.core.data.dataset_builder import DatasetBuilder
from data_juicer.core.data.ray_dataset import RayDataset
from data_juicer.core.executor import ExecutorBase
from data_juicer.core.executor.dag_execution_mixin import DAGExecutionMixin
from data_juicer.core.executor.event_logging_mixin import EventLoggingMixin, EventType
from data_juicer.core.ray_exporter import RayExporter
from data_juicer.ops import load_ops
from data_juicer.ops.op_fusion import fuse_operators
from data_juicer.utils.ckpt_utils import CheckpointStrategy, RayCheckpointManager
from data_juicer.utils.config_utils import ConfigAccessor
from data_juicer.utils.file_utils import is_remote_path
from data_juicer.utils.lazy_loader import LazyLoader

ray = LazyLoader("ray")

_AUTO_CPU_PARTITION_CAP = 4
_PARTITION_CONTENT_HASH_ALGORITHM = "sha256-multiset-v1"
_PARTITION_CONTENT_HASH_MODULUS = 1 << 256


def _canonical_row_bytes(row: Dict) -> bytes:
    """Serialize one row consistently for partition content hashing."""
    try:
        serialized = json.dumps(row, sort_keys=True, default=str, separators=(",", ":"), ensure_ascii=False)
    except Exception:
        serialized = str(row)
    return serialized.encode("utf-8")


def _hash_partition_batch(batch):
    """Return an order-independent hash accumulator for one Arrow batch."""
    import pyarrow

    digest_sum = 0
    digest_xor = 0
    rows = batch.to_pylist()
    for row in rows:
        digest = int.from_bytes(hashlib.sha256(_canonical_row_bytes(row)).digest(), "big")
        digest_sum = (digest_sum + digest) % _PARTITION_CONTENT_HASH_MODULUS
        digest_xor ^= digest

    return pyarrow.table(
        {
            "row_count": [len(rows)],
            "digest_sum": [f"{digest_sum:064x}"],
            "digest_xor": [f"{digest_xor:064x}"],
        }
    )


def _combine_partition_hash_partials(partials: List[Dict]) -> tuple:
    """Combine batch accumulators into a stable partition content hash."""
    row_count = 0
    digest_sum = 0
    digest_xor = 0
    for partial in partials:
        row_count += int(partial["row_count"])
        digest_sum = (digest_sum + int(partial["digest_sum"], 16)) % _PARTITION_CONTENT_HASH_MODULUS
        digest_xor ^= int(partial["digest_xor"], 16)

    payload = f"{_PARTITION_CONTENT_HASH_ALGORITHM}:" f"{row_count}:{digest_sum:064x}:{digest_xor:064x}"
    return hashlib.sha256(payload.encode("ascii")).hexdigest(), row_count


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
            # in some cases, such as we mount OSS bucket with fuse device
            # using Fluid, cleaning up temporary directories via
            # shutil.rmtree() fails, but os.rmdir() succeeds.
            try:
                shutil.rmtree(self.tmp_dir)
            except OSError as e:
                logger.warning(f"Remove tmp dir with shutil.rmtree() failed: {e}, " "will try os.rmdir()")
                os.rmdir(self.tmp_dir)


# Note: Using Ray Data's built-in map_batches for parallel processing instead of custom remote functions


# Simplified classes for basic functionality
@dataclass
class PartitionResult:
    """Simple result container for partition processing."""

    partition_id: int
    dataset: Optional[Any] = None
    success: bool = False
    error: Optional[str] = None


@dataclass
class PartitionMetadata:
    """Metadata for a single partition to enable validation on resume.

    Stores information about each partition that can be used to verify
    that re-partitioning produces the same result on job resumption.
    """

    partition_id: int
    row_count: int
    first_row_hash: str  # Hash of first row for validation
    last_row_hash: str  # Hash of last row for validation
    content_hash: str = ""  # Stable hash of the complete partition contents
    start_row: Optional[int] = None  # Inclusive row offset in the ordered input
    end_row: Optional[int] = None  # Exclusive row offset in the ordered input

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "PartitionMetadata":
        return cls(**data)


@dataclass
class PartitioningInfo:
    """Complete partitioning information for a job.

    Stored alongside checkpoints to enable validation that re-partitioning
    on resume produces identical partitions.
    """

    num_partitions: int
    total_rows: int
    partitions: List[PartitionMetadata] = field(default_factory=list)
    deterministic: bool = True  # Whether deterministic splitting was used
    hash_algorithm: str = _PARTITION_CONTENT_HASH_ALGORITHM

    def to_dict(self) -> Dict:
        return {
            "num_partitions": self.num_partitions,
            "total_rows": self.total_rows,
            "deterministic": self.deterministic,
            "hash_algorithm": self.hash_algorithm,
            "partitions": [p.to_dict() for p in self.partitions],
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "PartitioningInfo":
        partitions = [PartitionMetadata.from_dict(p) for p in data.get("partitions", [])]
        return cls(
            num_partitions=data["num_partitions"],
            total_rows=data["total_rows"],
            deterministic=data.get("deterministic", True),
            hash_algorithm=data.get("hash_algorithm", _PARTITION_CONTENT_HASH_ALGORITHM),
            partitions=partitions,
        )

    def save(self, path: str) -> None:
        """Save partitioning info to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved partitioning info to {path}")

    @classmethod
    def load(cls, path: str) -> Optional["PartitioningInfo"]:
        """Load partitioning info from JSON file."""
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r") as f:
                data = json.load(f)
            return cls.from_dict(data)
        except Exception as e:
            logger.warning(f"Failed to load partitioning info from {path}: {e}")
            return None


class PartitionedRayExecutor(ExecutorBase, DAGExecutionMixin, EventLoggingMixin):
    """
    Simplified Ray executor with dataset partitioning using .split().

    Features:
    - Single DatasetBuilder loads the full dataset
    - Uses Ray's .split() method for partitioning
    - Processes partitions in parallel with Ray tasks
    - Supports convergence points for global operations
    - Merges results back into a single dataset
    """

    def __init__(self, cfg: Optional[Namespace] = None):
        """Initialize the partitioned Ray executor."""
        super().__init__(cfg)

        self.executor_type = "ray_partitioned"
        self.work_dir = self._resolve_local_path(self.cfg.work_dir)
        self.job_id = self.cfg.get("job_id", None)

        # Initialize temporary directory for Ray operations
        self.tmp_dir = os.path.join(self.work_dir, ".tmp", ray.get_runtime_context().get_job_id())

        # Initialize EventLoggingMixin for job management and event logging
        EventLoggingMixin.__init__(self, cfg)

        # Initialize DAGExecutionMixin for AST/DAG functionality
        DAGExecutionMixin.__init__(self)

        # Override strategy methods for partitioned execution
        self._override_strategy_methods()

        self.datasetbuilder = DatasetBuilder(self.cfg, executor_type="ray")

        # Partition configuration
        self._configure_partitioning()

        # Checkpoint configuration and manager initialization
        checkpoint_cfg = getattr(self.cfg, "checkpoint", None)
        checkpoint_dir = self._resolve_local_path(
            getattr(self.cfg, "checkpoint_dir", os.path.join(self.work_dir, "checkpoints"))
        )

        if checkpoint_cfg:
            # Use ConfigAccessor to handle both dict and object configurations
            checkpoint_enabled = ConfigAccessor.get(checkpoint_cfg, "enabled", True)
            strategy_str = ConfigAccessor.get(checkpoint_cfg, "strategy", "every_op")
            checkpoint_n_ops = ConfigAccessor.get(checkpoint_cfg, "n_ops", 1)
            checkpoint_op_names = ConfigAccessor.get(checkpoint_cfg, "op_names", [])

            # Parse checkpoint strategy with validation
            try:
                checkpoint_strategy = CheckpointStrategy(strategy_str)
            except ValueError:
                logger.warning(f"Unknown checkpoint strategy: {strategy_str}, defaulting to EVERY_OP")
                checkpoint_strategy = CheckpointStrategy.EVERY_OP
        else:
            checkpoint_enabled = False
            checkpoint_strategy = CheckpointStrategy.DISABLED
            checkpoint_n_ops = 1
            checkpoint_op_names = []

        # Initialize Ray checkpoint manager
        self.ckpt_manager = RayCheckpointManager(
            ckpt_dir=checkpoint_dir,
            checkpoint_enabled=checkpoint_enabled,
            checkpoint_strategy=checkpoint_strategy,
            checkpoint_n_ops=checkpoint_n_ops,
            checkpoint_op_names=checkpoint_op_names,
            event_logger=self,
        )

        logger.info(f"Checkpointing: {'enabled' if self.ckpt_manager.checkpoint_enabled else 'disabled'}")
        if self.ckpt_manager.checkpoint_enabled:
            logger.info(f"Checkpoint strategy: {self.ckpt_manager.checkpoint_strategy.value}")
            logger.info(f"Checkpoint directory: {self.ckpt_manager.ckpt_dir}")

        # Initialize RayExporter for final output
        logger.info("Preparing exporter...")
        # Prepare export extra args, including S3 credentials if export_path is S3
        export_extra_args = dict(self.cfg.export_extra_args) if hasattr(self.cfg, "export_extra_args") else {}

        # If export_path is S3, extract AWS credentials with priority:
        # 1. export_aws_credentials (export-specific)
        # 2. dataset config (for backward compatibility)
        # 3. environment variables (handled by exporter)
        if self.cfg.export_path.startswith("s3://"):
            # Pass export-specific credentials if provided.
            # The RayExporter will handle falling back to environment variables or other credential mechanisms.
            if hasattr(self.cfg, "export_aws_credentials") and self.cfg.export_aws_credentials:
                export_aws_creds = self.cfg.export_aws_credentials
                if hasattr(export_aws_creds, "aws_access_key_id"):
                    export_extra_args["aws_access_key_id"] = export_aws_creds.aws_access_key_id
                if hasattr(export_aws_creds, "aws_secret_access_key"):
                    export_extra_args["aws_secret_access_key"] = export_aws_creds.aws_secret_access_key
                if hasattr(export_aws_creds, "aws_session_token"):
                    export_extra_args["aws_session_token"] = export_aws_creds.aws_session_token
                if hasattr(export_aws_creds, "aws_region"):
                    export_extra_args["aws_region"] = export_aws_creds.aws_region
                if hasattr(export_aws_creds, "endpoint_url"):
                    export_extra_args["endpoint_url"] = export_aws_creds.endpoint_url

        self.exporter = RayExporter(
            self.cfg.export_path,
            getattr(self.cfg, "export_type", None),
            getattr(self.cfg, "export_shard_size", 0),
            keep_stats_in_res_ds=getattr(self.cfg, "keep_stats_in_res_ds", True),
            keep_hashes_in_res_ds=getattr(self.cfg, "keep_hashes_in_res_ds", False),
            encrypt_before_export=getattr(self.cfg, "encrypt_before_export", False),
            encryption_key_path=getattr(self.cfg, "encryption_key_path", None),
            **export_extra_args,
        )

    @staticmethod
    def _resolve_local_path(path):
        """Convert a local, non-empty path to an absolute path.

        Ray's writers (e.g. write_parquet) run on workers whose working
        directory may differ from the main process, so relative paths like
        './tmp/...' cannot be resolved correctly and lead to empty checkpoint
        directories. Absolute conversion is applied only to local, non-empty
        paths: remote URIs (e.g. s3://, gs://, hdfs://) are left untouched to
        avoid corrupting their scheme, and empty/None values pass through
        unchanged to avoid raising TypeError.
        """
        if path and not is_remote_path(path):
            return os.path.abspath(path)
        return path

    def _configure_partitioning(self):
        """Configure partitioning based on manual or auto mode."""
        # Get partition configuration
        partition_cfg = getattr(self.cfg, "partition", {})

        # Use ConfigAccessor to handle both dict and object configurations
        mode = ConfigAccessor.get(partition_cfg, "mode", "auto")
        num_of_partitions = ConfigAccessor.get(partition_cfg, "num_of_partitions", 4)
        max_concurrent_partitions = ConfigAccessor.get(partition_cfg, "max_concurrent_partitions", "auto")
        partition_size = ConfigAccessor.get(partition_cfg, "size", 5000)
        max_size_mb = ConfigAccessor.get(partition_cfg, "max_size_mb", 64)

        # Fallback to legacy configuration if partition config is not available
        # or if legacy num_partitions is explicitly set
        if (
            not partition_cfg
            or hasattr(self.cfg, "num_partitions")
            and getattr(self.cfg, "num_partitions", None) is not None
        ):
            mode = "manual"
            num_of_partitions = getattr(self.cfg, "num_partitions", 4)
            if not partition_cfg:
                logger.warning("No partition configuration found, using legacy num_partitions")
            else:
                logger.warning("Legacy num_partitions detected, overriding partition configuration")

        self.partition_mode = mode
        self.num_partitions = num_of_partitions
        self.max_concurrent_partitions = max_concurrent_partitions
        self.partition_size = partition_size
        self.max_size_mb = max_size_mb

        if mode == "manual":
            logger.info(f"Manual partition mode: using {self.num_partitions} partitions")
        else:  # auto mode
            logger.info(f"Auto partition mode: will determine optimal partitioning based on data characteristics")
            logger.info(f"Fallback partition size: {self.partition_size} samples, max {self.max_size_mb} MB")

    def _configure_auto_partitioning(self, dataset, ops):
        """Configure partitioning using the partition size optimizer for auto mode."""
        try:
            from data_juicer.core.executor.partition_size_optimizer import (
                auto_configure_resources,
            )

            logger.info("🔧 Auto-configuring partition settings based on data characteristics...")

            # Use the partition size optimizer to determine optimal settings
            recommendations = auto_configure_resources(self.cfg, dataset, ops)

            # Update partition configuration based on recommendations
            recommended_size = ConfigAccessor.get(recommendations, "recommended_partition_size", self.partition_size)
            recommended_max_size_mb = ConfigAccessor.get(recommendations, "recommended_max_size_mb", self.max_size_mb)
            recommended_workers = ConfigAccessor.get(
                recommendations, "recommended_worker_count", getattr(self.cfg, "np", 4)
            )

            # Calculate optimal number of partitions based on dataset size and recommended partition size
            try:
                if hasattr(dataset, "count"):
                    total_samples = dataset.count()
                elif hasattr(dataset, "__len__"):
                    total_samples = len(dataset)
                else:
                    total_samples = 10000  # Fallback estimate

                # Calculate number of partitions needed
                self.num_partitions = max(1, int(total_samples / recommended_size))

                # Cap partitions at 2x recommended workers (scales with cluster size)
                max_partitions = max(32, recommended_workers * 2)
                self.num_partitions = min(self.num_partitions, max_partitions)

                logger.info(f"📊 Dataset analysis complete:")
                logger.info(f"  Total samples: {total_samples}")
                logger.info(f"  Recommended partition size: {recommended_size} samples")
                logger.info(f"  Calculated partitions: {self.num_partitions}")
                logger.info(f"  Recommended max size: {recommended_max_size_mb} MB")
                logger.info(f"  Recommended workers: {recommended_workers}")

                # Update worker count if not already set
                if not hasattr(self.cfg, "np") or self.cfg.np is None:
                    self.cfg.np = recommended_workers
                    logger.info(f"  Updated worker count to: {recommended_workers}")

            except Exception as e:
                logger.warning(f"Could not determine dataset size for partition calculation: {e}")
                logger.info(f"Using fallback partition count: {self.num_partitions}")

        except ImportError as e:
            logger.warning(f"Could not import partition size optimizer: {e}")
            logger.info("Falling back to manual partition configuration")
        except Exception as e:
            logger.warning(f"Auto partition configuration failed: {e}")
            logger.info("Falling back to manual partition configuration")

    def run(self, load_data_np: Optional[PositiveInt] = None, skip_return=False):
        """
        Run the simplified partitioned dataset processing pipeline.

        Args:
            load_data_np: Number of workers for loading dataset
            skip_return: Whether to skip returning the dataset
            job_id: Optional job ID to resume from checkpoints

        Returns:
            Processed dataset
        """
        # Use TempDirManager to ensure cleanup of temporary files
        with TempDirManager(self.tmp_dir):
            return self._run_impl(load_data_np, skip_return)

    def _run_impl(self, load_data_np: Optional[PositiveInt] = None, skip_return=False):
        """
        Internal implementation of the run method.
        """
        job_start_time = time.time()

        # Check if user provided a job_id (indicating resumption attempt)
        user_provided_job_id = getattr(self.cfg, "_user_provided_job_id", False)
        resume_requested = getattr(self.cfg, "_resume_requested", False)

        if user_provided_job_id and self.job_id:
            logger.info(f"🔄 User provided job_id: {self.job_id} - attempting to resume job")
            resume_result = self._resume_job(self.job_id)
            if resume_result == "completed":
                logger.info("✅ Job is already completed - nothing to do")
                return None  # Exit gracefully
            elif resume_result == "resuming":
                logger.info("✅ Job resumption successful - will use existing checkpoints")
                is_resuming = True
            else:  # resume_result == "failed"
                if resume_requested:
                    raise RuntimeError(
                        f"Unable to resume job {self.job_id}. " "The existing checkpoints were left unchanged."
                    )
                logger.info("❌ Job resumption failed - starting fresh")
                is_resuming = False
        else:
            if self.job_id:
                logger.info(f"🚀 Starting new job with auto-generated job_id: {self.job_id}")
            else:
                logger.info("🚀 Starting new job")
            is_resuming = False
            if self.job_id and self.ckpt_manager.checkpoint_enabled:
                logger.info(
                    f"Resume token: {self.job_id}. "
                    f"Rerun the original command with --resume {self.job_id} to resume this job."
                )

        if not is_resuming:
            logger.info("🚀 Starting simplified partitioned processing...")
        else:
            logger.info("🔄 Resuming partitioned processing from checkpoints...")

        # Log job start event
        self._log_event(
            event_type=EventType.JOB_START,
            message=(
                "Starting partitioned dataset processing"
                if not is_resuming
                else "Resuming partitioned dataset processing"
            ),
            metadata={
                "num_partitions": self.num_partitions,
                "checkpoint_enabled": self.ckpt_manager.checkpoint_enabled,
                "is_resuming": is_resuming,
                "job_id": self.job_id,
                "user_provided_job_id": user_provided_job_id,
                "resume_requested": resume_requested,
            },
        )

        # Note: Config validation is handled in _resume_job() if resuming

        # Load the full dataset using a single DatasetBuilder
        logger.info("Loading dataset with single DatasetBuilder...")

        # Ray Dataset captures a copy of DataContext when it is created. This
        # must therefore happen before DatasetBuilder.load_dataset() so saved
        # row boundaries have the same meaning when a job is resumed.
        self._enable_deterministic_execution()
        override_num_blocks = getattr(self.cfg, "override_num_blocks", None)
        dataset = self.datasetbuilder.load_dataset(num_proc=load_data_np, override_num_blocks=override_num_blocks)
        columns = dataset.schema().columns

        # Prepare operations
        logger.info("Preparing operations...")
        ops = self._prepare_operators()

        # A resumed job must reuse the saved partition count before DAG
        # initialization. Auto mode can otherwise choose a different count if
        # the Ray cluster resources changed between runs.
        if resume_requested:
            saved_info = self._load_partitioning_info()
            if saved_info is None:
                raise RuntimeError(
                    "Explicit resume requires saved partitioning_info.json; "
                    "existing checkpoints were left unchanged."
                )
            self.num_partitions = saved_info.num_partitions
            logger.info(f"Using saved partition count for resume: {self.num_partitions}")
        # Handle auto partition mode BEFORE initializing DAG
        # (DAG needs final partition count)
        elif self.partition_mode == "auto":
            self._configure_auto_partitioning(dataset, ops)

        # Record explicit actor-pool budgets and calculate automatic Ray
        # operator parallelism once on the driver. Every partition shares the
        # same operator objects, so resource planning must finish before the
        # concurrent partition threads start.
        self._configure_operator_parallelism(ops)
        self._resolve_max_concurrent_partitions(ops)

        # Initialize DAG execution planning with final partition count
        # Pass ops to avoid redundant loading
        self._initialize_dag_execution(self.cfg, ops=ops)

        # Log job start with DAG context
        # Handle both dataset_path (string) and dataset (dict) configurations
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
            "parallel_groups_count": len(self.pipeline_dag.parallel_groups) if self.pipeline_dag else 0,
        }
        self.log_job_start(job_config, len(ops))

        # Detect convergence points for global operations
        convergence_points = self._detect_convergence_points(self.cfg)

        if convergence_points:
            logger.info(f"Found convergence points at operations: {convergence_points}")
            final_dataset = self._process_with_convergence(dataset, ops, convergence_points)
        else:
            logger.info("No convergence points found, processing with simple partitioning")
            final_dataset = self._process_with_simple_partitioning(dataset, ops)

        # Export final dataset
        logger.info("Exporting final dataset...")
        self.exporter.export(final_dataset.data, columns=columns)

        job_duration = time.time() - job_start_time
        logger.info(f"✅ Job completed successfully in {job_duration:.2f}s")
        logger.info(f"📁 Output saved to: {self.cfg.export_path}")

        # Log job completion with DAG context
        self.log_job_complete(job_duration, self.cfg.export_path)

        if skip_return:
            return None

        return final_dataset

    def cleanup_temp_files(self):
        """Manually clean up temporary files from previous runs."""
        tmp_base_dir = os.path.join(self.work_dir, ".tmp")
        if os.path.exists(tmp_base_dir):
            logger.info(f"Cleaning up temporary files in {tmp_base_dir}")
            try:
                shutil.rmtree(tmp_base_dir)
            except OSError as e:
                logger.warning(f"Remove tmp dir with shutil.rmtree() failed: {e}, " "will try os.rmdir()")
                os.rmdir(tmp_base_dir)
            logger.info("Temporary files cleaned up successfully")
        else:
            logger.info("No temporary files found to clean up")

    def _process_with_simple_partitioning(self, dataset: RayDataset, ops: List):
        """
        Process dataset with real partitioning using Ray Data's split and union.

        Uses deterministic splitting to ensure reproducible partitions for
        checkpoint resumption.
        """
        logger.info("Processing with real partitioning using Ray Data's split and union...")

        # Split the dataset deterministically with metadata collection
        partitions, partitioning_info = self._split_dataset_deterministic(dataset)
        logger.info(
            f"Partitioning complete: {partitioning_info.num_partitions} partitions, "
            f"{partitioning_info.total_rows} total rows"
        )

        # Process partitions concurrently. Each worker thread drives one Ray
        # Dataset execution; the actual data processing still runs on Ray.
        requested_max_workers = min(len(partitions), self.max_concurrent_partitions)
        max_workers = self._limit_partition_workers_for_explicit_actors(ops, requested_max_workers)
        logger.info(
            f"Processing {len(partitions)} partitions with up to {max_workers} concurrent "
            "driver threads and checkpointing support..."
        )
        processed_partitions = [None] * len(partitions)
        original_concurrency = self._scale_operator_parallelism(ops, max_workers)

        try:
            # Capture the per-partition resource plan in an isolated template.
            # Restore the original operators before any worker starts so later
            # convergence stages never depend on thread-pool shutdown timing.
            isolate_operators = max_workers > 1
            partition_ops_template = self._clone_partition_operators(ops) if isolate_operators else ops
        finally:
            self._restore_operator_parallelism(original_concurrency)

        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ray-partition") as executor:
            futures = {
                executor.submit(
                    self._process_partition,
                    partition,
                    i,
                    len(partitions),
                    partition_ops_template,
                    isolate_operators,
                ): i
                for i, partition in enumerate(partitions)
            }
            try:
                for future in as_completed(futures):
                    partition_id, processed_data = future.result()
                    processed_partitions[partition_id] = processed_data
            except BaseException:
                # Running Ray Dataset executions cannot be forcefully
                # interrupted here, but queued driver work should not start
                # after the job has already failed.
                for future in futures:
                    future.cancel()
                raise

        # Merge all processed partitions back into a single dataset
        logger.info("Merging processed partitions...")
        if len(processed_partitions) == 1:
            merged_dataset = processed_partitions[0]
        else:
            # Union all partitions
            merged_dataset = processed_partitions[0]
            for partition in processed_partitions[1:]:
                merged_dataset = merged_dataset.union(partition)

        # Return as RayDataset wrapper
        return RayDataset(merged_dataset, cfg=self.cfg)

    def _process_partition(
        self,
        partition,
        partition_id: int,
        total_partitions: int,
        ops: List,
        isolate_operators: bool = True,
    ):
        """Process one partition while preserving its position in the final union."""
        logger.info(f"Processing partition {partition_id + 1}/{total_partitions}")

        self._log_event(
            event_type=EventType.PARTITION_START,
            message=f"Starting processing of partition {partition_id + 1}/{total_partitions}",
            partition_id=partition_id,
        )

        try:
            # RayDataset may temporarily mutate operator attributes while
            # building an execution plan (for example runtime_env fallback and
            # tracing). Give every concurrent partition its own instances.
            partition_ops = self._clone_partition_operators(ops) if isolate_operators else ops
            partition_dataset = self._wrap_with_precomputed_parallelism(partition)
            processed_partition = self._process_with_checkpointing(partition_dataset, partition_id, partition_ops)
        except Exception as error:
            self.log_partition_failed(partition_id, str(error), retry_count=0)
            raise

        self._log_event(
            event_type=EventType.PARTITION_COMPLETE,
            message=f"Completed processing of partition {partition_id + 1}/{total_partitions}",
            partition_id=partition_id,
        )

        return partition_id, processed_partition.data

    @staticmethod
    def _clone_partition_operators(ops: List) -> List:
        """Create an isolated operator graph for one partition."""
        try:
            return copy.deepcopy(ops)
        except Exception as error:
            raise RuntimeError(
                "Failed to isolate operators for concurrent partition execution. "
                "All partition operators must support deep copying."
            ) from error

    def _configure_operator_parallelism(self, ops: List) -> None:
        """Plan global actor resources once before partition threads.

        A positive explicit actor ``num_proc`` is treated as a global job
        budget, matching its meaning in the non-partitioned Ray executor. Auto
        parallelism is also calculated globally here and divided later.
        """
        self._auto_parallel_op_ids = set()
        self._explicit_actor_op_ids = {
            id(op) for op in ops if op.use_ray_actor() and self._actor_pool_capacity(op.num_proc) is not None
        }

        if not ConfigAccessor.get(self.cfg, "auto_op_parallelism", True):
            return

        auto_ops = [op for op in ops if op.use_auto_proc()]
        if not auto_ops:
            return

        from data_juicer.utils.process_utils import calculate_ray_np

        self._auto_parallel_op_ids = {id(op) for op in auto_ops}
        calculate_ray_np(ops)

    def _configure_auto_operator_parallelism(self, ops: List) -> None:
        """Compatibility wrapper for the former auto-only planner."""
        self._configure_operator_parallelism(ops)

    def _resolve_max_concurrent_partitions(self, ops: List) -> int:
        """Resolve automatic driver concurrency after operator planning.

        A GPU pipeline is bounded by the tightest per-worker CPU/GPU
        requirement. CPU-only pipelines use a small outer-pipeline cap because
        Ray Data can already parallelize work inside each partition.
        """
        raw_value = self.max_concurrent_partitions
        if not (isinstance(raw_value, str) and raw_value.lower() == "auto"):
            self.max_concurrent_partitions = max(1, int(raw_value))
            return self.max_concurrent_partitions

        try:
            cluster_resources = ray.cluster_resources()
            total_cpus = float(cluster_resources.get("CPU", 0))
            total_gpus = float(cluster_resources.get("GPU", 0))
        except Exception as error:
            logger.warning(
                "Could not inspect Ray cluster resources for automatic partition "
                f"concurrency; falling back to 1. Error: {error}"
            )
            self.max_concurrent_partitions = 1
            return self.max_concurrent_partitions

        if total_cpus <= 0:
            logger.warning(
                "Ray cluster reports no CPU resources for automatic partition " "concurrency; falling back to 1."
            )
            self.max_concurrent_partitions = 1
            return self.max_concurrent_partitions

        capacities = []
        gpu_operator_names = []
        for op in ops:
            num_cpus = getattr(op, "num_cpus", None)
            cpu_per_worker = float(num_cpus) if num_cpus and float(num_cpus) > 0 else 1.0
            capacity = math.floor(total_cpus / cpu_per_worker + 1e-9)

            num_gpus = getattr(op, "num_gpus", None)
            gpu_per_worker = float(num_gpus) if num_gpus and float(num_gpus) > 0 else None
            if gpu_per_worker is None:
                try:
                    if op.use_cuda():
                        gpu_per_worker = 1.0
                except (AttributeError, RuntimeError):
                    pass

            if gpu_per_worker is not None:
                gpu_operator_names.append(getattr(op, "_name", type(op).__name__))
                gpu_capacity = math.floor(total_gpus / gpu_per_worker + 1e-9)
                capacity = min(capacity, gpu_capacity)

            capacities.append(capacity)

        resource_capacity = min(capacities) if capacities else math.floor(total_cpus)
        if resource_capacity < 1:
            logger.warning(
                "Ray cluster resources cannot host one worker for every operator "
                "with the current resource requests; using one partition pipeline "
                "so Ray can surface the underlying scheduling error."
            )
            resource_capacity = 1

        if gpu_operator_names:
            resolved = resource_capacity
            workload = f"GPU operators: {', '.join(gpu_operator_names)}"
        else:
            resolved = min(resource_capacity, _AUTO_CPU_PARTITION_CAP)
            workload = f"CPU-only pipeline cap: {_AUTO_CPU_PARTITION_CAP}"

        self.max_concurrent_partitions = max(1, resolved)
        logger.info(
            "Auto-configured max_concurrent_partitions="
            f"{self.max_concurrent_partitions} from Ray resources "
            f"(CPU={total_cpus:g}, GPU={total_gpus:g}; {workload})"
        )
        return self.max_concurrent_partitions

    @staticmethod
    def _actor_pool_capacity(concurrency) -> Optional[int]:
        """Return the maximum size represented by an actor concurrency value."""
        if isinstance(concurrency, bool):
            return None
        if isinstance(concurrency, int) and concurrency > 0:
            return concurrency
        if isinstance(concurrency, (tuple, list)) and len(concurrency) in (2, 3):
            max_size = concurrency[1]
            if isinstance(max_size, int) and not isinstance(max_size, bool) and max_size > 0:
                return max_size
        return None

    def _limit_partition_workers_for_explicit_actors(self, ops: List, requested_max_workers: int) -> int:
        """Keep concurrent partitions within every explicit actor-pool budget."""
        explicit_actor_op_ids = getattr(self, "_explicit_actor_op_ids", set())
        actor_budgets = [
            (op._name, self._actor_pool_capacity(op.num_proc)) for op in ops if id(op) in explicit_actor_op_ids
        ]
        actor_budgets = [(name, budget) for name, budget in actor_budgets if budget is not None]
        if not actor_budgets:
            return requested_max_workers

        explicit_limit = min(budget for _, budget in actor_budgets)
        max_workers = max(1, min(requested_max_workers, explicit_limit))
        if max_workers < requested_max_workers:
            budget_summary = ", ".join(f"{name}={budget}" for name, budget in actor_budgets)
            logger.warning(
                f"Reducing concurrent partition workers from {requested_max_workers} to {max_workers} "
                f"to honor explicit global actor num_proc budget(s): {budget_summary}"
            )
        return max_workers

    @staticmethod
    def _scale_concurrency_for_partitions(concurrency, max_workers: int, *, round_up: bool = True):
        """Share a global actor pool size across concurrent partitions."""
        if max_workers <= 1 or concurrency is None:
            return concurrency

        def _scale(value):
            if value is None:
                return None
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                return value
            if round_up:
                return max(1, math.ceil(value / max_workers))
            return max(1, value // max_workers)

        if isinstance(concurrency, (tuple, list)):
            scaled = [_scale(value) for value in concurrency]
            if len(scaled) >= 2 and isinstance(scaled[0], int) and isinstance(scaled[1], int):
                scaled[0] = min(scaled[0], scaled[1])
            if (
                len(scaled) == 3
                and isinstance(scaled[0], int)
                and isinstance(scaled[1], int)
                and isinstance(scaled[2], int)
            ):
                scaled[2] = min(max(scaled[2], scaled[0]), scaled[1])
            return tuple(scaled)
        if isinstance(concurrency, int) and concurrency > 0:
            return _scale(concurrency)
        return concurrency

    def _scale_operator_parallelism(self, ops: List, max_workers: int):
        """Convert global auto and explicit actor budgets to per-partition values."""
        original_concurrency = []
        auto_parallel_op_ids = getattr(self, "_auto_parallel_op_ids", set())
        explicit_actor_op_ids = getattr(self, "_explicit_actor_op_ids", set())
        for op in ops:
            if id(op) in auto_parallel_op_ids:
                scaling_mode = "automatic"
                round_up = True
            elif id(op) in explicit_actor_op_ids:
                scaling_mode = "explicit"
                round_up = False
            else:
                continue

            original_concurrency.append((op, op.num_proc))
            op.num_proc = self._scale_concurrency_for_partitions(
                op.num_proc,
                max_workers,
                round_up=round_up,
            )
            if op.num_proc != original_concurrency[-1][1]:
                logger.info(
                    f"Op[{op._name}] {scaling_mode} global concurrency: "
                    f"{original_concurrency[-1][1]} -> {op.num_proc} "
                    f"across {max_workers} concurrent partitions"
                )
        return original_concurrency

    def _scale_auto_operator_parallelism(self, ops: List, max_workers: int):
        """Compatibility wrapper for the former auto-only scaler."""
        return self._scale_operator_parallelism(ops, max_workers)

    @staticmethod
    def _restore_operator_parallelism(original_concurrency) -> None:
        """Restore the global plan for convergence or later execution stages."""
        for op, concurrency in original_concurrency:
            op.num_proc = concurrency

    @staticmethod
    def _restore_auto_operator_parallelism(original_concurrency) -> None:
        """Compatibility wrapper for the former auto-only restorer."""
        PartitionedRayExecutor._restore_operator_parallelism(original_concurrency)

    def _wrap_with_precomputed_parallelism(self, dataset) -> RayDataset:
        """Wrap data without recalculating shared operator resources."""
        wrapped = RayDataset(dataset, cfg=self.cfg)
        wrapped._auto_proc = False
        return wrapped

    def _process_with_convergence(self, dataset: RayDataset, ops: List, convergence_points: List[int]):
        """
        Process dataset with convergence support for global operations.
        """
        logger.info("Processing with convergence support for global operations...")

        # Find the first convergence point
        first_convergence = min(convergence_points)
        logger.info(f"First convergence point at operation {first_convergence}")

        # Split operations into pre-convergence and post-convergence
        pre_convergence_ops = ops[:first_convergence]
        post_convergence_ops = ops[first_convergence:]

        logger.info(f"Pre-convergence operations: {len(pre_convergence_ops)}")
        logger.info(f"Post-convergence operations: {len(post_convergence_ops)}")

        # Process partitions up to convergence point
        if pre_convergence_ops:
            logger.info("Processing partitions up to convergence point...")
            processed_dataset = self._process_with_simple_partitioning(dataset, pre_convergence_ops)
        else:
            logger.info("No pre-convergence operations, using original dataset...")
            processed_dataset = dataset

        # Merge partitions for global operations
        logger.info("Merging partitions for global operations...")
        merged_dataset = processed_dataset.data

        # Process merged dataset with post-convergence operations
        if post_convergence_ops:
            logger.info("Processing merged dataset with global operations...")
            merged_ray_dataset = self._wrap_with_precomputed_parallelism(merged_dataset)

            # Pre-execute DAG monitoring (log operation start events)
            if self.pipeline_dag:
                self._pre_execute_operations_with_dag_monitoring(post_convergence_ops, partition_id=0)

            # Execute operations
            final_dataset = merged_ray_dataset.process(post_convergence_ops)

            # Post-execute DAG monitoring (log operation completion events)
            if self.pipeline_dag:
                self._post_execute_operations_with_dag_monitoring(post_convergence_ops, partition_id=0)

            logger.info("Global operations completed. Final dataset ready for export")
            return final_dataset
        else:
            # No post-convergence operations, just return the merged result
            return RayDataset(merged_dataset, cfg=self.cfg)

    def _process_with_checkpointing(self, dataset: RayDataset, partition_id: int, ops: List) -> RayDataset:
        """
        Process dataset with checkpointing support.
        Groups operations and checkpoints between groups based on strategy.
        """
        logger.info(f"Processing partition {partition_id} with checkpointing support...")

        if not self.ckpt_manager.checkpoint_enabled:
            logger.info(f"Checkpointing disabled, processing all operations at once for partition {partition_id}")

            # Get input row count before processing
            input_rows = dataset.data.count()
            start_time = time.time()

            # Pre-execute DAG monitoring (log operation start events)
            if self.pipeline_dag:
                self._pre_execute_operations_with_dag_monitoring(ops, partition_id=partition_id)

            # Execute operations (lazy)
            processed_dataset = dataset.process(ops)

            # Force materialization to get real execution (required for union anyway)
            processed_dataset.data = processed_dataset.data.materialize()

            # Get metrics after execution
            duration = time.time() - start_time
            output_rows = processed_dataset.data.count()

            logger.info(f"Partition {partition_id}: Processed {input_rows}→{output_rows} rows in {duration:.2f}s")

            # Post-execute DAG monitoring with real metrics
            if self.pipeline_dag:
                metrics = {"duration": duration, "input_rows": input_rows, "output_rows": output_rows}
                self._post_execute_operations_with_dag_monitoring(ops, partition_id=partition_id, metrics=metrics)

            return processed_dataset

        # check the latest checkpoint for the partition
        latest_checkpoint = self.ckpt_manager.find_latest_checkpoint(partition_id)

        # Group operations based on checkpoint strategy
        op_groups = self.ckpt_manager.group_operations_for_checkpointing(ops)
        logger.info(f"Grouped {len(ops)} operations into {len(op_groups)} groups for checkpointing")
        logger.info(f"Detailed op groups: {op_groups}")

        current_dataset = dataset

        for group_idx, (start_idx, end_idx, group_ops) in enumerate(op_groups):
            logger.info(
                f"Processing partition {partition_id}, group {group_idx + 1}/{len(op_groups)}: operations {start_idx}-{end_idx-1}"
            )

            if latest_checkpoint and latest_checkpoint[0] >= end_idx:
                logger.info(
                    f"Partition {partition_id}: All operations in group {group_idx + 1} already processed (checkpoint at op {latest_checkpoint[0]}, group ends at {end_idx-1}), skipping"
                )
                continue

            if latest_checkpoint and latest_checkpoint[0] >= start_idx:
                logger.info(f"Partition {partition_id}: Resuming from checkpoint at operation {latest_checkpoint[0]}")
                current_dataset = self.ckpt_manager.load_checkpoint(
                    latest_checkpoint[0], latest_checkpoint[1], partition_id, cfg=self.cfg
                )
                if current_dataset is None:
                    logger.warning(f"Partition {partition_id}: Failed to load checkpoint, starting from beginning")
                    current_dataset = dataset
                    group_ops = ops[start_idx:end_idx]  # Start from beginning of group
                    logger.info(
                        f"Partition {partition_id}: Will process {len(group_ops)} operations from beginning of group"
                    )
                else:
                    current_dataset._auto_proc = False
                    logger.info(
                        f"Partition {partition_id}: Successfully loaded checkpoint, resuming from operation {latest_checkpoint[0] + 1}"
                    )
                    group_ops = ops[latest_checkpoint[0] + 1 : end_idx]  # Resume from checkpoint
                    if not group_ops:
                        logger.info(
                            f"Partition {partition_id}: All operations in this group already processed, skipping"
                        )
                        continue
                    else:
                        logger.info(
                            f"Partition {partition_id}: Will process {len(group_ops)} remaining operations from checkpoint"
                        )

            # Process the group of operations
            if group_ops:
                logger.info(
                    f"Partition {partition_id}: Processing {len(group_ops)} operations in group {group_idx + 1}"
                )

                # Get input row count before processing
                input_rows = current_dataset.data.count()
                start_time = time.time()

                # Pre-execute DAG monitoring (log operation start events)
                if self.pipeline_dag:
                    self._pre_execute_operations_with_dag_monitoring(group_ops, partition_id=partition_id)

                # Execute operations (lazy)
                current_dataset = current_dataset.process(group_ops)

                # Force materialization (required for checkpointing anyway)
                current_dataset.data = current_dataset.data.materialize()

                # Get metrics after execution
                duration = time.time() - start_time
                output_rows = current_dataset.data.count()

                logger.info(
                    f"Partition {partition_id}, group {group_idx + 1}: Processed {input_rows}→{output_rows} rows in {duration:.2f}s"
                )

                # Post-execute DAG monitoring with real metrics
                if self.pipeline_dag:
                    metrics = {"duration": duration, "input_rows": input_rows, "output_rows": output_rows}
                    self._post_execute_operations_with_dag_monitoring(
                        group_ops, partition_id=partition_id, metrics=metrics
                    )

            # Checkpoint after the last operation in the group
            if group_ops:
                last_op_idx = end_idx - 1
                last_op_name = ops[last_op_idx]._name
                if self.ckpt_manager.should_checkpoint(last_op_idx, last_op_name):
                    logger.info(
                        f"Partition {partition_id}: Creating checkpoint after operation {last_op_idx}: {last_op_name}"
                    )
                    # Data already materialized above, safe to checkpoint
                    self.ckpt_manager.save_checkpoint(
                        current_dataset, last_op_idx, last_op_name, partition_id, cfg=self.cfg
                    )

        return current_dataset

    def _find_work_directory(self, job_id: str) -> Optional[str]:
        """Find the work directory based on job_id."""
        # Check if the current work_dir already contains the job_id
        current_work_dir = Path(self.work_dir)
        logger.info(f"Checking if current work_dir contains job_id: {current_work_dir}")

        if job_id in str(current_work_dir):
            # Current work_dir already contains job_id, check if it's a valid work directory
            logger.info(f"Current work_dir contains job_id '{job_id}', checking if it's a valid work directory")

            # Check if this directory has events files (indicating it's a work directory)
            latest_events_file = self.event_logger.find_latest_events_file(str(current_work_dir))
            if latest_events_file:
                logger.info(f"Found events file in current work_dir: {latest_events_file}")
                return str(current_work_dir)

            logger.warning(f"No events file found in current work_dir: {current_work_dir}")

        logger.warning(f"No directory found containing job_id '{job_id}' with events files")
        return None

    def _check_job_completion(self, work_dir: str, job_id: str) -> bool:
        """Check if the job is already completed."""
        latest_events_file = self.event_logger.find_latest_events_file(work_dir)
        if not latest_events_file:
            logger.info(f"No events file found in work directory: {work_dir}")
            return False

        is_completed = self.event_logger.check_job_completion(latest_events_file)
        if is_completed:
            logger.info(f"Job {job_id} is already completed - no need to resume")
        else:
            logger.info(f"Job {job_id} is not completed - resumption possible")

        return is_completed

    def _resume_job(self, job_id: str) -> str:
        """Resume a job from checkpoints.

        Returns:
            "completed": Job is already completed
            "resuming": Job can be resumed
            "failed": Job resumption failed
        """
        logger.info(f"Attempting to resume job: {job_id}")

        # Find work directory
        work_dir = self._find_work_directory(job_id)
        if not work_dir:
            logger.error(f"Work directory not found for job_id: {job_id}")
            return "failed"

        logger.info(f"Found work directory: {work_dir}")

        # Check if config validation passed (done during config initialization)
        if not getattr(self.cfg, "_same_yaml_config", False):
            logger.error("Config validation failed - configurations don't match")
            return "failed"

        # Check if job is already completed
        if self._check_job_completion(work_dir, job_id):
            return "completed"  # Job already completed

        # Update checkpoint directory to use the work directory's checkpoint directory
        work_checkpoint_dir = os.path.join(work_dir, "checkpoints")
        if os.path.exists(work_checkpoint_dir):
            self.ckpt_manager.ckpt_dir = work_checkpoint_dir
            logger.info(f"Using checkpoint directory from work directory: {self.ckpt_manager.ckpt_dir}")
        else:
            logger.warning(f"No checkpoint directory found in work directory: {work_checkpoint_dir}")

        return "resuming"

    def _prepare_operators(self):
        """Prepare process operators."""
        ops = load_ops(self.cfg.process)

        # Check for op_fusion configuration with safe attribute access
        if hasattr(self.cfg, "op_fusion") and self.cfg.op_fusion:
            logger.info(f"Start OP fusion and reordering with strategy [{self.cfg.fusion_strategy}]...")
            ops = fuse_operators(
                ops,
                mapper_fusion=getattr(self.cfg, "mapper_fusion", True),
                mapper_fusion_vram_limit=getattr(self.cfg, "mapper_fusion_vram_limit", 0.9),
            )

        return ops

    def _override_strategy_methods(self):
        """Override strategy methods for partitioned execution."""
        # Override DAG-related methods for partitioned execution
        # Note: Partition count is determined by the executor (self.num_partitions),
        # not by the DAG mixin, so we don't override _determine_partition_count here
        # Note: _detect_convergence_points is reused from DAGExecutionMixin (no override needed)
        self._get_dag_node_for_operation = self._get_dag_node_for_operation_partitioned

    def _get_dag_node_for_operation_partitioned(
        self, op_name: str, op_idx: int, partition_id: int = 0, **kwargs
    ) -> Optional[str]:
        """Get DAG node ID for partitioned operation."""
        if not self.dag_execution_strategy:
            return None

        return self.dag_execution_strategy.get_dag_node_id(op_name, op_idx, partition_id=partition_id, **kwargs)

    # ========== Deterministic Partitioning Methods ==========

    def _enable_deterministic_execution(self) -> None:
        """Enable deterministic execution order in Ray Data.

        This keeps the global row order stable so saved row boundaries can be
        reapplied with split_at_indices() during explicit resume.
        """
        try:
            ctx = ray.data.DataContext.get_current()
            ctx.execution_options.preserve_order = True
            logger.info("Enabled deterministic execution (preserve_order=True)")
        except Exception as e:
            logger.warning(f"Could not enable deterministic execution: {e}")

    def _compute_row_hash(self, row: Dict) -> str:
        """Compute a hash of a row for partition validation.

        Uses a stable JSON serialization to ensure consistent hashing.
        """
        # Sort keys for deterministic serialization
        try:
            row_str = json.dumps(row, sort_keys=True, default=str)
            return hashlib.md5(row_str.encode()).hexdigest()[:16]
        except Exception:
            # Fallback for non-serializable rows
            return hashlib.md5(str(row).encode()).hexdigest()[:16]

    def _collect_partition_metadata(
        self,
        partition,
        partition_id: int,
        start_row: Optional[int] = None,
        compute_content_hash: bool = True,
    ) -> PartitionMetadata:
        """Collect metadata from a partition for validation on resume.

        The complete content hash is independent of Ray block boundaries and
        row order within the partition. It is used for strict explicit resume
        validation; the first-row hash remains for backward compatibility.
        """
        if compute_content_hash:
            content_hash, row_count = self._compute_partition_content_hash(partition)
        else:
            content_hash = ""
            row_count = partition.count()

        # Get first row for hashing (cheap operation)
        first_row_hash = ""

        try:
            first_rows = partition.take(1)
            if first_rows:
                first_row_hash = self._compute_row_hash(first_rows[0])
        except Exception as e:
            logger.warning(f"Could not compute row hash for partition {partition_id}: {e}")

        return PartitionMetadata(
            partition_id=partition_id,
            row_count=row_count,
            first_row_hash=first_row_hash,
            last_row_hash="",  # Skip last_row_hash for efficiency
            content_hash=content_hash,
            start_row=start_row,
            end_row=start_row + row_count if start_row is not None else None,
        )

    def _compute_partition_content_hash(self, partition) -> tuple:
        """Compute a block-boundary-independent hash of a partition."""
        partials = partition.map_batches(_hash_partition_batch, batch_format="pyarrow").take_all()
        return _combine_partition_hash_partials(partials)

    def _get_partitioning_info_path(self) -> str:
        """Get the path to the partitioning info file."""
        return os.path.join(self.ckpt_manager.ckpt_dir, "partitioning_info.json")

    def _save_partitioning_info(self, info: PartitioningInfo) -> None:
        """Save partitioning info alongside checkpoints."""
        os.makedirs(self.ckpt_manager.ckpt_dir, exist_ok=True)
        info.save(self._get_partitioning_info_path())

    def _load_partitioning_info(self) -> Optional[PartitioningInfo]:
        """Load partitioning info from checkpoint directory."""
        return PartitioningInfo.load(self._get_partitioning_info_path())

    def _validate_partitions(self, partitions: List, saved_info: PartitioningInfo) -> bool:
        """Validate that current partitions match saved partitioning info.

        Returns True if partitions match (safe to use checkpoints),
        False if there's a mismatch (must restart from scratch).

        Validation checks:
        1. Partition count matches
        2. Row count per partition matches
        3. Complete partition content hash matches when available
        4. First row hash matches for backward compatibility
        """
        if len(partitions) != saved_info.num_partitions:
            logger.error(f"Partition count mismatch: current={len(partitions)}, " f"saved={saved_info.num_partitions}")
            return False

        for i, partition in enumerate(partitions):
            saved_meta = saved_info.partitions[i] if i < len(saved_info.partitions) else None

            if saved_meta is None:
                logger.error(f"No saved metadata for partition {i}")
                return False

            if saved_meta.content_hash:
                current_hash, current_count = self._compute_partition_content_hash(partition)
            else:
                current_hash = ""
                current_count = partition.count()

            if current_count != saved_meta.row_count:
                logger.error(
                    f"Partition {i} row count mismatch: current={current_count}, " f"saved={saved_meta.row_count}"
                )
                return False

            if saved_meta.content_hash and current_hash != saved_meta.content_hash:
                logger.error(
                    f"Partition {i} content hash mismatch: " f"current={current_hash}, saved={saved_meta.content_hash}"
                )
                return False

            # Validate first row hash (skip if not available)
            if saved_meta.first_row_hash:
                try:
                    first_rows = partition.take(1)
                    if first_rows:
                        current_hash = self._compute_row_hash(first_rows[0])
                        if current_hash != saved_meta.first_row_hash:
                            logger.error(
                                f"Partition {i} first row hash mismatch: "
                                f"current={current_hash}, saved={saved_meta.first_row_hash}"
                            )
                            return False
                except Exception as e:
                    logger.warning(f"Could not validate partition {i} hash: {e}")

        logger.info("Partition validation passed - safe to use checkpoints")
        return True

    def _split_at_saved_boundaries(self, dataset: RayDataset, saved_info: PartitioningInfo) -> List:
        """Recreate partitions using row boundaries saved by the first run."""
        if saved_info.num_partitions != self.num_partitions:
            raise RuntimeError(
                "Cannot resume with a different partition count: "
                f"current={self.num_partitions}, saved={saved_info.num_partitions}"
            )
        if len(saved_info.partitions) != saved_info.num_partitions:
            raise RuntimeError(
                "Saved partition metadata is incomplete: "
                f"expected={saved_info.num_partitions}, actual={len(saved_info.partitions)}"
            )

        if saved_info.num_partitions == 1:
            return [dataset.data.materialize()]

        split_indices = []
        expected_start = 0
        for partition_id, meta in enumerate(saved_info.partitions):
            start_row = meta.start_row if meta.start_row is not None else expected_start
            end_row = meta.end_row if meta.end_row is not None else start_row + meta.row_count
            if start_row != expected_start or end_row - start_row != meta.row_count:
                raise RuntimeError(
                    f"Saved row boundaries are invalid for partition {partition_id}: "
                    f"start={start_row}, end={end_row}, rows={meta.row_count}, "
                    f"expected_start={expected_start}"
                )
            expected_start = end_row
            if partition_id < saved_info.num_partitions - 1:
                split_indices.append(end_row)

        if expected_start != saved_info.total_rows:
            raise RuntimeError(
                "Saved row boundaries do not cover the saved total row count: "
                f"boundaries_end={expected_start}, total_rows={saved_info.total_rows}"
            )

        logger.info(f"Recreating partitions at saved row boundaries: {split_indices}")
        return dataset.data.split_at_indices(split_indices)

    def _split_dataset_deterministic(self, dataset: RayDataset) -> tuple:
        """Split dataset deterministically and collect metadata.

        Returns:
            tuple: (partitions, partitioning_info)
        """
        # Enable deterministic execution
        self._enable_deterministic_execution()

        # Check for existing partitioning info (resumption case)
        saved_info = self._load_partitioning_info()
        resume_requested = getattr(self.cfg, "_resume_requested", False)

        if resume_requested:
            if saved_info is None:
                raise RuntimeError(
                    "Explicit resume requires saved partitioning_info.json; "
                    "existing checkpoints were left unchanged."
                )
            if saved_info.hash_algorithm != _PARTITION_CONTENT_HASH_ALGORITHM:
                raise RuntimeError(
                    "Explicit resume cannot validate the saved partition hash algorithm: "
                    f"saved={saved_info.hash_algorithm}, "
                    f"supported={_PARTITION_CONTENT_HASH_ALGORITHM}. "
                    "Existing checkpoints were left unchanged."
                )
            if any(
                not meta.content_hash or meta.start_row is None or meta.end_row is None
                for meta in saved_info.partitions
            ):
                raise RuntimeError(
                    "Explicit resume requires content hashes and row boundaries "
                    "created by this version. Existing checkpoints were left unchanged."
                )

            logger.info("Explicit resume requested; recreating saved partition row boundaries...")
            partitions = self._split_at_saved_boundaries(dataset, saved_info)
            logger.info(f"Recreated {len(partitions)} partitions from saved boundaries")
            if not self._validate_partitions(partitions, saved_info):
                raise RuntimeError(
                    "Saved partition content hashes do not match the current input. "
                    "Refusing to resume; existing checkpoints were left unchanged."
                )
            logger.info("Saved partition hashes validated successfully - resuming checkpoints")
            return partitions, saved_info

        # Split the dataset
        logger.info(f"Splitting dataset into {self.num_partitions} partitions (deterministic mode)...")
        partitions = dataset.data.split(self.num_partitions)
        logger.info(f"Created {len(partitions)} partitions")

        # If resuming, validate partitions match
        if saved_info is not None:
            logger.info("Found existing partitioning info, validating...")
            if self._validate_partitions(partitions, saved_info):
                logger.info("Partitions validated successfully - resuming with existing checkpoints")
                return partitions, saved_info
            else:
                logger.warning(
                    "Partition validation FAILED - partitions don't match saved info. "
                    "This can happen if the input data changed or Ray's internal state differs. "
                    "Clearing checkpoints and starting fresh."
                )
                self._clear_invalid_checkpoints()
                saved_info = None

        # Collect metadata for new partitions
        logger.info("Collecting partition metadata for checkpoint validation...")
        partition_metadata = []
        next_start_row = 0
        compute_content_hash = self.ckpt_manager.checkpoint_enabled

        for i, partition in enumerate(partitions):
            meta = self._collect_partition_metadata(
                partition,
                i,
                start_row=next_start_row,
                compute_content_hash=compute_content_hash,
            )
            partition_metadata.append(meta)
            next_start_row = meta.end_row
            logger.debug(f"Partition {i}: {meta.row_count} rows, hash={meta.first_row_hash[:8]}...")

        total_rows = sum(meta.row_count for meta in partition_metadata)

        partitioning_info = PartitioningInfo(
            num_partitions=self.num_partitions,
            total_rows=total_rows,
            partitions=partition_metadata,
            deterministic=True,
        )

        # Save partitioning info
        self._save_partitioning_info(partitioning_info)

        return partitions, partitioning_info

    def _clear_invalid_checkpoints(self) -> None:
        """Clear checkpoints when partition validation fails."""
        if os.path.exists(self.ckpt_manager.ckpt_dir):
            logger.warning(f"Clearing invalid checkpoints in {self.ckpt_manager.ckpt_dir}")
            try:
                shutil.rmtree(self.ckpt_manager.ckpt_dir)
            except OSError as e:
                logger.warning(f"Remove ckpt dir with shutil.rmtree() failed: {e}, " "will try os.rmdir()")
                os.rmdir(self.ckpt_manager.ckpt_dir)
            os.makedirs(self.ckpt_manager.ckpt_dir, exist_ok=True)

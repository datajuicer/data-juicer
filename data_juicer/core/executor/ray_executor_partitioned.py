"""
Simplified Partitioned Ray Executor for Large Dataset Processing

This module implements a streamlined partitioned execution strategy for Ray mode that:
2. Splits the dataset into manageable partitions using Ray's .split() method
3. Processes each partition independently with Ray tasks
4. Merges results back into a single dataset for export
5. Supports convergence points for global operations (like deduplicators)
"""

import hashlib
import json
import os
import shutil
import time
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
from data_juicer.utils.lazy_loader import LazyLoader

ray = LazyLoader("ray")


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

    def to_dict(self) -> Dict:
        return {
            "num_partitions": self.num_partitions,
            "total_rows": self.total_rows,
            "deterministic": self.deterministic,
            "partitions": [p.to_dict() for p in self.partitions],
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "PartitioningInfo":
        partitions = [PartitionMetadata.from_dict(p) for p in data.get("partitions", [])]
        return cls(
            num_partitions=data["num_partitions"],
            total_rows=data["total_rows"],
            deterministic=data.get("deterministic", True),
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
        self.work_dir = self.cfg.work_dir
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
        checkpoint_dir = getattr(self.cfg, "checkpoint_dir", os.path.join(self.work_dir, "checkpoints"))

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
            **export_extra_args,
        )

    def _configure_partitioning(self):
        """Configure partitioning based on manual or auto mode."""
        # Get partition configuration
        partition_cfg = getattr(self.cfg, "partition", {})

        # Use ConfigAccessor to handle both dict and object configurations
        mode = ConfigAccessor.get(partition_cfg, "mode", "auto")
        num_of_partitions = ConfigAccessor.get(partition_cfg, "num_of_partitions", 4)
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
                logger.info("❌ Job resumption failed - starting fresh")
                is_resuming = False
        else:
            if self.job_id:
                logger.info(f"🚀 Starting new job with auto-generated job_id: {self.job_id}")
            else:
                logger.info("🚀 Starting new job")
            is_resuming = False

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
            },
        )

        # Note: Config validation is handled in _resume_job() if resuming

        # Load the full dataset using a single DatasetBuilder
        logger.info("Loading dataset with single DatasetBuilder...")

        dataset = self.datasetbuilder.load_dataset(num_proc=load_data_np)
        columns = dataset.schema().columns

        # Prepare operations
        logger.info("Preparing operations...")
        ops = self._prepare_operators()

        # Handle auto partition mode BEFORE initializing DAG
        # (DAG needs final partition count)
        if self.partition_mode == "auto":
            self._configure_auto_partitioning(dataset, ops)

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
            shutil.rmtree(tmp_base_dir)
            logger.info("✅ Temporary files cleaned up successfully")
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

        # Process each partition separately with checkpointing
        logger.info("Processing partitions with checkpointing support...")
        processed_partitions = []

        for i, partition in enumerate(partitions):
            logger.info(f"Processing partition {i+1}/{len(partitions)}")

            # Log partition start event
            self._log_event(
                event_type=EventType.PARTITION_START,
                message=f"Starting processing of partition {i+1}/{len(partitions)}",
                partition_id=i,
            )

            # Create a RayDataset wrapper for this partition
            partition_dataset = RayDataset(partition, cfg=self.cfg)

            # Apply operations with checkpointing support and DAG monitoring
            processed_partition = self._process_with_checkpointing(partition_dataset, i, ops)

            # Store the processed partition's data
            processed_partitions.append(processed_partition.data)

            # Log partition completion event
            self._log_event(
                event_type=EventType.PARTITION_COMPLETE,
                message=f"Completed processing of partition {i+1}/{len(partitions)}",
                partition_id=i,
            )

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
            merged_ray_dataset = RayDataset(merged_dataset, cfg=self.cfg)

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
            ops = fuse_operators(ops)

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

        This ensures that split() produces the same partitions on re-runs,
        which is critical for checkpoint resumption.
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

    def _collect_partition_metadata(self, partition, partition_id: int) -> PartitionMetadata:
        """Collect metadata from a partition for validation on resume.

        Only collects first_row_hash (not last_row_hash) for efficiency.
        Getting the last row requires take(all_rows) which is expensive.
        First row hash + row count is sufficient for detecting most mismatches.
        """
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
        )

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
        3. First row hash matches (efficient validation)
        """
        if len(partitions) != saved_info.num_partitions:
            logger.error(f"Partition count mismatch: current={len(partitions)}, " f"saved={saved_info.num_partitions}")
            return False

        for i, partition in enumerate(partitions):
            current_count = partition.count()
            saved_meta = saved_info.partitions[i] if i < len(saved_info.partitions) else None

            if saved_meta is None:
                logger.warning(f"No saved metadata for partition {i}")
                continue

            if current_count != saved_meta.row_count:
                logger.error(
                    f"Partition {i} row count mismatch: current={current_count}, " f"saved={saved_meta.row_count}"
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

    def _split_dataset_deterministic(self, dataset: RayDataset) -> tuple:
        """Split dataset deterministically and collect metadata.

        Returns:
            tuple: (partitions, partitioning_info)
        """
        # Enable deterministic execution
        self._enable_deterministic_execution()

        # Check for existing partitioning info (resumption case)
        saved_info = self._load_partitioning_info()

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
        total_rows = sum(p.count() for p in partitions)
        partition_metadata = []

        for i, partition in enumerate(partitions):
            meta = self._collect_partition_metadata(partition, i)
            partition_metadata.append(meta)
            logger.debug(f"Partition {i}: {meta.row_count} rows, hash={meta.first_row_hash[:8]}...")

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
            shutil.rmtree(self.ckpt_manager.ckpt_dir)
            os.makedirs(self.ckpt_manager.ckpt_dir, exist_ok=True)

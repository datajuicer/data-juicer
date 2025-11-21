"""
Simplified Partitioned Ray Executor for Large Dataset Processing

This module implements a streamlined partitioned execution strategy for Ray mode that:
2. Splits the dataset into manageable partitions using Ray's .split() method
3. Processes each partition independently with Ray tasks
4. Merges results back into a single dataset for export
5. Supports convergence points for global operations (like deduplicators)
"""

import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

from jsonargparse import Namespace
from loguru import logger
from pydantic import PositiveInt

from data_juicer.core.adapter import Adapter
from data_juicer.core.data.dataset_builder import DatasetBuilder
from data_juicer.core.data.ray_dataset import RayDataset
from data_juicer.core.executor import ExecutorBase
from data_juicer.core.executor.dag_execution_mixin import DAGExecutionMixin
from data_juicer.core.executor.event_logging_mixin import EventLoggingMixin, EventType
from data_juicer.core.ray_exporter import RayExporter
from data_juicer.ops import load_ops
from data_juicer.ops.op_fusion import fuse_operators
from data_juicer.utils.ckpt_utils import CheckpointStrategy, RayCheckpointManager
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


class PartitionedRayExecutor(ExecutorBase, EventLoggingMixin, DAGExecutionMixin):
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
        self.adapter = Adapter(self.cfg)
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
            # Handle both dict and object configurations
            if isinstance(checkpoint_cfg, dict):
                checkpoint_enabled = checkpoint_cfg.get("enabled", True)
                strategy_str = checkpoint_cfg.get("strategy", "every_op")
                checkpoint_n_ops = checkpoint_cfg.get("n_ops", 1)
                checkpoint_op_names = checkpoint_cfg.get("op_names", [])
            else:
                checkpoint_enabled = getattr(checkpoint_cfg, "enabled", True)
                strategy_str = getattr(checkpoint_cfg, "strategy", "every_op")
                checkpoint_n_ops = getattr(checkpoint_cfg, "n_ops", 1)
                checkpoint_op_names = getattr(checkpoint_cfg, "op_names", [])

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
        self.exporter = RayExporter(
            self.cfg.export_path,
            keep_stats_in_res_ds=getattr(self.cfg, "keep_stats_in_res_ds", True),
            keep_hashes_in_res_ds=getattr(self.cfg, "keep_hashes_in_res_ds", False),
        )

    def _configure_partitioning(self):
        """Configure partitioning based on manual or auto mode."""
        # Get partition configuration
        partition_cfg = getattr(self.cfg, "partition", {})

        # Handle both dict and object configurations
        if isinstance(partition_cfg, dict):
            mode = partition_cfg.get("mode", "auto")
            num_of_partitions = partition_cfg.get("num_of_partitions", 4)
            partition_size = partition_cfg.get("size", 5000)
            max_size_mb = partition_cfg.get("max_size_mb", 64)
        else:
            mode = getattr(partition_cfg, "mode", "auto")
            num_of_partitions = getattr(partition_cfg, "num_of_partitions", 4)
            partition_size = getattr(partition_cfg, "size", 5000)
            max_size_mb = getattr(partition_cfg, "max_size_mb", 64)

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
            if hasattr(recommendations, "get"):
                # Handle dict-like recommendations
                recommended_size = recommendations.get("recommended_partition_size", self.partition_size)
                recommended_max_size_mb = recommendations.get("recommended_max_size_mb", self.max_size_mb)
                recommended_workers = recommendations.get("recommended_worker_count", getattr(self.cfg, "np", 4))
            else:
                # Handle object-like recommendations
                recommended_size = getattr(recommendations, "recommended_partition_size", self.partition_size)
                recommended_max_size_mb = getattr(recommendations, "recommended_max_size_mb", self.max_size_mb)
                recommended_workers = getattr(recommendations, "recommended_worker_count", getattr(self.cfg, "np", 4))

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

                # Ensure we don't create too many partitions (max 32 for efficiency)
                self.num_partitions = min(self.num_partitions, 32)

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

        # Initialize DAG execution planning
        self._initialize_dag_execution(self.cfg)

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

        # Handle auto partition mode
        if self.partition_mode == "auto":
            self._configure_auto_partitioning(dataset, ops)

        # Detect convergence points for global operations
        convergence_points = self._detect_convergence_points_partitioned(self.cfg)

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
        """
        logger.info("Processing with real partitioning using Ray Data's split and union...")

        # Split the dataset into partitions
        logger.info(f"Splitting dataset into {self.num_partitions} partitions...")
        partitions = dataset.data.split(self.num_partitions)
        logger.info(f"Created {len(partitions)} partitions")

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

            # Use DAG-aware execution if available
            if self.pipeline_dag:
                final_dataset = self._execute_operations_with_dag_monitoring(
                    merged_ray_dataset, post_convergence_ops, partition_id=0
                )
            else:
                # Fallback to normal execution
                final_dataset = merged_ray_dataset.process(post_convergence_ops)

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
            # Still use DAG monitoring even when checkpointing is disabled
            return self._execute_operations_with_dag_monitoring(dataset, ops, partition_id)

        # check the latest checkpoint for the partition
        latest_checkpoint = self.ckpt_manager.find_latest_checkpoint(partition_id)

        # Group operations based on checkpoint strategy
        op_groups = self.ckpt_manager.group_operations_for_checkpointing(ops)
        logger.info(f"Grouped {len(ops)} operations into {len(op_groups)} groups for checkpointing")
        logger.info(f"Detailed op gruops: {op_groups}")

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

                # Use DAG-aware execution if available
                if self.pipeline_dag:
                    current_dataset = self._execute_operations_with_dag_monitoring(
                        current_dataset, group_ops, partition_id
                    )
                else:
                    # Fallback to normal execution with manual logging
                    # Log operation start events
                    for op_idx, op in enumerate(group_ops):
                        self._log_event(
                            event_type=EventType.OP_START,
                            message=f"Starting operation: {op._name}",
                            operation_name=op._name,
                            operation_idx=start_idx + op_idx,
                            partition_id=partition_id,
                        )

                    current_dataset = current_dataset.process(group_ops)

                    # Log operation completion events
                    for op_idx, op in enumerate(group_ops):
                        self._log_event(
                            event_type=EventType.OP_COMPLETE,
                            message=f"Completed operation: {op._name}",
                            operation_name=op._name,
                            operation_idx=start_idx + op_idx,
                            partition_id=partition_id,
                        )

            # Checkpoint after the last operation in the group
            if group_ops:
                last_op_idx = end_idx - 1
                last_op_name = ops[last_op_idx]._name
                if self.ckpt_manager.should_checkpoint(last_op_idx, last_op_name):
                    logger.info(
                        f"Partition {partition_id}: Creating checkpoint after operation {last_op_idx}: {last_op_name}"
                    )
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
            else:
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
            probe_res = None
            fusion_strategy = getattr(self.cfg, "fusion_strategy", "basic")
            if fusion_strategy == "probe":
                logger.info("Probe the OP speed for OP reordering...")
                probe_res, _ = self.adapter.probe_small_batch(self.dataset, ops)

            logger.info(f"Start OP fusion and reordering with strategy [{fusion_strategy}]...")
            ops = fuse_operators(ops, probe_res)

        return ops

    def _override_strategy_methods(self):
        """Override strategy methods for partitioned execution."""
        # Override partition count determination
        self._determine_partition_count = self._determine_partition_count_partitioned
        self._analyze_dataset_size = self._analyze_dataset_size_partitioned
        self._detect_convergence_points = self._detect_convergence_points_partitioned
        self._get_dag_node_for_operation = self._get_dag_node_for_operation_partitioned

    def _determine_partition_count_partitioned(self, cfg) -> int:
        """Determine partition count for partitioned execution."""
        return self.num_partitions

    def _analyze_dataset_size_partitioned(self, dataset_path: str) -> int:
        """Analyze dataset size for partition count determination."""
        try:
            file_size = os.path.getsize(dataset_path)
            # More accurate estimate for partitioned execution
            estimated_lines = file_size // 512  # Assume 512 bytes per line
            return estimated_lines
        except Exception as e:
            logger.error(f"Error analyzing dataset size: {e}")
            # Fallback to default
            return 100000

    def _detect_convergence_points_partitioned(self, cfg) -> List[int]:
        """Detect convergence points for partitioned execution."""
        # Get operations from config first
        operations = self._get_operations_from_config(cfg)
        convergence_points = []

        for op_idx, op in enumerate(operations):
            # Detect global operations (deduplicators, etc.)
            if self._is_global_operation_partitioned(op):
                convergence_points.append(op_idx)

            # Detect manual convergence points
            if hasattr(op, "converge_after") and op.converge_after:
                convergence_points.append(op_idx)

        return convergence_points

    def _is_global_operation_partitioned(self, operation) -> bool:
        """Check if an operation is a global operation for partitioned execution."""
        # Deduplicators are typically global operations
        if hasattr(operation, "_name") and "deduplicator" in operation._name:
            return True

        # Check for explicit global operation flag
        if hasattr(operation, "is_global_operation") and operation.is_global_operation:
            return True

        return False

    def _get_dag_node_for_operation_partitioned(
        self, op_name: str, op_idx: int, partition_id: int = 0, **kwargs
    ) -> Optional[str]:
        """Get DAG node ID for partitioned operation."""
        if not self.dag_execution_strategy:
            return None

        return self.dag_execution_strategy.get_dag_node_id(op_name, op_idx, partition_id=partition_id, **kwargs)

    def _execute_operations_with_dag_monitoring(self, dataset, ops: List, partition_id: int = 0):
        """Execute operations with DAG monitoring for partitioned execution."""
        if not self.pipeline_dag:
            logger.warning("Pipeline DAG not initialized, falling back to normal execution")
            return dataset.process(ops)

        # Log operation start events for all operations
        for op_idx, op in enumerate(ops):
            op_name = op._name
            node_id = self._get_dag_node_for_operation(op_name, op_idx, partition_id=partition_id)

            if node_id:
                # Mark DAG node as started
                self._mark_dag_node_started(node_id)

                # Log operation start with DAG context
                self._log_operation_with_dag_context(op_name, op_idx, "op_start", partition_id)
            else:
                # Log operation start without DAG context
                logger.warning(f"DAG node not found for operation {op_name}, logging without DAG context")
                if hasattr(self, "log_op_start"):
                    self.log_op_start(0, op_name, op_idx, {})

        # Execute all operations normally (this is what actually processes the data)
        processed_dataset = dataset.process(ops)

        # Log operation completion events for all operations
        for op_idx, op in enumerate(ops):
            op_name = op._name
            node_id = self._get_dag_node_for_operation(op_name, op_idx, partition_id=partition_id)

            if node_id:
                # Mark DAG node as completed
                self._mark_dag_node_completed(node_id, 0.0)  # Duration will be updated from events

                # Log operation completion with DAG context
                self._log_operation_with_dag_context(
                    op_name, op_idx, "op_complete", partition_id, duration=0.0, input_rows=0, output_rows=0
                )
            else:
                # Log operation completion without DAG context
                if hasattr(self, "log_op_complete"):
                    self.log_op_complete(0, op_name, op_idx, 0.0, None, 0, 0)

        return processed_dataset

    def _log_operation_with_dag_context(
        self, op_name: str, op_idx: int, event_type: str, partition_id: int = 0, **kwargs
    ) -> None:
        """Log an operation event with DAG context for partitioned execution."""
        # Get the corresponding DAG node
        node_id = self._get_dag_node_for_operation(op_name, op_idx, partition_id=partition_id)

        # Add DAG node ID to metadata if found
        if "metadata" not in kwargs:
            kwargs["metadata"] = {}

        if node_id:
            kwargs["metadata"]["dag_node_id"] = node_id
        else:
            # Log warning if DAG node not found
            logger.warning(f"DAG node not found for operation {op_name} (idx {op_idx})")

        # Call the original logging method with correct parameters
        if event_type == "op_start" and hasattr(self, "log_op_start"):
            self.log_op_start(0, op_name, op_idx, kwargs.get("metadata", {}))
        elif event_type == "op_complete" and hasattr(self, "log_op_complete"):
            self.log_op_complete(
                0,
                op_name,
                op_idx,
                kwargs.get("duration", 0),
                kwargs.get("checkpoint_path"),
                kwargs.get("input_rows", 0),
                kwargs.get("output_rows", 0),
            )
        elif event_type == "op_failed" and hasattr(self, "log_op_failed"):
            self.log_op_failed(0, op_name, op_idx, kwargs.get("error", "Unknown error"), kwargs.get("retry_count", 0))

    def log_op_start(self, partition_id, operation_name, operation_idx, op_args, metadata=None):
        """Override to add DAG context to operation start events."""
        # Get the corresponding DAG node
        node_id = self._get_dag_node_for_operation(operation_name, operation_idx)

        # Create metadata with DAG context
        if metadata is None:
            metadata = {}
        if node_id:
            metadata["dag_node_id"] = node_id
        else:
            logger.warning(f"DAG node not found for operation {operation_name} (idx {operation_idx})")

        # Call the parent method with metadata
        super().log_op_start(partition_id, operation_name, operation_idx, op_args, metadata=metadata)

    def log_op_complete(
        self,
        partition_id,
        operation_name,
        operation_idx,
        duration,
        checkpoint_path,
        input_rows,
        output_rows,
        metadata=None,
    ):
        """Override to add DAG context to operation complete events."""
        # Get the corresponding DAG node
        node_id = self._get_dag_node_for_operation(operation_name, operation_idx)

        # Create metadata with DAG context
        if metadata is None:
            metadata = {}
        if node_id:
            metadata["dag_node_id"] = node_id
        else:
            logger.warning(f"DAG node not found for operation {operation_name} (idx {operation_idx})")

        # Call the parent method with metadata
        super().log_op_complete(
            partition_id,
            operation_name,
            operation_idx,
            duration,
            checkpoint_path,
            input_rows,
            output_rows,
            metadata=metadata,
        )

    def log_op_failed(self, partition_id, operation_name, operation_idx, error_message, retry_count, metadata=None):
        """Override to add DAG context to operation failed events."""
        # Get the corresponding DAG node
        node_id = self._get_dag_node_for_operation(operation_name, operation_idx)

        # Create metadata with DAG context
        if metadata is None:
            metadata = {}
        if node_id:
            metadata["dag_node_id"] = node_id
        else:
            logger.warning(f"DAG node not found for operation {operation_name} (idx {operation_idx})")

        # Call the parent method with metadata
        super().log_op_failed(
            partition_id, operation_name, operation_idx, error_message, retry_count, metadata=metadata
        )

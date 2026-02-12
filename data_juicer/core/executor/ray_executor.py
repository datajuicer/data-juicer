import os
import shutil
import time
from typing import Optional

from jsonargparse import Namespace
from loguru import logger
from pydantic import PositiveInt

from data_juicer.core.data.dataset_builder import DatasetBuilder
from data_juicer.core.executor import ExecutorBase
from data_juicer.core.executor.dag_execution_mixin import DAGExecutionMixin
from data_juicer.core.executor.event_logging_mixin import EventLoggingMixin
from data_juicer.core.ray_exporter import RayExporter
from data_juicer.core.tracer.ray_tracer import RayTracer
from data_juicer.ops import OPEnvManager, load_ops
from data_juicer.ops.op_fusion import fuse_operators
from data_juicer.utils.lazy_loader import LazyLoader

ray = LazyLoader("ray")


class TempDirManager:
    def __init__(self, tmp_dir):
        self.tmp_dir = tmp_dir

    def __enter__(self):
        os.makedirs(self.tmp_dir, exist_ok=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if os.path.exists(self.tmp_dir):
            logger.info(f"Removing tmp dir {self.tmp_dir} ...")
            shutil.rmtree(self.tmp_dir)


class RayExecutor(ExecutorBase, DAGExecutionMixin, EventLoggingMixin):
    """
    Executor based on Ray.

    Run Data-Juicer data processing in a distributed cluster.

        1. Support Filter, Mapper and Exact Deduplicator operators for now.
        2. Only support loading `.json` files.
        3. Advanced functions, such as checkpoint, are not supported.

    """

    def __init__(self, cfg: Optional[Namespace] = None):
        """
        Initialization method.

        :param cfg: optional config dict.
        """
        super().__init__(cfg)

        self.executor_type = "ray"
        self.work_dir = self.cfg.work_dir

        # Initialize EventLoggingMixin for job management and event logging
        EventLoggingMixin.__init__(self, cfg)

        # Initialize DAGExecutionMixin for AST/DAG functionality
        DAGExecutionMixin.__init__(self)

        # init ray
        logger.info("Initializing Ray ...")

        from data_juicer.utils.ray_utils import initialize_ray

        initialize_ray(cfg=cfg, force=True)

        self.tmp_dir = os.path.join(self.work_dir, ".tmp", ray.get_runtime_context().get_job_id())

        # absolute path resolution logic

        # init dataset builder
        self.datasetbuilder = DatasetBuilder(self.cfg, executor_type="ray")

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
                # Iterate through the required fields directly, and copy them to export_extra_args if they exist.
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

    def run(self, load_data_np: Optional[PositiveInt] = None, skip_export: bool = False, skip_return: bool = False):
        """
        Running the dataset process pipeline

        :param load_data_np: number of workers when loading the dataset.
        :param skip_export: whether export the results into disk
        :param skip_return: skip return for API called.
        :return: processed dataset.
        """
        # 1. load data
        logger.info("Loading dataset with Ray...")
        dataset = self.datasetbuilder.load_dataset(num_proc=load_data_np)
        columns = dataset.data.columns()

        # 2. extract processes
        logger.info("Preparing process operators...")
        ops = load_ops(self.cfg.process, self.op_env_manager)

        # Initialize DAG execution planning (pass ops to avoid redundant loading)
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

        if self.cfg.op_fusion:
            logger.info(f"Start OP fusion and reordering with strategy " f"[{self.cfg.fusion_strategy}]...")
            ops = fuse_operators(ops)

        with TempDirManager(self.tmp_dir):
            # 3. data process with DAG monitoring
            logger.info("Processing data with DAG monitoring...")
            tstart = time.time()

            # Get input row count before processing
            input_rows = dataset.data.count()
            start_time = time.time()

            # Pre-execute DAG monitoring (log operation start events)
            if self.pipeline_dag:
                self._pre_execute_operations_with_dag_monitoring(ops)

            # Execute operations (Ray executor uses simple dataset.process)
            dataset = dataset.process(ops, tracer=self.tracer)

            # Force materialization to get real execution
            logger.info("Materializing dataset to collect real metrics...")
            dataset.data = dataset.data.materialize()

            # Get metrics after execution
            duration = time.time() - start_time
            output_rows = dataset.data.count()

            # Post-execute DAG monitoring (log operation completion events with real metrics)
            if self.pipeline_dag:
                metrics = {"duration": duration, "input_rows": input_rows, "output_rows": output_rows}
                self._post_execute_operations_with_dag_monitoring(ops, metrics=metrics)

            # 4. data export
            if not skip_export:
                logger.info("Exporting dataset to disk...")
                self.exporter.export(dataset.data, columns=columns)
            tend = time.time()
            logger.info(f"All Ops are done in {tend - tstart:.3f}s.")

        # Log job completion with DAG context
        job_duration = time.time() - tstart
        self.log_job_complete(job_duration, self.cfg.export_path)

        # 5. finalize the tracer results
        # Finalize sample-level traces after all operators have finished
        if self.tracer:
            ray.get(self.tracer.finalize_traces.remote())

        if not skip_return:
            return dataset

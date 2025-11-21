import os
from time import time
from typing import Optional, Union

from datasets import Dataset
from jsonargparse import Namespace
from loguru import logger
from pydantic import PositiveInt

from data_juicer.core.adapter import Adapter
from data_juicer.core.data import NestedDataset
from data_juicer.core.data.dataset_builder import DatasetBuilder
from data_juicer.core.executor import ExecutorBase
from data_juicer.core.executor.dag_execution_mixin import DAGExecutionMixin
from data_juicer.core.executor.event_logging_mixin import EventLoggingMixin
from data_juicer.core.exporter import Exporter
from data_juicer.core.tracer import Tracer
from data_juicer.ops import load_ops
from data_juicer.ops.op_fusion import fuse_operators
from data_juicer.ops.selector import (
    FrequencySpecifiedFieldSelector,
    TopkSpecifiedFieldSelector,
)
from data_juicer.utils import cache_utils
from data_juicer.utils.ckpt_utils import CheckpointManager
from data_juicer.utils.sample import random_sample


class DefaultExecutor(ExecutorBase, EventLoggingMixin, DAGExecutionMixin):
    """
    This Executor class is used to process a specific dataset.

    It will load the dataset and unify the format, then apply all the
    ops in the config file in order and generate a processed dataset.
    """

    def __init__(self, cfg: Optional[Namespace] = None):
        """
        Initialization method.

        :param cfg: optional jsonargparse Namespace.
        """
        super().__init__(cfg)
        # If work_dir contains job_id, all outputs go under it
        self.work_dir = self.cfg.work_dir

        # Initialize EventLoggingMixin for job management and event logging
        EventLoggingMixin.__init__(self, cfg)

        # Initialize DAGExecutionMixin for AST/DAG functionality
        DAGExecutionMixin.__init__(self)
        # Set executor type for strategy selection
        self.executor_type = "default"

        self.ckpt_manager = None

        self.adapter = Adapter(self.cfg)

        self.np = self.cfg.get("np", None) or 1

        # only enable it when using cache
        if self.cfg.use_cache:
            logger.info(f"Using cache compression method: " f"[{self.cfg.cache_compress}]")
            cache_utils.CACHE_COMPRESS = self.cfg.cache_compress

        # setup dataset builder
        logger.info("Setting up dataset builder...")
        self.dataset_builder = DatasetBuilder(self.cfg, executor_type=self.executor_type)

        # whether to use checkpoint mechanism. If it's true, Executor will
        # check if there are existing checkpoints first and try to load the
        # checkpoints. If the checkpoints are loaded successfully, ops that
        # have been processed will be skipped.
        if self.cfg.use_checkpoint:
            logger.info("Preparing checkpoint manager...")
            self.ckpt_dir = os.path.join(self.work_dir, "ckpt")
            self.ckpt_manager = CheckpointManager(self.ckpt_dir, self.cfg.process, self.np)
            if self.ckpt_manager.ckpt_available:
                logger.info("Found existed dataset checkpoint.")
                self.cfg.process = self.ckpt_manager.get_left_process_list()

        # prepare exporter and check export path suffix
        logger.info("Preparing exporter...")
        self.exporter = Exporter(
            self.cfg.export_path,
            self.cfg.export_type,
            self.cfg.export_shard_size,
            self.cfg.export_in_parallel,
            self.np,
            keep_stats_in_res_ds=self.cfg.keep_stats_in_res_ds,
            keep_hashes_in_res_ds=self.cfg.keep_hashes_in_res_ds,
            **self.cfg.export_extra_args,
        )

        # setup tracer
        self.open_tracer = self.cfg.open_tracer
        if self.open_tracer:
            logger.info("Preparing tracer...")
            self.tracer = Tracer(self.work_dir, self.cfg.op_list_to_trace, show_num=self.cfg.trace_num)

    def run(
        self,
        dataset: Union[Dataset, NestedDataset] = None,
        load_data_np: Optional[PositiveInt] = None,
        skip_export: bool = False,
        skip_return: bool = False,
    ):
        """
        Running the dataset process pipeline.

        :param dataset: a Dataset object to be executed.
        :param load_data_np: number of workers when loading the dataset.
        :param skip_export: whether export the results into disk
        :param skip_return: skip return for API called.
        :return: processed dataset.
        """
        # 1. format data
        if dataset is not None:
            logger.info(f"Using existing dataset {dataset}")
        elif self.cfg.use_checkpoint and self.ckpt_manager.ckpt_available:
            logger.info("Loading dataset from checkpoint...")
            dataset = self.ckpt_manager.load_ckpt()
        else:
            logger.info("Loading dataset from dataset builder...")
            if load_data_np is None:
                load_data_np = self.np
            dataset = self.dataset_builder.load_dataset(num_proc=load_data_np)

        # 2. extract processes and optimize their orders
        logger.info("Preparing process operators...")
        ops = load_ops(self.cfg.process)

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

        # OP fusion
        if self.cfg.op_fusion:
            probe_res = None
            if self.cfg.fusion_strategy == "probe":
                logger.info("Probe the OP speed for OP reordering...")
                probe_res, _ = self.adapter.probe_small_batch(dataset, ops)

            logger.info(f"Start OP fusion and reordering with strategy " f"[{self.cfg.fusion_strategy}]...")
            ops = fuse_operators(ops, probe_res)

        # adaptive batch size
        if self.cfg.adaptive_batch_size:
            # calculate the adaptive batch size
            bs_per_op = self.adapter.adapt_workloads(dataset, ops)
            assert len(bs_per_op) == len(ops)
            # update the adaptive batch size
            logger.info(f"Adapt batch sizes for each OP to {bs_per_op}")
            for i, op in enumerate(ops):
                if op.is_batched_op():
                    op.batch_size = bs_per_op[i]

        # 3. data process with DAG monitoring
        # - If tracer is open, trace each op after it's processed
        # - If checkpoint is open, clean the cache files after each process
        logger.info("Processing data with DAG monitoring...")
        tstart = time()

        # Use DAG-aware execution if available
        if self.pipeline_dag:
            self._execute_operations_with_dag_monitoring(dataset, ops)
        else:
            # Fallback to normal execution
            dataset = dataset.process(
                ops,
                work_dir=self.work_dir,
                exporter=self.exporter,
                checkpointer=self.ckpt_manager,
                tracer=self.tracer if self.cfg.open_tracer else None,
                adapter=self.adapter,
                open_monitor=self.cfg.open_monitor,
            )

        tend = time()
        logger.info(f"All OPs are done in {tend - tstart:.3f}s.")

        # 4. data export
        if not skip_export:
            logger.info("Exporting dataset to disk...")
            self.exporter.export(dataset)
        # compress the last dataset after exporting
        if self.cfg.use_cache and self.cfg.cache_compress:
            from data_juicer.utils.compress import compress

            compress(dataset)

        # Log job completion with DAG context
        job_duration = time() - tstart
        self.log_job_complete(job_duration, self.cfg.export_path)

        if not skip_return:
            return dataset

    def sample_data(
        self,
        dataset_to_sample: Dataset = None,
        load_data_np=None,
        sample_ratio: float = 1.0,
        sample_algo: str = "uniform",
        **kwargs,
    ):
        """
        Sample a subset from the given dataset.
        TODO add support other than LocalExecutor

        :param dataset_to_sample: Dataset to sample from. If None, will use
            the formatter linked by the executor. Default is None.
        :param load_data_np: number of workers when loading the dataset.
        :param sample_ratio: The ratio of the sample size to the original
            dataset size. Default is 1.0 (no sampling).
        :param sample_algo: Sampling algorithm to use. Options are "uniform",
            "frequency_specified_field_selector", or
            "topk_specified_field_selector".
            Default is "uniform".
        :return: A sampled Dataset.
        """
        # Determine the dataset to sample from
        if dataset_to_sample is not None:
            dataset = dataset_to_sample
        elif self.cfg.use_checkpoint and self.ckpt_manager.ckpt_available:
            logger.info("Loading dataset from checkpoint...")
            dataset = self.ckpt_manager.load_ckpt()
        else:
            logger.info("Loading dataset from dataset builder...")
            if load_data_np is None:
                load_data_np = self.np
            dataset = self.dataset_builder.load_dataset(num_proc=load_data_np)

        # Perform sampling based on the specified algorithm
        if sample_algo == "uniform":
            return random_sample(dataset, sample_ratio)
        elif sample_algo == "frequency_specified_field_selector":
            dj_op = FrequencySpecifiedFieldSelector(**kwargs)
            return dj_op.process(dataset)
        elif sample_algo == "topk_specified_field_selector":
            dj_op = TopkSpecifiedFieldSelector(**kwargs)
            return dj_op.process(dataset)
        else:
            raise ValueError(f"Unsupported sample_algo: {sample_algo}")

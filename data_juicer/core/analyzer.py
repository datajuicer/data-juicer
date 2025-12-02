import os
from typing import Optional, Union

from datasets import Dataset
from jsonargparse import Namespace
from loguru import logger
from pydantic import PositiveInt

from data_juicer.analysis import (
    ColumnWiseAnalysis,
    CorrelationAnalysis,
    OverallAnalysis,
)
from data_juicer.config import init_configs
from data_juicer.core.data.dataset_builder import DatasetBuilder
from data_juicer.ops import NON_STATS_FILTERS, TAGGING_OPS, Filter, load_ops
from data_juicer.ops.op_fusion import fuse_operators
from data_juicer.utils import cache_utils

from .adapter import Adapter
from .data import NestedDataset
from .exporter import Exporter


class Analyzer:
    """
    This Analyzer class is used to analyze a specific dataset.

    It will compute stats for all filter ops in the config file, apply
    multiple analysis (e.g. OverallAnalysis, ColumnWiseAnalysis, etc.)
    on these stats, and generate the analysis results (stats tables,
    distribution figures, etc.) to help users understand the input
    dataset better.
    """

    def __init__(self, cfg: Optional[Namespace] = None):
        """
        Initialization method.

        :param cfg: optional jsonargparse Namespace dict.
        """
        self.cfg = init_configs(which_entry=self) if cfg is None else cfg

        self.work_dir = self.cfg.work_dir

        if self.cfg.use_cache:
            logger.info(f"Using cache compression method: " f"[{self.cfg.cache_compress}]")
            cache_utils.CACHE_COMPRESS = self.cfg.cache_compress

        # setup dataset builder
        logger.info("Setting up dataset builder...")
        self.dataset_builder = DatasetBuilder(self.cfg, executor_type="default")

        # prepare exporter and check export path suffix
        # NOTICE: no need to export dataset texts for analyzer
        # (export_ds=False). Instead, only need to export stats
        # (export_stats=True).
        logger.info("Preparing exporter...")
        self.exporter = Exporter(
            self.cfg.export_path,
            self.cfg.export_type,
            self.cfg.export_shard_size,
            self.cfg.export_in_parallel,
            self.cfg.np,
            export_ds=self.cfg.export_original_dataset,
            keep_stats_in_res_ds=self.cfg.export_original_dataset,
            export_stats=True,
        )

        # parsed_res
        self.overall_result = None
        self.overall_single_plot_path = None
        self.analysis_path = os.path.join(self.cfg.work_dir, "analysis")

    def run(
        self,
        dataset: Union[Dataset, NestedDataset] = None,
        load_data_np: Optional[PositiveInt] = None,
        skip_export: bool = False,
        skip_return: bool = False,
    ):
        """
        Running the dataset analysis pipeline.

        :param dataset: a Dataset object to be analyzed.
        :param load_data_np: number of workers when loading the dataset.
        :param skip_export: whether export the results into disk
        :param skip_return: skip return for API called.
        :return: analyzed dataset.
        """
        # 1. format data
        if load_data_np is None:
            load_data_np = self.cfg.np
        if dataset is None:
            logger.info("Loading dataset from data formatter...")
            dataset = self.dataset_builder.load_dataset(num_proc=load_data_np)
        else:
            logger.info(f"Using existing dataset {dataset}")
        if self.cfg.auto:
            # if it's auto analysis, only analyze for a minor part of the input
            # dataset to save time and computing resource
            dataset = dataset.take(min(len(dataset), self.cfg.auto_num))

        # extract processes
        logger.info("Preparing process operators...")
        ops = load_ops(self.cfg.process)

        if self.cfg.op_fusion:
            probe_res = None
            if self.cfg.fusion_strategy == "probe":
                logger.info("Probe the OP speed for OP reordering...")
                adapter = Adapter(self.cfg)
                probe_res, _ = adapter.probe_small_batch(dataset, ops)

            logger.info(f"Start OP fusion and reordering with strategy " f"[{self.cfg.fusion_strategy}]...")
            ops = fuse_operators(ops, probe_res)

        # 2. stats precompute only for filter or tagging ops
        logger.info("Computing the stats of dataset...")
        stats_collected = False
        for op in ops:
            if isinstance(op, Filter) and op._name not in NON_STATS_FILTERS.modules:
                original_process = op.process
                op.process = None
                dataset = dataset.process(op, work_dir=self.work_dir, open_monitor=self.cfg.open_monitor)
                op.process = original_process
                stats_collected = True
            elif op._name in TAGGING_OPS.modules:
                dataset = dataset.process(op, work_dir=self.work_dir, open_monitor=self.cfg.open_monitor)
                stats_collected = True
        if not stats_collected:
            logger.warning(
                "No stats/meta collected. Please add some Filter OPs or " "Tagging OPs to the process list in configs."
            )
            if not skip_return:
                return dataset

        # 3. data export
        logger.info("Exporting dataset to disk...")
        self.exporter.export(dataset)
        if self.cfg.use_cache and self.cfg.cache_compress:
            from data_juicer.utils.compress import compress

            compress(dataset)

        # 4. analysis and output result to the export path
        # 4.1. Only consider fields in Fields.stats and Fields.meta
        # 4.2. For string fields, only consider its histogram
        # 4.3. For numeric fields, consider its histogram and box
        # 4.4. Otherwise, DO NOT analyze

        logger.info("Applying overall analysis on stats...")
        overall_analysis = OverallAnalysis(dataset, self.analysis_path)
        self.overall_result = overall_analysis.analyze(
            percentiles=self.cfg.percentiles, num_proc=self.cfg.np, skip_export=skip_export
        )

        logger.info(f"The overall analysis results are: {self.overall_result}")

        logger.info("Applying column-wise analysis on stats...")
        column_wise_analysis = ColumnWiseAnalysis(
            dataset,
            self.analysis_path,
            overall_result=self.overall_result,
            save_stats_in_one_file=self.cfg.save_stats_in_one_file,
        )
        column_wise_analysis.analyze(skip_export=skip_export)

        logger.info("Applying correlation analysis on stats...")
        correlation_analysis = CorrelationAnalysis(
            dataset,
            self.analysis_path,
        )
        correlation_analysis.analyze(skip_export=skip_export)

        if not skip_return:
            return dataset

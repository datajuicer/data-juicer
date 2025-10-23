import os
import shutil
import time
from typing import Optional

from jsonargparse import Namespace
from loguru import logger
from pydantic import PositiveInt

from data_juicer.core.data.dataset_builder import DatasetBuilder
from data_juicer.core.executor import ExecutorBase
from data_juicer.core.ray_exporter import RayExporter
from data_juicer.ops import load_ops
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


class RayExecutor(ExecutorBase):
    """
    Executor based on Ray.

    Run Data-Juicer data processing in a distributed cluster.

        1. Support Filter, Mapper and Exact Deduplicator operators for now.
        2. Only support loading `.json` files.
        3. Advanced functions such as checkpoint, tracer are not supported.

    """

    def __init__(self, cfg: Optional[Namespace] = None):
        """
        Initialization method.

        :param cfg: optional config dict.
        """
        super().__init__(cfg)
        self.executor_type = "ray"
        self.work_dir = self.cfg.work_dir
        # TODO: support ray
        # self.adapter = Adapter(self.cfg)

        # init ray
        logger.info("Initializing Ray ...")

        from data_juicer.utils.ray_utils import initialize_ray

        initialize_ray(cfg=cfg, force=True)

        self.tmp_dir = os.path.join(self.work_dir, ".tmp", ray.get_runtime_context().get_job_id())

        # absolute path resolution logic

        # init dataset builder
        self.datasetbuilder = DatasetBuilder(self.cfg, executor_type="ray")

        logger.info("Preparing exporter...")
        self.exporter = RayExporter(
            self.cfg.export_path,
            self.cfg.export_type,
            self.cfg.export_shard_size,
            keep_stats_in_res_ds=self.cfg.keep_stats_in_res_ds,
            keep_hashes_in_res_ds=self.cfg.keep_hashes_in_res_ds,
            **self.cfg.export_extra_args,
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
        columns = dataset.schema().columns

        # 2. extract processes
        logger.info("Preparing process operators...")
        ops = load_ops(self.cfg.process)

        # Apply core optimizer if enabled
        from data_juicer.core.optimization_manager import apply_optimizations

        ops = apply_optimizations(ops, self.cfg)

        with TempDirManager(self.tmp_dir):
            # 3. data process
            logger.info("Processing data...")
            tstart = time.time()
            dataset.process(ops)

            # 4. data export
            if not skip_export:
                logger.info("Exporting dataset to disk...")
                self.exporter.export(dataset.data, columns=columns)
            tend = time.time()
            logger.info(f"All Ops are done in {tend - tstart:.3f}s.")

        if not skip_return:
            return dataset

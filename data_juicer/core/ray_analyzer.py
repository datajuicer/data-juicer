import os
from typing import Optional

import pandas as pd
import pyarrow
import pyarrow.compute as pc
from jsonargparse import Namespace
from loguru import logger
from pydantic import PositiveInt

from data_juicer.config import init_configs
from data_juicer.core.data.dataset_builder import (
    DatasetBuilder,
    deprecated_load_data_np_kwargs,
)
from data_juicer.core.ray_exporter import RayExporter
from data_juicer.ops import NON_STATS_FILTERS, TAGGING_OPS, Filter, load_ops
from data_juicer.ops.op_fusion import fuse_operators
from data_juicer.utils.constant import DEFAULT_PREFIX, Fields
from data_juicer.utils.lazy_loader import LazyLoader

ray = LazyLoader("ray")


def _flatten_stats_batch(table: pyarrow.Table):
    """Flatten the stats/meta dict column into separate top-level columns."""
    result_columns = {}
    for col_name in [Fields.stats, Fields.meta]:
        if col_name not in table.column_names:
            continue
        col = table.column(col_name)
        dicts = col.to_pylist()
        if not dicts or not isinstance(dicts[0], dict):
            continue
        keys = set()
        for d in dicts:
            if isinstance(d, dict):
                keys.update(d.keys())
        if col_name == Fields.meta:
            keys = {k for k in keys if k.startswith(DEFAULT_PREFIX)}
        for key in keys:
            result_columns[key] = [d.get(key) if isinstance(d, dict) else None for d in dicts]
    if not result_columns:
        return table
    arrays = {k: pyarrow.array(v) for k, v in result_columns.items()}
    return pyarrow.table(arrays)


class RayAnalyzer:
    """
    Analyzer that uses Ray for distributed stats computation.

    Computes filter stats in parallel via Ray, then uses Ray's native
    aggregation (Mean, Std, Min, Max, Count) for overall analysis.
    No data is materialized to pandas — all computation stays distributed.
    """

    def __init__(self, cfg: Optional[Namespace] = None):
        self.cfg = init_configs(allow_auto=True) if cfg is None else cfg
        self.work_dir = self.cfg.work_dir

        from data_juicer.utils.ray_utils import initialize_ray

        try:
            initialize_ray(cfg=self.cfg)
        except ConnectionError:
            logger.info("No existing Ray cluster found, starting a local one...")
            ray.init(ignore_reinit_error=True)

        self.dataset_builder = DatasetBuilder(self.cfg, executor_type="ray")

        self.exporter = RayExporter(
            self.cfg.export_path,
            self.cfg.export_type,
            self.cfg.export_shard_size,
            keep_stats_in_res_ds=True,
        )

        self.overall_result = None
        self.analysis_path = os.path.join(self.cfg.work_dir, "analysis")

    def run(
        self,
        dataset=None,
        load_data_np: Optional[PositiveInt] = None,
        skip_export: bool = False,
        skip_return: bool = False,
    ):
        if dataset is None:
            logger.info("Loading dataset via Ray...")
            dataset = self.dataset_builder.load_dataset(
                **deprecated_load_data_np_kwargs(load_data_np, self.dataset_builder.executor_type)
            )
        else:
            logger.info(f"Using existing dataset {dataset}")

        if self.cfg.auto:
            count = dataset.data.count()
            limit = min(count, self.cfg.auto_num)
            if limit < count:
                logger.info(f"Auto mode: sampling {limit}/{count} rows " f"for analysis")
                dataset.data = dataset.data.limit(limit)

        logger.info("Preparing process operators...")
        ops = load_ops(self.cfg.process)

        if self.cfg.op_fusion:
            probe_res = None
            logger.info(f"Start OP fusion and reordering with strategy " f"[{self.cfg.fusion_strategy}]...")
            ops = fuse_operators(
                ops,
                probe_res,
                mapper_fusion=False,
                mapper_fusion_vram_limit=getattr(self.cfg, "mapper_fusion_vram_limit", 0.9),
            )

        filter_ops = [op for op in ops if isinstance(op, Filter) and op._name not in NON_STATS_FILTERS.modules]
        tagging_ops = [
            op for op in ops if op._name in TAGGING_OPS.modules or getattr(op, "_contains_tagging_ops", False)
        ]

        if not filter_ops and not tagging_ops:
            logger.warning(
                "No stats/meta collected. Please add some Filter " "OPs or Tagging OPs to the process list in configs."
            )
            if not skip_return:
                return dataset

        logger.info(f"Computing stats with Ray ({len(filter_ops)} filters, " f"{len(tagging_ops)} tagging ops)...")

        if filter_ops:
            dataset = dataset.process(filter_ops, stats_only=True)

        if tagging_ops:
            dataset = dataset.process(tagging_ops)

        if not skip_export:
            logger.info("Exporting dataset to disk...")
            self.exporter.export(dataset.data)

        logger.info("Computing overall analysis via Ray aggregation...")
        self.overall_result = self._compute_overall(dataset)
        logger.info(f"The overall analysis results are: {self.overall_result}")

        if not skip_return:
            return dataset

    def _compute_overall(self, dataset):
        os.makedirs(self.analysis_path, exist_ok=True)

        from ray.data.aggregate import Count, Max, Mean, Min, Std

        select_cols = []
        available_cols = dataset.data.columns()
        if Fields.stats in available_cols:
            select_cols.append(Fields.stats)
        if Fields.meta in available_cols:
            select_cols.append(Fields.meta)

        if not select_cols:
            logger.warning("No stats or meta columns found in dataset")
            return pd.DataFrame()

        flat_data = dataset.data.select_columns(select_cols).map_batches(_flatten_stats_batch, batch_format="pyarrow")

        flat_cols = flat_data.columns()
        if not flat_cols:
            logger.warning("No stat columns after flattening")
            return pd.DataFrame()

        numeric_cols = []
        list_numeric_cols = []
        skipped_cols = []
        arrow_schema = flat_data.schema()
        if hasattr(arrow_schema, "base_schema"):
            arrow_schema = arrow_schema.base_schema
        for col_name in flat_cols:
            idx = arrow_schema.get_field_index(col_name)
            field_type = arrow_schema.field(idx).type
            if pyarrow.types.is_integer(field_type) or pyarrow.types.is_floating(field_type):
                numeric_cols.append(col_name)
            elif pyarrow.types.is_list(field_type) or pyarrow.types.is_large_list(field_type):
                value_type = field_type.value_type
                if pyarrow.types.is_integer(value_type) or pyarrow.types.is_floating(value_type):
                    list_numeric_cols.append(col_name)
                else:
                    skipped_cols.append(col_name)
            else:
                skipped_cols.append(col_name)

        if skipped_cols:
            logger.warning(f"Skipping non-numeric columns in overall analysis: {skipped_cols}")

        if not numeric_cols and not list_numeric_cols:
            logger.warning("No numeric stat columns found")
            return pd.DataFrame()

        rows = {}

        if numeric_cols:
            aggs = []
            for col in numeric_cols:
                aggs.extend([Count(col), Mean(col), Std(col), Min(col), Max(col)])

            agg_result = flat_data.aggregate(*aggs)

            for col in numeric_cols:
                rows[col] = {
                    "count": agg_result[f"count({col})"],
                    "mean": agg_result[f"mean({col})"],
                    "std": agg_result[f"std({col})"],
                    "min": agg_result[f"min({col})"],
                    "max": agg_result[f"max({col})"],
                }

        for col in list_numeric_cols:
            col_data = flat_data.select_columns([col]).map_batches(
                lambda table, c=col: pyarrow.table({c: pc.list_flatten(table.column(c))}),
                batch_format="pyarrow",
            )
            col_agg = col_data.aggregate(Count(col), Mean(col), Std(col), Min(col), Max(col))
            rows[col] = {
                "count": col_agg[f"count({col})"],
                "mean": col_agg[f"mean({col})"],
                "std": col_agg[f"std({col})"],
                "min": col_agg[f"min({col})"],
                "max": col_agg[f"max({col})"],
            }

        overall = pd.DataFrame(rows)
        overall.to_csv(os.path.join(self.analysis_path, "overall.csv"))
        overall.to_markdown(os.path.join(self.analysis_path, "overall.md"))

        return overall

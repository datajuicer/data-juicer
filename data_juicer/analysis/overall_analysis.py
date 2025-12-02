import os
from multiprocessing import Pool

import pandas as pd
from loguru import logger
from tqdm import tqdm

from data_juicer.utils.constant import DEFAULT_PREFIX, Fields


def _single_column_analysis(col, *args, **kwargs):
    col_overall = col.describe(*args, **kwargs)
    return col_overall


class OverallAnalysis:
    """Apply analysis on the overall stats, including mean, std, quantiles,
    etc."""

    def __init__(self, dataset, output_path):
        """
        Initialization method.

        :param dataset: the dataset to be analyzed
        :param output_path: path to store the analysis results.
        """
        self.stats = pd.DataFrame(dataset[Fields.stats])
        self.meta = pd.DataFrame(dataset[Fields.meta])
        # remove non-tag columns
        meta_columns = self.meta.columns
        for col_name in meta_columns:
            if not col_name.startswith(DEFAULT_PREFIX):
                self.meta = self.meta.drop(col_name, axis=1)
        self.output_path = output_path
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        # default percentiles to analyze
        self.default_percentiles = [0.25, 0.5, 0.75]
        # supported dtypes of column to be analyzed
        # Notice: there won't be mixed types in a column because the stats is
        # obtained from Dataset, which doesn't allow mixed types.
        # Notice: for now, stats can only be:
        # {numbers, string, list of one of before}
        self.supported_object_types = {str, list}

    def refine_single_column(self, col):
        if col.dtype != "object":
            # not an object, return directly
            return col
        # if the type of this column is object, we can decide the actual type
        # according to the first element.
        first = col[0]
        if type(first) not in self.supported_object_types:
            logger.warning(
                f"There is a column of stats with type "
                f"[{type(first)}], which is not supported to be "
                f"analyzed for now."
            )
            return None
        if type(first) is str:
            # describe(include = 'all') can analyze the string type
            return col
        elif type(first) is list:
            # flatten and infer the type
            col = col.explode().infer_objects()
            return col

    def analyze(self, percentiles=[], num_proc=1, skip_export=False):
        """
        Apply overall analysis on the whole dataset based on the describe
        method of pandas.

        :param percentiles: percentiles to analyze
        :param num_proc: number of processes to analyze the dataset
        :param skip_export: whether export the results to disk
        :return: the overall analysis result.
        """
        # merge default and customized percentiles and get overall information
        percentiles = list(set(percentiles + self.default_percentiles))

        # merge stats and meta
        stats_and_meta = pd.concat([self.stats, self.meta], axis=1)
        all_columns = stats_and_meta.columns

        results = []
        pool = Pool(num_proc)
        for col_name in all_columns:
            this_col = self.refine_single_column(stats_and_meta[col_name])
            if this_col is None:
                continue
            res = pool.apply_async(
                _single_column_analysis,
                kwds={
                    "col": this_col,
                    "percentiles": percentiles,
                    "include": "all",
                },
            )
            results.append(res)
        pool.close()
        pool.join()
        result_cols = [res.get() for res in tqdm(results)]
        overall = pd.DataFrame(result_cols).T

        # export to result report file
        if not skip_export:
            overall.to_csv(os.path.join(self.output_path, "overall.csv"))
            overall.to_markdown(os.path.join(self.output_path, "overall.md"))

        return overall

import logging
import math
import os

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from wordcloud import WordCloud

from data_juicer.utils.constant import DEFAULT_PREFIX, Fields

from .overall_analysis import OverallAnalysis

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
FONT = os.environ.get("ANALYZER_FONT", "Heiti SC")
FONT_PATH = fm.findfont(FONT)

plt.rcParams["font.sans-serif"] = [FONT]
plt.rcParams["axes.unicode_minus"] = False


def get_row_col(total_num, factor=2):
    """
    Given the total number of stats figures, get the "best" number of rows and
    columns. This function is needed when we need to store all stats figures
    into one image.

    :param total_num: Total number of stats figures
    :param factor: Number of sub-figure types in each figure. In
        default, it's 2, which means there are histogram and box plot
        for each stat figure
    :return: "best" number of rows and columns, and the grid list
    """
    if factor <= 0 or total_num <= 0:
        return 0, 0, []
    n = total_num * factor  # actual number of figures
    now_col = factor  # search from the minimum number of columns
    now_row = total_num
    for col in range(factor, n + 1, factor):
        row = n * 1.0 / col
        if row != int(row):  # skip non-integer results
            continue
        if col > row:
            # object: minimum the difference between number of columns and rows
            if abs(col - row) > abs(now_col - now_row):
                break
            else:
                now_row = row
                now_col = col
                break
        now_row = row
        now_col = col

    # different sub-figures of the same stats should be in the same row
    now_col = now_col // factor

    # get grid indexes
    grids = []
    for i in range(total_num):
        grids.append((i // now_col, i % now_col))

    return int(now_row), int(now_col), grids


class ColumnWiseAnalysis:
    """Apply analysis on each column of stats respectively."""

    def __init__(self, dataset, output_path, overall_result=None, save_stats_in_one_file=True):
        """
        Initialization method

        :param dataset: the dataset to be analyzed
        :param output_path: path to store the analysis results
        :param overall_result: optional precomputed overall stats result
        :param save_stats_in_one_file: whether save all analysis figures of all
            stats into one image file
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

        # if no overall description provided, analyze it from scratch
        if overall_result is None:
            oa = OverallAnalysis(dataset, output_path)
            overall_result = oa.analyze()
        self.overall_result = overall_result

        self.save_stats_in_one_file = save_stats_in_one_file

    def analyze(self, show_percentiles=False, show=False, skip_export=False):
        """
        Apply analysis and draw the analysis figure for stats.

        :param show_percentiles: whether to show the percentile line in
            each sub-figure. If it's true, there will be several red
            lines to indicate the quantiles of the stats distributions
        :param show: whether to show in a single window after drawing
        :param skip_export: whether save the results into disk
        :return:
        """
        # number of sub-figures for each stat. There are histogram and box plot
        # for now, so it's 2.
        num_subcol = 2

        # Default width and height unit for each sub-figure
        width_unit = 4
        height_unit = 6

        stats_and_meta = pd.concat([self.stats, self.meta], axis=1)
        all_columns = [
            col_name for col_name in stats_and_meta.columns.to_list() if col_name in self.overall_result.columns
        ]
        num = len(all_columns)

        # get the recommended "best" number of columns and rows
        rec_row, rec_col, grid_indexes = get_row_col(num, num_subcol)

        if self.save_stats_in_one_file:
            # if save_stats_in_one_file is opened, use recommended "best"
            # number of columns and rows to initialize the image panel.
            rec_width = rec_col * num_subcol * width_unit
            rec_height = rec_row * height_unit
            fig = plt.figure(figsize=(rec_width, rec_height), layout="constrained")
            subfigs = fig.subfigures(rec_row, rec_col, wspace=0.01)
        for i, column_name in enumerate(tqdm(all_columns, desc="Column")):
            data = stats_and_meta[column_name]
            # explode data to flatten inner list
            data = data.explode().infer_objects()
            grid = grid_indexes[i]
            if self.save_stats_in_one_file:
                if rec_col == 1:
                    grid = grid[0]
                elif rec_row == 1:
                    grid = grid[1]

                if rec_col == 1 and rec_row == 1:
                    subfig = subfigs
                else:
                    subfig = subfigs[grid]
                subfig.set_facecolor("0.85")

            # numeric or string via nan. Apply different plot method for them.
            sampled_top = self.overall_result[column_name].get("top")
            if pd.isna(sampled_top):
                # numeric or numeric list -- draw histogram and box plot for
                # this stat
                percentiles = self.overall_result[column_name] if show_percentiles else None

                # get axes for each subplot
                if self.save_stats_in_one_file:
                    axes = subfig.subplots(1, num_subcol)
                else:
                    axes = [None] * num_subcol

                if not skip_export:
                    # draw histogram
                    self.draw_hist(
                        axes[0],
                        data,
                        os.path.join(self.output_path, f"{column_name}-hist.png"),
                        percentiles=percentiles,
                    )

                    # draw box
                    self.draw_box(
                        axes[1], data, os.path.join(self.output_path, f"{column_name}-box.png"), percentiles=percentiles
                    )
            else:
                # object (string) or string list -- only draw histogram for
                # this stat
                if self.save_stats_in_one_file:
                    axes = subfig.subplots(1, num_subcol)
                else:
                    axes = [None] * num_subcol

                if not skip_export:
                    self.draw_hist(axes[0], data, os.path.join(self.output_path, f"{column_name}-hist.png"))

                    self.draw_wordcloud(axes[1], data, os.path.join(self.output_path, f"{column_name}-wordcloud.png"))

            # add a title to the figure of this stat
            if self.save_stats_in_one_file:
                subfig.suptitle(f"{data.name}", fontsize="x-large", fontweight="bold")

        if self.save_stats_in_one_file:
            fig = plt.gcf()
            if not skip_export:
                fig.savefig(os.path.join(self.output_path, "all-stats.png"))
            if show:
                plt.show()
            else:
                pass
                # TODO: (fixme) the saved png sometime are blank
                plt.clf()

    def draw_hist(self, ax, data, save_path, percentiles=None, show=False):
        """
        Draw the histogram for the data.

        :param ax: the axes to draw
        :param data: data to draw
        :param save_path: the path to save the histogram figure
        :param percentiles: the overall analysis result of the data
            including percentile information
        :param show: whether to show in a single window after drawing
        :return:
        """
        # recommended number of bins
        data_num = len(data)
        rec_bins = max(int(math.sqrt(data_num)), 10)

        # if ax is None, using plot method in pandas
        if ax is None:
            ax = data.hist(bins=rec_bins, figsize=(20, 16))
        else:
            ax.hist(data, bins=rec_bins)

        # set axes
        ax.set_xlabel(data.name)
        ax.set_ylabel("Count")

        # draw percentile lines if it's not None
        if percentiles is not None:
            ymin, ymax = ax.get_ylim()
            for percentile in percentiles.keys():
                # skip other information
                if percentile in {"count", "unique", "top", "freq", "std"}:
                    continue
                value = percentiles[percentile]

                ax.vlines(x=value, ymin=ymin, ymax=ymax, colors="r")
                ax.text(x=value, y=ymax, s=percentile, rotation=30, color="r")
                ax.text(x=value, y=ymax * 0.97, s=str(round(value, 3)), rotation=30, color="r")

        if not self.save_stats_in_one_file:
            # save into file
            plt.savefig(save_path)

            if show:
                plt.show()
            else:
                # if no showing, we need to clear this axes to avoid
                # accumulated overlapped figures in different draw_xxx function
                # calling
                ax.clear()
        else:
            # add a little rotation on labels of x axis to avoid overlapping
            ax.tick_params(axis="x", rotation=25)

    def draw_box(self, ax, data, save_path, percentiles=None, show=False):
        """
        Draw the box plot for the data.

        :param ax: the axes to draw
        :param data: data to draw
        :param save_path: the path to save the box figure
        :param percentiles: the overall analysis result of the data
            including percentile information
        :param show: whether to show in a single window after drawing
        :return:
        """
        # if ax is None, using plot method in pandas
        if ax is None:
            ax = data.plot.box(figsize=(20, 16))
        else:
            ax.boxplot(data)

        # set axes
        ax.set_ylabel(data.name)

        # draw percentile lines if it's not None
        if percentiles is not None:
            xmin, xmax = ax.get_xlim()
            for percentile in percentiles.keys():
                # skip other information
                if percentile in {"count", "unique", "top", "freq", "std"}:
                    continue
                value = percentiles[percentile]

                ax.hlines(y=value, xmin=xmin, xmax=xmax, colors="r")
                ax.text(y=value, x=xmin + (xmax - xmin) * 0.6, s=f"{percentile}: {round(value, 3)}", color="r")

        if not self.save_stats_in_one_file:
            # save into file
            plt.savefig(save_path)

            if show:
                plt.show()
            else:
                # if no showing, we need to clear this axes to avoid
                # accumulated overlapped figures in different draw_xxx function
                # calling
                ax.clear()

    def draw_wordcloud(self, ax, data, save_path, show=False):
        word_list = data.tolist()
        word_nums = {}
        for w in word_list:
            if w is None:
                continue
            if w in word_nums:
                word_nums[w] += 1
            else:
                word_nums[w] = 1

        wc = WordCloud(font_path=FONT_PATH, width=400, height=320)
        wc.generate_from_frequencies(word_nums)

        if ax is None:
            ax = plt.figure(figsize=(20, 16))
        else:
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")

        if not self.save_stats_in_one_file:
            # save into file
            wc.to_file(save_path)

            if show:
                plt.show()
            else:
                # if no showing, we need to clear this axes to avoid
                # accumulated overlapped figures in different draw_xxx function
                # calling
                ax.clear()

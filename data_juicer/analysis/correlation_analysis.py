import numbers
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_juicer.utils.constant import Fields


def draw_heatmap(data, row_labels, col_labels, ax=None, cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current Axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}", textcolors=("black", "white"), threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.0

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            im.axes.text(j, i, valfmt(data[i, j], None), **kw)


def is_numeric_list_series(series):
    """
    Whether a series is a numerical-list column.
    """
    # drop nan
    non_null = series.dropna()
    if non_null.empty:
        return False

    # check if all values are lists
    all_lists = non_null.apply(lambda x: isinstance(x, list)).all()
    if not all_lists:
        return False

    # check if there are non-empty lists
    has_non_empty_list = non_null.apply(lambda x: isinstance(x, list) and len(x) > 0).any()
    if not has_non_empty_list:
        return False

    # check if all values in the list are numeric
    all_numeric = non_null.apply(lambda x: all(isinstance(i, numbers.Number) for i in x) if len(x) > 0 else True).all()

    return all_numeric


class CorrelationAnalysis:
    """Analyze the correlations among different stats. Only for numerical stats."""

    def __init__(self, dataset, output_path):
        """
        Initialization method.

        :param dataset: the dataset to be analyzed
        :param output_path: path to store the analysis results
        """
        self.stats = pd.DataFrame(dataset[Fields.stats])
        # only keep the numeric columns
        for col_name in self.stats.columns:
            if np.issubdtype(self.stats[col_name].dtype, np.number):
                continue
            elif is_numeric_list_series(self.stats[col_name]):
                self.stats[col_name] = self.stats[col_name].apply(
                    lambda x: np.mean(x) if isinstance(x, list) and len(x) > 0 else 0
                )
            else:
                self.stats = self.stats.drop(col_name, axis=1)

        self.output_path = output_path
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def analyze(self, method="pearson", show=False, skip_export=False):
        assert method in {"pearson", "kendall", "spearman"}
        columns = self.stats.columns
        if len(columns) <= 0:
            return None
        corr = self.stats.corr(method)

        fig, ax = plt.subplots(figsize=(16, 14))
        im, cbar = draw_heatmap(corr, columns, columns, ax=ax, cmap="YlGn", cbarlabel="correlation coefficient")
        annotate_heatmap(im, valfmt="{x:.2f}")
        if not skip_export:
            plt.savefig(
                os.path.join(self.output_path, f"stats-corr-{method}.png"),
                bbox_inches="tight",
                dpi=fig.dpi,
                pad_inches=0,
            )
            if show:
                plt.show()
            else:
                ax.clear()

        return corr

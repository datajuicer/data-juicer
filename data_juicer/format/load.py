import os

from data_juicer.format.formatter import FORMATTERS, BaseFormatter
from data_juicer.utils.file_utils import find_files_with_suffix


def load_formatter(dataset_path, text_keys=None, suffixes=None, add_suffix=False, **kwargs) -> BaseFormatter:
    """
    Load the appropriate formatter for different types of data formats.

    :param dataset_path: Path to dataset file or dataset directory
    :param text_keys: key names of field that stores sample text.
        Default: None
    :param suffixes: the suffix of files that will be read.
        Default: None
    :param add_suffix: whether to add the file suffix to dataset meta.
        Default: False
    :return: a dataset formatter.
    """

    if suffixes is None:
        suffixes = []
    ext_num = {}
    if os.path.isdir(dataset_path) or os.path.isfile(dataset_path):
        file_dict = find_files_with_suffix(dataset_path, suffixes)
        if not file_dict:
            raise IOError("Unable to find files matching the suffix from {}".format(dataset_path))
        for ext in file_dict:
            ext_num[ext] = len(file_dict[ext])

    # local dataset
    if ext_num:
        formatter_num = {}
        for name, formatter in FORMATTERS.modules.items():
            formatter_num[name] = 0
            for ext in ext_num:
                if ext in formatter.SUFFIXES:
                    formatter_num[name] += ext_num[ext]
        formatter = max(formatter_num, key=lambda x: formatter_num[x])
        target_suffixes = set(ext_num.keys()).intersection(set(FORMATTERS.modules[formatter].SUFFIXES))
        if not target_suffixes:
            raise ValueError(
                f"No suitable formatter found for {dataset_path}. "
                f"Supported extensions: "
                f"{[f.SUFFIXES for f in FORMATTERS.modules.values()]}"
            )
        return FORMATTERS.modules[formatter](
            dataset_path, text_keys=text_keys, suffixes=target_suffixes, add_suffix=add_suffix, **kwargs
        )

    else:
        raise ValueError(
            f"Unable to load the dataset from [{dataset_path}]. "
            f"It might be because Data-Juicer doesn't support "
            f"the format of this dataset, or the path of this "
            f"dataset is incorrect.Please check if it's a valid "
            f"dataset path and retry."
        )

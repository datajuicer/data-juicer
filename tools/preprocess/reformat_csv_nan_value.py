# This tool is used to reformat csv or tsv files which may contain Nan values
# in some field to several jsonl files.

import os
import pathlib
from multiprocessing import Pool

import fire
from datasets import Dataset


def reformat_nan_value(fp, jsonl_fp, keep_default_na, kwargs):
    """
    Reformat a csv/tsv file with kwargs.
    :param fp: a csv/tsv file
    :param jsonl_fp: path to save jsonl file
    :param keep_default_na: if False, no string will be parsed as NaN,
            otherwise only the default NaN values are used for parsing.
    :param kwargs: for tsv file,  kwargs["sep'} is `\t`
    :return: iterator over files,
    """
    ds = Dataset.from_csv(fp, keep_default_na=keep_default_na, **kwargs)
    ds.to_json(jsonl_fp, force_ascii=False)
    pass


def fp_iter(src_dir, suffix):
    """
    Find all files endswith the specified suffix in the source directory.
    :param src_dir: path to source dataset directory
    :return: iterator over files,
    """
    for fp in pathlib.Path(src_dir).glob(f"*{suffix}"):
        yield fp


def main(src_dir, target_dir, suffixes=[".csv"], is_tsv=False, keep_default_na=False, num_proc=1, **kwargs):
    """
    Reformat csv or tsv files that may contain Nan values using HuggingFace
    to load with extra args, e.g. set `keep_default_na` to False
    :param src_dir: path that's stores filenames are like "*.csv" or "*.tsv".
    :param target_dir: path to store the converted jsonl files.
    :param suffixes: files with suffixes to be to process, multi-suffixes args
                   like `--suffixes "'.tsv', '.csv'"
    :param is_tsv: if True, sep will be set to '\t'. Default ','.
    :param keep_default_na: if False, no strings will be parsed as NaN,
                otherwise only the default NaN values are used for parsing.
    :param num_proc: number of process workers, Default 1.
    :param kwargs: optional extra args for Dataset loading csv/tsv
    """
    # check if the source directory exists
    if not os.path.exists(src_dir):
        raise ValueError("The raw source data directory does not exist," " Please check and retry.")
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)

    if kwargs is None:
        kwargs = {}

    if is_tsv:
        kwargs["sep"] = "\t"

    if isinstance(suffixes, str):
        suffixes = [suffixes]

    pool = Pool(num_proc)
    for suffix in suffixes:
        for fp in fp_iter(src_dir, suffix):
            jsonl_fp = os.path.join(target_dir, fp.name.replace(suffix, ".jsonl"))
            pool.apply_async(reformat_nan_value, args=(str(fp), jsonl_fp, keep_default_na, kwargs))
    pool.close()
    pool.join()


if __name__ == "__main__":
    fire.Fire(main)

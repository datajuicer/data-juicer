# This tool is used to reformat jsonl files which may have Nan values
# in some field.

import os
import pathlib
from multiprocessing import Pool

import fire
import jsonlines
import pandas as pd
from datasets import Dataset


def check_dict_non_nan(obj):
    """
    Check if all fields in the dict object are non-Nan
    :param: a dict object
    :return: True if all fields in the dict object are non-Nan,
            else False
    """
    no_nan = True
    for key, value in obj.items():
        if isinstance(value, dict):
            no_nan = no_nan & check_dict_non_nan(value)
        elif pd.isna(value) or pd.isnull(value):
            return False
    return no_nan


def get_non_nan_features(src_dir):
    """
    Get the first object feature which does not contain Nan value.
    :param src_dir: path which stores jsonl files.
    :return: reference feature of dataset.
    """
    for fp in fp_iter(src_dir):
        with jsonlines.open(fp, "r") as reader:
            for obj in reader:
                if check_dict_non_nan(obj):
                    ds = Dataset.from_list([obj])
                    return ds.features
    return None


def reformat_jsonl(fp, jsonl_fp, features):
    """
    Reformat a jsonl file with reference features
    :param fp: input jsonl file
    :param jsonl_fp: formatted jsonl file
    :param features: reference feature to use for dataset.
    """
    with jsonlines.open(fp, "r") as reader:
        objs = [obj for obj in reader]
    ds = Dataset.from_list(objs, features=features)
    ds.to_json(jsonl_fp, force_ascii=False)


def fp_iter(src_dir):
    """
    Find all jsonl files in the source directory.
    :param src_dir: path to source dataset directory
    :return: iterator over jsonl files
    """
    for fp in pathlib.Path(src_dir).glob("*.jsonl"):
        yield fp


def main(src_dir, target_dir, num_proc=1):
    """
    Reformat the jsonl files which may contain Nan values. Traverse jsonl
    files to find the first object that does not contain Nan as a
    reference feature type, then set it for loading all jsonl files.
    :param src_dir: path that's stores jsonl files.
    :param target_dir: path to store the converted jsonl files.
    :param num_proc: number of process workers. Default it's 1.
    """

    # check if the source directory exists
    if not os.path.exists(src_dir):
        raise ValueError("The raw source data directory does not exist," " Please check and retry.")
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)

    pool = Pool(num_proc)
    features = get_non_nan_features(src_dir)
    for fp in fp_iter(src_dir):
        print(fp)
        jsonl_fp = os.path.join(target_dir, fp.name)
        pool.apply_async(reformat_jsonl, args=(str(fp), jsonl_fp, features))

    pool.close()
    pool.join()


if __name__ == "__main__":
    fire.Fire(main)

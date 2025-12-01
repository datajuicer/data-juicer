# This tool is used to split datasets to sub-datasets
# by fast-text language model.

import os

import fire
import pandas as pd
from loguru import logger

from data_juicer.config.config import get_default_cfg
from data_juicer.core.data import add_same_content_to_new_column
from data_juicer.core.data.dataset_builder import DatasetBuilder
from data_juicer.ops.filter.language_id_score_filter import LanguageIDScoreFilter
from data_juicer.utils.constant import Fields, StatsKeys


def keep_by_lang(sample, lang):
    """
    Keep samples with the specified language.
    :param sample: a sample in dataset
    :param lang: the specified language
    :return: True to keep,  False to discard
    """
    if sample[Fields.stats][StatsKeys.lang] == lang:
        return True
    return False


def main(src_dir, target_dir, text_key=None, suffixes=[], num_proc=1):
    """
    Load dataset from the source directory, then apply language identification
    using the operation filter called `LanguageIDScoreFilter`,
    finally, split the dataset by language and save it.
    :param src_dir: path to store the dataset.
    :param target_dir: path to store subset files(`jsonl` format)
    :param text_key: key name of field that stores sample text, default `text`:
    :param suffixes: files with suffixes to be loaded, default None
    :param num_proc: number of processes to process dataset, default 1.
    """
    if text_key is None:
        text_key = "text"

    # check if the source directory exists.
    if not os.path.exists(src_dir):
        raise ValueError("The raw source data directory does not exist," " Please check and retry.")
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)

    cfg = get_default_cfg()
    cfg.dataset_path = src_dir
    builder = DatasetBuilder(cfg)
    dataset = builder.load_dataset(num_proc=num_proc)

    op = LanguageIDScoreFilter(text_key=text_key)

    if Fields.stats not in dataset.features:
        # only add stats when calling filter op
        dataset = dataset.map(
            add_same_content_to_new_column,
            fn_kwargs={"new_column_name": Fields.stats, "initial_value": {}},
            num_proc=num_proc,
            desc="Adding new column for stats",
        )

    # identify language
    dataset = dataset.map(op.compute_stats, num_proc=num_proc)

    langs = pd.DataFrame(dataset[Fields.stats])[StatsKeys.lang]
    unique_langs = list(set(langs))

    logger.info(f"There are {len(dataset)} in dataset")
    logger.info(f"Languages in dataset are {unique_langs}")

    # split and save subset of dataset by language
    for lang in unique_langs:
        ds = dataset.filter(keep_by_lang, num_proc=num_proc, fn_kwargs=dict(lang=lang))

        logger.info(f"There are {len(ds)} with language [{lang}]")
        jsonl_fp = os.path.join(target_dir, lang + ".jsonl")
        ds.to_json(jsonl_fp, force_ascii=False)


if __name__ == "__main__":
    fire.Fire(main)

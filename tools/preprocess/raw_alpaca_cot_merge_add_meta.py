# This tool is used for converting the raw Alpaca-Cot data downloaded
# from Huggingface (ref: https://huggingface.co/datasets/QingyiSi/Alpaca-CoT)
# to several jsonl files.

import os
import pathlib
from multiprocessing import Pool

import fire
from datasets import load_dataset
from loguru import logger

meta_dict = {
    "Chain-of-Thought": {  # sub directory
        "Task": "MT",  # Alpaca-Cot original Task
        "Gen": "HG",  # Alpaca-Cot original Gen
        "Lang": "EN/CN",  # Alpaca-Cot original Language
        "Dataset": "Chain-of-Thought",  # sub directory
        "CFT-MR": False,  # whether is Multi-round Dialog data, added by Data-Juicer
        "IFT": True,  # whether is IFT data, added by Data-Juicer
        "CFT-SR": False,  # whether is CFT single-round data, added by
        # Data-Juicer
        "CFT-P": False,  # whether is Preference data, added by Data-Juicer
    },
    "GPT4all": {
        "Task": "MT",
        "Gen": "COL",
        "Lang": "EN",
        "Dataset": "GPT4all",
        "CFT-MR": False,
        "IFT": True,
        "CFT-SR": True,
        "CFT-P": False,
    },
    "GPTeacher": {
        "Task": "MT",
        "Gen": "SI",
        "Lang": "EN",
        "Dataset": "GPTeacher",
        "CFT-MR": False,
        "IFT": False,
        "CFT-SR": True,
        "CFT-P": False,
    },
    "Guanaco": {
        "Task": "MT",
        "Gen": "SI",
        "Lang": "ML",
        "Dataset": "Guanaco",
        "CFT-MR": False,
        "IFT": False,
        "CFT-SR": True,
        "CFT-P": False,
    },
    "HC3": {
        "Task": "TS",
        "Gen": "MIX",
        "Lang": "EN/CN",
        "Dataset": "HC3",
        "CFT-MR": False,
        "IFT": False,
        "CFT-SR": True,
        "CFT-P": True,
    },
    "alpaca": {
        "Task": "MT",
        "Gen": "SI",
        "Lang": "EN",
        "Dataset": "alpaca",
        "CFT-MR": False,
        "IFT": False,
        "CFT-SR": True,
        "CFT-P": False,
    },
    "Natural-Instructions": {
        "Task": "MT",
        "Gen": "COL",
        "Lang": "ML",
        "Dataset": "Natural-Instructions",
        "CFT-MR": False,
        "IFT": True,
        "CFT-SR": False,
        "CFT-P": False,
    },
    "belle_cn": {
        "Task": "TS/MT",
        "Gen": "SI",
        "Lang": "CN",
        "Dataset": "belle_cn",
        "CFT-MR": False,
        "IFT": False,
        "CFT-SR": True,
        "CFT-P": False,
    },
    "instinwild": {
        "Task": "MT",
        "Gen": "SI",
        "Lang": "EN/CN",
        "Dataset": "instinwild",
        "CFT-MR": False,
        "IFT": False,
        "CFT-SR": True,
        "CFT-P": False,
    },
    "prosocial-dialog": {
        "Task": "TS",
        "Gen": "MIX",
        "Lang": "EN",
        "Dataset": "prosocial-dialog",
        "CFT-MR": False,
        "IFT": False,
        "CFT-SR": True,
        "CFT-P": False,
    },
    "finance": {
        "Task": "TS",
        "Gen": "COL",
        "Lang": "EN",
        "Dataset": "finance",
        "CFT-MR": False,
        "IFT": False,
        "CFT-SR": True,
        "CFT-P": False,
    },
    "xP3": {
        "Task": "MT",
        "Gen": "COL",
        "Lang": "ML",
        "Dataset": "xP3",
        "CFT-MR": False,
        "IFT": True,
        "CFT-SR": False,
        "CFT-P": False,
    },
    "firefly": {
        "Task": "MT",
        "Gen": "COL",
        "Lang": "CN",
        "Dataset": "firefly",
        "CFT-MR": False,
        "IFT": True,
        "CFT-SR": False,
        "CFT-P": False,
    },
    "instruct": {
        "Task": "MT",
        "Gen": "COL",
        "Lang": "EN",
        "Dataset": "instruct",
        "CFT-MR": False,
        "IFT": False,
        "CFT-SR": True,
        "CFT-P": False,
    },
    "CodeAlpaca": {
        "Task": "TS",
        "Gen": "SI",
        "Lang": "EN",
        "Dataset": "CodeAlpaca",
        "CFT-MR": False,
        "IFT": True,
        "CFT-SR": False,
        "CFT-P": False,
    },
    "alpacaGPT4": {
        "Task": "MT",
        "Gen": "SI",
        "Lang": "EN/CN",
        "Dataset": "alpacaGPT4",
        "CFT-MR": False,
        "IFT": False,
        "CFT-SR": True,
        "CFT-P": True,
    },
    "webGPT": {
        "Task": "TS",
        "Gen": "MIX",
        "Lang": "EN",
        "Dataset": "webGPT",
        "CFT-MR": False,
        "IFT": True,
        "CFT-SR": False,
        "CFT-P": True,
    },
    "dolly": {
        "Task": "TS",
        "Gen": "HG",
        "Lang": "EN",
        "Dataset": "dolly",
        "CFT-MR": False,
        "IFT": False,
        "CFT-SR": True,
        "CFT-P": False,
    },
    "baize": {
        "Task": "MT",
        "Gen": "COL",
        "Lang": "EN",
        "Dataset": "baize",
        "CFT-MR": False,
        "IFT": False,
        "CFT-SR": True,
        "CFT-P": False,
    },
    "hh-rlhf": {
        "Task": "TS",
        "Gen": "MIX",
        "Lang": "EN",
        "Dataset": "hh-rlhf",
        "CFT-MR": True,
        "IFT": False,
        "CFT-SR": True,
        "CFT-P": True,
    },
    "OIG": {
        "Task": "MT",
        "Gen": "COL",
        "Lang": "EN",
        "Dataset": "OIG",
        "CFT-MR": False,
        "IFT": False,
        "CFT-SR": True,
        "CFT-P": False,
    },
    "GAOKAO": {
        "Task": "MT",
        "Gen": "COL",
        "Lang": "CN",
        "Dataset": "GAOKAO",
        "CFT-MR": False,
        "IFT": True,
        "CFT-SR": False,
        "CFT-P": False,
    },
    "camel": {
        "Task": "MT",
        "Gen": "SI",
        "Lang": "EN",
        "Dataset": "camel",
        "CFT-MR": False,
        "IFT": True,
        "CFT-SR": False,
        "CFT-P": False,
    },
    "FLAN-Muffin": {
        "Task": "MT",
        "Gen": "COL",
        "Lang": "EN",
        "Dataset": "FLAN-Muffin",
        "CFT-MR": False,
        "IFT": True,
        "CFT-SR": False,
        "CFT-P": False,
    },
    "COIG": {
        "Task": "MT",
        "Gen": "COL",
        "Lang": "CN",
        "Dataset": "COIG",
        "CFT-MR": False,
        "IFT": False,
        "CFT-SR": True,
        "CFT-P": False,
    },
    "gpt4tools": {
        "Task": "MT",
        "Gen": "SI",
        "Lang": "EN",
        "Dataset": "gpt4tools",
        "CFT-MR": False,
        "IFT": True,
        "CFT-SR": False,
        "CFT-P": False,
    },
    "ShareGPT": {
        "Task": "MT",
        "Gen": "MIX",
        "Lang": "EN",
        "Dataset": "ShareGPT",
        "CFT-MR": True,
        "IFT": False,
        "CFT-SR": True,
        "CFT-P": False,
    },
    "Auto-CoT": {
        "Task": "MT",
        "Gen": "COL",
        "Lang": "EN",
        "Dataset": "Auto-CoT",
        "CFT-MR": False,
        "IFT": True,
        "CFT-SR": False,
        "CFT-P": False,
    },
    "MOSS": {
        "Task": "TS",
        "Gen": "SI",
        "Lang": "EN/CN",
        "Dataset": "MOSS",
        "CFT-MR": False,
        "IFT": False,
        "CFT-SR": True,
        "CFT-P": False,
    },
    "ultrachat": {
        "Task": "TS",
        "Gen": "SI",
        "Lang": "EN",
        "Dataset": "ultrachat",
        "CFT-MR": False,
        "IFT": False,
        "CFT-SR": True,
        "CFT-P": False,
    },
    "Chinese-medical": {
        "Task": "TS",
        "Gen": "COL",
        "Lang": "CN",
        "Dataset": "Chinese-medical",
        "CFT-MR": False,
        "IFT": False,
        "CFT-SR": True,
        "CFT-P": False,
    },
    "CSL": {
        "Task": "MT",
        "Gen": "COL",
        "Lang": "CN",
        "Dataset": "CSL",
        "CFT-MR": False,
        "IFT": True,
        "CFT-SR": False,
        "CFT-P": False,
    },
    "pCLUE": {
        "Task": "MT",
        "Gen": "COL",
        "Lang": "CN",
        "Dataset": "pCLUE",
        "CFT-MR": False,
        "IFT": True,
        "CFT-SR": False,
        "CFT-P": False,
    },
    "news_commentary": {
        "Task": "TS",
        "Gen": "COL",
        "Lang": "CN",
        "Dataset": "news_commentary",
        "CFT-MR": False,
        "IFT": True,
        "CFT-SR": False,
        "CFT-P": False,
    },
    "StackExchange": {
        "Task": "MT",
        "Gen": "COL",
        "Lang": "EN",
        "Dataset": "StackExchange",
        "CFT-MR": False,
        "IFT": False,
        "CFT-SR": True,
        "CFT-P": True,
    },
    "ConvAI2": {
        "Task": "TS",
        "Gen": "HG",
        "Lang": "EN",
        "Dataset": "ConvAI2",
        "CFT-MR": False,
        "IFT": False,
        "CFT-SR": True,
        "CFT-P": False,
    },
    "FastChat": {
        "Task": "MT",
        "Gen": "SI",
        "Lang": "EN",
        "Dataset": "FastChat",
        "CFT-MR": False,
        "IFT": False,
        "CFT-SR": True,
        "CFT-P": False,
    },
    "Tabular-LLM-Data": {
        "Task": "MT",
        "Gen": "COL",
        "Lang": "EN/CN",
        "Dataset": "Tabular-LLM-Data",
        "CFT-MR": False,
        "IFT": True,
        "CFT-SR": False,
        "CFT-P": False,
    },
    "ThoughtSource": {
        "Task": "MT",
        "Gen": "COL",
        "Lang": "EN",
        "Dataset": "ThoughtSource",
        "CFT-MR": False,
        "IFT": True,
        "CFT-SR": False,
        "CFT-P": False,
    },
}


def merge_and_add_meta(filename, target_dir):
    """
    Merge `instruction`/`input`/`output` to `text` for process,
    and add meta info.
    :param filename: input dataset file
    :param target_dir: path to save updated dataset
    """

    ds = load_dataset("json", data_files=[filename], split="train")

    if "instruction" in ds.features and "input" in ds.features and "output" in ds.features:
        for column_name in ds.column_names:
            if column_name not in ["instruction", "input", "output"]:
                ds = ds.remove_columns(column_name)
    else:
        logger.warning(
            f'Can not find ["instruction", "input", "output"] in \
             {filename}, do nothing.'
        )
        return

    meta = None
    for key in meta_dict.keys():
        if key in filename:
            meta = meta_dict[key]

    if meta is None:
        logger.warning(f"Can not find meta in {filename}, do nothing.")
        return

    def _merge_and_add_meta(sample, path, meta):
        """
        Merge `instruction`/`input`/`output` to `text` for process,
        and add meta info.
        :param sample: a dict sample in dataset
        :param path: sample in which file
        :param meta: meta added to sample
        :return: updated sample
        """
        sample["text"] = " ".join([sample["instruction"], sample["input"], sample["output"]])
        sample["meta"] = meta
        sample["meta"]["origin_path"] = path
        return sample

    path = "".join(["Alpaca-CoT", filename.split("Alpaca-CoT")[1]])
    ds = ds.map(_merge_and_add_meta, num_proc=48, fn_kwargs={"path": path, "meta": meta})

    if len(ds) > 0:
        out_file = "".join([target_dir, filename.split("Alpaca-CoT")[1]])
        out_file = out_file.replace(".json", ".jsonl")
        dir_name = os.path.dirname(out_file)
        os.makedirs(dir_name, exist_ok=True)
        ds.to_json(out_file, force_ascii=False)


def fp_iter(src_dir):
    """
    Find all tar files in the source directory.
    :param src_dir: path to source dataset directory
    :return: iterator over json files
    """
    for fp in pathlib.Path(src_dir).rglob("*.json"):
        yield fp


def main(src_dir, target_dir, num_proc=4):
    """
    Load dataset from the source directory, then apply language identification
    using the operation filter called `LanguageIDScoreFilter`,
    finally, split the dataset by language and save it.
    :param src_dir: path that's store dataset directory
    :param target_dir: path to store subset files(`jsonl` format)
    :param num_proc: number of processes to process dataset, default 1.
    """

    # check if the source directory exists.
    if not os.path.exists(src_dir):
        raise ValueError("The raw source data directory does not exist," " Please check and retry.")
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)

    pool = Pool(num_proc)

    for fp in fp_iter(src_dir):
        pool.apply_async(merge_and_add_meta, args=(str(fp), target_dir))

    pool.close()
    pool.join()


if __name__ == "__main__":
    fire.Fire(main)

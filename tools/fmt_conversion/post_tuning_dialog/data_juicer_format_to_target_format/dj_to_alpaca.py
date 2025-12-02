# This tool is used to convert dataset in Data-Juicer format to a
# target dataset in Alpaca-like format.
#
# Data-Juicer format (query-response format):
# [
#   {
#     "system": "<system>",
#     "instruction": "<query-inst>",
#     "query": "<query-input>",
#     "response": "<response>",
#     "history": [
#       ["human instruction in the first round (optional)", "model response in the first round (optional)"],  # noqa: E501
#       ["human instruction in the second round (optional)", "model response in the second round (optional)"]  # noqa: E501
#     ],
#   },
#   ...
# ]
#
# Corresponding Alpaca format:
# [
#   {
#     "system": "<system>",
#     "instruction": "<query-inst>",
#     "input": "<query-input>",
#     "output": "<response>",
#     "history": [
#       ["human instruction in the first round (optional)", "model response in the first round (optional)"],  # noqa: E501
#       ["human instruction in the second round (optional)", "model response in the second round (optional)"]  # noqa: E501
#     ],
#   },
#   ......
# ]
#
# Reference:
# https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Customization/Custom-dataset.md
# https://github.com/hiyouga/LLaMA-Factory/blob/v0.9.1/data/README.md#alpaca-format

import json
import os

import fire
import jsonlines as jl
from loguru import logger
from tqdm import tqdm


def dj_to_alpaca(
    sample,
    input_key: str = "input",
    output_key: str = "output",
):
    modified_keys = {"query", "response"}
    new_sample = {key: sample[key] for key in sample if key not in modified_keys and sample[key]}

    # key mapping
    if "query" in sample:
        new_sample[input_key] = sample["query"]
    if "response" in sample:
        new_sample[output_key] = sample["response"]

    return new_sample


@logger.catch(reraise=True)
def main(
    src_ds_path: str,
    tgt_ds_path: str,
    input_key: str = "input",
    output_key: str = "output",
):
    """
    Convert a Data-Juicer dataset to the Alpaca-like format.

    :param src_ds_path: the path to the source dataset.
    :param tgt_ds_path: the path to store the converted target dataset.
    :param input_key: the field key to store the query sentence from human.
    :param output_key: the field key to store the response sentence from
        assistant.
    """

    # check arguments
    # check paths
    if not os.path.exists(src_ds_path):
        raise FileNotFoundError(f"Input dataset [{src_ds_path}] can not be found.")
    if not tgt_ds_path.endswith(".json"):
        raise ValueError('Only support "json" target dataset file now.')
    if os.path.dirname(tgt_ds_path) and not os.path.exists(os.path.dirname(tgt_ds_path)):
        logger.info(f"Create directory [{os.path.dirname(tgt_ds_path)}] " f"for the target dataset.")
        os.makedirs(os.path.dirname(tgt_ds_path))

    samples = []
    with jl.open(src_ds_path, "r") as reader:
        for sample in tqdm(reader):
            converted_sample = dj_to_alpaca(sample, input_key=input_key, output_key=output_key)
            samples.append(converted_sample)

    logger.info(f"Store the target dataset into [{tgt_ds_path}].")
    json.dump(samples, open(tgt_ds_path, "w", encoding="utf-8"))


if __name__ == "__main__":
    fire.Fire(main)

# This tool is used to convert dataset in Alpaca format to a
# target dataset in Data-Juicer query-response format.
#
# Alpaca format:
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
# Corresponding Data-Juicer format (query-response format):
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
# Reference:
# https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Customization/Custom-dataset.md
# https://github.com/hiyouga/LLaMA-Factory/blob/v0.9.1/data/README.md#alpaca-format

import json
import os
from typing import List, Union

import fire
import jsonlines as jl
from loguru import logger
from tqdm import tqdm


def alpaca_to_dj(
    sample,
    input_key: str = "input",
    output_key: str = "output",
    multimodal_keys: Union[str, List[str]] = None,
):
    modified_keys = {input_key, output_key}
    if multimodal_keys:
        modified_keys = modified_keys.union(set(multimodal_keys))
    new_sample = {key: sample[key] for key in sample if key not in modified_keys}

    # key mapping for input and output
    if input_key in sample:
        new_sample["query"] = sample[input_key]
    if output_key in sample:
        new_sample["response"] = sample[output_key]

    # update multimodal data
    if multimodal_keys:
        for mm_key in multimodal_keys:
            if not isinstance(sample[mm_key], list):
                new_sample[mm_key] = [sample[mm_key]]
            else:
                new_sample[mm_key] = sample[mm_key]

    return new_sample


@logger.catch(reraise=True)
def main(
    src_ds_path: str,
    tgt_ds_path: str,
    input_key: str = "input",
    output_key: str = "output",
    multimodal_keys: Union[str, List[str]] = None,
):
    """
    Convert an Alpaca-like dataset to the Data-Juicer query-response format.

    :param src_ds_path: the path to the source dataset.
    :param tgt_ds_path: the path to store the converted target dataset.
    :param input_key: the field key to store the query sentence from human.
    :param output_key: the field key to store the response sentence from
        assistant.
    :param multimodal_keys: optional keys to store multimodal data.
    """

    # check arguments
    # check paths
    if not os.path.exists(src_ds_path):
        raise FileNotFoundError(f"Input dataset [{src_ds_path}] can not be found.")
    if not tgt_ds_path.endswith(".jsonl"):
        raise ValueError('Only support "jsonl" target dataset file now.')
    if os.path.dirname(tgt_ds_path) and not os.path.exists(os.path.dirname(tgt_ds_path)):
        logger.info(f"Create directory [{os.path.dirname(tgt_ds_path)}] " f"for the target dataset.")
        os.makedirs(os.path.dirname(tgt_ds_path))

    if isinstance(multimodal_keys, str):
        multimodal_keys = [multimodal_keys]

    # load Alpaca dataset
    logger.info("Loading original dataset.")
    src_ds = json.load(open(src_ds_path, "r", encoding="utf-8"))
    logger.info(f"Load [{len(src_ds)}] samples.")

    with jl.open(tgt_ds_path, "w") as writer:
        for sample in tqdm(src_ds):
            converted_sample = alpaca_to_dj(
                sample, input_key=input_key, output_key=output_key, multimodal_keys=multimodal_keys
            )
            writer.write(converted_sample)
    logger.info(f"Store the target dataset into [{tgt_ds_path}].")


if __name__ == "__main__":
    fire.Fire(main)

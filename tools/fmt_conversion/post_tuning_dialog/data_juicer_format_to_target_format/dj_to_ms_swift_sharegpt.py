# This tool is used to convert dataset in Data-Juicer format to a
# target dataset in ModelScope-Swift ShareGPT format.
#
# Data-Juicer format (query-response format):
# [
#   {
#     "system": "<system>",
#     "query": "<query2>",
#     "response": "<response2>"
#     "history": [
#       [
#         "<query1>",
#         "<response1>"
#       ],
#     ]
#   },
#   ...
# ]
#
# Corresponding ModelScope-Swift ShareGPT format:
# [
#   {
#     "system": "<system>",
#     "conversation": [
#       {
#         "human": "<query1>",
#         "assistant": "<response1>"
#       },
#       {
#         "human": "<query2>",
#         "assistant": "<response2>"
#       }
#     ]
#   },
#   ......
# ]
#
# Reference:
# https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Customization/Custom-dataset.md

import json
import os

import fire
import jsonlines as jl
from loguru import logger
from tqdm import tqdm


def dj_to_ms_swift_sharegpt(
    sample,
    conversation_key: str = "conversation",
    human_key: str = "human",
    assistant_key: str = "assistant",
    system_key: str = "system",
    instruction_key: str = "instruction",
):
    modified_keys = {"query", "response", "history", "system", "instruction"}
    new_sample = {key: sample[key] for key in sample if key not in modified_keys}

    # find system prompt and instruction
    if "system" in sample:
        new_sample[system_key] = sample["system"]
    if "instruction" in sample:
        new_sample[instruction_key] = sample["instruction"]

    # construct conversation
    conversation = []
    # add dialogs
    for query, response in sample["history"]:
        conversation.append(
            {
                human_key: query,
                assistant_key: response,
            }
        )
    conversation.append({human_key: sample["query"], assistant_key: sample["response"] if "response" in sample else ""})

    new_sample[conversation_key] = conversation

    return new_sample


@logger.catch(reraise=True)
def main(
    src_ds_path: str,
    tgt_ds_path: str,
    conversation_key: str = "conversation",
    human_key: str = "human",
    assistant_key: str = "assistant",
    system_key: str = "system",
    instruction_key: str = "instruction",
):
    """
    Convert a Data-Juicer query-response dataset to the ModelScope-Swift
    ShareGPT-like format.

    :param src_ds_path: the path to the source dataset.
    :param tgt_ds_path: the path to store the converted target dataset.
    :param conversation_key: the field key to store conversions.
    :param human_key: the field key to store the sentence from human.
    :param assistant_key: the field key to store the sentence from assistant.
    :param system_key: the field key to store the system prompt.
    :param instruction_key: the field key to store the instruction content.
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

    # load dataset
    samples = []
    with jl.open(src_ds_path, "r") as reader:
        for sample in tqdm(reader):
            converted_sample = dj_to_ms_swift_sharegpt(
                sample,
                conversation_key=conversation_key,
                human_key=human_key,
                assistant_key=assistant_key,
                system_key=system_key,
                instruction_key=instruction_key,
            )
            samples.append(converted_sample)
    logger.info(f"Store the target dataset into [{tgt_ds_path}].")
    json.dump(samples, open(tgt_ds_path, "w", encoding="utf-8"))


if __name__ == "__main__":
    fire.Fire(main)

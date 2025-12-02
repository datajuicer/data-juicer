# This tool is used to convert dataset in Data-Juicer format to a
# target dataset in LLaMA-Factory ShareGPT-like format.
#
# Data-Juicer format (query-response format):
# [
#   {
#     "images": ["coco/train2017/000000033471.jpg"],
#     "query": "Is the bus driving down the street or pulled off to the side?",
#     "response": "The bus is driving down the street, which is crowded with people and other vehicles."  # noqa: E501
#     "history": [
#       [
#         "<image>\nWhat are the colors of the bus in the image?",
#         "The bus in the image is white and red."
#       ],
#       [
#         "What feature can be seen on the back of the bus?",
#         "The back of the bus features an advertisement."
#       ],
#     ]
#   },
#   ...
# ]
#
# Corresponding LLaMA-Factory ShareGPT format:
#   - usually in json format
# [
#   {
#     "images": ["coco/train2017/000000033471.jpg"],
#     "conversations": [
#       {
#         "from": "human",
#         "value": "<image>\nWhat are the colors of the bus in the image?"
#       },
#       {
#         "from": "gpt",
#         "value": "The bus in the image is white and red."
#       },
#       {
#         "from": "human",
#         "value": "What feature can be seen on the back of the bus?"
#       },
#       {
#         "from": "gpt",
#         "value": "The back of the bus features an advertisement."
#       },
#       {
#         "from": "human",
#         "value": "Is the bus driving down the street or pulled off to the side?"  # noqa: E501
#       },
#       {
#         "from": "gpt",
#         "value": "The bus is driving down the street, which is crowded with people and other vehicles."  # noqa: E501
#       }
#     ]
#   },
#   ...
# ]
#
# Reference:
# https://github.com/hiyouga/LLaMA-Factory/blob/v0.9.1/data/README.md#sharegpt-format

import json
import os

import fire
import jsonlines as jl
from loguru import logger
from tqdm import tqdm


def dj_to_llama_factory_sharegpt(
    sample,
    conversations_key: str = "conversations",
    from_key: str = "from",
    value_key: str = "value",
    human_role: str = "user",
    assistant_role: str = "assistant",
    system_role: str = "system",
    instruction_role: str = "instruction",
):
    modified_keys = {"query", "response", "history", "system", "instruction"}
    new_sample = {key: sample[key] for key in sample if key not in modified_keys and sample[key]}

    # construct conversations
    conversations = []
    # add system prompt and instruction
    if "system" in sample and sample["system"] != "":
        conversations.append({from_key: system_role, value_key: sample["system"]})
    if "instruction" in sample and sample["instruction"] != "":
        conversations.append({from_key: instruction_role, value_key: sample["instruction"]})

    # add dialogs
    for query, response in sample["history"]:
        conversations.append(
            {
                from_key: human_role,
                value_key: query,
            }
        )
        conversations.append(
            {
                from_key: assistant_role,
                value_key: response,
            }
        )
    conversations.append(
        {
            from_key: human_role,
            value_key: sample["query"],
        }
    )
    if "response" in sample and sample["response"] != "":
        conversations.append(
            {
                from_key: assistant_role,
                value_key: sample["response"],
            }
        )

    # get the result sample
    new_sample[conversations_key] = conversations

    return new_sample


@logger.catch(reraise=True)
def main(
    src_ds_path: str,
    tgt_ds_path: str,
    conversations_key: str = "conversations",
    from_key: str = "from",
    value_key: str = "value",
    human_role: str = "user",
    assistant_role: str = "assistant",
    system_role: str = "system",
    instruction_role: str = "instruction",
):
    """
    Convert a Data-Juicer dataset to the LLaMA-Factory ShareGPT-like format.

    :param src_ds_path: the path to the source dataset.
    :param tgt_ds_path: the path to store the converted target dataset.
    :param conversations_key: the field key to store conversions.
    :param from_key: the field key to store the sentence from.
    :param value_key: the field key to store the sentence content.
    :param human_role: the role to store the human prompt.
    :param assistant_role: the role to store the instruction content.
    :param system_role: the role to store the system prompt.
    :param instruction_role: the role to store the instruction content.
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
            converted_sample = dj_to_llama_factory_sharegpt(
                sample,
                conversations_key=conversations_key,
                from_key=from_key,
                value_key=value_key,
                human_role=human_role,
                assistant_role=assistant_role,
                system_role=system_role,
                instruction_role=instruction_role,
            )
            samples.append(converted_sample)

    logger.info(f"Store the target dataset into [{tgt_ds_path}].")
    json.dump(samples, open(tgt_ds_path, "w", encoding="utf-8"))


if __name__ == "__main__":
    fire.Fire(main)

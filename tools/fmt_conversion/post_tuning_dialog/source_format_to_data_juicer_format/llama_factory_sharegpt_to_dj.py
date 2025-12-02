# This tool is used to convert dataset in LLaMA-Factory ShareGPT format to a
# target dataset in Data-Juicer query-response format.
#
# LLaMA-Factory ShareGPT format:
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
# Corresponding Data-Juicer format (query-response format):
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
# Reference:
# https://github.com/hiyouga/LLaMA-Factory/blob/v0.9.1/data/README.md#sharegpt-format

import json
import os
from typing import List, Union

import fire
import jsonlines as jl
from loguru import logger
from tqdm import tqdm


def llama_factory_sharegpt_to_dj(
    sample,
    conversations_key: str = "conversations",
    from_key: str = "from",
    value_key: str = "value",
    system_role: str = "system",
    instruction_role: str = "instruction",
    multimodal_keys: Union[str, List[str]] = None,
):
    modified_keys = {conversations_key}
    if multimodal_keys:
        modified_keys = modified_keys.union(set(multimodal_keys))
    new_sample = {key: sample[key] for key in sample if key not in modified_keys}

    # conversations to query, response, history
    conversations = sample[conversations_key]
    # find system prompt and instruction
    system_prompt = ""
    instruction = ""
    remove_idx = []
    for i, conv in enumerate(conversations):
        if conv[from_key] == system_role:
            if system_prompt != "":
                raise NotImplementedError("DO NOT support more than 1 system prompts in the " "conversation for now.")
            system_prompt = conv[value_key]
            remove_idx.append(i)
        elif conv[from_key] == instruction_role:
            if instruction != "":
                raise NotImplementedError("DO NOT support more than 1 instructions in the " "conversation for now.")
            instruction = conv[value_key]
            remove_idx.append(i)
    if len(remove_idx) > 0:
        for i in remove_idx:
            conversations.pop(i)

    # reconstruct conversations
    conv_num = len(conversations)
    if conv_num == 0:
        query = ""
        response = ""
        history = []
    elif conv_num % 2 == 0:
        # the last 2 sentences are query and response
        query = conversations[-2][value_key]
        response = conversations[-1][value_key]
        history = [[conversations[i][value_key], conversations[i + 1][value_key]] for i in range(0, conv_num - 2, 2)]
    else:
        # the last 1 sentence is query and response is empty
        query = conversations[-1][value_key]
        response = ""
        history = [[conversations[i][value_key], conversations[i + 1][value_key]] for i in range(0, conv_num - 1, 2)]

    # get the result sample
    new_sample.update(
        {
            "system": system_prompt,
            "instruction": instruction,
            "query": query,
            "response": response,
            "history": history,
        }
    )

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
    conversations_key: str = "conversations",
    from_key: str = "from",
    value_key: str = "value",
    system_role: str = "system",
    instruction_role: str = "instruction",
    multimodal_keys: Union[str, List[str]] = None,
):
    """
    Convert a LLaMA-Factory ShareGPT-like dataset to the Data-Juicer
    query-response format.

    :param src_ds_path: the path to the source dataset.
    :param tgt_ds_path: the path to store the converted target dataset.
    :param conversations_key: the field key to store conversions.
    :param from_key: the field key to store the sentence from.
    :param value_key: the field key to store the sentence content.
    :param system_role: the field key to store the system prompt.
    :param instruction_role: the field key to store the instruction content.
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

    # load dataset
    logger.info("Loading original dataset.")
    src_ds = json.load(open(src_ds_path, "r", encoding="utf-8"))
    logger.info(f"Load [{len(src_ds)}] samples.")

    with jl.open(tgt_ds_path, "w") as writer:
        for sample in tqdm(src_ds):
            converted_sample = llama_factory_sharegpt_to_dj(
                sample,
                conversations_key=conversations_key,
                from_key=from_key,
                value_key=value_key,
                system_role=system_role,
                instruction_role=instruction_role,
                multimodal_keys=multimodal_keys,
            )
            writer.write(converted_sample)
    logger.info(f"Store the target dataset into [{tgt_ds_path}].")


if __name__ == "__main__":
    fire.Fire(main)

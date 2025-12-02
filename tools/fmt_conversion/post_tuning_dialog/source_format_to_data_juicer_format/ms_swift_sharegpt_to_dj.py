# This tool is used to convert dataset in ModelScope-Swift ShareGPT format to a
# target dataset in Data-Juicer query-response format.
#
# ModelScope-Swift ShareGPT format:
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
# Corresponding Data-Juicer format (query-response format):
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
# Reference:
# https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Customization/Custom-dataset.md

import json
import os
from typing import List, Union

import fire
import jsonlines as jl
from loguru import logger
from tqdm import tqdm


def ms_swift_sharegpt_to_dj(
    sample,
    conversation_key: str = "conversation",
    human_key: str = "human",
    assistant_key: str = "assistant",
    system_key: str = "system",
    instruction_key: str = "instruction",
    multimodal_keys: Union[str, List[str]] = None,
):
    modified_keys = {conversation_key, system_key, instruction_key}
    if multimodal_keys:
        modified_keys = modified_keys.union(set(multimodal_keys))
    new_sample = {key: sample[key] for key in sample if key not in modified_keys}

    # find system prompt and instruction
    if system_key in sample:
        new_sample["system"] = sample[system_key]
    if instruction_key in sample:
        new_sample["instruction"] = sample[instruction_key]

    # conversations to query, response, history
    conversation = sample[conversation_key]
    # reconstruct conversations
    conv_num = len(conversation)
    if conv_num == 0:
        query = ""
        response = ""
        history = []
    else:
        # the last 1 sentence is query and response is empty
        query = conversation[-1][human_key]
        response = conversation[-1][assistant_key]
        history = [[conv[human_key], conv[assistant_key]] for conv in conversation[:-1]]

    # get the result sample
    new_sample.update(
        {
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
    conversation_key: str = "conversation",
    human_key: str = "human",
    assistant_key: str = "assistant",
    system_key: str = "system",
    instruction_key: str = "instruction",
    multimodal_keys: Union[str, List[str]] = None,
):
    """
    Convert a ModelScope-Swift ShareGPT-like dataset to the Data-Juicer
    query-response format.

    :param src_ds_path: the path to the source dataset.
    :param tgt_ds_path: the path to store the converted target dataset.
    :param conversation_key: the field key to store conversions.
    :param human_key: the field key to store the sentence from human.
    :param assistant_key: the field key to store the sentence from assistant.
    :param system_key: the field key to store the system prompt.
    :param instruction_key: the field key to store the instruction content.
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
            converted_sample = ms_swift_sharegpt_to_dj(
                sample,
                conversation_key=conversation_key,
                human_key=human_key,
                assistant_key=assistant_key,
                system_key=system_key,
                instruction_key=instruction_key,
                multimodal_keys=multimodal_keys,
            )
            writer.write(converted_sample)
    logger.info(f"Store the target dataset into [{tgt_ds_path}].")


if __name__ == "__main__":
    fire.Fire(main)

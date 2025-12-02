# This tool is used to convert dataset in ModelScope-Swift Messages format to a
# target dataset in Data-Juicer query-response format.
#
# ModelScope-Swift Messages format:
#   - usually in json format
# [
#   {
#     "images": ["coco/train2017/000000033471.jpg"],
#     "messages": [
#       {
#         "role": "human",
#         "content": "<image>\nWhat are the colors of the bus in the image?"
#       },
#       {
#         "role": "gpt",
#         "content": "The bus in the image is white and red."
#       },
#       {
#         "role": "human",
#         "content": "What feature can be seen on the back of the bus?"
#       },
#       {
#         "role": "gpt",
#         "content": "The back of the bus features an advertisement."
#       },
#       {
#         "role": "human",
#         "content": "Is the bus driving down the street or pulled off to the side?"  # noqa: E501
#       },
#       {
#         "role": "gpt",
#         "content": "The bus is driving down the street, which is crowded with people and other vehicles."  # noqa: E501
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
# https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Customization/Custom-dataset.md
#
# This format is nearly the same as the LLaMA-Factory ShareGPT format, so we
# reuse the code in that conversion tools.

from typing import List, Union

import fire
import llama_factory_sharegpt_to_dj
from loguru import logger


@logger.catch(reraise=True)
def main(
    src_ds_path: str,
    tgt_ds_path: str,
    messages_key: str = "messages",
    role_key: str = "role",
    content_key: str = "content",
    system_role: str = "system",
    instruction_role: str = "instruction",
    multimodal_keys: Union[str, List[str]] = None,
):
    """
    Convert a Messages-like dataset to the Data-Juicer query-response format.

    :param src_ds_path: the path to the source dataset.
    :param tgt_ds_path: the path to store the converted target dataset.
    :param messages_key: the field key to store messages.
    :param role_key: the field key to store the sentence from.
    :param content_key: the field key to store the sentence content.
    :param system_role: the field key to store the system prompt.
    :param instruction_role: the field key to store the instruction content.
    :param multimodal_keys: optional keys to store multimodal data.
    """
    llama_factory_sharegpt_to_dj.main(
        src_ds_path,
        tgt_ds_path,
        conversations_key=messages_key,
        from_key=role_key,
        value_key=content_key,
        system_role=system_role,
        instruction_role=instruction_role,
        multimodal_keys=multimodal_keys,
    )


if __name__ == "__main__":
    fire.Fire(main)

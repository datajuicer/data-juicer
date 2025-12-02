# This tool is used to convert dataset in Data-Juicer format to a
# target dataset in ModelScope-Swift Messages-like format.
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
# Corresponding ModelScope-Swift Messages format:
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
# Reference:
# https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Customization/Custom-dataset.md
#
# This format is nearly the same as the LLaMA-Factory ShareGPT format, so we
# reuse the code in that conversion tools.

import dj_to_llama_factory_sharegpt
import fire
from loguru import logger


@logger.catch(reraise=True)
def main(
    src_ds_path: str,
    tgt_ds_path: str,
    messages_key: str = "messages",
    role_key: str = "role",
    content_key: str = "content",
    human_role: str = "user",
    assistant_role: str = "assistant",
    system_role: str = "system",
    instruction_role: str = "instruction",
):
    """
    Convert a Data-Juicer query-response dataset to the ModelScope-Swift
    Message format.

    :param src_ds_path: the path to the source dataset.
    :param tgt_ds_path: the path to store the converted target dataset.
    :param messages_key: the field key to store messages.
    :param role_key: the field key to store the sentence from.
    :param content_key: the field key to store the sentence content.
    :param human_role: the role to store the human prompt.
    :param assistant_role: the role to store the instruction content.
    :param system_role: the role to store the system prompt.
    :param instruction_role: the role to store the instruction content.
    """
    dj_to_llama_factory_sharegpt.main(
        src_ds_path,
        tgt_ds_path,
        conversations_key=messages_key,
        from_key=role_key,
        value_key=content_key,
        human_role=human_role,
        assistant_role=assistant_role,
        system_role=system_role,
        instruction_role=instruction_role,
    )


if __name__ == "__main__":
    fire.Fire(main)

# This tool is used to convert multimodal dataset in LLaVA format to a target
# dataset in Data-Juicer format.
#
# LLaVA format:
#   - single/multi-turn conversation
#   - in json
# [
#   {
#     "id": "000000033471",
#     "image": "coco/train2017/000000033471.jpg",
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
# Corresponding Data-Juicer format:
#   - multi-chunk interleaved image-text sequence
#   - in jsonl
# {'id': '000000033471',
#  'images': ['coco/train2017/000000033471.jpg'],
#  'text': '[[human]]: <image>\n'
#          'What are the colors of the bus in the image?\n'
#          '[[gpt]]: The bus in the image is white and red.\n'
#          '[[human]]: What feature can be seen on the back of the bus?\n'
#          '[[gpt]]: The back of the bus features an advertisement.\n'
#          '[[human]]: Is the bus driving down the street or pulled off to'
#          'the side?\n'
#          '[[gpt]]: The bus is driving down the street, which is crowded '
#          'with people and other vehicles. <|__dj__eoc|>',
#   'only_caption': False,}
#
# Reference:
# https://github.com/haotian-liu/LLaVA/blob/main/docs/Finetune_Custom_Data.md

import json
import os
import random

import fire
import jsonlines as jl
from loguru import logger
from tqdm import tqdm

from data_juicer.utils.mm_utils import SpecialTokens


@logger.catch(reraise=True)
def main(
    llava_ds_path: str,
    target_ds_path: str,
    str_id: bool = True,
    split_chunk: bool = False,
    image_broadcast: bool = False,
    image_broadcast_pos: str = "random",
    eoc_special_token: str = SpecialTokens.eoc,
    image_special_token: str = "<image>",
    add_eoc_at_last: bool = True,
    sent_separator: str = "\n",
    only_keep_caption: bool = False,
):
    """
    Convert a LLaVA-like dataset to the Data-Juicer format.

    :param llava_ds_path: path to the input LLaVA-like dataset.
    :param target_ds_path: path to store the converted dataset in Data-Juicer
        format.
    :param str_id: whether to convert all ids to str type. Default: True.
    :param split_chunk: whether to split each round of (human, robot)
        conversation pair into a single chunk. Default: False.
    :param image_broadcast: whether to broadcast the image token to all
        conversation rounds. If it's True, an image_special_token will be added
        to the human question in each conversation round. Default: False.
    :param image_broadcast_pos: the position to add the broadcast
        image_special_token. Should be one of ["before", "after", "random",
        "follow"], which means add this token before/after the human sentence,
        or ranomly choose "before" or "after", or follow the position of the
        first conversation round. Default: random.
    :param eoc_special_token: the special token for "end of a chunk". It's used
        to split conversation chunks explicitly. Default: <|__dj__eoc|> (from
        Data-Juicer).
    :param image_special_token: the special token for images. It's used to
        locate the images in the conversation. In typical LLaVA-like datasets,
        this token always be "<image>". You can change it to align with your
        own LLaVA-like datasets but should be careful of possible compatibility
        problems that come from this change. Default: <image>.
    :param add_eoc_at_last: whether to add an extra eoc_special_token at the
        end of text. Default: True.
    :param sent_separator: separator to split different sentences. Default: \n.
    :param only_keep_caption: only keep the caption in the single-turn dialog
        or not. This argument is mainly for the pretrain-type dataset of LLaVA,
        which only covers a limited number of questions and the corresponding
        answers are the captions of images. For fine-tuning dataset, there
        are some multi-turn dialogs which might not be suitable for setting
        this argument to True. Default: False.
    """
    # ----- Constant settings. Better not to change them. -----
    text_key = "text"  # default key of field to store the sample text
    image_key = "images"  # default key of field to store the image list
    from_format = "[[%s]]: "  # default handle method for the conversation role
    # ----- Constant settings. Better not to change them. -----

    # check arguments
    # check paths
    if not os.path.exists(llava_ds_path):
        raise FileNotFoundError(f"Input LLaVA dataset [{llava_ds_path}] can " f"not be found.")
    if not target_ds_path.endswith(".jsonl"):
        raise ValueError('Only support "jsonl" target dataset file now.')
    if os.path.dirname(target_ds_path) and not os.path.exists(os.path.dirname(target_ds_path)):
        logger.info(f"Create directory [{os.path.dirname(target_ds_path)}] " f"for the target dataset.")
        os.makedirs(os.path.dirname(target_ds_path))
    # check whether to split chunk and broadcast image token to each chunk
    if image_broadcast:
        if not split_chunk:
            raise ValueError("Arg split_chunk should be True when opening " "image_broadcast.")
        if image_broadcast_pos not in ["random", "before", "after", "follow"]:
            raise ValueError(
                f"Arg image_broadcast_pos should be one of ["
                f'"random", "before", "after", "follow"], but '
                f"given [{image_broadcast_pos}]."
            )
    # check if the default image special token is changed
    if image_special_token != "<image>":
        logger.warning(
            "The image_special_token used in the original LLaVA "
            'dataset is "<image>". It\'s better to align the this '
            "token. There might be some compatibility problem if "
            "you change it."
        )
    # check whether to add the eoc special token at last
    if not add_eoc_at_last:
        logger.warning(
            "You choose not to add special eoc token at the last, "
            "which might cause some compatibility problems for "
            "other type of datasets (e.g. OpenFlamingo)."
        )

    # load LLaVA dataset
    logger.info("Loading original LLaVA dataset.")
    llava_ds = json.load(open(llava_ds_path, "r", encoding="utf-8"))
    logger.info(f"Load [{len(llava_ds)}] samples.")

    with jl.open(target_ds_path, "w") as writer:
        for sample in tqdm(llava_ds):
            # id
            id = sample["id"]
            if str_id:
                id = str(id)

            # images and text
            image = sample.get("image", "")
            if image == "":
                logger.warning(
                    f"No images in the sample with id [{id}], "
                    f"which means this sample is not a multimodal "
                    f"sample. You'd better remove this sample "
                    f"before converting."
                )

            conversations = sample["conversations"]

            # assume the input dataset always contains multimodal conversations
            # and the conversations are always consists of (human, robot) pairs
            if len(conversations) % 2 != 0:
                raise ValueError(
                    f"The conversations in the sample with id "
                    f"[{id}] contains unbalance (human, robot) "
                    f"conversation round (number of conversation "
                    f"is [{len(conversations)}]). Please check "
                    f"and fix the dataset and retry."
                )

            if len(conversations) > 2 and only_keep_caption:
                logger.warning(
                    f"There are multi-turn-dialog sample with id "
                    f"[{id}] in this dataset. So this dataset "
                    f"might be a fine-tuning dataset."
                )

            # image list
            images = []
            # record the image token position in the first conversation round
            image_token_pos_in_first_round = ""
            # save the formatted conversations
            formatted_conversations = []
            # the number of conversation rounds
            num_round = len(conversations) // 2
            for i in range(num_round):
                # get the human question and robot answer in this round
                human_round = conversations[2 * i]
                robot_round = conversations[2 * i + 1]

                # get the role and sentence values
                role_human = from_format % human_round["from"]
                sent_human = human_round["value"]
                role_robot = from_format % robot_round["from"]
                sent_robot = robot_round["value"]

                if image == "":
                    # not a multimodal sample, keep everything still
                    pass
                elif i == 0:
                    # record the image token position in the first round
                    if sent_human.startswith(image_special_token):
                        image_token_pos_in_first_round = "before"
                    elif sent_human.endswith(image_special_token):
                        image_token_pos_in_first_round = "after"
                    else:
                        raise ValueError(
                            f"The position of image_special_token in the "
                            f"first round conversation of sample with id "
                            f"[{id}] is neither before nor after the text. "
                            f"The position might be wrong or there is no "
                            f"image_special_token in this sample. Please "
                            f"check and fix the dataset and retry."
                        )
                    images.append(image)
                else:
                    # whether broadcast image special token to following
                    # conversation rounds
                    if image_broadcast:
                        # broadcast image to each conversation round
                        if image_broadcast_pos == "before":
                            sent_human = image_special_token + sent_separator + sent_human
                        elif image_broadcast_pos == "after":
                            sent_human += sent_separator + image_special_token
                        elif image_broadcast_pos == "random":
                            if random.random() < 0.5:
                                # before
                                sent_human = image_special_token + sent_separator + sent_human
                            else:
                                # after
                                sent_human += sent_separator + image_special_token
                        else:
                            # follow the first round conversation
                            if image_token_pos_in_first_round == "before":
                                sent_human = image_special_token + sent_separator + sent_human
                            else:
                                sent_human += sent_separator + image_special_token
                        images.append(image)

                # combine these texts together
                if only_keep_caption:
                    new_sent = image_special_token + sent_separator + sent_robot
                else:
                    new_sent = role_human + sent_human + sent_separator + role_robot + sent_robot
                formatted_conversations.append(new_sent)

            join_sep = sent_separator
            if split_chunk:
                # split (human, robot) pairs into several chunks
                join_sep = f" {eoc_special_token}" + join_sep
            text = join_sep.join(formatted_conversations)
            if add_eoc_at_last:
                # add an extra eoc token after the whole sample text
                text += f" {eoc_special_token}"

            # get the new sample with Data-Juicer format
            new_sample = {
                "id": id,
                text_key: text,
                image_key: images,
            }
            writer.write(new_sample)
    logger.info(f"Store the target dataset into [{target_ds_path}].")


if __name__ == "__main__":
    fire.Fire(main)

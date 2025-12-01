# This tool is used to convert multimodal dataset in Youku-mPLUG format to a
# target dataset in Data-Juicer format.
#
# Youku-mPLUG format:
#   - 4 types: pretrain, classification, retrieval, captioning
#   - in csv
#   - text-video pair with other fields (label, ...)
#
# Youku-mPLUG-pretrain format:
# {'video_id:FILE': 'videos/pretrain/14111Y1211b-1134b18bAE55bFE7Jbb7135YE3aY54EaB14ba7CbAa1AbACB24527A.flv',  # noqa: E501
#  'title': '妈妈给宝宝听胎心，看看宝宝是怎么做的，太调皮了'}
#
# Youku-mPLUG-classification format:
# {'video_id:FILE': 'videos/classification/14111B1211bFBCBCJYF48B55b7523C51F3-8b3a5YbCa5817aBb38a5YAC-241F71J.mp4',  # noqa: E501
#  'title': '兔宝宝刚出生，为什么兔妈妈要把它们吃掉？看完涨见识了',
#  'label': '宠物'}
#
# Youku-mPLUG-retrieval format:
# {'clip_name:FILE': 'videos/retrieval/14111B1211bA1F31-E-2YB57--C518FCEBC553abJYFa5541a31C8a57522AYJbYF4aTdfofa112.mp4',  # noqa: E501
#  'caption': '身穿黑色上衣戴着头盔的女子在路上骑着摩托车四周还停放了一些车'}
#
# Youku-mPLUG-captioning format:
# {'video_id:FILE': 'videos/caption/14111B1211bEJB-1b3b-J3b7b8Y213BJ32-521a1EA8a53-3aBA72aA-4-2-CF1EJ8aTdfofa114.mp4',  # noqa: E501
#  'golden_caption': '穿白色球服的女生高高跳起，接住了球。']}
#
# Corresponding Data-Juicer format:
#   - two new fields are added:
#       - text: a chunk of text with the video special token.
#       - videos: video paths list
#   - other fields in the original format can be kept or not
#   - in jsonl
#
# Youku-mPLUG-pretrain Data-Juicer format:
# {'videos': ['videos/pretrain/14111Y1211b-1134b18bAE55bFE7Jbb7135YE3aY54EaB14ba7CbAa1AbACB24527A.flv'],  # noqa: E501
#  'text': '<__dj__video> 妈妈给宝宝听胎心，看看宝宝是怎么做的，太调皮了 <|__dj__eoc|>'}
#
# Youku-mPLUG-classification Data-Juicer format:
# {'videos': ['videos/classification/14111B1211bFBCBCJYF48B55b7523C51F3-8b3a5YbCa5817aBb38a5YAC-241F71J.mp4'],  # noqa: E501
#  'text': '<__dj__video> 兔宝宝刚出生，为什么兔妈妈要把它们吃掉？看完涨见识了 <|__dj__eoc|>',
#  'label': '宠物'}
#
# Youku-mPLUG-retrieval Data-Juicer format:
# {'videos': ['videos/retrieval/14111B1211bA1F31-E-2YB57--C518FCEBC553abJYFa5541a31C8a57522AYJbYF4aTdfofa112.mp4'],  # noqa: E501
#  'text': '<__dj__video> 身穿黑色上衣戴着头盔的女子在路上骑着摩托车四周还停放了一些车 <|__dj__eoc|>'}
#
# Youku-mPLUG-captioning Data-Juicer format:
# {'videos': ['videos/caption/14111B1211bEJB-1b3b-J3b7b8Y213BJ32-521a1EA8a53-3aBA72aA-4-2-CF1EJ8aTdfofa114.mp4'],  # noqa: E501
#  'text': '<__dj__video> 穿白色球服的女生高高跳起，接住了球。 <|__dj__eoc|> <__dj__video> 一排穿红色短袖的女生正在接受颁奖。 <|__dj__eoc|>']}  # noqa: E501
#
# Reference:
# https://modelscope.cn/datasets/modelscope/Youku-AliceMind/summary

import csv

import fire
import jsonlines as jl
from loguru import logger
from tqdm import tqdm

from data_juicer.utils.mm_utils import SpecialTokens
from tools.fmt_conversion.multimodal.utils import (
    check_args_load_to_dj_data,
    convert_text_to_dj,
)


@logger.catch(reraise=True)
def main(
    youku_ds_path: str,
    target_ds_path: str,
    eoc_special_token: str = SpecialTokens.eoc,
    video_special_token: str = SpecialTokens.video,
    add_eoc_at_last: bool = True,
    sent_separator: str = " ",
    video_special_token_insert_pos: str = "before",
    subset_type: str = "classification",
    keep_other_fields: bool = True,
):
    """
    Convert a Youku-mPLUG-like dataset to the Data-Juicer format.

    :param youku_ds_path: path to the input Youku-mPLUG-like dataset.
    :param target_ds_path: path to store the converted dataset in Data-Juicer
        format.
    :param eoc_special_token: the special token for "end of a chunk". It's used
        to split sentence chunks explicitly. Default: <|__dj__eoc|> (from
        Data-Juicer).
    :param video_special_token: the special token for videos. It's used to
        locate the videos in the text. In typical Youku-mPLUG-like datasets,
        this special token is not specified. So we simply use the default video
        special token from our Data-Juicer. Default: <__dj__video> (from
        Data-Juicer).
    :param add_eoc_at_last: whether to add an extra eoc_special_token at the
        end of text. Default: False.
    :param sent_separator: separator to split different sentences or tokens.
        Default: " "
    :param video_special_token_insert_pos: the position in the sentence to
        insert the corresponding video special token. Should be one of: [
        "before", "after", "random"]. Default: "before".
    :param subset_type: the subset type of the input dataset. Should be one of
        ["pretrain", "classification", "retrieval", "captioning"]. Default:
        "classification".
    :param keep_other_fields: whether to keep other fields in the original
        datasets. Default: False.
    """
    # ----- Constant settings. Better not to change them. -----
    text_key = "text"  # default key of field to store the sample text
    video_key = "videos"  # default key of field to store the video list
    fields_infos = {
        "pretrain": {
            "video_key": "video_id:FILE",
            "text_key": "title",
        },
        "classification": {
            "video_key": "video_id:FILE",
            "text_key": "title",
        },
        "retrieval": {
            "video_key": "clip_name:FILE",
            "text_key": "caption",
        },
        "captioning": {
            "video_key": "video_id:FILE",
            "text_key": "golden_caption",
        },
    }
    # ----- Constant settings. Better not to change them. -----

    # check arguments
    check_args_load_to_dj_data(
        add_eoc_at_last, keep_other_fields, target_ds_path, youku_ds_path, video_special_token_insert_pos, ".jsonl"
    )
    # check subset type
    if subset_type not in fields_infos:
        logger.error(
            f'Arg subset_type should be one of ["pretrain", '
            f'"classification", "retrieval", "captioning"], but '
            f"given [{subset_type}]."
        )
    ori_video_key = fields_infos[subset_type]["video_key"]
    ori_text_key = fields_infos[subset_type]["text_key"]

    # load Youku-mPLUG dataset
    logger.info("Start converting the original Youku-mPLUG dataset...")
    with open(youku_ds_path) as csvfile:
        reader = csv.DictReader(csvfile)
        with jl.open(target_ds_path, mode="w") as writer:
            for row in tqdm(reader):
                video = row[ori_video_key]
                text = row[ori_text_key]

                # convert text to data-juicer format
                # add video special token
                new_sample, text = convert_text_to_dj(
                    text,
                    row,
                    add_eoc_at_last,
                    eoc_special_token,
                    keep_other_fields,
                    sent_separator,
                    video_special_token,
                    video_special_token_insert_pos,
                )
                # all sentences correspond to the same video
                new_sample[video_key] = [video]
                new_sample[text_key] = text
                writer.write(new_sample)
    logger.info(f"Store the target dataset into [{target_ds_path}].")


if __name__ == "__main__":
    fire.Fire(main)

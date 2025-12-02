# This tool is used to convert multimodal dataset in Data-Juicer format to a
# target dataset in Youku-mPLUG-like format.
#
# Corresponding Data-Juicer format:
#   - two new fields to store the main data:
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
# Reference:
# https://modelscope.cn/datasets/modelscope/Youku-AliceMind/summary

import csv
import os

import fire
import jsonlines as jl
from loguru import logger
from tqdm import tqdm

from data_juicer.utils.mm_utils import SpecialTokens
from tools.fmt_conversion.multimodal.utils import remove_dj_special_tokens


def main(
    dj_ds_path: str,
    target_youku_ds_path: str,
    eoc_special_token: str = SpecialTokens.eoc,
    video_special_token: str = SpecialTokens.video,
    sent_separator: str = " ",
    subset_type: str = "classification",
):
    """
    Convert a Data-Juicer-format dataset to a Youku-mPLUG-like dataset.

    :param dj_ds_path: path to the input dataset in Data-Juicer format.
    :param target_youku_ds_path: path to store the converted dataset in
        Youku-mPLUG format.
    :param eoc_special_token: the special token for "end of a chunk". It's used
        to split sentence chunks explicitly. Default: <|__dj__eoc|> (from
        Data-Juicer).
    :param video_special_token: the special token for videos. It's used to
        locate the videos in the text. In typical Youku-mPLUG-like datasets,
        this special token is not specified. So we simply use the default video
        special token from our Data-Juicer. Default: <__dj__video> (from
        Data-Juicer).
    :param sent_separator: separator to split different sentences. Default: " "
    :param subset_type: the subset type of the input dataset. Should be one of
        ["pretrain", "classification", "retrieval", "captioning"]. Default:
        "classification".
    """
    # ----- Constant settings. Better not to change them. -----
    text_key = "text"  # default key of field to store the sample text
    video_key = "videos"  # default key of field to store the video list
    fields_infos = {
        "pretrain": {
            "video_key": "video_id:FILE",
            "text_key": "title",
            "other_required_keys": [],
        },
        "classification": {
            "video_key": "video_id:FILE",
            "text_key": "title",
            "other_required_keys": ["label"],
        },
        "retrieval": {
            "video_key": "clip_name:FILE",
            "text_key": "caption",
            "other_required_keys": [],
        },
        "captioning": {
            "video_key": "video_id:FILE",
            "text_key": "golden_caption",
            "other_required_keys": [],
        },
    }
    # ----- Constant settings. Better not to change them. -----

    # check arguments
    # check paths
    if not os.path.exists(dj_ds_path):
        raise FileNotFoundError(f"Input dataset [{dj_ds_path}] can not be found.")
    if not target_youku_ds_path.endswith(".csv"):
        raise ValueError('Only support "csv" target dataset file for Youku-mPLUG now.')
    if os.path.dirname(target_youku_ds_path) and not os.path.exists(os.path.dirname(target_youku_ds_path)):
        logger.info(f"Create directory [{os.path.dirname(target_youku_ds_path)}] for " f"the target dataset.")
        os.makedirs(os.path.dirname(target_youku_ds_path))
    # check subset type
    if subset_type not in fields_infos:
        logger.error(
            f'Arg subset_type should be one of ["pretrain", '
            f'"classification", "retrieval", "captioning"], but '
            f"given [{subset_type}]."
        )
    tgt_video_key = fields_infos[subset_type]["video_key"]
    tgt_text_key = fields_infos[subset_type]["text_key"]
    tgt_required_keys = fields_infos[subset_type]["other_required_keys"]

    # save Youku-mPLUG dataset from Data-Juicer format
    logger.info("Start converting the original dataset to Youku-mPLUG format...")
    with jl.open(dj_ds_path) as reader:
        with open(target_youku_ds_path, "w") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=[tgt_video_key, tgt_text_key] + tgt_required_keys)
            # write headers first
            writer.writeheader()
            for line_num, s in enumerate(tqdm(reader)):
                new_sample = {}
                # add required fields
                for key in tgt_required_keys:
                    if key not in s:
                        raise ValueError(f"Required key [{key}] is not in the " f"original Data-Juicer dataset.")
                    new_sample[key] = s[key]

                # add video, only keep the first one
                video = s[video_key][0]
                new_sample[tgt_video_key] = video

                # add text, remove extra special tokens
                text = s[text_key].strip()
                text = remove_dj_special_tokens(text, eoc_special_token, sent_separator, video_special_token)
                new_sample[tgt_text_key] = text

                writer.writerow(new_sample)
    logger.info(f"Store the target dataset into [{target_youku_ds_path}].")


if __name__ == "__main__":
    fire.Fire(main)

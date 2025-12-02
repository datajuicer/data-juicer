# This tool is used to convert multimodal dataset in Data-Juicer format to a
# target dataset in MSR-VTT format.
#
# Data-Juicer format:
#   - two new fields are added:
#       - text: a chunk of text with the video special token.
#       - videos: video paths list.
#   - other fields in the original format can be kept or not
#   - in jsonl
# {'videos': ['video6513.mp4'],
#  'text': '<__dj__video> a family is having conversation <|__dj__eoc|>',
#  'category': 14',
#  'url': 'https://www.youtube.com/watch?v=A9pM9iOuAzM',
#  'start time': 116.03,
#  'end time': 126.21,
#  'split': 'validate',
#  'id': 6513,
#  '__index_level_0__': 6128}
#
# Corresponding # MSR-VTT format:
#   - in jsonl
#   - caption-video pair with other fields (id, ...)
#   - videos are from load files, and start/end timestamps are given
#       - **Notice**: Refer to: https://cove.thecvf.com/datasets/839 to download video files.  # noqa: E501
#
# {'video_id': 'video6513.mp4',
#  'caption': 'a family is having conversation',
#  'category': 14',
#  'url': 'https://www.youtube.com/watch?v=A9pM9iOuAzM',
#  'start time': 116.03,
#  'end time': 126.21,
#  'split': 'validate',
#  'id': 6513,
#  '__index_level_0__': 6128}
#
# Reference:
# https://huggingface.co/datasets/AlexZigma/msr-vtt

import os

import fire
import jsonlines as jl
from loguru import logger
from tqdm import tqdm

from data_juicer.utils.mm_utils import SpecialTokens
from tools.fmt_conversion.multimodal.utils import remove_dj_special_tokens


def main(
    dj_ds_path: str,
    target_msr_vtt_ds_path: str,
    eoc_special_token: str = SpecialTokens.eoc,
    video_special_token: str = SpecialTokens.video,
    sent_separator: str = " ",
):
    """
    Convert a Data-Juicer-format dataset to a MSR-VTT-like dataset.

    :param dj_ds_path: path to the input dataset in Data-Juicer format.
    :param target_msr_vtt_ds_path: path to store the converted dataset in
        MSR-VTT format.
    :param eoc_special_token: the special token for "end of a chunk". It's used
        to split sentence chunks explicitly. Default: <|__dj__eoc|> (from
        Data-Juicer).
    :param video_special_token: the special token for videos. It's used to
        locate the videos in the text. In typical MSR-VTTe-like datasets,
        this special token is not specified. So we simply use the default video
        special token from our Data-Juicer. Default: <__dj__video> (from
        Data-Juicer).
    :param sent_separator: separator to split different sentences. Default: " "
    """
    # ----- Constant settings. Better not to change them. -----
    text_key = "text"  # default key of field to store the sample text
    video_key = "videos"  # default key of field to store the video list
    tgt_text_key = "caption"  # default target key of field to store texts
    tgt_video_key = "video_id"  # default target field to store videos
    # ----- Constant settings. Better not to change them. -----

    # check arguments
    # check paths
    if not os.path.exists(dj_ds_path):
        raise FileNotFoundError(f"Input dataset [{dj_ds_path}] can not be found.")
    if not target_msr_vtt_ds_path.endswith(".jsonl"):
        raise ValueError('Only support "jsonl" target dataset file for MSR-VTT now.')
    if os.path.dirname(target_msr_vtt_ds_path) and not os.path.exists(os.path.dirname(target_msr_vtt_ds_path)):
        logger.info(f"Create directory [{os.path.dirname(target_msr_vtt_ds_path)}] " f"for the target dataset.")
        os.makedirs(os.path.dirname(target_msr_vtt_ds_path))

    # save MSR-VTT dataset from Data-Juicer format
    logger.info("Start converting the original dataset to MSR-VTT format...")
    with jl.open(dj_ds_path) as reader:
        with jl.open(target_msr_vtt_ds_path, mode="w") as writer:
            for line_num, s in enumerate(tqdm(reader)):
                video = s.pop(video_key)[0]
                text = s.pop(text_key)

                new_sample = {}
                # add other fields
                for key in s:
                    new_sample[key] = s[key]

                # add video
                new_sample[tgt_video_key] = video

                # add caption
                text = remove_dj_special_tokens(text.strip(), eoc_special_token, sent_separator, video_special_token)

                new_sample[tgt_text_key] = text

                writer.write(new_sample)
    logger.info(f"Store the target dataset into [{target_msr_vtt_ds_path}].")


if __name__ == "__main__":
    fire.Fire(main)

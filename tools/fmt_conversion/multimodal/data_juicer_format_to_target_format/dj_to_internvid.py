# This tool is used to convert multimodal dataset in Data-Juicer format to a
# target dataset in InternVid format.
#
# Data-Juicer format:
#   - two extra fields:
#       - text: a chunk of text with the video special token.
#       - videos: video paths list, including cut videos according to their timestamps  # noqa: E501
#   - other fields in the original format can be kept or not
#   - in jsonl
# {'videos': ['videos/qJrOyggIB-w-cut.mp4'],
#  'text': '<__dj__video> a screen shot of heroes of the storm with people in action <|__dj__eoc|>', # noqa: E501
#  'Start_timestamp': '00:07:33.689',
#  'End_timestamp': '00:07:51.085',
#  'Aesthetic_Score': 4.29296875,
#  'UMT_Score': 0.4501953125}
#
# Corresponding InternVid format:
#   - in jsonl
#   - restore to Caption and YoutubeID
# {'YoutubeID': 'videos/qJrOyggIB-w.mp4',
#  'Start_timestamp': '00:07:33.689',
#  'End_timestamp': '00:07:51.085',
#  'Caption': 'a screen shot of heroes of the storm with people in action',
#  'Aesthetic_Score': 4.29296875,
#  'UMT_Score': 0.4501953125}
#
# Reference:
# https://huggingface.co/datasets/OpenGVLab/InternVid

import os

import fire
import jsonlines as jl
from loguru import logger
from tqdm import tqdm

from data_juicer.utils.mm_utils import SpecialTokens
from tools.fmt_conversion.multimodal.utils import remove_dj_special_tokens


def main(
    dj_ds_path: str,
    target_internvid_ds_path: str,
    eoc_special_token: str = SpecialTokens.eoc,
    video_special_token: str = SpecialTokens.video,
    sent_separator: str = " ",
):
    """
    Convert a Data-Juicer-format dataset to a InternVid-like dataset.

    :param dj_ds_path: path to the input dataset in Data-Juicer format.
    :param target_internvid_ds_path: path to store the converted dataset in
        InternVid format.
    :param eoc_special_token: the special token for "end of a chunk". It's used
        to split sentence chunks explicitly. Default: <|__dj__eoc|> (from
        Data-Juicer).
    :param video_special_token: the special token for videos. It's used to
        locate the videos in the text. In typical InternVide-like datasets,
        this special token is not specified. So we simply use the default video
        special token from our Data-Juicer. Default: <__dj__video> (from
        Data-Juicer).
    :param sent_separator: separator to split different sentences. Default: " "
    """
    # ----- Constant settings. Better not to change them. -----
    text_key = "text"  # default key of field to store the sample text
    video_key = "videos"  # default key of field to store the video list
    tgt_text_key = "Caption"  # default target key of field to store texts
    tgt_video_key = "YoutubeID"  # default target field to store videos
    # ----- Constant settings. Better not to change them. -----

    # check arguments
    # check paths
    if not os.path.exists(dj_ds_path):
        raise FileNotFoundError(f"Input dataset [{dj_ds_path}] can not be found.")
    if not target_internvid_ds_path.endswith(".jsonl"):
        raise ValueError('Only support "jsonl" target dataset file for InternVid now.')
    if os.path.dirname(target_internvid_ds_path) and not os.path.exists(os.path.dirname(target_internvid_ds_path)):
        logger.info(f"Create directory [{os.path.dirname(target_internvid_ds_path)}] " f"for the target dataset.")
        os.makedirs(os.path.dirname(target_internvid_ds_path))

    # save InternVid dataset from Data-Juicer format
    logger.info("Start converting the original dataset to InternVid format...")
    with jl.open(dj_ds_path) as reader:
        with jl.open(target_internvid_ds_path, mode="w") as writer:
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
    logger.info(f"Store the target dataset into [{target_internvid_ds_path}].")


if __name__ == "__main__":
    fire.Fire(main)

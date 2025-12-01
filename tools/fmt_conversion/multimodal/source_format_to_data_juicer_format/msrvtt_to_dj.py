# This tool is used to convert multimodal dataset in MSR-VTT format to a
# target dataset in Data-Juicer format.
#
# MSR-VTT format:
#   - in jsonl
#   - caption-video pair with other fields (id, ...)
#   - videos are from load files, and start/end timestamps are given
#       - **Notice**: Refer to: https://cove.thecvf.com/datasets/839 to download video files.  # noqa: E501
#
# {'video_id': 'video6513',
#  'caption': 'a family is having conversation',
#  'category': 14',
#  'url': 'https://www.youtube.com/watch?v=A9pM9iOuAzM',
#  'start time': 116.03,
#  'end time': 126.21,
#  'split': 'validate',
#  'id': 6513,
#  '__index_level_0__': 6128}
#
# Corresponding Data-Juicer format:
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
#
# Reference:
# https://huggingface.co/datasets/AlexZigma/msr-vtt

import fire
import jsonlines as jl
from loguru import logger
from tqdm import tqdm

from data_juicer.utils.mm_utils import SpecialTokens
from tools.fmt_conversion.multimodal.utils import (
    check_args_load_to_dj_data,
    convert_text_to_dj,
)


def main(
    msr_vtt_ds_path: str,
    target_ds_path: str,
    eoc_special_token: str = SpecialTokens.eoc,
    video_special_token: str = SpecialTokens.video,
    add_eoc_at_last: bool = True,
    sent_separator: str = " ",
    video_special_token_insert_pos: str = "before",
    keep_other_fields: bool = True,
):
    """
    Convert an MSR-VTT-like dataset to the Data-Juicer format.

    :param msr_vtt_ds_path: path to the input MSR-VTT-like dataset.
    :param target_ds_path: path to store the converted dataset in Data-Juicer
        format.
    :param eoc_special_token: the special token for "end of a chunk". It's used
        to split sentence chunks explicitly. Default: <|__dj__eoc|> (from
        Data-Juicer).
    :param video_special_token: the special token for videos. It's used to
        locate the videos in the text. In typical MSR-VTTe-like datasets,
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
    :param keep_other_fields: whether to keep other fields in the original
        datasets. Default: False.
    """
    # ----- Constant settings. Better not to change them. -----
    text_key = "text"  # default key of field to store the sample text
    video_key = "videos"  # default key of field to store the video list
    ori_text_key = "caption"  # default original key of field to store texts
    ori_video_key = "video_id"  # default original field to store videos
    # ----- Constant settings. Better not to change them. -----

    # check arguments
    check_args_load_to_dj_data(
        add_eoc_at_last, keep_other_fields, target_ds_path, msr_vtt_ds_path, video_special_token_insert_pos, ".jsonl"
    )

    # start conversion
    logger.info("Start converting the original MSR-VTT dataset...")
    with jl.open(msr_vtt_ds_path) as reader:
        with jl.open(target_ds_path, mode="w") as writer:
            for s in tqdm(reader):
                video = s.pop(ori_video_key)
                text = s.pop(ori_text_key)
                video += ".mp4"
                # convert text to data-juicer format
                # add video special token
                new_sample, text = convert_text_to_dj(
                    text,
                    s,
                    add_eoc_at_last,
                    eoc_special_token,
                    keep_other_fields,
                    sent_separator,
                    video_special_token,
                    video_special_token_insert_pos,
                )

                new_sample[video_key] = [video]
                new_sample[text_key] = text
                writer.write(new_sample)
    logger.info(f"Store the target dataset into [{target_ds_path}].")


if __name__ == "__main__":
    fire.Fire(main)

# This tool is used to convert multimodal dataset in InternVid format to a
# target dataset in Data-Juicer format.
#
# InternVid format:
#   - in jsonl
#   - caption-video pair with other fields (CLIP_Score, ...)
#   - videos are from Youtube, and start/end timestamps are given
#       - **Notice**: only YoutubeIDs are provided in the original dataset.
#           Here we suppose that users have downloaded these videos already,
#           and the YoutubeIDs are replaced with their video paths.
# {'YoutubeID': 'videos/qJrOyggIB-w.mp4',
#  'Start_timestamp': '00:07:33.689',
#  'End_timestamp': '00:07:51.085',
#  'Caption': 'a screen shot of heroes of the storm with people in action',
#  'Aesthetic_Score': 4.29296875,
#  'UMT_Score': 0.4501953125}
#
# Corresponding Data-Juicer format:
#   - two new fields are added:
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
#
# Reference:
# https://huggingface.co/datasets/OpenGVLab/InternVid

import os

import fire
import jsonlines as jl
from loguru import logger
from tqdm import tqdm

from data_juicer.utils.file_utils import add_suffix_to_filename
from data_juicer.utils.mm_utils import (
    SpecialTokens,
    cut_video_by_seconds,
    timecode_string_to_seconds,
)
from tools.fmt_conversion.multimodal.utils import (
    check_args_load_to_dj_data,
    convert_text_to_dj,
)


def main(
    internvid_ds_path: str,
    target_ds_path: str,
    eoc_special_token: str = SpecialTokens.eoc,
    video_special_token: str = SpecialTokens.video,
    add_eoc_at_last: bool = True,
    sent_separator: str = " ",
    video_special_token_insert_pos: str = "before",
    cut_videos: bool = True,
    cut_video_store_path: str = None,
    keep_other_fields: bool = True,
):
    """
    Convert an InternVid-like dataset to the Data-Juicer format.

    :param internvid_ds_path: path to the input InternVid-like dataset.
    :param target_ds_path: path to store the converted dataset in Data-Juicer
        format.
    :param eoc_special_token: the special token for "end of a chunk". It's used
        to split sentence chunks explicitly. Default: <|__dj__eoc|> (from
        Data-Juicer).
    :param video_special_token: the special token for videos. It's used to
        locate the videos in the text. In typical InternVide-like datasets,
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
    :param cut_videos: whether to cut the videos into smaller ones according to
        their start/end timestamps. Default: True. If you did this process
        before converting, please set it to False.
    :param cut_video_store_path: a path to store the cut videos. If cut_videos
        is True and this path is None, store the cut videos into the same
        directory as the original videos.
    :param keep_other_fields: whether to keep other fields in the original
        datasets. Default: False.
    """
    # ----- Constant settings. Better not to change them. -----
    text_key = "text"  # default key of field to store the sample text
    video_key = "videos"  # default key of field to store the video list
    ori_text_key = "Caption"  # default original key of field to store texts
    ori_video_key = "YoutubeID"  # default original field to store videos
    # ----- Constant settings. Better not to change them. -----

    input_ds_dir = os.path.dirname(internvid_ds_path)

    # check arguments
    check_args_load_to_dj_data(
        add_eoc_at_last, keep_other_fields, target_ds_path, internvid_ds_path, video_special_token_insert_pos, ".jsonl"
    )
    if cut_videos:
        logger.warning(
            "You set the cut_videos arg to True. This tool will "
            "take a video cut from the input video according to "
            "the start/end timestamps."
        )

    # start conversion
    logger.info("Start converting the original InternVid dataset...")
    with jl.open(internvid_ds_path) as reader:
        with jl.open(target_ds_path, mode="w") as writer:
            for s in tqdm(reader):
                video = s.pop(ori_video_key)
                text = s.pop(ori_text_key)

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

                # cut videos if needed
                if cut_videos:
                    video = os.path.join(input_ds_dir, video)
                    cut_video_path = None
                    if cut_video_store_path is None:
                        # set it to the directory stores the original videos
                        cut_video_path = os.path.dirname(os.path.abspath(video))
                    else:
                        cut_video_path = cut_video_store_path
                    # cut the video and store in a new path
                    video_basename = os.path.basename(video)
                    new_video = os.path.join(
                        cut_video_path,
                        add_suffix_to_filename(video_basename, f'_{s["Start_timestamp"]}_{s["End_timestamp"]}'),
                    )
                    start_pts = timecode_string_to_seconds(s["Start_timestamp"])
                    end_pts = timecode_string_to_seconds(s["End_timestamp"])
                    if cut_video_by_seconds(video, new_video, start_pts, end_pts):
                        video = new_video
                    else:
                        continue

                new_sample[video_key] = [video]
                new_sample[text_key] = text
                if cut_videos:
                    # add a meta field to record whether this video is cut
                    new_sample["is_cut"] = True
                else:
                    new_sample["is_cut"] = False
                writer.write(new_sample)
    logger.info(f"Store the target dataset into [{target_ds_path}].")


if __name__ == "__main__":
    fire.Fire(main)

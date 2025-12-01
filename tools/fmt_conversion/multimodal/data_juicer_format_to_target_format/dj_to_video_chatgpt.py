# This tool is used to convert multimodal dataset in Data-Juicer format to a
# target dataset in Video-ChatGPT format.
#
# Reference:
# https://github.com/mbzuai-oryx/Video-ChatGPT/tree/main/data
#
# # Corresponding Data-Juicer format:
#   - two new fields to store the main data: 'youtube_id' and 'text',
#       the 'videos' is actual path to the video files
# {'youtube_id': 'k_ZXmr8pmrs',
# 'videos': ['youtube_video_dir/v_k_ZXmr8pmrs.mp4'],
#  'text':
#       '<__dj__video>'
#       '[[q]]: What are the main activities that take place in the video? \n'
#       '[[a]]: The main activities that take place in the video are the
#       preparation of camera equipment by a man.... <|__dj__eoc|>'
#  }
#
#  Video-ChatGPT format:
#   - Topics for Video summarization; Description-based question-answers
#   (exploring spatial, temporal, relationships, and reasoning concepts);
#   and Creative/generative question-answers
#   - in json file, a single line storing text-video tuples in a list,
#   below is an example
#
# [{'q': 'What are the main activities that take place in the video?',
# 'a': 'The main activities that take place in the video are the preparation of
# camera equipment by a man, a group of men riding a helicopter, and a man
# sailing a boat through the water.',
# 'video_id': 'v_k_ZXmr8pmrs'}, ...]
#
import json
import os

import fire
import jsonlines as jl
from loguru import logger
from tqdm import tqdm

from data_juicer.utils.mm_utils import SpecialTokens
from tools.fmt_conversion.multimodal.utils import remove_dj_special_tokens


def main(
    dj_ds_path: str,
    target_video_chatgpt_ds_path: str,
    eoc_special_token: str = SpecialTokens.eoc,
    video_special_token: str = SpecialTokens.video,
    sent_separator: str = " ",
):
    """
    Convert a Data-Juicer-format dataset to a Video-ChatGPT-like dataset.

    :param dj_ds_path: path to the input dataset in Data-Juicer format.
    :param target_video_chatgpt_ds_path: path to store the converted dataset in
        Video-ChatGPT format.
    :param eoc_special_token: the special token for "end of a chunk". It's used
        to split sentence chunks explicitly. Default: <|__dj__eoc|> (from
        Data-Juicer).
    :param video_special_token: the special token for videos. It's used to
        locate the videos in the text. In typical Video-ChatGPT-like datasets,
        this special token is not specified. So we simply use the default video
        special token from our Data-Juicer. Default: <__dj__video> (from
        Data-Juicer).
    :param sent_separator: separator to split different sentences. Default: " "
    """
    # ----- Constant settings. Better not to change them. -----
    text_key = "text"  # default key of field to store the sample text
    video_key = "videos"  # default key of field to store video files path
    video_id_key = "youtube_id"  # default key of field to store youtube id
    tgt_q_key = "q"  # default original key of field to store texts
    tgt_a_key = "a"
    tgt_video_key = "video_id"  # default original field to store videos
    # ----- Constant settings. Better not to change them. -----

    # check arguments
    if not os.path.exists(dj_ds_path):
        raise FileNotFoundError(f"Input Video_ChatGPT dataset in dj format, " f"[{dj_ds_path}], can not be found.")
    if not target_video_chatgpt_ds_path.endswith(".json"):
        raise ValueError('Only support "json" target dataset file for Video_ChatGPT now.')
    if os.path.dirname(target_video_chatgpt_ds_path) and not os.path.exists(
        os.path.dirname(target_video_chatgpt_ds_path)
    ):
        logger.info(
            f"Create directory " f"[{os.path.dirname(target_video_chatgpt_ds_path)}] " f"for the target dataset."
        )
        os.makedirs(os.path.dirname(target_video_chatgpt_ds_path))

    # save Video-ChatGPT dataset from Data-Juicer format
    logger.info("Start converting the DJ dataset to Video-ChatGPT format...")
    all_samples = []
    with jl.open(dj_ds_path) as reader:
        for line_num, s in enumerate(tqdm(reader)):
            video_path = s.pop(video_key)[0]
            new_sample = {}

            video_id = s.pop(video_id_key)
            new_sample[tgt_video_key] = "v_" + video_id
            # add video
            new_sample[video_key] = video_path

            # add question and answer
            text = s.pop(text_key).strip()
            text = remove_dj_special_tokens(text, eoc_special_token, sent_separator, video_special_token)
            # get the question and answer
            parts = text.split(f"[[{tgt_q_key}]]:")[1]
            q, a = parts.split(f"[[{tgt_a_key}]]:")
            new_sample[tgt_q_key] = q.strip()
            new_sample[tgt_a_key] = a.strip()

            # add other fields
            for key in s:
                if key not in [tgt_q_key, tgt_a_key]:
                    new_sample[key] = s[key]

            all_samples.append(new_sample)
    with open(target_video_chatgpt_ds_path, "w") as file:
        json.dump(all_samples, file)
    logger.info(f"Store the dataset into [{target_video_chatgpt_ds_path}].")


if __name__ == "__main__":
    fire.Fire(main)

# This tool is used to convert multimodal dataset in Video_Chatgpt format to a
# target dataset in Data-Juicer format.
#
#  Video-ChatGPT format:
#   - Topics for Video summarization; Description-based question-answers
#   (exploring spatial, temporal, relationships, and reasoning concepts);
#   and Creative/generative question-answers
#   - in json file, a single line storing text-video pairs in a list,
#   below is an example
#
# [{'q': 'What are the main activities that take place in the video?',
# 'a': 'The main activities that take place in the video are the preparation of
# camera equipment by a man, a group of men riding a helicopter, and a man
# sailing a boat through the water.',
# 'video_id': 'v_k_ZXmr8pmrs'}, ...]
#
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

import json
import os

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
    video_chatgpt_ds_path: str,
    target_ds_dj_path: str,
    eoc_special_token: str = SpecialTokens.eoc,
    video_special_token: str = SpecialTokens.video,
    add_eoc_at_last: bool = True,
    sent_separator: str = " ",
    video_special_token_insert_pos: str = "before",
    keep_other_fields: bool = True,
):
    """
    Convert a Video_Chatgpt-like dataset to the Data-Juicer format.

    :param video_chatgpt_ds_path: path to the input Video_Chatgpt-like dataset.
    :param target_ds_dj_path: path to store the converted dataset in
        Data-Juicer format.
    :param eoc_special_token: the special token for "end of a chunk". It's used
        to split sentence chunks explicitly. Default: <|__dj__eoc|> (from
        Data-Juicer).
    :param video_special_token: the special token for videos. It's used to
        locate the videos in the text. In typical Video_Chatgpt-like datasets,
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
    video_id_key = "youtube_id"  # default original field to store video id
    ori_text_key_q = "q"  # default original key of field to store texts
    ori_text_key_a = "a"  # default original key of field to store texts
    ori_video_key = "video_id"  # default original field to store video ids

    def format_dj_text(text_q, text_a):
        """
        This function returns a formatted string.

        :param text_q: Text for the question
        :param text_a: Text for the answer
        :return: Formatted string
        """
        return f"[[{ori_text_key_q}]]:{text_q} \n[[{ori_text_key_a}]]:{text_a}"

    # ----- Constant settings. Better not to change them. -----

    input_ds_dir = os.path.dirname(video_chatgpt_ds_path)

    # check arguments
    check_args_load_to_dj_data(
        add_eoc_at_last,
        keep_other_fields,
        target_ds_dj_path,
        video_chatgpt_ds_path,
        video_special_token_insert_pos,
        ".jsonl",
    )

    # start conversion
    logger.info(f"Start converting the original Video_Chatgpt dataset " f"from {video_chatgpt_ds_path}...")
    with open(video_chatgpt_ds_path, "r") as json_file:
        ori_data = json.load(json_file)
    with jl.open(target_ds_dj_path, mode="w") as writer:
        for s in tqdm(ori_data):
            # v_k_ZXmr8pmrs --> k_ZXmr8pmrs
            video_id = s.pop(ori_video_key)[2:]
            text_q = s.pop(ori_text_key_q)
            text_a = s.pop(ori_text_key_a)

            text = format_dj_text(text_q, text_a)

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

            new_sample[video_key] = [os.path.join(input_ds_dir, video_id)]
            new_sample[text_key] = text
            new_sample[video_id_key] = video_id

            writer.write(new_sample)
    logger.info(f"Store the target dataset into [{target_ds_dj_path}].")


if __name__ == "__main__":
    fire.Fire(main)

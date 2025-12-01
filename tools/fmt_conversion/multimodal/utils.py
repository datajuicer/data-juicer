import os
import random
from copy import deepcopy

from loguru import logger


def remove_dj_special_tokens(text, eoc_special_token, sent_separator, video_special_token):
    # remove possible sentence separator
    if text.startswith(sent_separator):
        text = text[len(sent_separator) :].strip()
    if text.endswith(sent_separator):
        text = text[: -len(sent_separator)].strip()
    # remove eoc token
    if text.endswith(eoc_special_token):
        text = text[: -len(eoc_special_token)].strip()
    # remove possible video special token
    if text.startswith(video_special_token):
        text = text[len(video_special_token) :].strip()
        if text.startswith(sent_separator):
            text = text[len(sent_separator) :].strip()
    elif text.endswith(video_special_token):
        text = text[: -len(video_special_token)].strip()
        if text.endswith(sent_separator):
            text = text[: -len(sent_separator)].strip()
    return text


def check_args_load_to_dj_data(
    add_eoc_at_last,
    keep_other_fields,
    target_ds_dj_path,
    video_ds_path,
    video_special_token_insert_pos,
    target_ds_path_suffix,
):
    if not os.path.exists(video_ds_path):
        raise FileNotFoundError(f"Input dataset " f"[{video_ds_path}] can not be found.")
    if not target_ds_dj_path.endswith(target_ds_path_suffix):
        raise ValueError(f'Only support "{target_ds_path_suffix}" target dataset file now.')
    if os.path.dirname(target_ds_dj_path) and not os.path.exists(os.path.dirname(target_ds_dj_path)):
        logger.info(f"Create directory [{os.path.dirname(target_ds_dj_path)}] " f"for the target dataset.")
        os.makedirs(os.path.dirname(target_ds_dj_path))
    # check insert position
    if video_special_token_insert_pos not in ["random", "before", "after"]:
        raise ValueError(
            f"Arg video_special_token_insert_pos should be one "
            f'of ["before", "after", "random"], but given '
            f"[{video_special_token_insert_pos}]"
        )
    # check whether to add the eoc special token at last
    if not add_eoc_at_last:
        logger.warning(
            "You choose not to add special eoc token at the last, "
            "which might cause some compatibility problems for "
            "other type of datasets (e.g. OpenFlamingo)."
        )
    if not keep_other_fields:
        logger.warning(
            "You choose not to keep other fields in the original "
            "dataset. Thus some information might be lost in the "
            "processed anc converted-back dataset!"
        )


def convert_text_to_dj(
    text,
    original_sample,
    add_eoc_at_last,
    eoc_special_token,
    keep_other_fields,
    sent_separator,
    video_special_token,
    video_special_token_insert_pos,
):
    if video_special_token_insert_pos == "before":
        text = video_special_token + sent_separator + text
    elif video_special_token_insert_pos == "after":
        text += sent_separator + video_special_token
    else:
        if random.random() < 0.5:
            # before
            text = video_special_token + sent_separator + text
        else:
            # after
            text += sent_separator + video_special_token
    if add_eoc_at_last:
        text += f"{sent_separator}{eoc_special_token}"
    if keep_other_fields:
        new_sample = deepcopy(original_sample)
    else:
        new_sample = {}
    return new_sample, text

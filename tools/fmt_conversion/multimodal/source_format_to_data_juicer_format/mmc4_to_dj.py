# This tool is used to convert multimodal dataset in MMC4 format to a target
# dataset in Data-Juicer format.
#
# MMC4 format:
#   - interleaved image-text sequence
#   - in jsonl
# {'image_info': [{'face_detections': None,
#                  'image_name': 'b9040a0dbb22.jpg',
#                  'matched_sim': 0.27694183588027954,
#                  'matched_text_index': 2,
#                  'raw_url': 'http://www.hfitinfo.com/honda_fit_pics/3/2/index.90.jpg'},  # noqa: E501
#                 {'face_detections': None,
#                  'image_name': 'db1c21bc8474.jpg',
#                  'matched_sim': 0.3234919607639313,
#                  'matched_text_index': 1,
#                  'raw_url': 'http://www.hfitinfo.com/honda_fit_pics/3/2/index.91.jpg'}],  # noqa: E501
#  'similarity_matrix': [[0.24363446235656738,
#                         0.31758785247802734,
#                         0.27694183588027954],
#                        [0.2233106791973114,
#                         0.3234919607639313,
#                         0.26118797063827515]],
#  'text_list': ['When you lock the door using the lock tab on the driver’s '
#                'door, all of the other doors and tailgate lock at the same '
#                'time.',
#                'Press the master door lock switch in as shown to lock or '
#                'unlock all doors and the tailgate.',
#                'When you lock/unlock the driver’s door and tailgate using the '  # noqa: E501
#                'master lock switch, all the other doors lock/ unlock at the '
#                'same time.'],
#  'url': 'http://www.hfitinfo.com/hofi-48.html',
#  'could_have_url_duplicate': 0 }
#
# Corresponding Data-Juicer format:
#   - two new fields are added:
#       - text: multi-chunk interleaved image-text sequence in one string. Each
#           sentence in the original dataset is a chunk in this text string.
#       - images: image paths list
#   - other fields in the original format can be kept or not
#   - in jsonl
# {'text': 'When you lock the door using the lock tab on the driver’s door, '
#          'all of the other doors and tailgate lock at the same time. '
#          '<|__dj__eoc|> <__dj__image> Press the master door lock switch in '
#          'as shown to lock or unlock all doors and the tailgate. '
#          '<|__dj__eoc|> <__dj__image> When you lock/unlock the driver’s '
#          'door and tailgate using the master lock switch, all the other '
#          'doors lock/ unlock at the same time. <|__dj__eoc|>',
#  'images': ['db1c21bc8474.jpg', 'b9040a0dbb22.jpg'],
#  'image_info': [{'face_detections': None,
#                  'image_name': 'b9040a0dbb22.jpg',
#                  'matched_sim': 0.27694183588027954,
#                  'matched_text_index': 2,
#                  'raw_url': 'http://www.hfitinfo.com/honda_fit_pics/3/2/index.90.jpg'},  # noqa: E501
#                 {'face_detections': None,
#                  'image_name': 'db1c21bc8474.jpg',
#                  'matched_sim': 0.3234919607639313,
#                  'matched_text_index': 1,
#                  'raw_url': 'http://www.hfitinfo.com/honda_fit_pics/3/2/index.91.jpg'}],  # noqa: E501
#  'similarity_matrix': [[0.24363446235656738,
#                         0.31758785247802734,
#                         0.27694183588027954],
#                        [0.2233106791973114,
#                         0.3234919607639313,
#                         0.26118797063827515]],
#  'text_list': ['When you lock the door using the lock tab on the driver’s '
#                'door, all of the other doors and tailgate lock at the same '
#                'time.',
#                'Press the master door lock switch in as shown to lock or '
#                'unlock all doors and the tailgate.',
#                'When you lock/unlock the driver’s door and tailgate using the '  # noqa: E501
#                'master lock switch, all the other doors lock/ unlock at the '
#                'same time.'],
#  'url': 'http://www.hfitinfo.com/hofi-48.html',
#  'could_have_url_duplicate': 0 }
#
# Reference:
# https://github.com/allenai/mmc4#documents

import os
import random
from copy import deepcopy

import fire
import jsonlines as jl
from loguru import logger
from tqdm import tqdm

from data_juicer.utils.mm_utils import SpecialTokens


@logger.catch(reraise=True)
def main(
    mmc4_ds_path: str,
    target_ds_path: str,
    image_dir: str = None,
    eoc_special_token: str = SpecialTokens.eoc,
    image_special_token: str = SpecialTokens.image,
    image_special_token_insert_pos: str = "before",
    add_eoc_at_last: bool = True,
    sent_separator: str = " ",
    keep_other_fields: bool = True,
):
    """
    Convert a MMC4-like dataset to the Data-Juicer format.

    :param mmc4_ds_path: path to the input MMC4-like dataset.
    :param target_ds_path: path to store the converted dataset in Data-Juicer
        format.
    :param image_dir: directory to store images. If it's None, it means the
        "image_name" for each image includes this information already. Default:
        None.
    :param eoc_special_token: the special token for "end of a chunk". It's used
        to split sentence chunks explicitly. Default: <|__dj__eoc|> (from
        Data-Juicer).
    :param image_special_token: the special token for images. It's used to
        locate the images in the conversation. In typical MMC4-like datasets,
        this special token is not specified. So we simply use the default image
        special token from our Data-Juicer. Default: <__dj__image> (from
        Data-Juicer).
    :param image_special_token_insert_pos: the position in the sentence to
        insert the corresponding image special token. Should be one of: [
        "before", "after", "random"]. Default: "before", which is aligned with
        Flamingo format.
    :param add_eoc_at_last: whether to add an extra eoc_special_token at the
        end of text. Default: True.
    :param sent_separator: separator to split different sentences. Default: " "
    :param keep_other_fields: whether to keep other fields in the original
        datasets. Default: False.
    """
    # ----- Constant settings. Better not to change them. -----
    # default key of field to store the sample text
    text_key = "text"
    # default key of field to store the image list
    image_key = "images"
    # required fields in the original dataset
    REQUIRED_FIELDS = {"image_info", "text_list"}
    # ----- Constant settings. Better not to change them. -----

    # check arguments
    # check paths
    if not os.path.exists(mmc4_ds_path):
        raise FileNotFoundError(f"Input MMC4 dataset [{mmc4_ds_path}] can " f"not be found.")
    if not target_ds_path.endswith(".jsonl"):
        raise ValueError('Only support "jsonl" target dataset file now.')
    if os.path.dirname(target_ds_path) and not os.path.exists(os.path.dirname(target_ds_path)):
        logger.info(f"Create directory [{os.path.dirname(target_ds_path)}] " f"for the target dataset.")
        os.makedirs(os.path.dirname(target_ds_path))
    # check image dir
    if not image_dir:
        image_dir = ""
    # check insert position
    if image_special_token_insert_pos not in ["random", "before", "after"]:
        raise ValueError(
            f"Arg image_special_token_insert_pos should be one "
            f'of ["before", "after", "random"], but given '
            f"[{image_special_token_insert_pos}]"
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

    # load MMC4 dataset
    logger.info("Start converting the original MMC4 dataset...")
    # record the failed samples: (line_number, fail_reason_info)
    failed_samples = []
    with jl.open(mmc4_ds_path, "r") as reader:
        with jl.open(target_ds_path, "w") as writer:
            for line_num, sample in enumerate(tqdm(reader)):
                # check required fields
                fields_ok = True
                for key in REQUIRED_FIELDS:
                    if key not in sample:
                        failed_samples.append(
                            (
                                line_num,
                                f"There is no key [{key}] in the sample whose line"
                                f" number is [{line_num}], which is required for "
                                f"MMC4-like dataset conversion.",
                            )
                        )
                        fields_ok = False
                        break
                if not fields_ok:
                    continue

                new_sample = {}
                if keep_other_fields:
                    # if other fields need to be kept, initialize the new
                    # sample with the original sample
                    new_sample = deepcopy(sample)

                # convert text_list and image_info to text and images
                image_infos = sample["image_info"]
                sentences = sample["text_list"]

                # sort image infos by their matched_text_index
                image_infos.sort(key=lambda s: s["matched_text_index"])

                # get the image path list directly
                images = [os.path.join(image_dir, s["image_name"]) for s in image_infos]

                # construct the text string in Data-Juicer format
                img_idx = 0
                new_sents = []
                for sent_idx, sent in enumerate(sentences):
                    # find the matched sentence of the current image
                    image_num_this_sent = 0
                    while img_idx < len(image_infos) and image_infos[img_idx]["matched_text_index"] == sent_idx:
                        image_num_this_sent += 1
                        img_idx += 1

                    if image_num_this_sent > 0:
                        # insert several image_special_tokens to specific
                        # position.
                        image_special_tokens = sent_separator.join([image_special_token] * image_num_this_sent)
                        if image_special_token_insert_pos == "before":
                            sent = image_special_tokens + sent_separator + sent
                        elif image_special_token_insert_pos == "after":
                            sent += sent_separator + image_special_tokens
                        else:
                            if random.random() < 0.5:
                                # before
                                sent = image_special_tokens + sent_separator + sent
                            else:
                                # after
                                sent += sent_separator + image_special_tokens
                    new_sents.append(sent)

                join_sep = f" {eoc_special_token}{sent_separator}"
                text = join_sep.join(new_sents)
                if add_eoc_at_last:
                    text += f" {eoc_special_token}"

                # construct the new sample
                new_sample[image_key] = images
                new_sample[text_key] = text

                writer.write(new_sample)
    logger.info(f"Store the target dataset into [{target_ds_path}].")
    if len(failed_samples) > 0:
        failed_samples_path = target_ds_path + "_failed.txt"
        logger.warning(
            f"[{len(failed_samples)} samples fail to be converted, "
            f"whose line number and failed reasons are store in "
            f"[{failed_samples_path}]"
        )
        with open(failed_samples_path, "w") as fout:
            for line_num, reason in failed_samples:
                fout.write(f"{line_num}\t{reason}\n")


if __name__ == "__main__":
    fire.Fire(main)

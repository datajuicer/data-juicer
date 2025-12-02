# This tool is used to convert multimodal dataset in Data-Juicer format to a
# target dataset in MMC4 format. Notice: if the similarity matrix is included
# in the dataset, it might not be able to be restored to the original
# correlation and could be with wrong shape due to some images or text
# sentences might be removed. So this tool will do nothing to the similarity
# matrix.
#
# MMC4 in Data-Juicer format:
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
# MMC4 format:
#   - interleaved image-text sequence
#   - in jsonl
#   - extra information except "image_name", "matched_text_index", "text_list"
#       will be included only if they are kept when converting the original
#       MMC4 format to Data-Juicer format. (keep_other_fields is True)
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
# Reference:
# https://github.com/allenai/mmc4#documents

import os
from copy import deepcopy

import fire
import jsonlines as jl
from loguru import logger
from tqdm import tqdm

from data_juicer.utils.mm_utils import SpecialTokens


@logger.catch(reraise=True)
def main(
    dj_ds_path: str,
    target_mmc4_ds_path: str,
    eoc_special_token: str = SpecialTokens.eoc,
    image_special_token: str = SpecialTokens.image,
    sent_separator: str = " ",
    keep_dj_fields: bool = False,
):
    """
    Convert a Data-Juicer-format dataset to an MMC4-like format. Notice: if
    the similarity matrix is included in the dataset, it might not be able to
    be restored to the original correlation and could be with wrong shape due
    to some images or text sentences might be removed. So this tool will do
    nothing to the similarity matrix.

    :param dj_ds_path: path to the input dataset in Data-Juicer format.
    :param target_mmc4_ds_path: path to store the converted dataset in MMC4
        format.
    :param eoc_special_token: the special token for "end of a chunk". It's used
        to split sentence chunks explicitly. Default: <|__dj__eoc|> (from
        Data-Juicer).
    :param image_special_token: the special token for images. It's used to
        locate the images in the conversation. In typical MMC4-like datasets,
        this special token is not specified. So we simply use the default image
        special token from our Data-Juicer. Default: <__dj__image> (from
        Data-Juicer).
    :param sent_separator: separator to split different sentences. Default: " "
    :param keep_dj_fields: whether to keep intermediate fields from
        Data-Juicer, such as "images", "text", ... Default: False.
    """
    # ----- Constant settings. Better not to change them. -----
    # default key of field to store the sample text
    text_key = "text"
    # default key of field to store the image list
    image_key = "images"
    # ----- Constant settings. Better not to change them. -----

    # check arguments
    # check paths
    if not os.path.exists(dj_ds_path):
        raise FileNotFoundError(f"Input dataset [{dj_ds_path}] can not be found.")
    if not target_mmc4_ds_path.endswith(".jsonl"):
        raise ValueError('Only support "jsonl" target dataset file for MMC4 now.')
    if os.path.dirname(target_mmc4_ds_path) and not os.path.exists(os.path.dirname(target_mmc4_ds_path)):
        logger.info(f"Create directory [{os.path.dirname(target_mmc4_ds_path)}] for " f"the target dataset.")
        os.makedirs(os.path.dirname(target_mmc4_ds_path))

    # whether to keep dj fields
    if keep_dj_fields:
        logger.warning(
            "You choose to keep intermediate fields added when "
            "converting to Data-Juicer format, which are usually "
            "useless in the final dataset but it will increase the "
            "size of the whole dataset file."
        )

    # load MMC4 dataset
    logger.info("Start converting the original dataset to MMC4 format...")
    with jl.open(dj_ds_path, "r") as reader:
        with jl.open(target_mmc4_ds_path, "w") as writer:
            for line_num, sample in enumerate(tqdm(reader)):
                text = sample[text_key]
                images = sample[image_key]

                # skip empty samples
                if len(text) == 0:
                    continue

                # image_infos are kept or not?
                image_infos = []
                ori_image_infos = []
                if "image_info" in sample:
                    ori_image_infos = sample["image_info"]

                # Only keep those image_infos that are still contained by
                # processed images.
                for processed_img in images:
                    found = False
                    for img in ori_image_infos:
                        img_name = img["image_name"]
                        if processed_img.endswith(img_name):
                            found = True
                            # update to new image name
                            img["image_name"] = processed_img
                            image_infos.append(img)
                            break
                    if not found:
                        image_infos.append(
                            {
                                "image_name": processed_img,
                            }
                        )

                # split text into a list of several sentences (chunks)
                # remove empty chunks (e.g. the last chunk '' after eoc)
                chunks = [sent.strip() for sent in text.split(eoc_special_token) if sent.strip()]

                # construct text_list and update matched_text_index for the
                # final image_infos
                sentences = []
                curr_image_idx = 0
                for text_idx, sent in enumerate(chunks):
                    # remove possible sentence separator
                    if sent.endswith(sent_separator):
                        sent = sent[: -len(sent_separator)].strip()
                    if sent.startswith(sent_separator):
                        sent = sent[len(sent_separator) :].strip()

                    # remove possible image_special_token and update
                    # matched_text_index for corresponding image_info
                    found_image_num = 0
                    while sent.startswith(image_special_token):
                        sent = sent[len(image_special_token) :].strip()
                        found_image_num += 1
                        if sent.startswith(sent_separator):
                            sent = sent[len(sent_separator) :].strip()
                    while sent.endswith(image_special_token):
                        sent = sent[: -len(image_special_token)].strip()
                        found_image_num += 1
                        if sent.endswith(sent_separator):
                            sent = sent[: -len(sent_separator)].strip()
                    sentences.append(sent)
                    if found_image_num > 0:
                        for _ in range(found_image_num):
                            if curr_image_idx < len(image_infos):
                                image_infos[curr_image_idx]["matched_text_index"] = text_idx
                                curr_image_idx += 1
                            else:
                                # if there are extra images, just skip them and
                                # report a warning
                                logger.warning(
                                    f"Sample with line number "
                                    f"[{line_num}] contains "
                                    f"unaligned numbers of images "
                                    f"and image tokens. Please "
                                    f"check and retry if needed."
                                )

                # reorder image_info to the same order as the original dataset
                final_image_info = []
                for img in ori_image_infos:
                    img_name = img["image_name"]
                    for processed_img in image_infos:
                        processed_img_name = processed_img["image_name"]
                        if processed_img_name.endswith(img_name):
                            final_image_info.append(processed_img)
                            break

                # construct the new sample structure
                new_sample = deepcopy(sample)
                new_sample["image_info"] = final_image_info
                new_sample["text_list"] = sentences
                if not keep_dj_fields:
                    _ = new_sample.pop(image_key)
                    _ = new_sample.pop(text_key)

                writer.write(new_sample)

    logger.info(f"Store the target dataset into [{target_mmc4_ds_path}].")


if __name__ == "__main__":
    fire.Fire(main)

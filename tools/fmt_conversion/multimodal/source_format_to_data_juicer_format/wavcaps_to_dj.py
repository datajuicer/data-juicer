# This tool is used to convert multimodal dataset in WavCaps format to a target
# dataset in Data-Juicer format.
#
# WavCps format:
# { 'num_captions_per_audio': 1,
#   'data': [{
#       'title': 'Airplane Landing Airport',
#       'description': 'Large commercial airplane landing at an airport runway.',  # noqa: E501
#       'author': 'Daniel Simion',
#       'href': '2219-Airplane-Landing-Airport.html',
#       'caption': 'An airplane is landing.',
#       'id': '2219',
#       'duration': 14.1424375,
#       'audio': 'wav_path',
#       'download_link': 'http://soundbible.com/grab.php?id=2219&type=wav'
#   },  {
#       'title': 'Service Bell Help',
#       'description': 'Customer ringing service bell in need of help in a store.',  # noqa: E501
#       'author': 'Daniel Simion',
#       'href': '2218-Service-Bell-Help.html',
#       'caption': 'Someone is ringing a bell.',
#       'id': '2218',
#       'duration': 1.5698125,
#       'audio': 'wav_path',
#       'download_link': 'http://soundbible.com/grab.php?id=2218&type=wav'
#   },
#   ...]
# }
#
# Corresponding Data-Juicer format:
# {'id': 2219,
#  'audios': ['./path/to/audio/2219.flac'],
#  'text': '<__dj__audio>\n'
#          'An airplane is landing. <|__dj__eoc|>',
#  '__dj__meta__': {
#       'num_captions_per_audio': 1,
#       'title': 'Airplane Landing Airport',
#       'description': 'Large commercial airplane landing at an airport runway.',  # noqa: E501
#       'author': 'Daniel Simion',
#       'href': '2219-Airplane-Landing-Airport.html',
#       'caption': 'An airplane is landing.',
#       'id': '2219',
#       'duration': 14.1424375,
#       'audio': 'wav_path',
#       'download_link': 'http://soundbible.com/grab.php?id=2219&type=wav',
#       'category': '',
#       'tags': '' }}
# {'id': 2218,
#  'audios': ['./path/to/audio/2218.flac'],
#  'text': '<__dj__audio>\n'
#          'Someone is ringing a bell. <|__dj__eoc|>',
#  '__dj__meta__': {
#       'num_captions_per_audio': 1,
#       'title': 'Service Bell Help',
#       'description': 'Customer ringing service bell in need of help in a store.',  # noqa: E501
#       'author': 'Daniel Simion',
#       'href': '2218-Service-Bell-Help.html',
#       'caption': 'Someone is ringing a bell.',
#       'id': '2218',
#       'duration': 1.5698125,
#       'audio': 'wav_path',
#       'download_link': 'http://soundbible.com/grab.php?id=2218&type=wav',
#       'category': '',
#       'tags': '' }}

import json
import os

import fire
import jsonlines as jl
from loguru import logger
from tqdm import tqdm

from data_juicer.utils.constant import Fields
from data_juicer.utils.mm_utils import SpecialTokens


def creat_meta_filed(num_captions_per_audio, source_meta):
    meta_dict = {
        "num_captions_per_audio": num_captions_per_audio,
        "title": "",
        "description": "",
        "author": "",
        "href": "",
        "caption": "",
        "id": "",
        "duration": "",
        "audio": "",
        "download_link": "",
        "category": "",
        "tags": "",
    }
    for key in source_meta:
        meta_dict[key] = source_meta[key]
    return meta_dict


def get_all_files(dirname):
    result = {}
    for maindir, subdir, file_name_list in os.walk(dirname):
        for filename in file_name_list:
            filepath = os.path.join(maindir, filename)
            result[filename] = filepath
    return result


@logger.catch(reraise=True)
def main(
    wavcaps_json_path: str,
    wavcaps_audio_path: str,
    target_ds_path: str,
    str_id: bool = True,
    target_field: str = "caption",
    eoc_special_token: str = SpecialTokens.eoc,
    audio_special_token: str = SpecialTokens.audio,
    add_eoc_at_last: bool = True,
    add_target_field_token: bool = False,
    sent_separator: str = "\n",
):
    """
    Convert a WavCaps-like dataset to the Data-Juicer format.

    :param wavcaps_json_path: path to the json files of WavCaps-like dataset.
    :param wavcaps_audio_path: path to the audio files of WavCaps-like dataset.
    :param target_ds_path: path to store the converted dataset in Data-Juicer
        format.
    :param target_field: the field used to describe audio in the WavCaps-like
        dataset, which can be one of ['caption','title','description'].
    :param eoc_special_token: the special token for "end of a chunk". It's used
        to split conversation chunks explicitly. Default: <|__dj__eoc|> (from
        Data-Juicer).
    :param audio_special_token: the special token for audios. It's used to
        locate the audios in the text.
    :param add_eoc_at_last: whether to add an extra eoc_special_token at the
        end of text. Default: True.
    :param add_target_field_token: whether to add an extra target_field_token
        into text.
    :param sent_separator: separator to split different sentences. Default: \n.
    """
    # ----- Constant settings. Better not to change them. -----
    text_key = "text"  # default key of field to store the sample text
    audio_key = "audios"  # default key of field to store the audio list
    from_format = "[[%s]]: "  # default handle method for the text label
    # ----- Constant settings. Better not to change them. -----

    # check arguments
    # check paths
    if not os.path.exists(wavcaps_json_path):
        raise FileNotFoundError(f"Input WavCaps json path [{wavcaps_json_path}] can " f"not be found.")
    if not os.path.exists(wavcaps_audio_path):
        raise FileNotFoundError(f"Input WavCaps audio path [{wavcaps_audio_path}] can " f"not be found.")
    if not target_ds_path.endswith(".jsonl"):
        raise ValueError('Only support "jsonl" target dataset file now.')

    if target_field not in ["caption", "description", "title"]:
        raise ValueError("target_field must be in '['caption', 'description', 'title']'")

    if os.path.dirname(target_ds_path) and not os.path.exists(os.path.dirname(target_ds_path)):
        logger.info(f"Create directory [{os.path.dirname(target_ds_path)}] " f"for the target dataset.")
        os.makedirs(os.path.dirname(target_ds_path))

    # check whether to add the eoc special token at last
    if not add_eoc_at_last:
        logger.warning(
            "You choose not to add special eoc token at the last, "
            "which might cause some compatibility problems for "
            "other type of datasets (e.g. OpenFlamingo)."
        )

    # load WavCaps dataset
    logger.info("Loading original WavCaps dataset.")
    wavcaps_ds = json.load(open(wavcaps_json_path, "r", encoding="utf-8"))
    num_captions_per_audio = wavcaps_ds["num_captions_per_audio"]
    wavcaps_ds = wavcaps_ds["data"]
    logger.info(f"Load [{len(wavcaps_ds)}] samples.")
    all_audio_files = get_all_files(wavcaps_audio_path)

    with jl.open(target_ds_path, "w") as writer:
        for sample in tqdm(wavcaps_ds):
            # id
            id = sample["id"]
            if str_id:
                id = str(id)

            audio_name = id.strip().split(".")[0] + ".flac"
            target_meta = creat_meta_filed(num_captions_per_audio, sample)

            # audio and text
            if audio_name not in all_audio_files:
                logger.warning(
                    f"No audios in the sample with id [{id}], "
                    f"which means this sample is not a multimodal "
                    f"sample. You'd better remove this sample "
                    f"before converting."
                )
                continue
            audio = [all_audio_files[audio_name]]
            text = audio_special_token + sent_separator
            if target_field not in sample.keys():
                logger.warning(f"{target_field} does not exist in this sample with " f"id [{id}].")
                continue

            if add_target_field_token:
                text += from_format % target_field
            text += sample[target_field]
            if add_eoc_at_last:
                text += eoc_special_token

            # get the new sample with Data-Juicer format
            new_sample = {"id": id, text_key: text, audio_key: audio, Fields.meta: target_meta}
            writer.write(new_sample)
    logger.info(f"Store the target dataset into [{target_ds_path}].")


if __name__ == "__main__":
    fire.Fire(main)

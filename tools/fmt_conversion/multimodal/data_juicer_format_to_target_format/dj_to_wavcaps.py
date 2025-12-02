# This tool is used to convert multimodal dataset in Data-Juicer format to a
# target dataset in WavCaps format.
#
# Data-Juicer format:
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
#
# Corresponding WavCps format:
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

import json
import os

import fire
import jsonlines as jl
from loguru import logger
from tqdm import tqdm

from data_juicer.utils.constant import Fields
from data_juicer.utils.mm_utils import SpecialTokens


@logger.catch(reraise=True)
def main(
    dj_ds_path: str,
    target_wavcaps_ds_path: str,
    target_field: str = "caption",
    eoc_special_token: str = SpecialTokens.eoc,
    audio_special_token: str = SpecialTokens.audio,
    remove_eoc_at_last: bool = True,
    remove_target_field_token: bool = False,
    sent_separator: str = "\n",
):
    """
    Convert a Data-Juicer-format dataset to a WavCaps-like dataset.

    :param dj_ds_path: path to the input dataset in Data-Juicer format.
    :param target_wavcaps_ds_path: path to store the converted dataset in
        WavCaps format.
    :param target_field: the field used to describe audio in the WavCaps-like
        dataset, which can be one of ['caption','title','description'].
    :param eoc_special_token: the special token for "end of a chunk". It's used
        to split conversation chunks explicitly. Default: <|__dj__eoc|> (from
        Data-Juicer).
    :param audio_special_token: the special token for audios. It's used to
        locate the audios in the text.
    :param remove_eoc_at_last: whether to remove the extra eoc_special_token at
        the end of text. Default: True.
    :param remove_target_field_token: whether to remove the extra
        target_field_token at text.
    :param sent_separator: separator to split different sentences. Default: \n.
    """
    # ----- Constant settings. Better not to change them. -----
    from_format = "[[%s]]: "  # default handle method for the text label
    # ----- Constant settings. Better not to change them. -----

    if not os.path.exists(dj_ds_path):
        raise FileNotFoundError(f"Input dataset [{dj_ds_path}] can not be found.")
    if not target_wavcaps_ds_path.endswith(".json"):
        raise ValueError('Only support "json" target dataset file for WavCaps now.')
    if os.path.dirname(target_wavcaps_ds_path) and not os.path.exists(os.path.dirname(target_wavcaps_ds_path)):
        logger.info(f"Create directory [{os.path.dirname(target_wavcaps_ds_path)}] " f"for the target dataset.")
        os.makedirs(os.path.dirname(target_wavcaps_ds_path))

    if target_field not in ["caption", "description", "title"]:
        raise ValueError("target_field must be in '['caption', 'description', 'title']'")

    logger.info("Start to convert.")
    samples = {"num_captions_per_audio": 1, "data": []}
    with jl.open(dj_ds_path, "r") as reader:
        for sample in tqdm(reader):
            id = sample["id"]
            if Fields.meta not in sample:
                logger.warning(f"{Fields.meta} does not exist in this sample with " f"id [{id}].")
                continue

            if target_field not in sample[Fields.meta].keys():
                logger.warning(f"{target_field} does not exist in this sample with " f"id [{id}].")
                continue
            samples["num_captions_per_audio"] = sample[Fields.meta]["num_captions_per_audio"]
            del sample[Fields.meta]["num_captions_per_audio"]

            sample[Fields.meta][target_field] = sample["text"].replace(audio_special_token + sent_separator, "")
            if remove_eoc_at_last:
                sample[Fields.meta][target_field] = sample[Fields.meta][target_field].replace(eoc_special_token, "")
            if remove_target_field_token:
                sample[Fields.meta][target_field] = sample[Fields.meta][target_field].replace(
                    from_format % target_field, ""
                )
            samples["data"].append(sample[Fields.meta])

    logger.info(f"Start to write the converted dataset to " f"[{target_wavcaps_ds_path}]...")
    json.dump(samples, open(target_wavcaps_ds_path, "w", encoding="utf-8"))


if __name__ == "__main__":
    fire.Fire(main)

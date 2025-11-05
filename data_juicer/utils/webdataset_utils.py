import io
import json
from typing import Any, Dict, Optional, Union

import numpy as np
import PIL.Image

from data_juicer.utils.mm_utils import load_audio

_VIDEO_EXTENTIONS = ["mp4", "mov", "avi", "mkv"]
_AUDIO_EXTENTIONS = ["flac", "mp3", "sox", "wav", "m4a", "ogg", "wma"]
_IMAGE_EXTENTIONS = ["jpg", "png", "ppm", "pgm", "pbm", "pnm"]


def read_file_as_bytes(file_path):
    with open(file_path, "rb") as f:
        return f.read()


def _load_image(value, format="PIL"):
    import numpy as np
    import PIL.Image

    if format == "PIL":
        return PIL.Image.open(io.BytesIO(value))
    else:
        return np.asarray(PIL.Image.open(io.BytesIO(value)))


def _custom_default_decoder(sample: Dict[str, Any], format: Optional[Union[bool, str]] = True):
    """A custom decoder for webdataset. Support multiple images list decoding.

    This handles common file extensions: .txt, .cls, .cls2,
        .jpg, .png, .json, .npy, .mp, .pt, .pth, .pickle, .pkl.
    These are the most common extensions used in webdataset.
    For other extensions, users can provide their own decoder.

    Args:
        sample: sample, modified in place
    """
    sample = dict(sample)
    for key, value in sample.items():
        extension = key.split(".")[-1]
        if key.startswith("__"):
            continue
        elif extension in ["txt", "text"]:
            sample[key] = value.decode("utf-8")
        elif extension in ["cls", "cls2"]:
            sample[key] = int(value.decode("utf-8"))
        elif extension in _IMAGE_EXTENTIONS:
            sample[key] = _load_image(value, format)
        elif extension in [s + "s" for s in _IMAGE_EXTENTIONS]:
            import pickle

            value = pickle.loads(value)

            sample[key] = [_load_image(v, format) for v in value]
        elif extension == "json":
            sample[key] = json.loads(value)
        elif extension == "npy":
            import numpy as np

            sample[key] = np.load(io.BytesIO(value))
        elif extension == "mp":
            import msgpack

            sample[key] = msgpack.unpackb(value, raw=False)
        elif extension in ["pt", "pth"]:
            import torch

            sample[key] = torch.load(io.BytesIO(value))
        elif extension in ["pickle", "pkl"]:
            import pickle

            sample[key] = pickle.loads(value)
        elif extension in _AUDIO_EXTENTIONS:
            sample[key] = load_audio(value)
        elif extension in [s + "s" for s in _AUDIO_EXTENTIONS]:
            sample[key] = [load_audio(v) for v in pickle.loads(value)]
        elif extension in _VIDEO_EXTENTIONS:
            import pickle

            value = pickle.loads(value)
            sample[key] = [_load_image(frame, format) for frame in value]
        elif extension in [s + "s" for s in _VIDEO_EXTENTIONS]:
            import pickle

            videos_frames_list = pickle.loads(value)
            videos_frames_decode = []
            for video_frames in videos_frames_list:
                videos_frames_decode.append([_load_image(frame) for frame in video_frames])
            # list in list
            sample[key] = videos_frames_decode

    return sample


def _encode_image(value, extension):
    from ray.data._internal.datasource.webdataset_datasource import extension_to_format

    if isinstance(value, np.ndarray):
        value = PIL.Image.fromarray(value)
    elif isinstance(value, bytes):
        return value
    elif isinstance(value, str):
        value = PIL.Image.open(value)
    assert isinstance(value, PIL.Image.Image)
    stream = io.BytesIO()
    value.save(stream, format=extension_to_format.get(extension.lower(), extension))
    return stream.getvalue()


def _encode_audio(value):
    if isinstance(value, str):
        return read_file_as_bytes(value)
    elif isinstance(value, bytes):
        return value
    assert isinstance(value, bytes), f"value should be a bytes, got {type(value)}"

    return value


def _custom_default_encoder(sample: Dict[str, Any], format: Optional[Union[str, bool]] = True):
    """A custom encoder for webdataset.
    In addition to the original encoding, it also supports encode image lists and byte type images.

    This handles common file extensions: .txt, .cls, .cls2, .jpg,
        .png, .json, .npy, .mp, .pt, .pth, .pickle, .pkl, .jpgs (images list),
        .jpegs (images list), .pngs (images list).
    These are the most common extensions used in webdataset.
    For other extensions, users can provide their own encoder.

    Args:
        sample (Dict[str, Any]): sample
    """
    sample = dict(sample)
    for key, value in sample.items():
        extension = key.split(".")[-1]
        if key.startswith("__"):
            continue
        elif extension in ["txt"]:
            if isinstance(value, list):
                sample[key] = [v.encode("utf-8") for v in value]
            else:
                sample[key] = value.encode("utf-8")
        elif extension in ["cls", "cls2"]:
            sample[key] = str(value).encode("utf-8")
        elif extension in _IMAGE_EXTENTIONS:
            sample[key] = _encode_image(value, extension)
        elif extension in [s + "s" for s in _IMAGE_EXTENTIONS]:
            import pickle

            extension = extension.rstrip("s")
            sample[key] = pickle.dumps([_encode_image(v, extension) for v in value])
        elif extension == "json":
            sample[key] = json.dumps(value).encode("utf-8")
        elif extension == "npy":
            import numpy as np

            stream = io.BytesIO()
            np.save(stream, value)
            sample[key] = stream.getvalue()
        elif extension == "mp":
            import msgpack

            sample[key] = msgpack.dumps(value)
        elif extension in ["pt", "pth"]:
            import torch

            stream = io.BytesIO()
            torch.save(value, stream)
            sample[key] = stream.getvalue()
        elif extension in ["pickle", "pkl"]:
            import pickle

            stream = io.BytesIO()
            pickle.dump(value, stream)
            sample[key] = stream.getvalue()
        elif extension in _AUDIO_EXTENTIONS:
            sample[key] = _encode_audio(value)
        elif extension in [s + "s" for s in _AUDIO_EXTENTIONS]:
            import pickle

            extension = extension.rstrip("s")
            sample[key] = pickle.dumps([_encode_audio(v) for v in value])
        elif extension in _VIDEO_EXTENTIONS:
            import pickle

            extension = "jpg"
            sample[key] = pickle.dumps([_encode_image(frame, extension) for frame in value])
        elif extension in [s + "s" for s in _VIDEO_EXTENTIONS]:
            import pickle

            videos_frames_list = value
            videos_frames_decode = []
            for video_frames in videos_frames_list:
                cur_decode_frames = []
                for frame in video_frames:
                    if isinstance(frame, str):
                        frame = _encode_image(frame)
                    else:
                        assert isinstance(frame, bytes), "frame should be string path or bytes"
                    cur_decode_frames.append(frame)

                videos_frames_decode.append(cur_decode_frames)
            # list in list
            sample[key] = pickle.dumps(videos_frames_decode)

    return sample


def reconstruct_custom_webdataset_format(samples, field_mapping: Optional[Dict[str, str]] = None):
    """
    Reconstruct the original dataset to the WebDataset format.
    For all keys, they can be specified by `field_mapping` argument, which is a dict mapping from the target
    field key in the result format to the source field key in the original samples.

    :param samples: the input samples batch to be reconstructed
    :param field_mapping: the field mapping to construct the left fields.
    """
    if field_mapping is None:
        field_mapping = {}
    assert isinstance(field_mapping, dict)

    # not specified -- return the original samples
    if len(field_mapping) == 0:
        return samples

    # construct the left fields
    reconstructed_sample = {}
    for tgt_field, src_field in field_mapping.items():
        assert isinstance(src_field, str) or isinstance(src_field, list)
        if isinstance(src_field, str):
            reconstructed_sample[tgt_field] = samples[src_field]
        elif isinstance(src_field, list):
            reconstructed_sample[tgt_field] = {src_field_item: samples[src_field_item] for src_field_item in src_field}

    return reconstructed_sample

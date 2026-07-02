import copy
import os
import re
import uuid

import numpy as np
from loguru import logger

from data_juicer.utils.constant import Fields
from data_juicer.utils.file_utils import add_suffix_to_filename, transfer_filename
from data_juicer.utils.mm_utils import SpecialTokens
from data_juicer.utils.video_utils import create_video_reader

from ..base_op import OPERATORS, Mapper
from ..op_fusion import LOADED_VIDEOS


def create_replacer(replacements):
    def replacer(match):
        return replacements.pop(0)

    return replacer


OP_NAME = "video_split_by_duration_mapper"


@OPERATORS.register_module(OP_NAME)
@LOADED_VIDEOS.register_module(OP_NAME)
class VideoSplitByDurationMapper(Mapper):
    """Splits videos into segments based on a specified duration.

    This operator splits each video in the dataset into smaller segments, each with a fixed
    duration. The last segment is discarded if its duration is less than the specified
    minimum last split duration. The original sample can be kept or removed based on the
    `keep_original_sample` parameter. The generated video files are saved in the specified
    directory or, if not provided, in the same directory as the input files. The key metric
    for this operation is the duration of each segment, which is character-based (seconds).

    - Splits videos into segments of a specified duration.
    - Discards the last segment if it is shorter than the minimum allowed duration.
    - Keeps or removes the original sample based on the `keep_original_sample` parameter.
    - Saves the generated video files in the specified directory or the input file's
      directory.
    - Uses the duration in seconds to determine the segment boundaries."""

    _batched_op = True

    def __init__(
        self,
        split_duration: float = 10,
        overlap_duration: float = 0,
        min_last_split_duration: float = 0,
        keep_original_sample: bool = True,
        save_dir: str = None,
        video_backend: str = "ffmpeg",
        ffmpeg_extra_args: str = "",
        output_format: str = "path",
        save_field: str = None,
        legacy_split_by_text_token: bool = True,
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param split_duration: duration of each video split in seconds.
        :param overlap_duration: overlap duration in seconds between
            consecutive splits. For example, with split_duration=20 and
            overlap_duration=5, clips will be [0-20, 15-35, 30-50, ...].
            Must be non-negative and less than split_duration. Default: 0
            (no overlap).
        :param min_last_split_duration: The minimum allowable duration in
            seconds for the last video split. If the duration of the last
            split is less than this value, it will be discarded.
        :param keep_original_sample: whether to keep the original sample. If
            it's set to False, there will be only cut sample in the
            final datasets and the original sample will be removed. It's True
            in default.
        :param save_dir: The directory where generated video files will be stored.
            If not specified, outputs will be saved in the same directory as their corresponding input files.
            This path can alternatively be defined by setting the `DJ_PRODUCED_DATA_DIR` environment variable.
        :param video_backend: video backend, can be `ffmpeg`, `av`.
        :param ffmpeg_extra_args: Extra ffmpeg args for splitting video, only valid when `video_backend` is `ffmpeg`.
        :param output_format: The output format of the videos.
            Supported formats are: ["path", "bytes"].
            If format is "path", the output is a list of lists, where each inner
            list contains the path of the split videos.
            If format is "bytes", the output is a list of lists, where each inner
            list contains the bytes of the split videos.
        :param save_field: The new field name to save generated video files path.
            If not specified, will overwrite the original video field.
        :param legacy_split_by_text_token: Whether to split by special tokens (e.g. <__dj__video>)
            in the text field and read videos in order, or use the 'videos' or 'frames' field directly.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self._init_parameters = self.remove_extra_parameters(locals())
        self._init_parameters.pop("save_dir", None)

        self.split_duration = split_duration
        self.overlap_duration = overlap_duration
        assert self.overlap_duration >= 0, f"overlap_duration must be >= 0, got {overlap_duration}"
        assert self.overlap_duration < self.split_duration, (
            f"overlap_duration ({overlap_duration}) must be less than " f"split_duration ({split_duration})"
        )
        self.min_last_split_duration = min_last_split_duration
        self.keep_original_sample = keep_original_sample
        self.extra_args = kwargs
        self.save_dir = save_dir
        self.video_backend = video_backend
        assert self.video_backend in ["ffmpeg", "av"]
        self.ffmpeg_extra_args = ffmpeg_extra_args
        self.output_format = output_format.lower()
        assert self.output_format in [
            "path",
            "bytes",
        ], f"output_format '{output_format}' is not supported. Can only be one of ['path', 'bytes']."
        self.save_field = save_field
        self.legacy_split_by_text_token = legacy_split_by_text_token
        if self.legacy_split_by_text_token:
            logger.warning(
                "`legacy_split_by_text_token` is set to true, "
                "spliting the text field by special tokens "
                "(e.g. <__dj__video>) to read videos in order. "
                "This behavior will be deprecated in future versions. "
                "Please set `legacy_split_by_text_token` to False, "
                'and use the "videos" or "frames" field directly.'
            )

    def split_videos_by_duration(self, container, video_key: str = None):
        video_duration = container.metadata.duration
        if video_duration <= self.split_duration:
            if video_key:
                return [video_key]
            return []

        # Step size: split_duration - overlap_duration
        # e.g. split=20, overlap=5 → step=15 → starts=[0, 15, 30, ...]
        step = self.split_duration - self.overlap_duration
        start_times = np.arange(0, video_duration, step).tolist()

        count = 0
        split_video_keys = []

        if video_key:
            unique_video_key = transfer_filename(video_key, OP_NAME, self.save_dir, **self._init_parameters)
        else:
            unique_video_key = os.path.join(self.save_dir, f"{uuid.uuid4().hex}.mp4")

        if self.video_backend == "ffmpeg" and self.ffmpeg_extra_args:
            kwargs = {"ffmpeg_extra_args": self.ffmpeg_extra_args}
        else:
            kwargs = {}

        for start in start_times:
            end = start + self.split_duration

            if end >= video_duration:
                # Last segment: check minimum duration
                remaining = video_duration - start
                if remaining >= self.min_last_split_duration:
                    split_video_key = add_suffix_to_filename(unique_video_key, f"_{count}")
                    if container.extract_clip(start, None, split_video_key, **kwargs):
                        split_video_keys.append(split_video_key)
                        count += 1
                break
            else:
                split_video_key = add_suffix_to_filename(unique_video_key, f"_{count}")
                if container.extract_clip(start, end, split_video_key, **kwargs):
                    split_video_keys.append(split_video_key)
                    count += 1

        return split_video_keys

    def _process_single_sample(self, sample):
        # there is no video in this sample
        if self.video_key not in sample or sample[self.video_key] is None or len(sample[self.video_key]) == 0:
            sample[Fields.source_file] = []
            return []

        is_video_path = isinstance(sample[self.video_key][0], str)
        if Fields.source_file not in sample or not sample[Fields.source_file]:
            if is_video_path:
                sample[Fields.source_file] = sample[self.video_key]

        # the split results
        split_sample = copy.deepcopy(sample)
        split_sample[self.text_key] = ""
        split_sample[Fields.source_file] = []

        # load all video(s)
        loaded_videos = sample[self.video_key]
        videos = {}
        for video_idx, loaded_video in enumerate(loaded_videos):
            if video_idx not in videos:
                # avoid loading the same videos
                video = create_video_reader(loaded_video, backend=self.video_backend)
                videos[video_idx] = video

        split_video_keys = []

        if self.legacy_split_by_text_token:
            offset = 0
            # split each video chunk by chunk
            for chunk in sample[self.text_key].split(SpecialTokens.eoc):
                # skip empty chunks or contents after the last eoc token
                if not chunk.strip():
                    continue
                else:
                    video_count = chunk.count(SpecialTokens.video)
                    place_holders = []
                    for idx in range(offset, offset + video_count):
                        video = videos[idx]
                        if is_video_path:
                            video_path = loaded_videos[idx]
                            new_video_keys = self.split_videos_by_duration(video, video_path)
                            split_sample[Fields.source_file].extend([video_path] * len(new_video_keys))
                        else:
                            new_video_keys = self.split_videos_by_duration(video, None)
                            split_sample[Fields.source_file].extend(new_video_keys)
                        video.close()
                        split_video_keys.extend(new_video_keys)
                        place_holders.append(SpecialTokens.video * len(new_video_keys))

                    # insert the generated text according to given mode
                    replacer_function = create_replacer(place_holders)
                    new_split_text_per_chunk = re.sub(SpecialTokens.video, replacer_function, chunk)
                    split_sample[self.text_key] += f"{new_split_text_per_chunk}{SpecialTokens.eoc}"  # noqa: E501
                    offset += video_count
        else:
            # TODO: handle the text field update
            for video_idx, video in videos.items():
                if is_video_path:
                    video_path = loaded_videos[video_idx]
                    new_video_keys = self.split_videos_by_duration(video, video_path)
                    split_sample[Fields.source_file].extend([video_path] * len(new_video_keys))
                else:
                    new_video_keys = self.split_videos_by_duration(video, None)
                    split_sample[Fields.source_file].extend(new_video_keys)
                video.close()
                split_video_keys.extend(new_video_keys)

        if self.output_format == "bytes":
            from data_juicer.utils.mm_utils import load_file_byte

            split_videos = [load_file_byte(f) for f in split_video_keys]
        else:
            split_videos = split_video_keys

        if self.save_field:
            split_sample[self.save_field] = split_videos
        else:
            split_sample[self.video_key] = split_videos
        return [split_sample]

    def process_batched(self, samples):
        # reconstruct samples from "dict of lists" to "list of dicts"
        reconstructed_samples = []
        for i in range(len(samples[self.text_key])):
            reconstructed_samples.append({key: samples[key][i] for key in samples})
        samples_after_split = []
        # do split for each sample within the batch
        for ori_sample in reconstructed_samples:
            if self.keep_original_sample:
                samples_after_split.append(ori_sample)
            generated_samples = self._process_single_sample(ori_sample)
            if len(generated_samples) != 0:
                samples_after_split.extend(generated_samples)
        # reconstruct samples from "list of dicts" to "dict of lists"
        keys = samples_after_split[0].keys()
        res_samples = {}
        for key in keys:
            res_samples[key] = [s[key] for s in samples_after_split]
        return res_samples

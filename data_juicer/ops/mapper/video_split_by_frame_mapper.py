import copy
import math
import os
import re
import subprocess
from typing import Optional

from data_juicer.utils.constant import Fields
from data_juicer.utils.file_utils import add_suffix_to_filename
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.mm_utils import SpecialTokens

from ..base_op import OPERATORS, Mapper
from ..op_fusion import LOADED_VIDEOS

cv2 = LazyLoader("cv2")


def create_replacer(replacements):
    def replacer(match):
        return replacements.pop(0)

    return replacer


OP_NAME = "video_split_by_frame_mapper"


@OPERATORS.register_module(OP_NAME)
@LOADED_VIDEOS.register_module(OP_NAME)
class VideoSplitByFrameMapper(Mapper):
    """Splits videos into segments based on a specified frame count or duration.

    This operator splits each video in the dataset into smaller segments, each with a fixed
    frame count (or duration). It supports overlapping segments. The original sample can be
    kept or removed based on the `keep_original_sample` parameter. The generated video files
    are saved in the specified directory or, if not provided, in the same directory as the
    input files.

    - Splits videos into segments of a specified frame count or duration.
    - Supports overlapping segments.
    - Keeps or removes the original sample based on the `keep_original_sample` parameter.
    - Saves the generated video files in the specified directory or the input file's
      directory.
    """

    _batched_op = True

    def __init__(
        self,
        split_len: float = 10,
        overlap_len: float = 0,
        unit: str = "frame",
        keep_original_sample: bool = True,
        save_dir: Optional[str] = None,
        ffmpeg_extra_args: str = "",
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param split_len: length of each video split. Can be frame count (int) or duration in seconds (float).
        :param overlap_len: overlap length between adjacent splits. Can be frame count (int) or duration in seconds (float).
        :param unit: unit of split_len and overlap_len. Can be 'frame' or 'second'. Default is 'frame'.
        :param keep_original_sample: whether to keep the original sample. If
            it's set to False, there will be only cut sample in the
            final datasets and the original sample will be removed. It's True
            in default.
        :param save_dir: The directory where generated video files will be stored.
            If not specified, outputs will be saved in the same directory as their corresponding input files.
            This path can alternatively be defined by setting the `DJ_PRODUCED_DATA_DIR` environment variable.
        :param ffmpeg_extra_args: Extra ffmpeg args for splitting video.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self._init_parameters = self.remove_extra_parameters(locals())
        self._init_parameters.pop("save_dir", None)

        self.split_len = split_len
        self.overlap_len = overlap_len
        self.unit = unit
        if self.unit not in ["frame", "second"]:
            raise ValueError(f"Unit must be 'frame' or 'second', but got {self.unit}")

        self.keep_original_sample = keep_original_sample
        self.save_dir = save_dir
        self.ffmpeg_extra_args = ffmpeg_extra_args

    def split_videos_by_frame(self, video_key):
        # 1. Get video metadata using OpenCV
        cap = cv2.VideoCapture(video_key)
        if not cap.isOpened():
            # If video cannot be opened, return empty list (or maybe original key?)
            # Following other mappers, we might skip or return empty.
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        if fps <= 0 or total_frames <= 0:
            return []

        # Convert seconds to frames if unit is 'second'
        if self.unit == "second":
            clip_len_frames = int(self.split_len * fps)
            overlap_len_frames = int(self.overlap_len * fps)
        else:
            clip_len_frames = int(self.split_len)
            overlap_len_frames = int(self.overlap_len)

        if clip_len_frames <= 0:
            return []

        # Calculate number of parts
        # We want to cover the whole video.
        # If overlap is 0: num_parts = ceil(total / clip)
        # If overlap > 0: stride = clip - overlap.
        # We start at 0, stride, 2*stride...
        # Last start point must be such that start + clip <= total (if we don't want partials at end?)
        # The user's code:
        # num_parts = math.ceil(total_frames / clip_len) -> This logic in user code seems to assume 0 overlap for count calculation?
        # Wait, user code:
        # num_parts = math.ceil(total_frames / clip_len)
        # for part_idx in range(num_parts):
        #    start_frame = part_idx * clip_len
        #    frames_to_extract = clip_len + overlap_len
        # This logic implies the "stride" is `clip_len`. And `overlap_len` is EXTRA frames added to the end.
        # So it's not "sliding window with overlap" in the traditional sense (where stride = window - overlap).
        # It is "consecutive chunks + extra overlap at the end".
        # Let's stick to the user's logic as requested: "split_video_for_vitra" logic.
        # User code:
        # num_parts = math.ceil(total_frames / clip_len)
        # start_frame = part_idx * clip_len
        # frames_to_extract = clip_len + overlap_len

        # However, if I use `split_len` as the "stride" and `overlap_len` as the overlap...
        # The user's variable is `CLIP_LEN` and `OVERLAP`.
        # `num_parts = math.ceil(total_frames / CLIP_LEN)`
        # `start_frame = part_idx * CLIP_LEN`
        # `frames_to_extract = CLIP_LEN + OVERLAP`
        # This means the actual duration of the output video is `CLIP_LEN + OVERLAP`.
        # And the next video starts at `CLIP_LEN`.
        # So the overlap is indeed `OVERLAP`.
        # Example: Clip=100, Overlap=20.
        # Part 0: Start 0, Len 120. (0-120)
        # Part 1: Start 100, Len 120. (100-220)
        # Overlap is 20 frames (100-120).
        # Yes, this matches "stride = split_len".

        stride = clip_len_frames
        extra_overlap = overlap_len_frames

        num_parts = math.ceil(total_frames / stride)

        split_video_keys = []
        # transfer_filename can be imported from data_juicer.utils.file_utils if needed
        # unique_video_key = transfer_filename(video_key, OP_NAME, self.save_dir, **self._init_parameters)

        if self.save_dir is not None:
            output_dir = self.save_dir
        else:
            output_dir = os.path.dirname(video_key)

        unique_video_key = os.path.join(output_dir, os.path.basename(video_key))

        for part_idx in range(num_parts):
            start_frame = part_idx * stride
            frames_to_extract = stride + extra_overlap

            # Boundary check
            if start_frame + frames_to_extract > total_frames:
                frames_to_extract = total_frames - start_frame

            # If the remaining frames are too few (e.g. less than a threshold?), user code doesn't check.
            # But user code does: if start_frame + frames_to_extract > total_frames: frames_to_extract = ...
            # And if frames_to_extract <= 0? (Should not happen if loop is correct)

            start_time = start_frame / fps

            output_path = add_suffix_to_filename(unique_video_key, f"_part{part_idx + 1}")

            # Construct FFmpeg command
            # -y: overwrite
            # -ss: start time
            # -i: input
            # -frames:v: frames to extract
            # -c:v libx264 ...
            # -an: remove audio (User code has -an. Should I keep it? Maybe make it optional? User code says "VITRA seems to only focus on visual". I'll keep it for now or make it configurable via ffmpeg_extra_args. But user asked to imitate the code.)
            # I will include -an by default if it's in the user code, but maybe it's better to let ffmpeg_extra_args handle encoding options.
            # User code: "-c:v", "libx264", "-preset", "fast", "-crf", "22", "-an"

            cmd = [
                "ffmpeg",
                "-y",
                "-ss",
                f"{start_time:.4f}",
                "-i",
                video_key,
                "-frames:v",
                str(frames_to_extract),
            ]

            # Default encoding args from user code
            # We can allow overriding via ffmpeg_extra_args
            if self.ffmpeg_extra_args:
                import shlex

                cmd.extend(shlex.split(self.ffmpeg_extra_args))
            else:
                # Default to user's sample logic if no args provided
                cmd.extend(["-c:v", "libx264", "-preset", "fast", "-crf", "22", "-an"])

            cmd.extend(["-loglevel", "error", output_path])

            try:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                split_video_keys.append(output_path)
            except subprocess.CalledProcessError:
                # If failed, maybe skip?
                continue

        return split_video_keys

    def _process_single_sample(self, sample):
        # there is no video in this sample
        if self.video_key not in sample or sample[self.video_key] is None or len(sample[self.video_key]) == 0:
            sample[Fields.source_file] = []
            return []

        if Fields.source_file not in sample or not sample[Fields.source_file]:
            sample[Fields.source_file] = sample[self.video_key]

        # the split results
        split_sample = copy.deepcopy(sample)
        split_sample[self.text_key] = ""
        split_sample[Fields.source_file] = []

        # load all video(s)
        loaded_video_keys = sample[self.video_key]

        # We don't need to pre-load video objects like in DurationMapper because we use cv2/ffmpeg per file.

        split_video_keys = []
        offset = 0
        # split each video chunk by chunk
        for chunk in sample[self.text_key].split(SpecialTokens.eoc):
            # skip empty chunks or contents after the last eoc token
            if not chunk.strip():
                continue
            else:
                video_count = chunk.count(SpecialTokens.video)
                place_holders = []
                for video_key in loaded_video_keys[offset : offset + video_count]:
                    new_video_keys = self.split_videos_by_frame(video_key)
                    split_video_keys.extend(new_video_keys)
                    place_holders.append(SpecialTokens.video * len(new_video_keys))
                    split_sample[Fields.source_file].extend([video_key] * len(new_video_keys))

                # insert the generated text according to given mode
                replacer_function = create_replacer(place_holders)
                new_split_text_per_chunk = re.sub(SpecialTokens.video, replacer_function, chunk)
                split_sample[self.text_key] += f"{new_split_text_per_chunk}{SpecialTokens.eoc}"  # noqa: E501
                offset += video_count

        split_sample[self.video_key] = split_video_keys
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

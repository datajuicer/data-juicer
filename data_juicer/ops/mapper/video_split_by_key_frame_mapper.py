import copy
import re

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


OP_NAME = "video_split_by_key_frame_mapper"


@OPERATORS.register_module(OP_NAME)
@LOADED_VIDEOS.register_module(OP_NAME)
class VideoSplitByKeyFrameMapper(Mapper):
    """Splits a video into segments based on key frames.

    This operator processes video data by splitting it into multiple segments at key frame
    boundaries. It uses the key frames to determine where to make the splits. The original
    sample can be kept or discarded based on the `keep_original_sample` parameter. If
    `save_dir` is specified, the split video files will be saved in that directory;
    otherwise, they will be saved in the same directory as the input files. The operator
    processes each video in the sample and updates the sample with the new video keys and
    text placeholders. The `Fields.source_file` field is updated to reflect the new video
    segments. This operator works in batch mode, processing multiple samples at once."""

    _batched_op = True

    def __init__(
        self,
        keep_original_sample: bool = True,
        save_dir: str = None,
        video_backend: str = "av",
        ffmpeg_extra_args: str = "",
        output_format: str = "path",
        save_field: str = None,
        legacy_split_by_text_token: bool = True,
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param keep_original_sample: whether to keep the original sample. If
            it's set to False, there will be only split sample in the
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
            e.g.[
                    [video1_split1_path, video1_split2_path, ...],
                    [video2_split1_path, video2_split2_path, ...],
                    ...
                ] (In the order of the videos).
            If format is "bytes", the output is a list of lists, where each inner
            list contains the bytes of the split videos.
            e.g. [
                    [video1_split1_byte, video1_split2_byte, ...],
                    [video2_split1_byte, video2_split2_byte, ...],
                    ...
                ] (In the order of the videos).
        :param save_field: The new field name to save generated video files path.
            If not specified, will overwrite the original video field.
        :param legacy_split_by_text_token: Whether to split by special tokens (e.g. <__dj__video>)
            in the text field and read videos in order, or use the 'videos' field directly.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self._init_parameters = self.remove_extra_parameters(locals())
        self._init_parameters.pop("save_dir", None)
        self.keep_original_sample = keep_original_sample
        self.extra_args = kwargs
        self.save_dir = save_dir
        self.video_backend = video_backend
        assert self.video_backend in ["ffmpeg", "av"]
        self.ffmpeg_extra_args = ffmpeg_extra_args
        self.output_format = output_format.lower()
        self.save_field = save_field
        assert self.output_format in [
            "path",
            "bytes",
        ], f"output_format '{output_format}' is not supported. Can only be one of ['path', 'bytes']."
        self.legacy_split_by_text_token = legacy_split_by_text_token
        if legacy_split_by_text_token:
            logger.warning(
                "`legacy_split_by_text_token` is set to true, "
                "spliting the text field by special tokens "
                "(e.g. <__dj__video>) to read videos in order. "
                "This behavior will be deprecated in future versions. "
                "Please set `legacy_split_by_text_token` to False, "
                'and use the "videos" field directly.'
            )

    def get_split_key_frame(self, video_key, container):
        timestamps = container.extract_keyframes().pts_time

        if self.video_backend == "ffmpeg" and self.ffmpeg_extra_args:
            kwargs = {"ffmpeg_extra_args": self.ffmpeg_extra_args}
        else:
            kwargs = {}
        count = 0
        split_video_keys = []
        unique_video_key = transfer_filename(video_key, OP_NAME, self.save_dir, **self._init_parameters)
        for i in range(1, len(timestamps)):
            split_video_key = add_suffix_to_filename(unique_video_key, f"_{count}")
            if container.extract_clip(timestamps[i - 1], timestamps[i], split_video_key, **kwargs):
                split_video_keys.append(split_video_key)
                count += 1

        split_video_key = add_suffix_to_filename(unique_video_key, f"_{count}")
        if timestamps and container.extract_clip(timestamps[-1], None, split_video_key, **kwargs):
            split_video_keys.append(split_video_key)
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
        videos = {}
        for loaded_video_key in loaded_video_keys:
            if loaded_video_key not in videos:
                # avoid loading the same videos
                video = create_video_reader(loaded_video_key, backend=self.video_backend)
                videos[loaded_video_key] = video

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
                    for video_key in loaded_video_keys[offset : offset + video_count]:
                        video = videos[video_key]
                        new_video_keys = self.get_split_key_frame(video_key, video)
                        video.close()
                        split_video_keys.extend(new_video_keys)
                        place_holders.append(SpecialTokens.video * len(new_video_keys))
                        split_sample[Fields.source_file].extend([video_key] * len(new_video_keys))

                    # insert the generated text according to given mode
                    replacer_function = create_replacer(place_holders)
                    new_split_text_per_chunk = re.sub(SpecialTokens.video, replacer_function, chunk)
                    split_sample[self.text_key] += f"{new_split_text_per_chunk}{SpecialTokens.eoc}"  # noqa: E501
                    offset += video_count
        else:
            # TODO: handle the text field update
            for video_key in loaded_video_keys:
                video = videos[video_key]
                new_video_keys = self.get_split_key_frame(video_key, video)
                video.close()
                split_video_keys.extend(new_video_keys)
                split_sample[Fields.source_file].extend([video_key] * len(new_video_keys))

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

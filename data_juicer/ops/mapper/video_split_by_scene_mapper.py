import math
import os
import re
import shutil
import tempfile
import uuid
from itertools import chain

from loguru import logger
from pydantic import NonNegativeFloat, NonNegativeInt

from data_juicer.utils.constant import Fields
from data_juicer.utils.file_utils import add_suffix_to_filename, transfer_filename
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.mm_utils import SpecialTokens

from ..base_op import OPERATORS, Mapper

scenedetect = LazyLoader("scenedetect")

OP_NAME = "video_split_by_scene_mapper"


def replace_func(match, scene_counts_iter):
    try:
        count = next(scene_counts_iter)
        return SpecialTokens.video * count
    except StopIteration:
        return match.group(0)


@OPERATORS.register_module(OP_NAME)
class VideoSplitBySceneMapper(Mapper):
    """Splits videos into scene clips based on detected scene changes.

    This operator uses a specified scene detector to identify and split video scenes. It
    supports three types of detectors: ContentDetector, ThresholdDetector, and
    AdaptiveDetector. The operator processes each video in the sample, detects scenes, and
    splits the video into individual clips. The minimum length of a scene can be set, and
    progress can be shown during processing. The resulting clips are saved in the specified
    directory or the same directory as the input files if no save directory is provided. The
    operator also updates the text field in the sample to reflect the new video clips. If a
    video does not contain any scenes, it remains unchanged."""

    # Define shared detector keys and their properties
    available_detectors = {
        "ContentDetector": ["weights", "luma_only", "kernel_size"],
        "AdaptiveDetector": [
            "window_width",
            "min_content_val",
            "weights",
            "luma_only",
            "kernel_size",
            "video_manager",
            "min_delta_hsv",
        ],
        "ThresholdDetector": ["fade_bias", "add_final_scene", "method", "block_size"],
    }

    def __init__(
        self,
        detector: str = "ContentDetector",
        threshold: NonNegativeFloat = 27.0,
        min_scene_len: NonNegativeInt = 15,
        show_progress: bool = False,
        save_dir: str = None,
        save_field: str = None,
        ffmpeg_extra_args: str = "-movflags frag_keyframe+empty_moov",
        output_format: str = "path",
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param detector: Algorithm from `scenedetect.detectors`. Should be one
            of ['ContentDetector', 'ThresholdDetector', 'AdaptiveDetector`].
        :param threshold: Threshold passed to the detector.
        :param min_scene_len: Minimum length of any scene.
        :param show_progress: Whether to show progress from scenedetect.
        :param save_dir: The directory where generated video files will be stored.
            If not specified, outputs will be saved in the same directory as their corresponding input files.
            This path can alternatively be defined by setting the `DJ_PRODUCED_DATA_DIR` environment variable.
        :param save_field: The new field name to save generated video files path.
            If not specified, will overwrite the original video field.
        :param ffmpeg_extra_args: Extra ffmpeg args for splitting video.
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
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self._init_parameters = self.remove_extra_parameters(locals())
        self._init_parameters.pop("save_dir", None)

        if detector not in self.available_detectors:
            raise ValueError(
                f"Scene detector {detector} is not supported. "
                f"Can only be one of {list(self.available_detectors.keys())}"
            )

        self.detector = detector
        self.threshold = threshold
        self.min_scene_len = min_scene_len
        self.show_progress = show_progress
        self.save_dir = save_dir
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
        self.save_field = save_field
        self.ffmpeg_extra_args = ffmpeg_extra_args
        self.output_format = output_format.lower()
        assert self.output_format in [
            "path",
            "bytes",
        ], f"output_format '{output_format}' is not supported. Can only be one of ['path', 'bytes']."

        # prepare detector args
        available_kwargs = self.available_detectors[self.detector]
        self.detector_class = getattr(scenedetect.detectors, self.detector)
        self.detector_kwargs = {key: kwargs[key] for key in available_kwargs if key in kwargs}

    def _detect_scenes_and_split_video(self, video, detector, is_video_path, redirected_video_key):
        output_template = add_suffix_to_filename(redirected_video_key, "_$SCENE_NUMBER")
        if is_video_path:
            scene_list = scenedetect.detect(video, detector, show_progress=self.show_progress, start_in_scene=True)

            if len(scene_list) > 1:
                output_files = self._split_video(video, scene_list, output_template)
            else:
                output_files = [video]  # raw video path
            return scene_list, output_files
        else:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as temp_file:
                temp_file.write(video)
                temp_file.flush()
                scene_list = scenedetect.detect(
                    temp_file.name, detector, show_progress=self.show_progress, start_in_scene=True
                )
                if len(scene_list) > 1:
                    output_files = self._split_video(temp_file.name, scene_list, output_template)
                else:
                    shutil.copyfile(temp_file.name, redirected_video_key)
                    output_files = [redirected_video_key]
                return scene_list, output_files

    def _split_video(self, video_path, scene_list, output_template):
        scene_num_format = f"%0{max(3, math.floor(math.log(len(scene_list), 10)) + 1)}d"
        output_files = [
            output_template.replace("$SCENE_NUMBER", scene_num_format % (i + 1)) for i in range(len(scene_list))
        ]
        scenedetect.split_video_ffmpeg(
            input_video_path=video_path,
            scene_list=scene_list,
            output_file_template=output_template,
            show_progress=self.show_progress,
            arg_override=self.ffmpeg_extra_args,
        )
        return output_files

    def process_single(self, sample, context=False):
        # there is no video in this sample
        if self.video_key not in sample or not sample[self.video_key]:
            sample[Fields.source_file] = []
            return sample

        # load videos
        loaded_videos = sample[self.video_key]
        output_video_keys = {}
        scene_counts = {}

        for video_idx, loaded_video in enumerate(loaded_videos):
            # skip duplicate
            if video_idx in output_video_keys:
                continue

            is_video_path = isinstance(loaded_video, str)
            redirected_video_key = (
                transfer_filename(loaded_video, OP_NAME, self.save_dir, **self._init_parameters)
                if is_video_path
                else os.path.join(self.save_dir, f"{uuid.uuid4().hex}.mp4")
            )
            # detect scenes
            detector = self.detector_class(self.threshold, self.min_scene_len, **self.detector_kwargs)
            try:
                scene_list, output_keys = self._detect_scenes_and_split_video(
                    loaded_video, detector, is_video_path, redirected_video_key
                )
                output_video_keys[video_idx] = output_keys
                scene_counts[video_idx] = len(scene_list)
            except Exception as e:
                # Log or handle the error gracefully
                output_video_keys[video_idx] = [redirected_video_key]
                scene_counts[video_idx] = 0
                logger.error(f"Error processing video {loaded_video}: {e}")

            if self.output_format == "bytes":
                from data_juicer.utils.mm_utils import load_file_byte

                output_video_keys[video_idx] = [load_file_byte(f) for f in output_video_keys[video_idx]]

        # replace split video tokens
        if self.text_key in sample:
            scene_counts_iter = iter([scene_counts[key] for key in range(len(loaded_videos))])
            sample[self.text_key] = re.sub(
                re.escape(SpecialTokens.video),
                lambda match: replace_func(match, scene_counts_iter),
                sample[self.text_key],
            )

        # update source file and save field
        sample[Fields.source_file] = list(
            chain.from_iterable(
                [
                    (
                        [loaded_videos[idx]] * len(output_video_keys[idx])
                        if isinstance(loaded_videos[idx], str)
                        else output_video_keys[idx]
                    )
                    for idx in range(len(loaded_videos))
                ]
            )
        )
        if self.save_field:
            sample[self.save_field] = list(chain.from_iterable(output_video_keys.values()))
        else:
            sample[self.video_key] = list(chain.from_iterable(output_video_keys.values()))

        return sample

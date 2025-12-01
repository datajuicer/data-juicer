import math
import re
from itertools import chain

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

        # prepare detector args
        available_kwargs = self.available_detectors[self.detector]
        self.detector_class = getattr(scenedetect.detectors, self.detector)
        self.detector_kwargs = {key: kwargs[key] for key in available_kwargs if key in kwargs}

    def process_single(self, sample, context=False):
        # there is no video in this sample
        if self.video_key not in sample or not sample[self.video_key]:
            sample[Fields.source_file] = []
            return sample

        # load videos
        loaded_video_keys = sample[self.video_key]
        output_video_keys = {}
        scene_counts = {}

        for video_key in loaded_video_keys:
            # skip duplicate
            if video_key in output_video_keys:
                continue

            redirected_video_key = transfer_filename(video_key, OP_NAME, self.save_dir, **self._init_parameters)
            output_template = add_suffix_to_filename(redirected_video_key, "_$SCENE_NUMBER")

            # detect scenes
            detector = self.detector_class(self.threshold, self.min_scene_len, **self.detector_kwargs)
            scene_list = scenedetect.detect(video_key, detector, show_progress=self.show_progress, start_in_scene=True)
            scene_counts[video_key] = len(scene_list)

            if len(scene_list) > 1:
                # sync with split_video_ffmpeg internal
                scene_num_format = f"%0{max(3, math.floor(math.log(len(scene_list), 10)) + 1)}d"  # noqa: E501
                output_video_keys[video_key] = [
                    output_template.replace("$SCENE_NUMBER", scene_num_format % (i + 1)) for i in range(len(scene_list))
                ]
                # split video into clips
                scenedetect.split_video_ffmpeg(
                    input_video_path=video_key,
                    scene_list=scene_list,
                    output_file_template=output_template,
                    show_progress=self.show_progress,
                )
            else:
                output_video_keys[video_key] = [video_key]

        # replace split video tokens
        if self.text_key in sample:
            scene_counts_iter = iter([scene_counts[key] for key in loaded_video_keys])
            updated_text = re.sub(
                re.escape(SpecialTokens.video),
                lambda match: replace_func(match, scene_counts_iter),
                sample[self.text_key],
            )
            sample[self.text_key] = updated_text

        # when the file is modified, its source file needs to be updated.
        sample[Fields.source_file] = []
        for value in loaded_video_keys:
            sample[Fields.source_file].extend([value] * len(output_video_keys[value]))

        sample[self.video_key] = list(chain.from_iterable([output_video_keys[key] for key in loaded_video_keys]))
        return sample

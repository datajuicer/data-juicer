import math
import os
import sys

from pydantic import PositiveInt

from data_juicer.utils.constant import Fields
from data_juicer.utils.file_utils import transfer_filename
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.logger_utils import HiddenPrints
from data_juicer.utils.mm_utils import close_video, load_video

from ..base_op import OPERATORS, Mapper
from ..op_fusion import LOADED_VIDEOS

with HiddenPrints():
    ffmpeg = LazyLoader("ffmpeg", "ffmpeg-python")

OP_NAME = "video_resize_resolution_mapper"


@OPERATORS.register_module(OP_NAME)
@LOADED_VIDEOS.register_module(OP_NAME)
class VideoResizeResolutionMapper(Mapper):
    """Resizes video resolution based on specified width and height constraints.

    This operator resizes videos to fit within the provided minimum and maximum width and
    height limits. It can optionally maintain the original aspect ratio by adjusting the
    dimensions accordingly. The resized videos are saved in the specified directory or the
    same directory as the input if no save directory is provided. The key metric for
    resizing is the video's width and height, which are adjusted to meet the constraints
    while maintaining the aspect ratio if configured. The `force_divisible_by` parameter
    ensures that the output dimensions are divisible by a specified integer, which must be a
    positive even number when used with aspect ratio adjustments."""

    def __init__(
        self,
        min_width: int = 1,
        max_width: int = sys.maxsize,
        min_height: int = 1,
        max_height: int = sys.maxsize,
        force_original_aspect_ratio: str = "disable",
        force_divisible_by: PositiveInt = 2,
        save_dir: str = None,
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param min_width: Videos with width less than 'min_width' will be
            mapped to videos with equal or bigger width.
        :param max_width: Videos with width more than 'max_width' will be
            mapped to videos with equal of smaller width.
        :param min_height: Videos with height less than 'min_height' will be
            mapped to videos with equal or bigger height.
        :param max_height: Videos with height more than 'max_height' will be
            mapped to videos with equal or smaller height.
        :param force_original_aspect_ratio: Enable decreasing or \
            increasing output video width or height if necessary \
            to keep the original aspect ratio, including ['disable', \
            'decrease', 'increase'].
        :param force_divisible_by: Ensures that both the output dimensions, \
            width and height, are divisible by the given integer when used \
            together with force_original_aspect_ratio, must be a positive \
            even number.
        :param save_dir: The directory where generated video files will be stored.
            If not specified, outputs will be saved in the same directory as their corresponding input files.
            This path can alternatively be defined by setting the `DJ_PRODUCED_DATA_DIR` environment variable.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self._init_parameters = self.remove_extra_parameters(locals())
        self._init_parameters.pop("save_dir", None)

        force_original_aspect_ratio = force_original_aspect_ratio.lower()

        if force_original_aspect_ratio not in ["disable", "decrease", "increase"]:
            raise ValueError(
                f"force_original_aspect_ratio [{force_original_aspect_ratio}]"
                f" is not supported. "
                f"Can only be one of ['disable', 'decrease', 'increase']. "
            )
        if (force_divisible_by <= 1 or force_divisible_by % 2 == 1) and force_original_aspect_ratio != "disable":
            raise ValueError(f"force_divisible_by [{force_divisible_by}] must be a positive" f" even number. ")

        self.min_width = min_width
        self.max_width = max_width
        self.min_height = min_height
        self.max_height = max_height
        self.scale_method = "scale"
        self.force_original_aspect_ratio = force_original_aspect_ratio
        self.force_divisible_by = force_divisible_by
        self.save_dir = save_dir

    def process_single(self, sample, context=False):
        # there is no video in this sample
        if self.video_key not in sample or not sample[self.video_key]:
            sample[Fields.source_file] = []
            return sample

        if Fields.source_file not in sample or not sample[Fields.source_file]:
            sample[Fields.source_file] = sample[self.video_key]

        loaded_video_keys = sample[self.video_key]

        for index, video_key in enumerate(loaded_video_keys):
            container = load_video(video_key)
            video = container.streams.video[0]
            width = video.codec_context.width
            height = video.codec_context.height
            origin_ratio = width / height
            close_video(container)

            if (
                width >= self.min_width
                and width <= self.max_width
                and height >= self.min_height
                and height <= self.max_height
            ):
                continue

            # keep the original aspect ratio as possible
            if width < self.min_width:
                height = self.min_width / origin_ratio
                width = self.min_width
            if width > self.max_width:
                height = self.max_width / origin_ratio
                width = self.max_width
            if height < self.min_height:
                width = self.min_height * origin_ratio
                height = self.min_height
            if height > self.max_height:
                width = self.max_height * origin_ratio
                height = self.max_height

            # the width and height of a video must be divisible by 2.
            if self.force_original_aspect_ratio == "disable":
                force_divisible_by = 2
            else:
                force_divisible_by = self.force_divisible_by

            # make sure in the range if possible
            width = int(max(width, self.min_width))
            width = math.ceil(width / force_divisible_by) * force_divisible_by
            width = int(min(width, self.max_width))
            width = int(width / force_divisible_by) * force_divisible_by
            height = int(max(height, self.min_height))
            height = math.ceil(height / force_divisible_by) * force_divisible_by
            height = int(min(height, self.max_height))
            height = int(height / force_divisible_by) * force_divisible_by

            # keep the origin aspect ratio
            if self.force_original_aspect_ratio == "increase":
                if width / height < origin_ratio:
                    width = height * origin_ratio
                elif width / height > origin_ratio:
                    height = width / origin_ratio
            elif self.force_original_aspect_ratio == "decrease":
                if width / height < origin_ratio:
                    height = width / origin_ratio
                elif width / height > origin_ratio:
                    width = height * origin_ratio
            width = int(round(width / force_divisible_by)) * force_divisible_by
            height = int(round(height / force_divisible_by)) * force_divisible_by

            # resize
            resized_video_key = transfer_filename(video_key, OP_NAME, self.save_dir, **self._init_parameters)
            if not os.path.exists(resized_video_key) or resized_video_key not in loaded_video_keys:
                args = ["-nostdin", "-v", "quiet", "-y"]  # close the ffmpeg log
                stream = ffmpeg.input(video_key)
                stream = stream.filter("scale", width=width, height=height)
                stream = stream.output(resized_video_key).global_args(*args)
                stream.run()

            loaded_video_keys[index] = resized_video_key

        # when the file is modified, its source file needs to be updated.
        for i, value in enumerate(sample[self.video_key]):
            if sample[Fields.source_file][i] != value:
                if loaded_video_keys[i] != value:
                    sample[Fields.source_file][i] = value

        sample[self.video_key] = loaded_video_keys
        return sample

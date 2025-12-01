from fractions import Fraction

import numpy as np

from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.mm_utils import close_video, load_data_with_context, load_video

from ..base_op import OPERATORS, Filter
from ..op_fusion import LOADED_VIDEOS


@OPERATORS.register_module("video_aspect_ratio_filter")
@LOADED_VIDEOS.register_module("video_aspect_ratio_filter")
class VideoAspectRatioFilter(Filter):
    """Filter to keep samples with video aspect ratio within a specific range.

    This operator filters samples based on the aspect ratios of their videos. It keeps
    samples where the video aspect ratios fall within a specified range. The aspect ratio is
    calculated as the width divided by the height (W / H). The operator supports two
    strategies for keeping samples: 'any' and 'all'. In 'any' mode, a sample is kept if at
    least one video meets the aspect ratio condition. In 'all' mode, all videos in the
    sample must meet the condition for the sample to be kept. The aspect ratios are computed
    and stored in the 'video_aspect_ratios' field of the sample's stats."""

    def __init__(self, min_ratio: str = "9/21", max_ratio: str = "21/9", any_or_all: str = "any", *args, **kwargs):
        """
        Initialization method.

        :param min_ratio: The minimum aspect ratio to keep samples,
            supported format is a string, such as "9:21" or "9/21".
        :param max_ratio: The maximum aspect ratio to keep samples,
            supported format is a string, such as "21:9" or "21/9".
        :param any_or_all: keep this sample with 'any' or 'all' strategy of
            all videos. 'any': keep this sample if any videos meet the
            condition. 'all': keep this sample only if all videos meet the
            condition.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.min_ratio = Fraction(str(min_ratio).replace(":", "/"))
        self.max_ratio = Fraction(str(max_ratio).replace(":", "/"))
        if any_or_all not in ["any", "all"]:
            raise ValueError(f"Keep strategy [{any_or_all}] is not supported. " f'Can only be one of ["any", "all"].')
        self.any = any_or_all == "any"

    def compute_stats_single(self, sample, context=False):
        # check if it's computed already
        if StatsKeys.video_aspect_ratios in sample[Fields.stats]:
            return sample

        # there is no video in this sample
        if self.video_key not in sample or not sample[self.video_key]:
            sample[Fields.stats][StatsKeys.video_aspect_ratios] = np.array([], dtype=np.float64)
            return sample

        # load videos
        loaded_video_keys = sample[self.video_key]
        sample, videos = load_data_with_context(sample, context, loaded_video_keys, load_video)

        # compute aspect ratios for each video with W/H
        video_aspect_ratios = {}
        for key, video in videos.items():
            stream = video.streams.video[0]
            video_aspect_ratios[key] = stream.codec_context.width / stream.codec_context.height
            if not context:
                close_video(video)

        sample[Fields.stats][StatsKeys.video_aspect_ratios] = [video_aspect_ratios[key] for key in loaded_video_keys]

        return sample

    def process_single(self, sample):
        video_aspect_ratios = sample[Fields.stats][StatsKeys.video_aspect_ratios]

        keep_bools = np.array(
            [
                self.get_keep_boolean(aspect_ratio, self.min_ratio, self.max_ratio)
                for aspect_ratio in video_aspect_ratios
            ]
        )
        if len(keep_bools) <= 0:
            return True

        # different strategies
        if self.any:
            return keep_bools.any()
        else:
            return keep_bools.all()

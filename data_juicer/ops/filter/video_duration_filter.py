import sys

import numpy as np

from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.mm_utils import close_video, load_data_with_context, load_video

from ..base_op import OPERATORS, Filter
from ..op_fusion import LOADED_VIDEOS

OP_NAME = "video_duration_filter"


@OPERATORS.register_module(OP_NAME)
@LOADED_VIDEOS.register_module(OP_NAME)
class VideoDurationFilter(Filter):
    """Keep data samples whose videos' durations are within a specified range.

    This operator filters data samples based on the duration of their associated videos. It
    keeps samples where the video durations fall within a specified minimum and maximum
    range. The filtering strategy can be set to 'any' or 'all':
    - 'any': Keep the sample if any of its videos meet the duration criteria.
    - 'all': Keep the sample only if all of its videos meet the duration criteria.
    The video durations are computed and stored in the 'video_duration' field of the
    sample's stats. If no videos are present, an empty array is stored."""

    def __init__(
        self, min_duration: float = 0, max_duration: float = sys.maxsize, any_or_all: str = "any", *args, **kwargs
    ):
        """
        Initialization method.

        :param min_duration: The min video duration to keep samples in seconds.
            It's 0 by default.
        :param max_duration: The max video duration to keep samples in seconds.
            It's sys.maxsize by default.
        :param any_or_all: keep this sample with 'any' or 'all' strategy of
            all videos. 'any': keep this sample if any videos meet the
            condition. 'all': keep this sample only if all videos meet the
            condition.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.min_duration = min_duration
        self.max_duration = max_duration
        if any_or_all not in ["any", "all"]:
            raise ValueError(f"Keep strategy [{any_or_all}] is not supported. " f'Can only be one of ["any", "all"].')
        self.any = any_or_all == "any"

    def compute_stats_single(self, sample, context=False):
        # check if it's computed already
        if StatsKeys.video_duration in sample[Fields.stats]:
            return sample

        # there is no video in this sample
        if self.video_key not in sample or not sample[self.video_key]:
            sample[Fields.stats][StatsKeys.video_duration] = np.array([], dtype=np.float64)
            return sample

        # load videos
        loaded_video_keys = sample[self.video_key]
        sample, videos = load_data_with_context(sample, context, loaded_video_keys, load_video)

        video_durations = {}
        for video_key, video in videos.items():
            stream = video.streams.video[0]
            video_durations[video_key] = round(stream.duration * stream.time_base)
            if not context:
                close_video(video)

        # get video durations
        sample[Fields.stats][StatsKeys.video_duration] = [
            video_durations[video_key] for video_key in sample[self.video_key]
        ]

        return sample

    def process_single(self, sample):
        video_durations = sample[Fields.stats][StatsKeys.video_duration]
        keep_bools = np.array(
            [self.get_keep_boolean(duration, self.min_duration, self.max_duration) for duration in video_durations]
        )
        if len(keep_bools) <= 0:
            return True

        # different strategies
        if self.any:
            return keep_bools.any()
        else:
            return keep_bools.all()

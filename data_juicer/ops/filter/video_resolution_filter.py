import sys

import numpy as np

from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.mm_utils import close_video, load_data_with_context, load_video

from ..base_op import OPERATORS, Filter
from ..op_fusion import LOADED_VIDEOS

OP_NAME = "video_resolution_filter"


@OPERATORS.register_module(OP_NAME)
@LOADED_VIDEOS.register_module(OP_NAME)
class VideoResolutionFilter(Filter):
    """Keep data samples whose videos' resolutions are within a specified range.

    This operator filters data samples based on the resolution of the videos they contain.
    It keeps samples if the video resolutions fall within the defined width and height
    ranges. The filtering strategy can be set to 'any' or 'all':
    - 'any': Keeps the sample if any video meets the resolution criteria.
    - 'all': Keeps the sample only if all videos meet the resolution criteria.

    The operator computes and caches the 'video_width' and 'video_height' for each video in
    the sample. If no videos are present, it sets these fields to empty arrays. These cached
    values are used to determine whether to keep or filter out the sample."""

    def __init__(
        self,
        min_width: int = 1,
        max_width: int = sys.maxsize,
        min_height: int = 1,
        max_height: int = sys.maxsize,
        any_or_all: str = "any",
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param min_width: The min horizontal resolution.
        :param max_width: The max horizontal resolution.
        :param min_height: The min vertical resolution.
        :param max_height: The max vertical resolution.
        :param any_or_all: keep this sample with 'any' or 'all' strategy of
            all videos. 'any': keep this sample if any videos meet the
            condition. 'all': keep this sample only if all videos meet the
            condition.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.min_width = min_width
        self.max_width = max_width
        self.min_height = min_height
        self.max_height = max_height
        if any_or_all not in ["any", "all"]:
            raise ValueError(f"Keep strategy [{any_or_all}] is not supported. " f'Can only be one of ["any", "all"].')
        self.any = any_or_all == "any"

    def compute_stats_single(self, sample, context=False):
        # check if it's computed already
        if StatsKeys.video_width in sample[Fields.stats] and StatsKeys.video_height in sample[Fields.stats]:
            return sample

        # there is no video in this sample
        if self.video_key not in sample or not sample[self.video_key]:
            sample[Fields.stats][StatsKeys.video_width] = np.array([], dtype=np.int64)
            sample[Fields.stats][StatsKeys.video_height] = np.array([], dtype=np.int64)
            return sample

        # load videos
        loaded_video_keys = sample[self.video_key]
        sample, videos = load_data_with_context(sample, context, loaded_video_keys, load_video)

        video_width, video_height = dict(), dict()
        for video_key, video in videos.items():
            # default to load the first stream
            video_stream = video.streams.video[0]

            # fail in loading video
            if video_stream is None:
                return sample

            video_width[video_key] = video_stream.codec_context.width
            video_height[video_key] = video_stream.codec_context.height

        # get video resolutions
        sample[Fields.stats][StatsKeys.video_width] = [video_width[video_key] for video_key in sample[self.video_key]]
        sample[Fields.stats][StatsKeys.video_height] = [video_height[video_key] for video_key in sample[self.video_key]]

        if not context:
            for vid_key in videos:
                close_video(videos[vid_key])

        return sample

    def process_single(self, sample):
        ws = sample[Fields.stats][StatsKeys.video_width]
        hs = sample[Fields.stats][StatsKeys.video_height]
        keep_bools = np.array(
            [
                self.get_keep_boolean(w, self.min_width, self.max_width)
                and self.get_keep_boolean(h, self.min_height, self.max_height)
                for w, h in zip(ws, hs)
            ]
        )
        if len(keep_bools) <= 0:
            return True

        # different strategies
        if self.any:
            return keep_bools.any()
        else:
            return keep_bools.all()

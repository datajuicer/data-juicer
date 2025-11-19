import sys
from contextlib import contextmanager
from typing import Optional, Tuple, Union

import numpy as np
from pydantic import PositiveFloat, PositiveInt

from data_juicer.utils.constant import Fields, MetaKeys, StatsKeys
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.mm_utils import calculate_resized_dimensions

from ..base_op import OPERATORS, UNFORKABLE, Filter

cv2 = LazyLoader("cv2", "opencv-python")

OP_NAME = "video_motion_score_filter"


@contextmanager
def VideoCapture(*args, **kwargs):
    cap = cv2.VideoCapture(*args, **kwargs)
    try:
        yield cap
    finally:
        cap.release()


@UNFORKABLE.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
class VideoMotionScoreFilter(Filter):
    """Filter to keep samples with video motion scores within a specific range.

    The operator uses Farneback's algorithm from OpenCV to compute dense optical flow. It
    calculates the average motion score for each video and retains samples based on the
    specified minimum and maximum score thresholds. The 'any' or 'all' strategy determines
    whether to keep a sample if any or all videos meet the criteria. The motion score is
    computed as the mean magnitude of the optical flow, which can be normalized relative to
    the frame's diagonal length. The stats are cached under the key 'video_motion_score'."""

    _default_kwargs = {
        "pyr_scale": 0.5,
        "levels": 3,
        "winsize": 15,
        "iterations": 3,
        "poly_n": 5,
        "poly_sigma": 1.2,
        "flags": 0,
    }

    def __init__(
        self,
        min_score: float = 0.25,
        max_score: float = sys.float_info.max,
        sampling_fps: PositiveFloat = 2,
        size: Union[PositiveInt, Tuple[PositiveInt], Tuple[PositiveInt, PositiveInt], None] = None,
        max_size: Optional[PositiveInt] = None,
        divisible: PositiveInt = 1,
        relative: bool = False,
        any_or_all: str = "any",
        if_output_optical_flow: bool = False,
        optical_flow_key: str = MetaKeys.video_optical_flow,
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param min_score: The minimum motion score to keep samples.
        :param max_score: The maximum motion score to keep samples.
        :param sampling_fps: The sampling rate in frames_per_second for
            optical flow calculations.
        :param size: Resize frames before computing optical flow. If size is a
            sequence like (h, w), frame size will be matched to this. If size
            is an int, smaller edge of frames will be matched to this number.
            i.e, if height > width, then frame will be rescaled to (size *
            height / width, size). Default `None` to keep the original size.
        :param max_size: The maximum allowed for the longer edge of resized
            frames. If the longer edge of frames is greater than max_size after
            being resized according to size, size will be overruled so that the
            longer edge is equal to max_size. As a result, the smaller edge may
            be shorter than size. This is only supported if size is an int.
        :param divisible: The number that the dimensions must be divisible by.
        :param relative: If `True`, the optical flow magnitude is normalized to
            a [0, 1] range, relative to the frame's diagonal length.
        :param any_or_all: keep this sample with 'any' or 'all' strategy of
            all videos. 'any': keep this sample if any videos meet the
            condition. 'all': keep this sample only if all videos meet the
            condition.
        :param if_output_camera_parameters: Determines whether to output
            the computed optical flows into the metas. The optical flows for each
            video will be stored in the shape of (num_frame, H, W, 2)
        :param optical_flow_key: The field name to store the optical flows. It's
            "video_optical_flow" in default.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.min_score = min_score
        self.max_score = max_score
        self.sampling_fps = sampling_fps

        if isinstance(size, (list, tuple)):
            if len(size) not in [1, 2]:
                raise ValueError(
                    f"Size must be an int or a 1 or 2 element tuple/list," f"not a {len(size)} element tuple/list."
                )
        if isinstance(size, int):
            size = (size,)
        self.size = size
        self.max_size = max_size
        self.divisible = divisible
        self.relative = relative

        self.extra_kwargs = self._default_kwargs
        for key in kwargs:
            if key in self.extra_kwargs:
                self.extra_kwargs[key] = kwargs[key]

        if any_or_all not in ["any", "all"]:
            raise ValueError(f"Keep strategy [{any_or_all}] is not supported. " f'Can only be one of ["any", "all"].')
        self.any = any_or_all == "any"

        self.if_output_optical_flow = if_output_optical_flow
        self.optical_flow_key = optical_flow_key

    def setup_model(self, rank=None):
        self.model = cv2.calcOpticalFlowFarneback

    def compute_flow(self, prev_frame, curr_frame):
        curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        if prev_frame is None:
            flow = None
        else:
            flow = self.model(prev_frame, curr_frame, None, **self.extra_kwargs)
        return flow, curr_frame

    def compute_stats_single(self, sample, rank=None, context=False):
        self.rank = rank

        # check if it's computed already
        if StatsKeys.video_motion_score in sample[Fields.stats]:
            return sample

        # there is no video in this sample
        if self.video_key not in sample or not sample[self.video_key]:
            sample[Fields.stats][StatsKeys.video_motion_score] = np.array([], dtype=np.float64)
            return sample

        self.setup_model(rank)

        # load videos
        loaded_video_keys = sample[self.video_key]
        unique_motion_scores = {}
        video_optical_flows = {}
        for video_key in loaded_video_keys:
            # skip duplicate videos
            if video_key in unique_motion_scores:
                continue

            video_motion_scores = []
            optical_flows = []
            with VideoCapture(video_key) as cap:
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    sampling_fps = min(self.sampling_fps, fps)
                    sampling_step = round(fps / sampling_fps)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    # at least two frames for computing optical flow
                    sampling_step = max(min(sampling_step, total_frames - 1), 1)
                    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    new_size = calculate_resized_dimensions((height, width), self.size, self.max_size, self.divisible)

                prev_frame = None
                frame_count = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        # If the frame can't be read, it could be due to
                        # a corrupt frame or reaching the end of the video.
                        break

                    if new_size != (height, width):
                        frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)

                    # return flow of shape (H, W, 2) and transformed frame
                    # of shape (H, W, 3) in BGR mode
                    flow, prev_frame = self.compute_flow(prev_frame, frame)
                    if flow is None:
                        continue
                    optical_flows.append(flow)
                    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    frame_motion_score = np.mean(mag)
                    if self.relative:
                        frame_motion_score /= np.hypot(*frame.shape[:2])
                    video_motion_scores.append(frame_motion_score)

                    # quickly skip frames
                    frame_count += sampling_step
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)

            # may due to frame corruption
            if not video_motion_scores:
                unique_motion_scores[video_key] = -1
            else:
                unique_motion_scores[video_key] = np.mean(video_motion_scores or [-1])

            video_optical_flows[video_key] = np.stack(optical_flows)

        sample[Fields.stats][StatsKeys.video_motion_score] = [unique_motion_scores[key] for key in loaded_video_keys]
        if self.if_output_optical_flow:
            sample[Fields.meta][self.optical_flow_key] = [video_optical_flows[key] for key in loaded_video_keys]
        return sample

    def process_single(self, sample):
        video_motion_scores = sample[Fields.stats][StatsKeys.video_motion_score]

        keep_bools = np.array(
            [
                self.get_keep_boolean(motion_score, self.min_score, self.max_score)
                for motion_score in video_motion_scores
            ]
        )
        if len(keep_bools) <= 0:
            return True

        # different strategies
        if self.any:
            return keep_bools.any()
        else:
            return keep_bools.all()

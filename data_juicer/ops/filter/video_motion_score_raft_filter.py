import sys
from typing import Optional, Tuple, Union

from pydantic import PositiveFloat, PositiveInt

from data_juicer.ops.filter.video_motion_score_filter import VideoMotionScoreFilter
from data_juicer.utils.constant import MetaKeys
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.resource_utils import cuda_device_count

from ..base_op import OPERATORS, UNFORKABLE

torch = LazyLoader("torch")
tvm = LazyLoader("torchvision.models")
tvt = LazyLoader("torchvision.transforms")

OP_NAME = "video_motion_score_raft_filter"


@UNFORKABLE.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
class VideoMotionScoreRaftFilter(VideoMotionScoreFilter):
    """Filter to keep samples with video motion scores within a specified range.

    This operator utilizes the RAFT (Recurrent All-Pairs Field Transforms) model from
    torchvision to predict optical flow between video frames. It keeps samples where the
    video motion score is within the given min and max score range. The motion score is
    computed based on the optical flow between frames, which is estimated using the RAFT
    model. The operator can sample frames at a specified FPS and apply transformations to
    the frames before computing the flow.

    - The RAFT model is used to estimate the optical flow.
    - Frames are preprocessed using a series of transformations including normalization and
      color channel flipping.
    - The motion score is calculated from the optical flow data.
    - The operator can be configured to filter based on any or all frames in the video.
    - The device for model inference (CPU or CUDA) is automatically detected and set.

    For further details, refer to the official torchvision documentation:
    https://pytorch.org/vision/main/models/raft.html

    The original paper on RAFT is available here:
    https://arxiv.org/abs/2003.12039
    """

    _accelerator = "cuda"
    _default_kwargs = {}

    def __init__(
        self,
        min_score: float = 1.0,
        max_score: float = sys.float_info.max,
        sampling_fps: PositiveFloat = 2,
        size: Union[PositiveInt, Tuple[PositiveInt], Tuple[PositiveInt, PositiveInt], None] = None,
        max_size: Optional[PositiveInt] = None,
        divisible: PositiveInt = 8,
        relative: bool = False,
        any_or_all: str = "any",
        if_output_optical_flow: bool = False,
        optical_flow_key: str = MetaKeys.video_optical_flow,
        *args,
        **kwargs,
    ):
        super().__init__(
            min_score,
            max_score,
            sampling_fps,
            size,
            max_size,
            divisible,
            relative,
            any_or_all,
            if_output_optical_flow,
            optical_flow_key,
            *args,
            **kwargs,
        )

    def setup_model(self, rank=None):
        self.model = tvm.optical_flow.raft_large(weights=tvm.optical_flow.Raft_Large_Weights.DEFAULT, progress=False)
        if self.use_cuda():
            rank = rank if rank is not None else 0
            rank = rank % cuda_device_count()
            self.device = f"cuda:{rank}"
        else:
            self.device = "cpu"
        self.model.to(self.device)
        self.model.eval()

        self.transforms = tvt.Compose(
            [
                tvt.ToTensor(),
                tvt.Normalize(mean=0.5, std=0.5),  # map [0, 1] into [-1, 1]
                tvt.Lambda(lambda img: img.flip(-3).unsqueeze(0)),  # BGR to RGB
            ]
        )

    def compute_flow(self, prev_frame, curr_frame):
        curr_frame = self.transforms(curr_frame).to(self.device)
        if prev_frame is None:
            flow = None
        else:
            with torch.inference_mode():
                flows = self.model(prev_frame, curr_frame)
            flow = flows[-1][0].cpu().numpy().transpose((1, 2, 0))  # 2, H, W -> H, W, 2
        return flow, curr_frame

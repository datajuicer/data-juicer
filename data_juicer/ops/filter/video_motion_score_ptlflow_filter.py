import sys
from typing import Optional, Tuple, Union

from jsonargparse import dict_to_namespace
from pydantic import PositiveFloat, PositiveInt

from data_juicer.ops.filter.video_motion_score_filter import VideoMotionScoreFilter
from data_juicer.utils.constant import MetaKeys
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.resource_utils import cuda_device_count

from ..base_op import OPERATORS, UNFORKABLE

torch = LazyLoader("torch")
tvm = LazyLoader("torchvision.models")
tvt = LazyLoader("torchvision.transforms")
ptlflow = LazyLoader("ptlflow")
ptlflow_io_adapter = LazyLoader("ptlflow.utils.io_adapter")

OP_NAME = "video_motion_score_ptlflow_filter"


@UNFORKABLE.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
class VideoMotionScorePtlflowFilter(VideoMotionScoreFilter):
    """Filter to keep samples with video motion scores within a specified range.

    This operator utilizes the ptlflow library (https://github.com/hmorimitsu/ptlflow) to
    predict optical flow between video frames. It keeps samples where the
    video motion score is within the given min and max score range. The motion score is
    computed based on the optical flow between frames, which is estimated using the models
    supported in ptlflow. The operator can sample frames at a specified FPS and apply
    transformations to the frames before computing the flow.

    - The models in ptlflow is used to estimate the optical flow.
    - Frames are preprocessed using a series of transformations including normalization and
      color channel flipping.
    - The motion score is calculated from the optical flow data.
    - The operator can be configured to filter based on any or all frames in the video.
    - The device for model inference (CPU or CUDA) is automatically detected and set.

    For further details, refer to the official documentation:
    https://ptlflow.readthedocs.io/
    """

    _accelerator = "cuda"
    _default_kwargs = {}

    def __init__(
        self,
        min_score: float = 1.0,
        max_score: float = sys.float_info.max,
        model_name: str = "dpflow",
        ckpt_path: Optional[str] = None,
        get_model_args: Optional[dict] = None,
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

        self.model_name = model_name
        self.ckpt_path = ckpt_path
        if get_model_args is not None:
            get_model_args = dict_to_namespace(get_model_args)
        self.get_model_args = get_model_args

    def setup_model(self, rank=None):
        self.model = ptlflow.get_model(self.model_name, ckpt_path=self.ckpt_path, args=self.get_model_args)
        if self.use_cuda():
            rank = rank if rank is not None else 0
            rank = rank % cuda_device_count()
            self.device = f"cuda:{rank}"
        else:
            self.device = "cpu"
        self.model.to(self.device)
        self.model.eval()

    def compute_flow(self, prev_frame, curr_frame):
        if prev_frame is None:
            flow = None
        else:
            io_adapter = ptlflow_io_adapter.IOAdapter(self.model, prev_frame.shape[:2])
            frames = [prev_frame, curr_frame]
            inputs = io_adapter.prepare_inputs(frames)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            predictions = self.model(inputs)
            flows = predictions.get("flows")  # shape: (1, 1, 2, H, W)
            flow = flows[-1][0].cpu().detach().numpy().transpose((1, 2, 0))  # 2, H, W -> H, W, 2
        return flow, curr_frame

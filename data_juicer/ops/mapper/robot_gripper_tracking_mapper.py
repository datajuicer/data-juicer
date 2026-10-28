import os
from pathlib import Path

from data_juicer.utils.robot_utils import episode_root, perception_settings

from ..base_op import OPERATORS, UNFORKABLE, Mapper

OP_NAME = "robot_gripper_tracking_mapper"


@OPERATORS.register_module(OP_NAME)
@UNFORKABLE.register_module(OP_NAME)
class RobotGripperTrackingMapper(Mapper):
    """Segment the robot gripper and propagate its mask across all frames."""

    _accelerator = "cuda"

    def __init__(
        self,
        model: str = None,
        endpoint: str = None,
        sam3_model: str = None,
        output_dir: str = "output/episodes",
        *args,
        **kwargs,
    ):
        """Initialize the mapper.

        :param model: Served VLM model name. Defaults to VLM_MODEL.
        :param endpoint: OpenAI-compatible VLM endpoint. Defaults to VLM_URL.
        :param sam3_model: SAM3 checkpoint directory or model ID. Defaults to SAM3_MODEL or facebook/sam3.
        :param output_dir: Directory for episode artifacts.
        :param args: Additional positional arguments for Mapper.
        :param kwargs: Additional keyword arguments for Mapper.
        """
        super().__init__(*args, **kwargs)
        self.output_dir = output_dir
        (self.model, self.endpoint) = (model, endpoint)
        self.sam3_model = sam3_model or os.environ.get("SAM3_MODEL", "facebook/sam3")
        self._context = None

    def process_single(self, sample, rank=None):
        from data_juicer.utils import robot_perception_utils as perception

        root = episode_root(sample, self.output_dir)
        (model, endpoint, device) = perception_settings(self.model, self.endpoint, self.use_cuda(), rank)
        args = perception.gripper_tracking_config()
        args.video = root / "operator_input" / (sample["episode_id"] + ".mp4")
        args.parquet = Path(sample["metadata_path"])
        args.output_dir = root / "gripper_operator"
        (args.vlm_model, args.vllm_base_url, args.device) = (model, endpoint, device)
        args.vllm_api_key = os.environ.get("VLM_API_KEY", args.vllm_api_key)
        args.sam3_text_model = args.sam3_video_model = self.sam3_model
        if self._context is None:
            self._context = perception.GripperTrackingContext(args)
        result = perception.track_episode_gripper(args, self._context)
        sample["gripper_output"] = str(result["output_json"])
        return sample

import os
from pathlib import Path

from data_juicer.utils.robot_utils import episode_root, perception_settings, write_json

from ..base_op import OPERATORS, UNFORKABLE, Mapper

OP_NAME = "robot_object_tracking_mapper"


@OPERATORS.register_module(OP_NAME)
@UNFORKABLE.register_module(OP_NAME)
class RobotObjectTrackingMapper(Mapper):
    """Localize the instructed object with VLM/SAM3 and track the full episode."""

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
        from data_juicer.utils.robot_utils import decode_camera_views

        root = episode_root(sample, self.output_dir)
        (model, endpoint, device) = perception_settings(self.model, self.endpoint, self.use_cuda(), rank)
        views = decode_camera_views(Path(sample["raw_dir"]), root / "frames")
        primary = sample["primary_view"]
        if primary not in views:
            raise ValueError("primary_view is absent from the episode")
        write_json(root / "ir/sampling.json", {"operator": "full_episode_decode", "views": views})
        inputs = root / "operator_input"
        inputs.mkdir(exist_ok=True)
        parquet = inputs / (sample["episode_id"] + ".parquet")
        for link, target in [
            (parquet, Path(sample["metadata_path"])),
            (parquet.with_suffix(".mp4"), Path(views[primary]["sampled_video"])),
        ]:
            if link.is_symlink():
                link.unlink()
            if link.exists():
                raise FileExistsError(link)
            link.symlink_to(target.resolve())
        args = perception.object_tracking_config()
        args.output_root = root / "object_operator"
        (args.vlm_model, args.vllm_base_url, args.device) = (model, endpoint, device)
        args.vllm_api_key = os.environ.get("VLM_API_KEY", args.vllm_api_key)
        args.sam3_model = args.sam3_video_model = self.sam3_model
        args.run_tracking = True
        if self._context is None:
            self._context = perception.ObjectTrackingContext(args)
        perception.track_episode_object(args, self._context, parquet)
        folder = args.output_root / sample["episode_id"]
        sample["object_outputs"] = [
            str(folder / "video_object_detection_new.json"),
            str(folder / "video_object_tracking.json"),
        ]
        return sample

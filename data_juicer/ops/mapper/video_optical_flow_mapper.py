import os

from pydantic import PositiveInt

from data_juicer.ops.load import load_ops
from data_juicer.utils.cache_utils import DATA_JUICER_ASSETS_CACHE
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.model_utils import get_model, prepare_model
from data_juicer.utils.video_perception_utils import (
    get_video_frame_groups,
    get_video_visualization_dir,
)

from ..base_op import OPERATORS, TAGGING_OPS, UNFORKABLE, Mapper
from ..op_fusion import LOADED_VIDEOS

OP_NAME = "video_optical_flow_mapper"
cv2 = LazyLoader("cv2", package_name="opencv-python")

torch = LazyLoader("torch")
torchvision = LazyLoader("torchvision")


@TAGGING_OPS.register_module(OP_NAME)
@UNFORKABLE.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
@LOADED_VIDEOS.register_module(OP_NAME)
class VideoOpticalFlowMapper(Mapper):
    """Generate optical flow information for videos."""

    _accelerator = "cuda"

    def __init__(
        self,
        if_save_visualization: bool = True,
        save_visualization_dir: str = DATA_JUICER_ASSETS_CACHE,
        frame_num: PositiveInt = 3,
        duration: float = 0,
        frame_dir: str = DATA_JUICER_ASSETS_CACHE,
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param if_save_visualization: Whether to save visualization results.
        :param save_visualization_dir: The path for saving visualization results.
        :param frame_num: The number of frames to be extracted uniformly from
            the video. If it's 1, only the middle frame will be extracted. If
            it's 2, only the first and the last frames will be extracted. If
            it's larger than 2, in addition to the first and the last frames,
            other frames will be extracted uniformly within the video duration.
            If "duration" > 0, frame_num is the number of frames per segment.
        :param duration: The duration of each segment in seconds.
            If 0, frames are extracted from the entire video.
            If duration > 0, the video is segmented into multiple segments
            based on duration, and frames are extracted from each segment.
        :param frame_dir: Output directory to save extracted frames.

        """

        super().__init__(*args, **kwargs)
        LazyLoader.check_packages(["torchvision"])

        self.model_key = prepare_model(model_type="optical_flow_raft")
        self.if_save_visualization = if_save_visualization
        self.save_visualization_dir = save_visualization_dir or DATA_JUICER_ASSETS_CACHE
        self.frame_field = MetaKeys.video_frames
        self.tag_field_name = MetaKeys.video_optical_flow_tags
        self.frame_num = frame_num
        self.duration = duration
        self.frame_dir = frame_dir or DATA_JUICER_ASSETS_CACHE

        self.video_extract_frames_mapper_args = {
            "frame_sampling_method": "uniform",
            "frame_num": frame_num,
            "duration": duration,
            "frame_dir": self.frame_dir,
            "frame_field": self.frame_field,
            "video_key": self.video_key,
            "legacy_split_by_text_token": False,
            "num_proc": None,  # Disable multiprocessing to avoid nested process pool issue
            "auto_op_parallelism": False,  # Disable auto parallelism to avoid nested process pool issue
        }
        self.fused_ops = load_ops([{"video_extract_frames_mapper": self.video_extract_frames_mapper_args}])

    def raft_preprocess(self, img1_batch, img2_batch, transforms):
        img1_batch = torchvision.transforms.functional.resize(img1_batch, size=[520, 960], antialias=False)
        img2_batch = torchvision.transforms.functional.resize(img2_batch, size=[520, 960], antialias=False)
        return transforms(img1_batch, img2_batch)

    def process_single(self, sample=None, rank=None):
        metadata = sample.setdefault(Fields.meta, {})
        if self.tag_field_name in metadata:
            return sample
        groups = get_video_frame_groups(sample, self.frame_field, self.video_key, self.fused_ops[0])
        result = {"pred_flow": [], "frame_pairs": [], "video_indices": []}
        if not any(len(paths) > 1 for paths in groups):
            metadata[self.tag_field_name] = result
            return sample
        model, transforms = get_model(model_key=self.model_key, rank=rank, use_cuda=self.use_cuda())
        device = next(model.parameters()).device
        for video_index, paths in enumerate(groups):
            previous = None
            for frame_index, path in enumerate(paths):
                image = cv2.imread(path)
                if image is None:
                    raise ValueError(f"Could not read video frame: {path}")
                current = torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).unsqueeze(0)
                if previous is not None:
                    first, second = self.raft_preprocess(previous.to(device), current.to(device), transforms)
                    with torch.no_grad():
                        predicted_flow = model(first, second)[-1]
                    result["pred_flow"].append(predicted_flow[0].detach().cpu().tolist())
                    result["frame_pairs"].append([paths[frame_index - 1], path])
                    result["video_indices"].append(video_index)
                    if self.if_save_visualization:
                        directory = get_video_visualization_dir(
                            self.save_visualization_dir, OP_NAME, paths, video_index
                        )
                        visualization = (
                            torchvision.utils.flow_to_image(predicted_flow)[0].cpu().permute(1, 2, 0).numpy()
                        )
                        cv2.imwrite(
                            os.path.join(directory, f"vis_{frame_index - 1}.jpg"),
                            cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR),
                        )
                previous = current
        metadata[self.tag_field_name] = result
        return sample

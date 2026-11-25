import os

from pydantic import PositiveInt

from data_juicer.ops.load import load_ops
from data_juicer.utils.cache_utils import DATA_JUICER_ASSETS_CACHE
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.model_utils import get_model, prepare_model
from data_juicer.utils.video_perception_utils import (
    get_video_frame_groups,
    get_video_output_features,
    get_video_visualization_dir,
    nested_video_feature,
)

from ..base_op import OPERATORS, Mapper
from ..op_fusion import LOADED_VIDEOS

OP_NAME = "video_whole_body_pose_estimation_mapper"

cv2 = LazyLoader("cv2", "opencv-contrib-python")


@OPERATORS.register_module(OP_NAME)
@LOADED_VIDEOS.register_module(OP_NAME)
class VideoWholeBodyPoseEstimationMapper(Mapper):
    """Input a video containing people, and use the DWPose
    model to extract the body, hand, feet, and face keypoints
    of the human subjects in the video, i.e., 2D Whole-body
    Pose Estimation."""

    _accelerator = "cuda"

    def __init__(
        self,
        onnx_det_model: str = "yolox_l.onnx",
        onnx_pose_model: str = "dw-ll_ucoco_384.onnx",
        frame_num: PositiveInt = 3,
        duration: float = 0,
        tag_field_name: str = MetaKeys.pose_estimation_tags,
        frame_dir: str = DATA_JUICER_ASSETS_CACHE,
        if_save_visualization: bool = False,
        save_visualization_dir: str = DATA_JUICER_ASSETS_CACHE,
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param onnx_det_model: The path to 'yolox_l.onnx'.
        :param onnx_pose_model: The path to 'dw-ll_ucoco_384.onnx'.
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
        :param tag_field_name: The field name to store the tags. It's
            "pose_estimation_tags" in default.
        :param frame_dir: Output directory to save extracted frames.
        :param if_save_visualization: Whether to save visualization results.
        :param save_visualization_dir: The path for saving visualization results.
        :param args: extra args
        :param kwargs: extra args

        """

        super().__init__(*args, **kwargs)

        self.video_extract_frames_mapper_args = {
            "frame_sampling_method": "uniform",
            "frame_num": frame_num,
            "duration": duration,
            "frame_dir": frame_dir,
            "frame_field": MetaKeys.video_frames,
            "video_key": self.video_key,
            "legacy_split_by_text_token": False,
            "num_proc": None,  # Disable multiprocessing to avoid nested process pool issue
            "auto_op_parallelism": False,  # Disable auto parallelism to avoid nested process pool issue
        }
        self.fused_ops = load_ops([{"video_extract_frames_mapper": self.video_extract_frames_mapper_args}])

        self.model_key = prepare_model(
            model_type="dwpose", onnx_det_model=onnx_det_model, onnx_pose_model=onnx_pose_model
        )

        self.frame_num = frame_num
        self.frame_field = MetaKeys.video_frames
        self.duration = duration
        self.tag_field_name = tag_field_name
        self.frame_dir = frame_dir
        self.if_save_visualization = if_save_visualization
        self.save_visualization_dir = save_visualization_dir

    def _get_output_features(self, input_features):
        if input_features is None:
            return None
        fields = {
            name: nested_video_feature("float64", 4)
            for name in ["body_keypoints", "foot_keypoints", "faces_keypoints", "hands_keypoints"]
        }
        fields.update(
            bbox_results_list=nested_video_feature("float64", 3),
            frame_paths=nested_video_feature("string"),
            video_indices=nested_video_feature("int64"),
        )
        return get_video_output_features(input_features, self.tag_field_name, fields)

    def process_single(self, sample=None, rank=None):
        meta = sample.setdefault(Fields.meta, {})
        if self.tag_field_name in meta:
            return sample
        groups = get_video_frame_groups(sample, self.frame_field, self.video_key, self.fused_ops[0])
        names = ["body_keypoints", "foot_keypoints", "faces_keypoints", "hands_keypoints", "bbox_results_list"]
        tags = {name: [] for name in names}
        tags.update(frame_paths=[], video_indices=[])
        if not any(groups):
            meta[self.tag_field_name] = tags
            return sample

        dwpose_model = get_model(self.model_key, rank, self.use_cuda())
        for video_index, paths in enumerate(groups):
            if not paths:
                continue
            if self.if_save_visualization:
                visualization_dir = get_video_visualization_dir(
                    self.save_visualization_dir, OP_NAME, paths, video_index
                )
            for frame_index, path in enumerate(paths):
                image = cv2.imread(path)
                if image is None:
                    raise ValueError(f"Could not read video frame: {path}")
                results = dwpose_model(image)
                for name, values in zip(names, results[:5]):
                    tags[name].append(values.tolist() if hasattr(values, "tolist") else values)
                tags["frame_paths"].append(path)
                tags["video_indices"].append(video_index)
                if self.if_save_visualization:
                    plt = LazyLoader("matplotlib.pyplot")
                    plt.imsave(os.path.join(visualization_dir, f"temp_frame_pose_{frame_index}.jpg"), results[5])
        meta[self.tag_field_name] = tags
        return sample

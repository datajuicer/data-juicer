import os

import matplotlib.pyplot as plt
from pydantic import PositiveInt

import data_juicer
from data_juicer.ops.load import load_ops
from data_juicer.utils.cache_utils import DATA_JUICER_ASSETS_CACHE
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.mm_utils import SpecialTokens
from data_juicer.utils.model_utils import get_model, prepare_model

from ..base_op import OPERATORS, Mapper
from ..op_fusion import LOADED_VIDEOS

OP_NAME = "video_whole_body_pose_estimation_mapper"

cv2 = LazyLoader("cv2", "opencv-python")


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
            "frame_key": MetaKeys.video_frames,
        }
        self.fused_ops = load_ops([{"video_extract_frames_mapper": self.video_extract_frames_mapper_args}])

        self.model_key = prepare_model(
            model_type="dwpose", onnx_det_model=onnx_det_model, onnx_pose_model=onnx_pose_model
        )

        self.frame_num = frame_num
        self.duration = duration
        self.tag_field_name = tag_field_name
        self.frame_dir = frame_dir
        self.if_save_visualization = if_save_visualization
        self.save_visualization_dir = save_visualization_dir

    def process_single(self, sample=None, rank=None):

        # check if it's generated already
        if self.tag_field_name in sample[Fields.meta]:
            return sample

        # there is no video in this sample
        if self.video_key not in sample or not sample[self.video_key]:
            return []

        # load videos
        ds_list = [{"text": SpecialTokens.video, "videos": sample[self.video_key]}]

        dataset = data_juicer.core.data.NestedDataset.from_list(ds_list)
        dataset = self.fused_ops[0].run(dataset)

        dwpose_model = get_model(self.model_key, rank, self.use_cuda())

        frames_root = os.path.join(self.frame_dir, os.path.splitext(os.path.basename(sample[self.video_key][0]))[0])
        frame_names = os.listdir(frames_root)
        frames_path = sorted([os.path.join(frames_root, frame_name) for frame_name in frame_names])

        body_keypoints = []
        foot_keypoints = []
        faces_keypoints = []
        hands_keypoints = []
        bbox_results_list = []

        if self.if_save_visualization:
            if not os.path.exists(self.save_visualization_dir):
                os.makedirs(self.save_visualization_dir, exist_ok=True)
            frame_dir_for_temp_video = os.path.join(self.save_visualization_dir, frames_root + "_pose_output")
            if not os.path.exists(frame_dir_for_temp_video):
                os.makedirs(frame_dir_for_temp_video, exist_ok=True)

        for temp_frame_id, temp_frame in enumerate(frames_path):
            oriImg = cv2.imread(temp_frame)
            body, foot, faces, hands, bbox_results, draw_pose = dwpose_model(oriImg)
            body_keypoints.append(body)
            foot_keypoints.append(foot)
            faces_keypoints.append(faces)
            hands_keypoints.append(hands)
            bbox_results_list.append(bbox_results)

            if self.if_save_visualization:
                plt.imsave(
                    os.path.join(frame_dir_for_temp_video, f"temp_frame_pose_{str(temp_frame_id)}.jpg"), draw_pose
                )

        sample[Fields.meta][self.tag_field_name] = {}
        sample[Fields.meta][self.tag_field_name]["body_keypoints"] = body_keypoints
        sample[Fields.meta][self.tag_field_name]["foot_keypoints"] = foot_keypoints
        sample[Fields.meta][self.tag_field_name]["faces_keypoints"] = faces_keypoints
        sample[Fields.meta][self.tag_field_name]["hands_keypoints"] = hands_keypoints
        sample[Fields.meta][self.tag_field_name]["bbox_results_list"] = bbox_results_list

        return sample

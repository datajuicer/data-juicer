import json
import os

import numpy as np
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

OP_NAME = "video_camera_calibration_static_moge_mapper"

cv2 = LazyLoader("cv2", "opencv-python")
torch = LazyLoader("torch")


@OPERATORS.register_module(OP_NAME)
@LOADED_VIDEOS.register_module(OP_NAME)
class VideoCameraCalibrationStaticMogeMapper(Mapper):
    """Compute the camera intrinsics and field of view (FOV)
    for a static camera using Moge-2 (more accurate
    than DeepCalib)."""

    _accelerator = "cuda"

    def __init__(
        self,
        model_path: str = "Ruicheng/moge-2-vitl",
        frame_num: PositiveInt = 3,
        duration: float = 0,
        tag_field_name: str = MetaKeys.static_camera_calibration_moge_tags,
        frame_dir: str = DATA_JUICER_ASSETS_CACHE,
        if_output_info: bool = True,
        output_info_dir: str = DATA_JUICER_ASSETS_CACHE,
        if_output_points_info: bool = True,
        if_output_depth_info: bool = True,
        if_output_mask_info: bool = True,
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param model_path: The path to the Moge-2 model.
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
            "static_camera_calibration_moge_tags" in default.
        :param frame_dir: Output directory to save extracted frames.
        :param if_output_info: Whether to save the camera parameters results
            to an JSON file.
        :param output_info_dir: Output directory for saving camera parameters.
        :param if_output_points_info: Determines whether to output point map
            in OpenCV camera coordinate system (x right, y down, z forward).
            For MoGe-2, the point map is in metric scale.
        :param if_output_depth_info: Determines whether to output
            depth maps.
        :param if_output_mask_info: Determines whether to output a
            binary mask for valid pixels.
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
        self.model_key = prepare_model(model_type="moge", model_path=model_path)

        self.frame_num = frame_num
        self.duration = duration
        self.tag_field_name = tag_field_name
        self.frame_dir = frame_dir
        self.output_info_dir = output_info_dir
        self.if_output_points_info = if_output_points_info
        self.if_output_depth_info = if_output_depth_info
        self.if_output_mask_info = if_output_mask_info
        self.if_output_info = if_output_info

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

        frames_root = os.path.join(self.frame_dir, os.path.splitext(os.path.basename(sample[self.video_key][0]))[0])
        frame_names = os.listdir(frames_root)
        frames_path = sorted([os.path.join(frames_root, frame_name) for frame_name in frame_names])
        model = get_model(self.model_key, rank, self.use_cuda())

        final_k_list = []
        final_hfov_list = []
        final_vfov_list = []
        final_points_list = []
        final_depth_list = []
        final_mask_list = []

        if rank is not None:
            device = f"cuda:{rank}" if self.use_cuda() else "cpu"
        else:
            device = "cuda" if self.use_cuda() else "cpu"

        for i, path in enumerate(frames_path):

            input_image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            height, width, channels = input_image.shape
            input_image = torch.tensor(input_image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)

            output = model.infer(input_image)

            points = output["points"].cpu().tolist()
            depth = output["depth"].cpu().tolist()
            mask = output["mask"].cpu().tolist()
            intrinsics = output["intrinsics"].cpu().tolist()

            temp_k = [
                [intrinsics[0][0] * width, 0, intrinsics[0][2] * width],
                [0, intrinsics[1][1] * height, intrinsics[1][2] * height],
                [0, 0, 1],
            ]

            temp_hfov = 2 * np.arctan(1 / 2 / intrinsics[0][0])  # rad
            temp_vfov = 2 * np.arctan(1 / 2 / intrinsics[1][1])

            final_k_list.append(temp_k)
            final_hfov_list.append(temp_hfov)
            final_vfov_list.append(temp_vfov)

            if self.if_output_points_info:
                final_points_list.append(points)

            if self.if_output_depth_info:
                final_depth_list.append(depth)

            if self.if_output_mask_info:
                final_mask_list.append(mask)

        sample[Fields.meta][self.tag_field_name] = {
            "frames_folder": frames_root,
            "frame_names": frame_names,
            "intrinsics_list": final_k_list,
            "hfov_list": final_hfov_list,
            "vfov_list": final_vfov_list,
            "points_list": final_points_list,
            "depth_list": final_depth_list,
            "mask_list": final_mask_list,
        }

        if self.if_output_info:
            os.makedirs(self.output_info_dir, exist_ok=True)
            with open(
                os.path.join(
                    self.output_info_dir, os.path.splitext(os.path.basename(sample[self.video_key][0]))[0] + ".json"
                ),
                "w",
            ) as f:
                json.dump(sample[Fields.meta][self.tag_field_name], f)

        return sample

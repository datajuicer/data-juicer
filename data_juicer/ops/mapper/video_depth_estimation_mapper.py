import os
import subprocess

import numpy as np

from data_juicer.utils.cache_utils import DATA_JUICER_ASSETS_CACHE
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.model_utils import get_model, prepare_model

from ..base_op import OPERATORS, TAGGING_OPS, UNFORKABLE, Mapper
from ..op_fusion import LOADED_VIDEOS

OP_NAME = "video_depth_estimation_mapper"


cv2 = LazyLoader("cv2", "opencv-contrib-python")
torch = LazyLoader("torch")
open3d = LazyLoader("open3d")


@TAGGING_OPS.register_module(OP_NAME)
@UNFORKABLE.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
@LOADED_VIDEOS.register_module(OP_NAME)
class VideoDepthEstimationMapper(Mapper):
    """Perform depth estimation on the video."""

    _accelerator = "cuda"

    def __init__(
        self,
        video_depth_model_path: str = "video_depth_anything_vitb.pth",
        if_save_point_cloud: bool = True,
        point_cloud_dir_for_metric: str = DATA_JUICER_ASSETS_CACHE,
        max_res: int = 1280,
        torch_dtype: str = "fp16",
        if_save_visualization: bool = False,
        save_visualization_dir: str = DATA_JUICER_ASSETS_CACHE,
        grayscale: bool = False,
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param video_depth_model_path: The path to the Video-Depth-Anything model.
            If the model is a 'metric' model, the code will automatically switch
            to metric mode.
        :param if_save_point_cloud: Whether to save point cloud results (the user
            should input the path for storing point clouds).
        :param point_cloud_dir_for_metric: The path for storing point
            clouds (for a 'metric' model).
        :param max_res: The maximum resolution threshold for videos; videos exceeding
            this threshold will be resized.
        :param torch_dtype: The floating point type used for model inference. Can be
            one of ['fp32', 'fp16']
        :param if_save_visualization: Whether to save visualization results.
        :param save_visualization_dir: The path for saving visualization results.
        :param grayscale: If True, the colorful palette will not be applied.

        """

        super().__init__(*args, **kwargs)
        LazyLoader.check_packages(["easydict", "xformers", "imageio", "imageio-ffmpeg"])

        video_depth_anything_repo_path = os.path.join(DATA_JUICER_ASSETS_CACHE, "Video-Depth-Anything")
        if not os.path.exists(video_depth_anything_repo_path):
            subprocess.run(
                [
                    "git",
                    "clone",
                    "https://github.com/DepthAnything/Video-Depth-Anything.git",
                    video_depth_anything_repo_path,
                ],
                check=True,
            )
        import sys

        sys.path.append(video_depth_anything_repo_path)
        print(f"====exist====: {os.path.exists(os.path.join(sys.path[-1], 'utils', 'dc_utils.py'))}")
        from utils.dc_utils import read_video_frames, save_video

        if "metric" in video_depth_model_path:
            self.metric = True
        else:
            self.metric = False

        self.read_video_frames = read_video_frames
        self.save_video = save_video

        self.tag_field_name = MetaKeys.video_depth_tags
        self.max_res = max_res
        self.torch_dtype = torch_dtype
        self.if_save_point_cloud = if_save_point_cloud
        self.point_cloud_dir_for_metric = point_cloud_dir_for_metric
        self.if_save_visualization = if_save_visualization
        self.save_visualization_dir = save_visualization_dir
        self.grayscale = grayscale
        self.frame_field = MetaKeys.video_frames
        self.model_key = prepare_model(model_type="video_depth_anything", model_path=video_depth_model_path)

    def process_single(self, sample=None, rank=None):

        # check if it's generated already
        if self.tag_field_name in sample[Fields.meta]:
            return sample

        # there is no video in this sample
        if (self.video_key not in sample or not sample[self.video_key]) and self.frame_field not in sample:
            sample[Fields.meta][self.tag_field_name] = {"depth_data": [], "fps": -1}
            return sample

        video_depth_anything_model = get_model(model_key=self.model_key, rank=rank, use_cuda=self.use_cuda())

        if rank is not None:
            device = f"cuda:{str(rank)}"
        else:
            device = "cuda"

        if self.frame_field in sample:
            if "fps" not in sample:
                raise ValueError("If inputting extracted frames instead of a video, the 'fps' must be provided.")
            target_fps = sample["fps"]

            first_frame = cv2.imread(sample[self.frame_field][0])
            original_height = int(first_frame.shape[0])
            original_width = int(first_frame.shape[1])

            if self.max_res > 0 and max(original_height, original_width) > self.max_res:
                scale = self.max_res / max(original_height, original_width)
                height = round(original_height * scale)
                width = round(original_width * scale)

            frames = []
            for temp_frame_path in sample[self.frame_field]:
                temp_frame = cv2.imread(temp_frame_path)
                temp_frame = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB)
                if self.max_res > 0 and max(original_height, original_width) > self.max_res:
                    temp_frame = cv2.resize(temp_frame, (width, height))
                frames.append(temp_frame)
            frames = np.stack(frames, axis=0)

        else:
            frames, target_fps = self.read_video_frames(sample[self.video_key][0], -1, -1, self.max_res)

        depths, fps = video_depth_anything_model.infer_video_depth(
            frames,
            target_fps,
            input_size=518,
            device=device if self.use_cuda() else "cpu",
            fp32=False if self.torch_dtype == "fp16" else True,
        )

        if self.if_save_visualization:
            if self.video_key in sample:
                video_name = os.path.basename(sample[self.video_key][0])
            else:
                video_name = sample[self.frame_field][0].split("/")[-2]
            os.makedirs(self.save_visualization_dir, exist_ok=True)
            processed_video_path = os.path.join(
                self.save_visualization_dir, os.path.splitext(video_name)[0] + "_src.mp4"
            )
            depth_vis_path = os.path.join(self.save_visualization_dir, os.path.splitext(video_name)[0] + "_vis.mp4")
            self.save_video(frames, processed_video_path, fps=fps)
            self.save_video(depths, depth_vis_path, fps=fps, is_depths=True, grayscale=self.grayscale)

        if self.metric and self.if_save_point_cloud:
            os.makedirs(self.point_cloud_dir_for_metric, exist_ok=True)
            width, height = depths[0].shape[-1], depths[0].shape[-2]
            x, y = np.meshgrid(np.arange(width), np.arange(height))
            x = (x - width / 2) / 470.4
            y = (y - height / 2) / 470.4

            for i, (color_image, depth) in enumerate(zip(frames, depths)):
                z = np.array(depth)
                points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)
                colors = np.array(color_image).reshape(-1, 3) / 255.0

                pcd = open3d.geometry.PointCloud()
                pcd.points = open3d.utility.Vector3dVector(points)
                pcd.colors = open3d.utility.Vector3dVector(colors)
                open3d.io.write_point_cloud(
                    os.path.join(self.point_cloud_dir_for_metric, "point" + str(i).zfill(4) + ".ply"), pcd
                )

        sample[Fields.meta][self.tag_field_name] = {}
        sample[Fields.meta][self.tag_field_name]["depth_data"] = depths
        sample[Fields.meta][self.tag_field_name]["fps"] = fps

        return sample

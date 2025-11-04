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


LazyLoader.check_packages(["easydict", "xformers"])
cv2 = LazyLoader("cv2", "opencv-python")
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
            to metric mode, and the user should input the path for storing point
            clouds.
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
        self.point_cloud_dir_for_metric = point_cloud_dir_for_metric
        self.if_save_visualization = if_save_visualization
        self.save_visualization_dir = save_visualization_dir
        self.grayscale = grayscale
        self.model_key = prepare_model(model_type="video_depth_anything", model_path=video_depth_model_path)

    def process_single(self, sample=None, rank=None):
        # check if it's generated already
        if self.tag_field_name in sample[Fields.meta]:
            return sample

        # there is no video in this sample
        if self.video_key not in sample or not sample[self.video_key]:
            sample[Fields.meta][self.tag_field_name] = {"depth_data": [], "fps": -1}
            return sample

        video_depth_anything_model = get_model(model_key=self.model_key, rank=rank, use_cuda=self.use_cuda())

        if rank is not None:
            device = f"cuda:{str(rank)}"
        else:
            device = "cuda"
        frames, target_fps = self.read_video_frames(sample[self.video_key][0], -1, -1, self.max_res)
        depths, fps = video_depth_anything_model.infer_video_depth(
            frames,
            target_fps,
            input_size=518,
            device=device if self.use_cuda() else "cpu",
            fp32=False if self.torch_dtype == "fp16" else True,
        )

        if self.if_save_visualization:
            video_name = os.path.basename(sample[self.video_key][0])
            os.makedirs(self.save_visualization_dir, exist_ok=True)
            processed_video_path = os.path.join(
                self.save_visualization_dir, os.path.splitext(video_name)[0] + "_src.mp4"
            )
            depth_vis_path = os.path.join(self.save_visualization_dir, os.path.splitext(video_name)[0] + "_vis.mp4")
            self.save_video(frames, processed_video_path, fps=fps)
            self.save_video(depths, depth_vis_path, fps=fps, is_depths=True, grayscale=self.grayscale)

        if self.metric:
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

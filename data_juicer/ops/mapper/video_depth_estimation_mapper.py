import os
import subprocess

import numpy as np

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

    def _get_output_features(self, input_features):
        if input_features is None:
            return None
        from datasets import Value

        fields = {
            "depth_data": nested_video_feature("float64", 3),
            "fps": Value("float64"),
            "video_fps": nested_video_feature("float64"),
            "frame_paths": nested_video_feature("string"),
            "video_indices": nested_video_feature("int64"),
            "frame_indices": nested_video_feature("int64"),
        }
        return get_video_output_features(input_features, self.tag_field_name, fields)

    def process_single(self, sample=None, rank=None):
        meta = sample.setdefault(Fields.meta, {})
        if self.tag_field_name in meta:
            return sample

        from_frames = self.frame_field in sample
        groups = (
            get_video_frame_groups(sample, self.frame_field, self.video_key, None)
            if from_frames
            else [[video] for video in (sample.get(self.video_key) or [])]
        )
        tags = {
            "depth_data": [],
            "fps": -1,
            "video_fps": [-1] * len(groups),
            "frame_paths": [],
            "video_indices": [],
            "frame_indices": [],
        }
        if not any(groups):
            meta[self.tag_field_name] = tags
            return sample
        if from_frames:
            input_fps = sample.get("fps")
            video_fps = input_fps if isinstance(input_fps, (list, tuple)) else [input_fps] * len(groups)
            if len(video_fps) != len(groups):
                raise ValueError("'fps' must contain one frame rate per video frame sequence.")
            for paths, fps in zip(groups, video_fps):
                if paths and (not isinstance(fps, (int, float)) or not np.isfinite(fps) or fps <= 0):
                    raise ValueError("A positive finite 'fps' must be provided for each nonempty frame sequence.")

        model = None
        device = f"cuda:{rank}" if self.use_cuda() and rank is not None else ("cuda" if self.use_cuda() else "cpu")
        for video_index, paths in enumerate(groups):
            if not paths:
                continue
            if from_frames:
                frames = []
                for path in paths:
                    image = cv2.imread(path)
                    if image is None:
                        raise ValueError(f"Could not read video frame: {path}")
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    height, width = image.shape[:2]
                    if self.max_res > 0 and max(height, width) > self.max_res:
                        scale = self.max_res / max(height, width)
                        image = cv2.resize(image, (round(width * scale), round(height * scale)))
                    frames.append(image)
                frames = np.stack(frames, axis=0)
                target_fps = video_fps[video_index]
            else:
                frames, target_fps = self.read_video_frames(paths[0], -1, -1, self.max_res)
            if len(frames) == 0:
                continue
            if model is None:
                model = get_model(model_key=self.model_key, rank=rank, use_cuda=self.use_cuda())
            depths, fps = model.infer_video_depth(
                frames, target_fps, input_size=518, device=device, fp32=self.torch_dtype != "fp16"
            )
            if len(depths) != len(frames):
                raise ValueError("Depth predictions must match the number of input frames.")
            if tags["fps"] == -1:
                tags["fps"] = fps
            tags["video_fps"][video_index] = fps
            tags["depth_data"].extend(np.asarray(depths).tolist())
            tags["video_indices"].extend([video_index] * len(frames))
            tags["frame_indices"].extend(range(len(frames)))
            if from_frames:
                tags["frame_paths"].extend(paths)

            if self.if_save_visualization:
                directory = get_video_visualization_dir(self.save_visualization_dir, OP_NAME, paths, video_index)
                self.save_video(frames, os.path.join(directory, "src.mp4"), fps=fps)
                self.save_video(
                    depths, os.path.join(directory, "vis.mp4"), fps=fps, is_depths=True, grayscale=self.grayscale
                )
            if self.metric and self.if_save_point_cloud:
                directory = get_video_visualization_dir(self.point_cloud_dir_for_metric, OP_NAME, paths, video_index)
                width, height = depths[0].shape[-1], depths[0].shape[-2]
                x, y = np.meshgrid(np.arange(width), np.arange(height))
                x = (x - width / 2) / 470.4
                y = (y - height / 2) / 470.4
                for index, (color_image, depth) in enumerate(zip(frames, depths)):
                    z = np.asarray(depth)
                    points = np.stack((x * z, y * z, z), axis=-1).reshape(-1, 3)
                    colors = np.asarray(color_image).reshape(-1, 3) / 255.0
                    cloud = open3d.geometry.PointCloud()
                    cloud.points = open3d.utility.Vector3dVector(points)
                    cloud.colors = open3d.utility.Vector3dVector(colors)
                    open3d.io.write_point_cloud(os.path.join(directory, f"point{index:04d}.ply"), cloud)
        meta[self.tag_field_name] = tags
        return sample

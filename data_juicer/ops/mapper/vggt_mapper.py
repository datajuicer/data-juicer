import os
import subprocess

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

OP_NAME = "vggt_mapper"

torch = LazyLoader("torch")


@OPERATORS.register_module(OP_NAME)
@LOADED_VIDEOS.register_module(OP_NAME)
class VggtMapper(Mapper):
    """Input a video of a single scene, and use VGGT to extract information including Camera
    Pose, Depth Maps, Point Maps, and 3D Point Tracks.

    - The operator processes a video and extracts frames based on the specified frame number
      and duration.
    - It uses the VGGT model to analyze the extracted frames and generate various outputs
      such as camera parameters, depth maps, point maps, and 3D point tracks.
    - If 3D point tracks are required, the user must provide query points in the format [x,
      y], relative to the top-left corner.
    - The results are stored in the sample's metadata under the specified tag field name,
      which defaults to 'vggt_tags'.
    - The operator can output camera parameters, depth maps, point maps from projection,
      point maps from unprojection, and 3D point tracks, depending on the configuration.
    - The VGGT model is loaded from the provided path, and the operator runs in CUDA mode if
      available."""

    _accelerator = "cuda"

    def __init__(
        self,
        vggt_model_path: str = "facebook/VGGT-1B",
        frame_num: PositiveInt = 3,
        duration: float = 0,
        tag_field_name: str = MetaKeys.vggt_tags,
        frame_dir: str = DATA_JUICER_ASSETS_CACHE,
        if_output_camera_parameters: bool = True,
        if_output_depth_maps: bool = True,
        if_output_point_maps_from_projection: bool = True,
        if_output_point_maps_from_unprojection: bool = True,
        if_output_point_tracks: bool = True,
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param vggt_model_path: The path to the VGGT model.
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
            "vggt_tags" in default.
        :param frame_dir: Output directory to save extracted frames.
        :param if_output_camera_parameters: Determines whether to output
            camera parameters.
        :param if_output_depth_maps: Determines whether to output
            depth maps.
        :param if_output_point_maps_from_projection: Determines whether to
            output point maps directly inferred by VGGT.
        :param if_output_point_maps_from_unprojection: Determines whether to
            output point maps constructed from depth maps and camera parameters.
        :param if_output_point_tracks: Determines whether to output point tracks.
            If point tracks are required, the user should provide a list where
            each element consists of 2D point coordinates (list shape: (N, 2)).
            The point coordinates should be specified in the format [x, y],
            relative to the top-left corner, where x/y values are non-normalized.
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

        vggt_repo_path = os.path.join(DATA_JUICER_ASSETS_CACHE, "vggt")
        if not os.path.exists(vggt_repo_path):
            subprocess.run(["git", "clone", "https://github.com/facebookresearch/vggt.git", vggt_repo_path], check=True)
        import sys

        sys.path.append(vggt_repo_path)
        from vggt.utils.geometry import unproject_depth_map_to_point_map
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        self.load_and_preprocess_images = load_and_preprocess_images
        self.pose_encoding_to_extri_intri = pose_encoding_to_extri_intri
        self.unproject_depth_map_to_point_map = unproject_depth_map_to_point_map

        self.frame_num = frame_num
        self.duration = duration
        self.tag_field_name = tag_field_name
        self.frame_dir = frame_dir
        self.dtype = torch.bfloat16 if self.use_cuda() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        self.model_key = prepare_model(model_type="vggt", model_path=vggt_model_path)

        self.if_output_camera_parameters = if_output_camera_parameters
        self.if_output_depth_maps = if_output_depth_maps
        self.if_output_point_maps_from_projection = if_output_point_maps_from_projection
        self.if_output_point_maps_from_unprojection = if_output_point_maps_from_unprojection
        self.if_output_point_tracks = if_output_point_tracks

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

        vggt_model = get_model(self.model_key, rank, self.use_cuda())

        frames_root = os.path.join(self.frame_dir, os.path.splitext(os.path.basename(sample[self.video_key][0]))[0])
        frame_names = os.listdir(frames_root)
        frames_path = sorted([os.path.join(frames_root, frame_name) for frame_name in frame_names])

        if rank is not None:
            images = self.load_and_preprocess_images(frames_path).to(f"cuda:{str(rank)}" if self.use_cuda() else "cpu")
        else:
            images = self.load_and_preprocess_images(frames_path).to("cuda" if self.use_cuda() else "cpu")

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=self.dtype):
                images = images[None]
                aggregated_tokens_list, ps_idx = vggt_model.aggregator(images)

        # Predict Cameras
        if self.if_output_camera_parameters or self.if_output_point_maps_from_unprojection:
            with torch.no_grad():
                pose_enc = vggt_model.camera_head(aggregated_tokens_list)[-1]
                # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
                extrinsic, intrinsic = self.pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        else:
            extrinsic = []
            intrinsic = []

        # Predict Depth Maps
        if self.if_output_depth_maps or self.if_output_point_maps_from_unprojection:
            with torch.no_grad():
                depth_map, depth_conf = vggt_model.depth_head(aggregated_tokens_list, images, ps_idx)
        else:
            depth_map = []
            depth_conf = []

        # Predict Point Maps
        if self.if_output_point_maps_from_projection:
            with torch.no_grad():
                point_map, point_conf = vggt_model.point_head(aggregated_tokens_list, images, ps_idx)
        else:
            point_map = []
            point_conf = []

        # Construct 3D Points from Depth Maps and Cameras
        if self.if_output_point_maps_from_unprojection:
            with torch.no_grad():
                # which usually leads to more accurate 3D points than point map branch
                point_maps_from_unprojection = self.unproject_depth_map_to_point_map(
                    depth_map.squeeze(0), extrinsic.squeeze(0), intrinsic.squeeze(0)
                )
        else:
            point_maps_from_unprojection = []

        # Predict Tracks
        # If point track output is required, users must provide a list of non-normalized [x, y]
        # coordinates (shape (N, 2)) relative to the top-left corner.
        # The tracking is done in 4 iterations. The last iteration should be the best one.
        query_points = sample.get("query_points")
        # choose your own points to track, with shape (N, 2) for one scene
        if self.if_output_point_tracks and query_points and len(query_points) > 0:
            with torch.no_grad():
                if rank is not None:
                    query_points_tensor = torch.FloatTensor(query_points).to(
                        f"cuda:{str(rank)}" if self.use_cuda() else "cpu"
                    )
                else:
                    query_points_tensor = torch.FloatTensor(query_points).to("cuda" if self.use_cuda() else "cpu")
                track_list, vis_score, conf_score = vggt_model.track_head(
                    aggregated_tokens_list, images, ps_idx, query_points=query_points_tensor[None]
                )
        else:
            track_list = []
            vis_score = []
            conf_score = []

        sample[Fields.meta][self.tag_field_name] = {}
        sample[Fields.meta][self.tag_field_name]["frames_folder"] = frames_root
        sample[Fields.meta][self.tag_field_name]["frames_path"] = frames_path
        sample[Fields.meta][self.tag_field_name]["camera_parameters"] = {"extrinsic": extrinsic, "intrinsic": intrinsic}
        sample[Fields.meta][self.tag_field_name]["depth_maps"] = {"depth_map": depth_map, "depth_conf": depth_conf}
        sample[Fields.meta][self.tag_field_name]["point_maps_from_projection"] = {
            "point_map": point_map,
            "point_conf": point_conf,
        }
        sample[Fields.meta][self.tag_field_name]["point_maps_from_unprojection"] = {
            "point_maps_from_unprojection": point_maps_from_unprojection
        }
        sample[Fields.meta][self.tag_field_name]["point_tracks"] = {
            "query_points": query_points,
            "track_list": track_list,
            "vis_score": vis_score,
            "conf_score": conf_score,
        }

        return sample

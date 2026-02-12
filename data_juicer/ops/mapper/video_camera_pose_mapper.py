import os
import subprocess

import numpy as np
from pydantic import PositiveInt

import data_juicer
from data_juicer.ops.load import load_ops
from data_juicer.utils.cache_utils import DATA_JUICER_ASSETS_CACHE
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.lazy_loader import LazyLoader

from ..base_op import OPERATORS, Mapper
from ..op_fusion import LOADED_VIDEOS

OP_NAME = "video_camera_pose_mapper"

cv2 = LazyLoader("cv2", "opencv-python")
torch = LazyLoader("torch")


@OPERATORS.register_module(OP_NAME)
@LOADED_VIDEOS.register_module(OP_NAME)
class VideoCameraPoseMapper(Mapper):
    """Extract camera poses by leveraging MegaSaM and MoGe-2."""

    _accelerator = "cuda"

    def __init__(
        self,
        moge_model_path: str = "Ruicheng/moge-2-vitl",
        frame_num: PositiveInt = 3,
        duration: float = 0,
        tag_field_name: str = MetaKeys.video_camera_pose_tags,
        frame_dir: str = DATA_JUICER_ASSETS_CACHE,
        if_output_moge_info: bool = False,
        moge_output_info_dir: str = DATA_JUICER_ASSETS_CACHE,
        if_save_info: bool = True,
        output_info_dir: str = DATA_JUICER_ASSETS_CACHE,
        max_frames: int = 1000,
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param moge_model_path: The path to the Moge-2 model.
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
            "video_camera_pose_tags" in default.
        :param frame_dir: Output directory to save extracted frames.
        :param if_output_moge_info: Whether to save the results from MoGe-2
             to an JSON file.
        :param moge_output_info_dir: Output directory for saving camera
            parameters.
        :param if_save_info: Whether to save the results to an npz file.
        :param output_info_dir: Path for saving the results.
        :param max_frames: Maximum number of frames to save.
        :param args: extra args
        :param kwargs: extra args

        """

        super().__init__(*args, **kwargs)

        self.video_camera_calibration_static_moge_mapper_args = {
            "model_path": moge_model_path,
            "frame_num": frame_num,
            "duration": duration,
            "frame_dir": frame_dir,
            "if_output_points_info": False,
            "if_output_depth_info": True,
            "if_output_mask_info": True,
            "if_output_info": if_output_moge_info,
            "output_info_dir": moge_output_info_dir,
        }
        self.fused_ops = load_ops(
            [{"video_camera_calibration_static_moge_mapper": self.video_camera_calibration_static_moge_mapper_args}]
        )

        megasam_repo_path = os.path.join(DATA_JUICER_ASSETS_CACHE, "mega-sam")
        if not os.path.exists(megasam_repo_path):
            subprocess.run(["git", "clone", "https://github.com/mega-sam/mega-sam.git", megasam_repo_path], check=True)
            subprocess.run(
                ["git", "submodule", "update", "--init", "--recursive"], cwd=os.path.join(megasam_repo_path, "base")
            )

            with open(os.path.join(megasam_repo_path, "base", "src", "altcorr_kernel.cu"), "r") as f:
                temp_file_content = f.read()
                temp_file_content = temp_file_content.replace(".type()", ".scalar_type()")

            with open(os.path.join(megasam_repo_path, "base", "src", "altcorr_kernel.cu"), "w") as f:
                f.write(temp_file_content)

            with open(os.path.join(megasam_repo_path, "base", "src", "correlation_kernels.cu"), "r") as f:
                temp_file_content = f.read()
                temp_file_content = temp_file_content.replace(".type()", ".scalar_type()")

            with open(os.path.join(megasam_repo_path, "base", "src", "correlation_kernels.cu"), "w") as f:
                f.write(temp_file_content)

            with open(os.path.join(megasam_repo_path, "base", "src", "droid_kernels.cu"), "r") as f:
                temp_file_content = f.read()
                temp_file_content = temp_file_content.replace(".type()", ".scalar_type()")

            with open(os.path.join(megasam_repo_path, "base", "src", "droid_kernels.cu"), "w") as f:
                f.write(temp_file_content)

            with open(
                os.path.join(megasam_repo_path, "base", "thirdparty", "lietorch", "lietorch", "src", "lietorch_gpu.cu"),
                "r",
            ) as f:
                temp_file_content = f.read()
                temp_file_content = temp_file_content.replace(".type()", ".scalar_type()")

            with open(
                os.path.join(megasam_repo_path, "base", "thirdparty", "lietorch", "lietorch", "src", "lietorch_gpu.cu"),
                "w",
            ) as f:
                f.write(temp_file_content)

            with open(
                os.path.join(
                    megasam_repo_path, "base", "thirdparty", "lietorch", "lietorch", "src", "lietorch_cpu.cpp"
                ),
                "r",
            ) as f:
                temp_file_content = f.read()
                temp_file_content = temp_file_content.replace(".type()", ".scalar_type()")

            with open(
                os.path.join(
                    megasam_repo_path, "base", "thirdparty", "lietorch", "lietorch", "src", "lietorch_cpu.cpp"
                ),
                "w",
            ) as f:
                f.write(temp_file_content)

        try:
            import droid_backends
            import lietorch

            self.droid_backends = droid_backends
            self.lietorch = lietorch
        except ImportError:
            subprocess.run(["python", "setup.py", "install"], cwd=os.path.join(megasam_repo_path, "base"))

        try:
            import torch_scatter

            self.torch_scatter = torch_scatter
        except ImportError:
            """ "Please refer to https://github.com/rusty1s/pytorch_scatter to locate the
            installation link that is compatible with your PyTorch and CUDA versions."""
            torch_version = "2.8.0"
            cuda_version = "cu128"
            subprocess.run(
                [
                    "pip",
                    "install",
                    "torch-scatter",
                    "-f",
                    f"https://data.pyg.org/whl/torch-{torch_version}+{cuda_version}.html",
                ],
                cwd=os.path.join(megasam_repo_path, "base"),
            )

        import sys

        sys.path.append(os.path.join(megasam_repo_path, "base", "droid_slam"))
        from droid import Droid
        from lietorch import SE3

        self.SE3 = SE3
        self.Droid = Droid

        self.tag_field_name = tag_field_name
        self.if_save_info = if_save_info
        self.output_info_dir = output_info_dir
        self.max_frames = max_frames
        self.frame_dir = frame_dir

    def image_stream(self, frames_path, depth_list, intrinsics_list):

        for t, (image_path, depth, intrinsics) in enumerate(zip(frames_path, depth_list, intrinsics_list)):
            image = cv2.imread(image_path)
            h0, w0, _ = image.shape
            h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
            w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

            image = cv2.resize(image, (w1, h1), interpolation=cv2.INTER_AREA)
            image = image[: h1 - h1 % 8, : w1 - w1 % 8]
            image = torch.as_tensor(image).permute(2, 0, 1)
            image = image[None]

            depth = torch.as_tensor(depth)
            depth = torch.nn.functional.interpolate(depth[None, None], (h1, w1), mode="nearest-exact").squeeze()
            depth = depth[: h1 - h1 % 8, : w1 - w1 % 8]

            mask = torch.ones_like(depth)

            intrinsics = torch.as_tensor([intrinsics[0][0], intrinsics[1][1], intrinsics[0][2], intrinsics[1][2]])
            intrinsics[0::2] *= w1 / w0
            intrinsics[1::2] *= h1 / h0

            yield t, image, depth, intrinsics, mask

    def process_single(self, sample=None, rank=None):
        # check if it's generated already
        if self.tag_field_name in sample[Fields.meta]:
            return sample

        # there is no video in this sample
        if self.video_key not in sample or not sample[self.video_key]:
            return []

        ds_list = [{"videos": sample[self.video_key]}]

        dataset = data_juicer.core.data.NestedDataset.from_list(ds_list)
        if Fields.meta not in dataset.features:
            dataset = dataset.add_column(name=Fields.meta, column=[{}] * dataset.num_rows)
        dataset = dataset.map(self.fused_ops[0].process, num_proc=1, with_rank=True)
        res_list = dataset.to_list()

        temp_frame_name = os.path.splitext(os.path.basename(sample[self.video_key][0]))[0]
        frames_root = os.path.join(self.frame_dir, temp_frame_name)
        frame_names = os.listdir(frames_root)
        frames_path = sorted([os.path.join(frames_root, frame_name) for frame_name in frame_names])

        depth_list = res_list[0][Fields.meta][MetaKeys.static_camera_calibration_moge_tags]["depth_list"]
        intrinsics_list = res_list[0][Fields.meta][MetaKeys.static_camera_calibration_moge_tags]["intrinsics_list"]

        valid_image_list = []
        valid_depth_list = []
        valid_intrinsics_list = []
        valid_mask_list = []

        # for t, (image_path, depth, intrinsics) in enumerate(zip(frames_path, depth_list, intrinsics_list)):

        for t, image, depth, intrinsics, mask in self.image_stream(frames_path, depth_list, intrinsics_list):

            valid_image_list.append(image[0])
            valid_depth_list.append(depth)
            valid_mask_list.append(mask)
            valid_intrinsics_list.append(intrinsics)

            if t == 0:
                args = droid_args(image_size=[image.shape[2], image.shape[3]])
                droid = self.Droid(args)

            droid.track(t, image, depth, intrinsics=intrinsics, mask=mask)

        droid.track_final(t, image, depth, intrinsics=intrinsics, mask=mask)

        traj_est, depth_est, motion_prob = droid.terminate(
            self.image_stream(frames_path, depth_list, intrinsics_list),
            _opt_intr=True,
            full_ba=True,
            scene_name=temp_frame_name,
        )

        t = traj_est.shape[0]
        images = np.array(valid_image_list[:t])
        disps = 1.0 / (np.array(valid_depth_list[:t]) + 1e-6)

        poses = traj_est
        intrinsics = droid.video.intrinsics[:t].cpu().numpy()

        intrinsics = intrinsics[0] * 8.0
        poses_th = torch.as_tensor(poses, device="cpu")
        cam_c2w = self.SE3(poses_th).inv().matrix().numpy()

        K = np.eye(3)
        K[0, 0] = intrinsics[0]
        K[1, 1] = intrinsics[1]
        K[0, 2] = intrinsics[2]
        K[1, 2] = intrinsics[3]

        max_frames = min(self.max_frames, images.shape[0])

        return_images = np.uint8(images[:max_frames, ::-1, ...].transpose(0, 2, 3, 1))
        return_depths = np.float32(1.0 / disps[:max_frames, ...])
        return_cam_c2w = cam_c2w[:max_frames]

        if self.if_save_info:
            os.makedirs(self.output_info_dir, exist_ok=True)

            np.savez(
                os.path.join(self.output_info_dir, "%s_droid.npz" % temp_frame_name),
                images=return_images,
                depths=return_depths,
                intrinsic=K,
                cam_c2w=return_cam_c2w,
            )

        sample[Fields.meta][self.tag_field_name] = {
            "frames_folder": frames_root,
            "frame_names": frame_names,
            "images": return_images,
            "depths": return_depths,
            "intrinsic": K,
            "cam_c2w": return_cam_c2w,
        }

        return sample


class droid_args:
    def __init__(self, image_size):
        self.weights = os.path.join(DATA_JUICER_ASSETS_CACHE, "mega-sam", "checkpoints", "megasam_final.pth")
        self.disable_vis = True
        self.image_size = image_size
        self.buffer = 1024
        self.stereo = False
        self.filter_thresh = 2.0

        self.warmup = 8
        self.beta = 0.3
        self.frontend_nms = 1
        self.keyframe_thresh = 2.0
        self.frontend_window = 25
        self.frontend_thresh = 12.0
        self.frontend_radius = 2

        self.upsample = False
        self.backend_thresh = 16.0
        self.backend_radius = 2
        self.backend_nms = 3

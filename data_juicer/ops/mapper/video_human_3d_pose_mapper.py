import glob
import os
import shutil
import tempfile
from copy import deepcopy

import cv2
import numpy as np
import PIL
from loguru import logger
from PIL.ImageOps import exif_transpose
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

OP_NAME = "video_human_3d_pose_mapper"

torch = LazyLoader("torch")
torchvision = LazyLoader("torchvision")


@OPERATORS.register_module(OP_NAME)
@LOADED_VIDEOS.register_module(OP_NAME)
class VideoHuman3DPoseMapper(Mapper):
    """Extract 3D human pose with Human3R (SMPL-X)."""

    _accelerator = "cuda"

    def __init__(
        self,
        model_path: str = "human3r_896L.pth",
        frame_num: PositiveInt = 3,
        duration: float = 0,
        frame_dir: str = DATA_JUICER_ASSETS_CACHE,
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param model_path: The path to the Human3R model checkpoint.
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

        LazyLoader.check_packages(["gsplat", "roma", "pyrender", "smplx"])

        os.environ["PYOPENGL_PLATFORM"] = "egl"
        self.frame_field = MetaKeys.video_frames
        self.tag_field_name = MetaKeys.video_human_3d_pose_tags
        self.frame_num = frame_num
        self.duration = duration
        self.frame_dir = frame_dir

        import torch.nn.functional as F
        import torchvision.transforms as tvf

        self.tvf = tvf
        self.F = F

        self.model_key = prepare_model(model_type="human_3d_pose_human3r", model_path=model_path)

        self.video_extract_frames_mapper_args = {
            "frame_sampling_method": "uniform",
            "frame_num": frame_num,
            "duration": duration,
            "frame_dir": frame_dir,
            "frame_key": MetaKeys.video_frames,
            "num_proc": None,  # Disable multiprocessing to avoid nested process pool issue
            "auto_op_parallelism": False,  # Disable auto parallelism to avoid nested process pool issue
        }
        self.fused_ops = load_ops([{"video_extract_frames_mapper": self.video_extract_frames_mapper_args}])

    def pad_image(self, img_tensor, target_size, pad_value=-1.0):
        """
        torch version of ImageOps.pad, equivalent to the combination of contain and pad

        Args:
            img_tensor: torch tensor, shape [C, H, W] or [B, C, H, W]
            target_size: int, target size (square)

        Returns:
            torch tensor, shape [C, target_size, target_size] or [B, C, target_size, target_size]
        """

        # process input dimension
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size, channels, height, width = img_tensor.shape

        # calculate scale (contain function)
        scale = min(target_size / height, target_size / width)

        # resize image
        new_height = int(height * scale)
        new_width = int(width * scale)

        img_resized = self.F.interpolate(
            img_tensor, size=(new_height, new_width), mode="bilinear", align_corners=False  # bicubic
        )

        # calculate padding (pad function)
        pad_height = target_size - new_height
        pad_width = target_size - new_width

        # center padding
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left

        # apply padding (left, right, top, bottom)
        img_padded = self.F.pad(
            img_resized, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=pad_value
        )

        if squeeze_output:
            img_padded = img_padded.squeeze(0)

        return img_padded

    def load_images(self, folder_or_list, size, square_ok=False, verbose=True):
        """open and convert all images in a list or folder to proper input format for DUSt3R"""

        def _resize_pil_image(img, long_edge_size):
            S = max(img.size)
            if S > long_edge_size:
                interp = PIL.Image.LANCZOS
            elif S <= long_edge_size:
                interp = PIL.Image.BICUBIC
            new_size = tuple(int(round(x * long_edge_size / S)) for x in img.size)
            return img.resize(new_size, interp)

        ImgNorm = self.tvf.Compose([self.tvf.ToTensor(), self.tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        if isinstance(folder_or_list, str):
            if verbose:
                logger.info(f">> Loading images from {folder_or_list}")

            root, folder_content = folder_or_list, sorted(os.listdir(folder_or_list))

        elif isinstance(folder_or_list, list):
            if verbose:
                logger.info(f">> Loading a list of {len(folder_or_list)} images")
            root, folder_content = "", folder_or_list

        else:
            raise ValueError(f"bad {folder_or_list}= ({type(folder_or_list)})")

        supported_images_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
        supported_images_extensions = tuple(supported_images_extensions)

        imgs = []
        for path in folder_content:
            if not path.lower().endswith(supported_images_extensions):
                continue
            img = exif_transpose(PIL.Image.open(os.path.join(root, path))).convert("RGB")
            W1, H1 = img.size
            if size == 224:

                img = _resize_pil_image(img, round(size * max(W1 / H1, H1 / W1)))
            else:

                img = _resize_pil_image(img, size)
            W, H = img.size
            cx, cy = W // 2, H // 2
            if size == 224:
                half = min(cx, cy)
                img = img.crop((cx - half, cy - half, cx + half, cy + half))
            else:
                halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
                if not (square_ok) and W == H:
                    halfh = 3 * halfw / 4
                img = img.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))

            W2, H2 = img.size
            if verbose:
                logger.info(f" - adding {path} with resolution {W1}x{H1} --> {W2}x{H2}")
            imgs.append(
                dict(
                    img=ImgNorm(img)[None],
                    true_shape=np.int32([img.size[::-1]]),
                    idx=len(imgs),
                    instance=str(len(imgs)),
                )
            )

        assert imgs, "no images found at " + root
        if verbose:
            logger.info(f" (Found {len(imgs)} images)")
        return imgs

    def get_focalLength_from_fieldOfView(self, fov=60, img_size=512):
        """
        Compute the focal length of the camera lens by assuming a certain FOV for the entire image
        Args:
            - fov: float, expressed in degree
            - img_size: int
        Return:
            focal: float
        """
        focal = img_size / (2 * np.tan(np.radians(fov) / 2))
        return focal

    def get_camera_parameters(self, img_size, fov=60, p_x=None, p_y=None, device=torch.device("cuda")):
        """Given image size, fov and principal point coordinates, return K the camera parameter matrix"""
        K = torch.eye(3)
        # Get focal length.
        focal = self.get_focalLength_from_fieldOfView(fov=fov, img_size=img_size)
        K[0, 0], K[1, 1] = focal, focal

        # Set principal point
        if p_x is not None and p_y is not None:
            K[0, -1], K[1, -1] = p_x * img_size, p_y * img_size
        else:
            K[0, -1], K[1, -1] = img_size // 2, img_size // 2

        # Add batch dimension
        K = K.unsqueeze(0).to(device)
        return K

    def parse_seq_path(self, p):
        if os.path.isdir(p):
            img_paths = sorted(glob.glob(f"{p}/*"))
            tmpdirname = None
        else:
            cap = cv2.VideoCapture(p)
            if not cap.isOpened():
                raise ValueError(f"Error opening video file {p}")
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if video_fps == 0:
                cap.release()
                raise ValueError(f"Error: Video FPS is 0 for {p}")
            frame_interval = 1
            frame_indices = list(range(0, total_frames, frame_interval))
            logger.info(
                f" - Video FPS: {video_fps}, Frame Interval: {frame_interval}, Total Frames to Read: {len(frame_indices)}"
            )
            img_paths = []
            tmpdirname = tempfile.mkdtemp()
            for i in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if not ret:
                    break
                frame_path = os.path.join(tmpdirname, f"frame_{i}.jpg")
                cv2.imwrite(frame_path, frame)
                img_paths.append(frame_path)
            cap.release()
        return img_paths, tmpdirname

    def prepare_input(
        self,
        img_paths,
        img_mask,
        size,
        raymaps=None,
        raymap_mask=None,
        revisit=1,
        update=True,
        img_res=None,
        reset_interval=100,
    ):
        """
        Prepare input views for inference from a list of image paths.

        Args:
            img_paths (list): List of image file paths.
            img_mask (list of bool): Flags indicating valid images.
            size (int): Target image size.
            raymaps (list, optional): List of ray maps.
            raymap_mask (list, optional): Flags indicating valid ray maps.
            revisit (int): How many times to revisit each view.
            update (bool): Whether to update the state on revisits.

        Returns:
            list: A list of view dictionaries.
        """

        images = self.load_images(img_paths, size=size)
        if img_res is not None:
            K_mhmr = self.get_camera_parameters(img_res, device="cpu")  # if use pseudo K

        views = []
        if raymaps is None and raymap_mask is None:
            # Only images are provided.
            for i in range(len(images)):
                view = {
                    "img": images[i]["img"],
                    "ray_map": torch.full(
                        (
                            images[i]["img"].shape[0],
                            6,
                            images[i]["img"].shape[-2],
                            images[i]["img"].shape[-1],
                        ),
                        torch.nan,
                    ),
                    "true_shape": torch.from_numpy(images[i]["true_shape"]),
                    "idx": i,
                    "instance": str(i),
                    "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(0),
                    "img_mask": torch.tensor(True).unsqueeze(0),
                    "ray_mask": torch.tensor(False).unsqueeze(0),
                    "update": torch.tensor(True).unsqueeze(0),
                    "reset": torch.tensor((i + 1) % reset_interval == 0).unsqueeze(0),
                }
                if img_res is not None:
                    view["img_mhmr"] = self.pad_image(view["img"], img_res)
                    view["K_mhmr"] = K_mhmr
                views.append(view)
                if (i + 1) % reset_interval == 0:
                    overlap_view = deepcopy(view)
                    overlap_view["reset"] = torch.tensor(False).unsqueeze(0)
                    views.append(overlap_view)
        else:
            # Combine images and raymaps.
            num_views = len(images) + len(raymaps)
            assert len(img_mask) == len(raymap_mask) == num_views
            assert sum(img_mask) == len(images) and sum(raymap_mask) == len(raymaps)

            j = 0
            k = 0
            for i in range(num_views):
                view = {
                    "img": (images[j]["img"] if img_mask[i] else torch.full_like(images[0]["img"], torch.nan)),
                    "ray_map": (raymaps[k] if raymap_mask[i] else torch.full_like(raymaps[0], torch.nan)),
                    "true_shape": (
                        torch.from_numpy(images[j]["true_shape"])
                        if img_mask[i]
                        else torch.from_numpy(np.int32([raymaps[k].shape[1:-1][::-1]]))
                    ),
                    "idx": i,
                    "instance": str(i),
                    "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(0),
                    "img_mask": torch.tensor(img_mask[i]).unsqueeze(0),
                    "ray_mask": torch.tensor(raymap_mask[i]).unsqueeze(0),
                    "update": torch.tensor(img_mask[i]).unsqueeze(0),
                    "reset": torch.tensor((i + 1) % reset_interval == 0).unsqueeze(0),
                }
                if img_res is not None:
                    view["img_mhmr"] = self.pad_image(view["img"], img_res)
                    view["K_mhmr"] = K_mhmr
                if img_mask[i]:
                    j += 1
                if raymap_mask[i]:
                    k += 1
                views.append(view)
                if (i + 1) % reset_interval == 0:
                    overlap_view = deepcopy(view)
                    overlap_view["reset"] = torch.tensor(False).unsqueeze(0)
                    views.append(overlap_view)
            assert j == len(images) and k == len(raymaps)

        if revisit > 1:
            new_views = []
            for r in range(revisit):
                for i, view in enumerate(views):
                    new_view = deepcopy(view)
                    new_view["idx"] = r * len(views) + i
                    new_view["instance"] = str(r * len(views) + i)
                    if r > 0 and not update:
                        new_view["update"] = torch.tensor(False).unsqueeze(0)
                    new_views.append(new_view)
            return new_views

        return views

    def process_single(self, sample=None, rank=None):

        # check if it's generated already
        if self.tag_field_name in sample[Fields.meta]:
            return sample

        # there is no video in this sample
        if (self.video_key not in sample or not sample[self.video_key]) and self.frame_field not in sample:
            sample[Fields.meta][self.tag_field_name] = {}
            sample[Fields.meta][self.tag_field_name]["valid_frame_list"] = []
            sample[Fields.meta][self.tag_field_name]["camera_pose"] = []
            sample[Fields.meta][self.tag_field_name]["conf"] = []
            sample[Fields.meta][self.tag_field_name]["smpl_shape"] = []
            sample[Fields.meta][self.tag_field_name]["smpl_transl"] = []
            sample[Fields.meta][self.tag_field_name]["smpl_rotmat"] = []
            sample[Fields.meta][self.tag_field_name]["smpl_expression"] = []
            sample[Fields.meta][self.tag_field_name]["smpl_id"] = []
            sample[Fields.meta][self.tag_field_name]["smpl_loc"] = []
            return sample

        model, inference_recurrent_lighter = get_model(model_key=self.model_key, rank=rank, use_cuda=self.use_cuda())

        if rank is not None:
            device = f"cuda:{str(rank)}"
        else:
            device = "cuda"

        if self.frame_field in sample:
            frames_path = sample[self.frame_field]
            frames_root = os.path.dirname(frames_path[0])
        else:
            # load videos
            ds_list = [{"text": SpecialTokens.video, "videos": sample[self.video_key]}]

            dataset = data_juicer.core.data.NestedDataset.from_list(ds_list)
            dataset = self.fused_ops[0].run(dataset)

            temp_frame_name = os.path.splitext(os.path.basename(sample[self.video_key][0]))[0]
            frames_root = os.path.join(self.frame_dir, temp_frame_name)
            frame_names = os.listdir(frames_root)
            frames_path = sorted([os.path.join(frames_root, frame_name) for frame_name in frame_names])

        img_paths, tmpdirname = self.parse_seq_path(frames_root)
        img_mask = [True] * len(img_paths)

        img_res = getattr(model, "mhmr_img_res", None)
        views = self.prepare_input(
            img_paths=img_paths,
            img_mask=img_mask,
            size=512,
            revisit=1,
            update=True,
            img_res=img_res,
            reset_interval=100,
        )

        if tmpdirname is not None:
            shutil.rmtree(tmpdirname)

        final_valid_frame_list = []
        final_camera_pose_list = []
        final_conf_list = []
        final_smpl_shape_list = []
        final_smpl_transl_list = []
        final_smpl_rotmat_list = []
        final_smpl_expression_list = []
        final_smpl_id_list = []
        final_smpl_loc_list = []

        outputs, _ = inference_recurrent_lighter(views, model, device, use_ttt3r=False)

        for temp_frame_output_id, temp_frame_output in enumerate(outputs["pred"]):
            if "smpl_shape" in temp_frame_output:
                final_valid_frame_list.append(temp_frame_output_id)
                final_camera_pose_list.append(torch.squeeze(temp_frame_output["camera_pose"], 0).cpu().numpy())
                final_conf_list.append(torch.squeeze(temp_frame_output["conf"], 0).cpu().numpy())
                final_smpl_shape_list.append(torch.squeeze(temp_frame_output["smpl_shape"], 0).cpu().numpy())
                final_smpl_transl_list.append(torch.squeeze(temp_frame_output["smpl_transl"], 0).cpu().numpy())
                final_smpl_rotmat_list.append(torch.squeeze(temp_frame_output["smpl_rotmat"], 0).cpu().numpy())
                final_smpl_expression_list.append(torch.squeeze(temp_frame_output["smpl_expression"], 0).cpu().numpy())
                final_smpl_id_list.append(torch.squeeze(temp_frame_output["smpl_id"], 0).cpu().numpy())
                final_smpl_loc_list.append(torch.squeeze(temp_frame_output["smpl_loc"], 0).cpu().numpy())

        sample[Fields.meta][self.tag_field_name] = {}
        sample[Fields.meta][self.tag_field_name]["valid_frame_list"] = final_valid_frame_list
        sample[Fields.meta][self.tag_field_name]["camera_pose"] = final_camera_pose_list
        sample[Fields.meta][self.tag_field_name]["conf"] = final_conf_list
        sample[Fields.meta][self.tag_field_name]["smpl_shape"] = final_smpl_shape_list
        sample[Fields.meta][self.tag_field_name]["smpl_transl"] = final_smpl_transl_list
        sample[Fields.meta][self.tag_field_name]["smpl_rotmat"] = final_smpl_rotmat_list
        sample[Fields.meta][self.tag_field_name]["smpl_expression"] = final_smpl_expression_list
        sample[Fields.meta][self.tag_field_name]["smpl_id"] = final_smpl_id_list
        sample[Fields.meta][self.tag_field_name]["smpl_loc"] = final_smpl_loc_list

        return sample

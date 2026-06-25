import os
import uuid

import numpy as np
from loguru import logger

from data_juicer.utils.constant import CameraCalibrationKeys, Fields, MetaKeys
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.model_utils import get_model, prepare_model

from ..base_op import OPERATORS, Mapper
from ..op_fusion import LOADED_VIDEOS

OP_NAME = "video_camera_calibration_moge_mapper"

cv2 = LazyLoader("cv2", "opencv-python")
torch = LazyLoader("torch")


@OPERATORS.register_module(OP_NAME)
@LOADED_VIDEOS.register_module(OP_NAME)
class VideoCameraCalibrationMogeMapper(Mapper):
    """Compute the camera intrinsics and field of view (FOV)
    for a static camera using Moge-2 (more accurate
    than DeepCalib)."""

    _accelerator = "cuda"

    def __init__(
        self,
        model_path: str = "Ruicheng/moge-2-vitl",
        tag_field_name: str = MetaKeys.camera_calibration_moge_tags,
        frame_field: str = MetaKeys.video_frames,
        output_intrinsics: bool = True,
        output_hfov: bool = True,
        output_vfov: bool = True,
        output_points: bool = True,
        output_depth: bool = True,
        output_mask: bool = True,
        frame_batch_size: int = 8,
        save_dir: str = None,
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param model_path: The path to the Moge-2 model.
        :param tag_field_name: The field name to store the tags. It's
            "camera_calibration_moge_tags" in default.
        :param frame_field: The field name where the video frames are stored.
        :param output_intrinsics: Determines whether to output camera intrinsics.
        :param output_hfov: Determines whether to output horizontal field of view.
        :param output_vfov: Determines whether to output vertical field of view.
        :param output_points: Determines whether to output point map
            in OpenCV camera coordinate system (x right, y down, z forward).
            For MoGe-2, the point map is in metric scale.
        :param output_depth: Determines whether to output depth maps.
        :param output_mask: Determines whether to output a binary mask for valid pixels.
        :param frame_batch_size: Number of frames to batch together for GPU
            inference. Larger values improve throughput but require more VRAM.
            Default: 8.
        :param save_dir: Directory to save large numpy arrays (depth, mask,
            points) as .npy files instead of storing them inline. When set,
            tag_dict stores file paths (strings) instead of numpy arrays,
            which avoids memory limit.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)

        self.model_key = prepare_model(model_type="moge", model_path=model_path)
        self.tag_field_name = tag_field_name
        self.frame_field = frame_field
        self.output_points = output_points
        self.output_depth = output_depth
        self.output_mask = output_mask
        self.output_intrinsics = output_intrinsics
        self.output_hfov = output_hfov
        self.output_vfov = output_vfov
        self.frame_batch_size = frame_batch_size
        self.save_dir = save_dir
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
        assert (
            self.output_points
            or self.output_depth
            or self.output_mask
            or self.output_intrinsics
            or self.output_hfov
            or self.output_vfov
        ), "At least one type of output info must be True."

    def _need_anything(self, sample) -> bool:
        """Whether this video still needs any requested outputs."""

        existing_tags = sample[Fields.meta].get(self.tag_field_name)
        if not existing_tags:
            return True

        if not isinstance(existing_tags[0], dict):
            raise ValueError(
                f"The existing field {self.tag_field_name} in sample[Fields.meta] should be a sequence of dict, but get {existing_tags}."
            )

        # Map: instance flag -> corresponding tag key
        requirements = {
            "output_intrinsics": CameraCalibrationKeys.intrinsics,
            "output_hfov": CameraCalibrationKeys.hfov,
            "output_vfov": CameraCalibrationKeys.vfov,
            "output_points": CameraCalibrationKeys.points,
            "output_depth": CameraCalibrationKeys.depth,
            "output_mask": CameraCalibrationKeys.mask,
        }

        for tag_dict in existing_tags:
            missing_any = any(getattr(self, flag, False) and key not in tag_dict for flag, key in requirements.items())
            if missing_any:
                return True

        return False

    def _save_numpy(self, arr: np.ndarray, prefix: str) -> str:
        """Save a numpy array to a .npy file and return the path."""
        filename = f"{prefix}_{uuid.uuid4().hex[:12]}.npy"
        path = os.path.join(self.save_dir, filename)
        np.save(path, arr)
        return path

    def _decode_frame(self, frame, device):
        """Decode a single frame to a (3, H, W) float32 tensor and return (tensor, H, W)."""
        if isinstance(frame, bytes):
            image_array = np.frombuffer(frame, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        else:
            image = cv2.imread(frame)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        tensor = torch.tensor(image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)
        return tensor, h, w

    def _process_video_frames_batched(self, frames, model, device, tag_dict):
        """Process all frames of one video using batched MoGe inference.

        MoGe v2 infer() natively supports (B, 3, H, W) batch input.
        Same-resolution frames (within a single clip) are stacked and
        inferred together for significantly better GPU utilization.
        """
        need_K = self.output_intrinsics and CameraCalibrationKeys.intrinsics not in tag_dict
        need_hfov = self.output_hfov and CameraCalibrationKeys.hfov not in tag_dict
        need_vfov = self.output_vfov and CameraCalibrationKeys.vfov not in tag_dict
        need_points = self.output_points and CameraCalibrationKeys.points not in tag_dict
        need_depth = self.output_depth and CameraCalibrationKeys.depth not in tag_dict
        need_mask = self.output_mask and CameraCalibrationKeys.mask not in tag_dict
        need_intrinsics_related = need_K or need_hfov or need_vfov

        # Step 1: Decode all frames and record their dimensions
        tensors = []
        heights = []
        widths = []
        for frame in frames:
            t, h, w = self._decode_frame(frame, device)
            tensors.append(t)
            heights.append(h)
            widths.append(w)

        num_frames = len(tensors)
        if num_frames == 0:
            return

        # Step 2: Check if all frames share the same resolution (typical for a single clip)
        all_same_size = all(h == heights[0] and w == widths[0] for h, w in zip(heights, widths))

        final_k_list = []
        final_hfov_list = []
        final_vfov_list = []
        final_points_list = []
        final_depth_list = []
        final_mask_list = []

        if all_same_size:
            # Batched inference path: stack frames and process in chunks
            height, width = heights[0], widths[0]
            for batch_start in range(0, num_frames, self.frame_batch_size):
                batch_end = min(batch_start + self.frame_batch_size, num_frames)
                batch_tensor = torch.stack(tensors[batch_start:batch_end], dim=0)  # (B, 3, H, W)

                output = model.infer(batch_tensor)

                batch_len = batch_end - batch_start
                for i in range(batch_len):
                    if need_intrinsics_related:
                        intr_np = output["intrinsics"][i].cpu().numpy()
                        if need_K:
                            final_k_list.append(
                                [
                                    [float(intr_np[0][0]) * width, 0, float(intr_np[0][2]) * width],
                                    [0, float(intr_np[1][1]) * height, float(intr_np[1][2]) * height],
                                    [0, 0, 1],
                                ]
                            )
                        if need_hfov:
                            final_hfov_list.append(float(2 * np.arctan(1 / 2 / intr_np[0][0])))
                        if need_vfov:
                            final_vfov_list.append(float(2 * np.arctan(1 / 2 / intr_np[1][1])))
                    if need_points:
                        final_points_list.append(output["points"][i].cpu().numpy())
                    if need_depth:
                        final_depth_list.append(output["depth"][i].cpu().numpy())
                    if need_mask:
                        final_mask_list.append(output["mask"][i].cpu().numpy())
        else:
            # Fallback: per-frame inference when frames have different sizes
            logger.debug("Frames have mixed resolutions, falling back to per-frame inference.")
            for i in range(num_frames):
                output = model.infer(tensors[i])
                height, width = heights[i], widths[i]

                if need_intrinsics_related:
                    intr_np = output["intrinsics"].cpu().numpy()
                    if need_K:
                        final_k_list.append(
                            [
                                [float(intr_np[0][0]) * width, 0, float(intr_np[0][2]) * width],
                                [0, float(intr_np[1][1]) * height, float(intr_np[1][2]) * height],
                                [0, 0, 1],
                            ]
                        )
                    if need_hfov:
                        final_hfov_list.append(float(2 * np.arctan(1 / 2 / intr_np[0][0])))
                    if need_vfov:
                        final_vfov_list.append(float(2 * np.arctan(1 / 2 / intr_np[1][1])))
                if need_points:
                    final_points_list.append(output["points"].cpu().numpy())
                if need_depth:
                    final_depth_list.append(output["depth"].cpu().numpy())
                if need_mask:
                    final_mask_list.append(output["mask"].cpu().numpy())

        # Step 3: Write results to tag_dict
        # For large numpy arrays (depth, mask, points), save to .npy files
        # when save_dir is configured, to avoid memory limit.
        if need_K:
            tag_dict[CameraCalibrationKeys.intrinsics] = final_k_list
        if need_hfov:
            tag_dict[CameraCalibrationKeys.hfov] = final_hfov_list
        if need_vfov:
            tag_dict[CameraCalibrationKeys.vfov] = final_vfov_list
        if need_points:
            if self.save_dir is not None:
                tag_dict[CameraCalibrationKeys.points] = [self._save_numpy(arr, "points") for arr in final_points_list]
            else:
                # Convert numpy arrays to standard Python lists for Ray/Arrow compatibility
                tag_dict[CameraCalibrationKeys.points] = [arr.tolist() for arr in final_points_list]
        if need_depth:
            if self.save_dir is not None:
                tag_dict[CameraCalibrationKeys.depth] = [self._save_numpy(arr, "depth") for arr in final_depth_list]
            else:
                # Convert numpy arrays to standard Python lists for Ray/Arrow compatibility
                tag_dict[CameraCalibrationKeys.depth] = [arr.tolist() for arr in final_depth_list]
        if need_mask:
            if self.save_dir is not None:
                tag_dict[CameraCalibrationKeys.mask] = [self._save_numpy(arr, "mask") for arr in final_mask_list]
            else:
                # Convert numpy arrays to standard Python lists for Ray/Arrow compatibility
                tag_dict[CameraCalibrationKeys.mask] = [arr.tolist() for arr in final_mask_list]

    def process_single(self, sample=None, rank=None):
        # there is no video in this sample
        if self.video_key not in sample or not sample[self.video_key]:
            return sample

        if sample.get(self.frame_field) is None:
            return sample

        if not self._need_anything(sample):
            return sample

        model = get_model(self.model_key, rank, self.use_cuda())

        videos_frames = sample[self.frame_field]
        num_videos = len(videos_frames)

        if self.tag_field_name not in sample[Fields.meta]:
            sample[Fields.meta][self.tag_field_name] = [{} for _ in range(num_videos)]

        tags_list = sample[Fields.meta][self.tag_field_name]

        if len(tags_list) != num_videos:
            raise ValueError(
                f"The field {self.tag_field_name} in sample[Fields.meta] "
                "should be a list of dict with the same length as the number of videos."
            )

        if rank is not None:
            device = f"cuda:{rank}" if self.use_cuda() else "cpu"
        else:
            device = "cuda" if self.use_cuda() else "cpu"

        for video_idx in range(num_videos):
            tag_dict = tags_list[video_idx]
            self._process_video_frames_batched(videos_frames[video_idx], model, device, tag_dict)

        return sample

import os

import cv2
import numpy as np
from pydantic import PositiveInt

from data_juicer.ops.load import load_ops
from data_juicer.utils.cache_utils import DATA_JUICER_ASSETS_CACHE
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.model_utils import get_model, prepare_model
from data_juicer.utils.video_perception_utils import (
    get_video_frame_groups,
    get_video_visualization_dir,
)

from ..base_op import OPERATORS, TAGGING_OPS, UNFORKABLE, Mapper
from ..op_fusion import LOADED_VIDEOS

OP_NAME = "video_normal_map_mapper"


@TAGGING_OPS.register_module(OP_NAME)
@UNFORKABLE.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
@LOADED_VIDEOS.register_module(OP_NAME)
class VideoNormalMapMapper(Mapper):
    """Generate normal maps for videos (with the Metric3D model)."""

    _accelerator = "cuda"

    def __init__(
        self,
        model_path: str = "onnx-community/metric3d-vit-large/onnx/model.onnx",
        if_save_visualization: bool = True,
        save_visualization_dir: str = DATA_JUICER_ASSETS_CACHE,
        frame_num: PositiveInt = 3,
        duration: float = 0,
        frame_dir: str = DATA_JUICER_ASSETS_CACHE,
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param model_path: The path to the Metric3D model.
        :param if_save_visualization: Whether to save visualization results.
        :param save_visualization_dir: The path for saving visualization results.
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
        providers = LazyLoader("onnxruntime", package_name="onnxruntime-gpu").get_available_providers()
        if self.use_cuda() and "CUDAExecutionProvider" not in providers:
            raise RuntimeError(
                "CUDAExecutionProvider is unavailable. Install a compatible onnxruntime-gpu build "
                "or set accelerator='cpu'."
            )

        self.model_key = prepare_model(model_type="normal_map_metric3d", model_path=model_path)
        self.if_save_visualization = if_save_visualization
        self.save_visualization_dir = save_visualization_dir or DATA_JUICER_ASSETS_CACHE
        self.frame_field = MetaKeys.video_frames
        self.tag_field_name = MetaKeys.video_normal_map_tags
        self.frame_num = frame_num
        self.duration = duration
        self.frame_dir = frame_dir or DATA_JUICER_ASSETS_CACHE
        self.input_size = (616, 1064)

        self.video_extract_frames_mapper_args = {
            "frame_sampling_method": "uniform",
            "frame_num": frame_num,
            "duration": duration,
            "frame_dir": self.frame_dir,
            "frame_field": self.frame_field,
            "video_key": self.video_key,
            "legacy_split_by_text_token": False,
            "num_proc": None,  # Disable multiprocessing to avoid nested process pool issue
            "auto_op_parallelism": False,  # Disable auto parallelism to avoid nested process pool issue
        }
        self.fused_ops = load_ops([{"video_extract_frames_mapper": self.video_extract_frames_mapper_args}])

    def prepare_input(self, rgb_image):

        input_size = self.input_size
        h, w = rgb_image.shape[:2]
        scale = min(input_size[0] / h, input_size[1] / w)
        rgb = cv2.resize(rgb_image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)

        padding = [123.675, 116.28, 103.53]
        h, w = rgb.shape[:2]
        pad_h = input_size[0] - h
        pad_w = input_size[1] - w
        pad_h_half = pad_h // 2
        pad_w_half = pad_w // 2
        rgb: np.ndarray = cv2.copyMakeBorder(
            rgb,
            pad_h_half,
            pad_h - pad_h_half,
            pad_w_half,
            pad_w - pad_w_half,
            cv2.BORDER_CONSTANT,
            value=padding,
        )
        pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]

        onnx_input = {
            "pixel_values": np.ascontiguousarray(np.transpose(rgb, (2, 0, 1))[None], dtype=np.float32),  # 1, 3, H, W
        }
        return onnx_input, pad_info

    def process_single(self, sample=None, rank=None):
        metadata = sample.setdefault(Fields.meta, {})
        if self.tag_field_name in metadata:
            return sample
        groups = get_video_frame_groups(sample, self.frame_field, self.video_key, self.fused_ops[0])
        result = {"pred_norm": [], "frame_paths": [], "video_indices": []}
        if not any(groups):
            metadata[self.tag_field_name] = result
            return sample
        session = get_model(model_key=self.model_key, rank=rank, use_cuda=self.use_cuda())
        for video_index, paths in enumerate(groups):
            for frame_index, path in enumerate(paths):
                image = cv2.imread(path)
                if image is None:
                    raise ValueError(f"Could not read video frame: {path}")
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                onnx_input, padding = self.prepare_input(rgb_image)
                normal = session.run(None, onnx_input)[1][0, :3]
                normal = normal[
                    :, padding[0] : self.input_size[0] - padding[1], padding[2] : self.input_size[1] - padding[3]
                ]
                normal = cv2.resize(
                    normal.transpose(1, 2, 0), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR
                )
                result["pred_norm"].append(normal.tolist())
                result["frame_paths"].append(path)
                result["video_indices"].append(video_index)
                if self.if_save_visualization:
                    directory = get_video_visualization_dir(self.save_visualization_dir, OP_NAME, paths, video_index)
                    visualization = ((normal + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
                    cv2.imwrite(
                        os.path.join(directory, f"vis_{frame_index}.jpg"),
                        cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR),
                    )
        metadata[self.tag_field_name] = result
        return sample

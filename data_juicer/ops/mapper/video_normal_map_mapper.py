import os

import cv2
import numpy as np
from pydantic import PositiveInt

import data_juicer
from data_juicer.ops.load import load_ops
from data_juicer.utils.cache_utils import DATA_JUICER_ASSETS_CACHE
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.mm_utils import SpecialTokens
from data_juicer.utils.model_utils import get_model, prepare_model

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
        LazyLoader.check_packages(["onnxruntime-gpu"])

        self.model_key = prepare_model(model_type="normal_map_metric3d", model_path=model_path)
        self.if_save_visualization = if_save_visualization
        self.save_visualization_dir = save_visualization_dir
        self.frame_field = MetaKeys.video_frames
        self.tag_field_name = MetaKeys.video_normal_map_tags
        self.frame_num = frame_num
        self.duration = duration
        self.frame_dir = frame_dir
        self.input_size = (616, 1064)

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

        # check if it's generated already
        if self.tag_field_name in sample[Fields.meta]:
            return sample

        # there is no video in this sample
        if (self.video_key not in sample or not sample[self.video_key]) and self.frame_field not in sample:
            sample[Fields.meta][self.tag_field_name] = {"pred_norm": []}
            return sample

        ort_session = get_model(model_key=self.model_key, rank=rank, use_cuda=self.use_cuda())

        if self.frame_field in sample:
            frames_path = sample[self.frame_field]
            video_name = os.path.basename(os.path.dirname(frames_path[0]))
        else:
            # load videos
            ds_list = [{"text": SpecialTokens.video, "videos": sample[self.video_key]}]

            dataset = data_juicer.core.data.NestedDataset.from_list(ds_list)
            dataset = self.fused_ops[0].run(dataset)

            temp_frame_name = os.path.splitext(os.path.basename(sample[self.video_key][0]))[0]
            frames_root = os.path.join(self.frame_dir, temp_frame_name)
            frame_names = os.listdir(frames_root)
            frames_path = sorted([os.path.join(frames_root, frame_name) for frame_name in frame_names])
            video_name = os.path.splitext(os.path.basename(sample[self.video_key][0]))[0]

        if self.if_save_visualization:
            os.makedirs(os.path.join(self.save_visualization_dir, video_name), exist_ok=True)

        final_pred_norm = []

        for temp_img_path_id, temp_img_path in enumerate(frames_path):
            rgb_image = cv2.imread(temp_img_path)[:, :, ::-1]  # BGR to RGB
            original_shape = rgb_image.shape[:2]
            onnx_input, pad_info = self.prepare_input(rgb_image)
            outputs = ort_session.run(None, onnx_input)

            # normal map
            normal = outputs[1].squeeze()
            normal = normal[
                :,
                pad_info[0] : self.input_size[0] - pad_info[1],
                pad_info[2] : self.input_size[1] - pad_info[3],
            ]
            normal = normal.transpose(1, 2, 0)
            normal = cv2.resize(normal, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LINEAR)
            final_pred_norm.append(normal)

            if self.if_save_visualization:
                normal_vis = (normal + 1.0) / 2.0
                normal_vis = (normal_vis * 255).clip(0, 255).astype(np.uint8)
                normal_vis = normal_vis[..., ::-1]

                cv2.imwrite(
                    os.path.join(self.save_visualization_dir, video_name, f"vis_{str(temp_img_path_id)}.jpg"),
                    normal_vis,
                )

        sample[Fields.meta][self.tag_field_name] = {}
        sample[Fields.meta][self.tag_field_name]["pred_norm"] = final_pred_norm

        return sample

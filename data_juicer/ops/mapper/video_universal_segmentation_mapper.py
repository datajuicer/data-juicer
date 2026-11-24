import os
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image
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

OP_NAME = "video_universal_segmentation_mapper"

torch = LazyLoader("torch")


@TAGGING_OPS.register_module(OP_NAME)
@UNFORKABLE.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
@LOADED_VIDEOS.register_module(OP_NAME)
class VideoUniversalSegmentationMapper(Mapper):
    """Generate semantic segmentation, instance segmentation,
    and panoptic segmentation information for videos
    (with the OneFormer model)."""

    _accelerator = "cuda"

    def __init__(
        self,
        model_path: str = "shi-labs/oneformer_ade20k_swin_large",
        if_output_semantic_segmentation: bool = True,
        if_output_instance_segmentation: bool = True,
        if_output_panoptic_segmentation: bool = True,
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

        :param model_path: The model path to the OneFormer model (huggingface).
        :param if_output_semantic_segmentation: Determines whether to
            output semantic segmentation inferred by OneFormer.
        :param if_output_instance_segmentation: Determines whether to
            output instance segmentation inferred by OneFormer.
        :param if_output_panoptic_segmentation: Determines whether to
            output panoptic segmentation inferred by OneFormer.
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
        LazyLoader.check_packages(["transformers"])

        self.model_key = prepare_model(model_type="huggingface", pretrained_model_name_or_path=model_path)
        self.if_save_visualization = if_save_visualization
        self.save_visualization_dir = save_visualization_dir or DATA_JUICER_ASSETS_CACHE
        self.frame_field = MetaKeys.video_frames
        self.tag_field_name = MetaKeys.video_universal_segmentation_tags
        self.frame_num = frame_num
        self.duration = duration
        self.frame_dir = frame_dir or DATA_JUICER_ASSETS_CACHE
        self.if_output_semantic_segmentation = if_output_semantic_segmentation
        self.if_output_instance_segmentation = if_output_instance_segmentation
        self.if_output_panoptic_segmentation = if_output_panoptic_segmentation

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

    def visualization_semantic(self, predicted_semantic_map, model, image_pil):

        predicted_semantic_map = predicted_semantic_map.cpu().numpy()
        unique_classes = np.unique(predicted_semantic_map)

        num_classes = len(model.config.id2label)
        rng = np.random.RandomState(42)
        color_palette = rng.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)

        img_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        seg_bgr = np.zeros_like(img_bgr)

        text_labels = []
        min_area_threshold = img_bgr.shape[0] * img_bgr.shape[1] * 0.005

        for class_id in unique_classes:
            seg_bgr[predicted_semantic_map == class_id] = color_palette[class_id]

            mask = np.uint8(predicted_semantic_map == class_id) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            label_name = model.config.id2label[class_id]

            for contour in contours:
                if cv2.contourArea(contour) > min_area_threshold:
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        text_labels.append((label_name, cx, cy))

        alpha = 0.6
        overlay_bgr = cv2.addWeighted(img_bgr, 1 - alpha, seg_bgr, alpha, 0)

        for name, cx, cy in text_labels:
            cv2.putText(overlay_bgr, name, (cx - 15, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(
                overlay_bgr, name, (cx - 15, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
            )

        return overlay_bgr

    def visualization_instance_or_panoptic(self, predicted_results, model, image_pil):
        segmentation_map = predicted_results["segmentation"].cpu().numpy()
        segments_info = predicted_results["segments_info"]

        image_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        seg_image_bgr = np.zeros_like(image_bgr, dtype=np.uint8)
        label_counters = defaultdict(int)

        rng = np.random.RandomState(42)
        text_labels = []

        for segment in segments_info:
            segment_id = segment["id"]
            label_id = segment["label_id"]
            label_name = model.config.id2label[label_id]

            label_counters[label_name] += 1
            instance_name = f"{label_name}-{label_counters[label_name]}"

            color_bgr = rng.randint(0, 255, size=(3,)).tolist()
            mask = segmentation_map == segment_id
            seg_image_bgr[mask] = color_bgr

            y_coords, x_coords = np.where(mask)
            if len(y_coords) > 0:
                center_x = int(np.mean(x_coords))
                center_y = int(np.mean(y_coords))
                text_labels.append((instance_name, center_x, center_y))

        alpha = 0.6
        overlay_bgr = cv2.addWeighted(seg_image_bgr, alpha, image_bgr, 1 - alpha, 0)

        for name, cx, cy in text_labels:
            cv2.putText(overlay_bgr, name, (cx - 15, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(
                overlay_bgr, name, (cx - 15, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
            )

        return overlay_bgr

    def process_single(self, sample=None, rank=None):
        metadata = sample.setdefault(Fields.meta, {})
        if self.tag_field_name in metadata:
            return sample
        tasks = [
            task
            for task, enabled in [
                ("semantic", self.if_output_semantic_segmentation),
                ("instance", self.if_output_instance_segmentation),
                ("panoptic", self.if_output_panoptic_segmentation),
            ]
            if enabled
        ]
        result = {
            "semantic_segmentation_map": [],
            "instance_segmentation_map": [],
            "instance_segmentation_info": [],
            "panoptic_segmentation_map": [],
            "panoptic_segmentation_info": [],
            "frame_paths": [],
            "video_indices": [],
        }
        if not tasks:
            metadata[self.tag_field_name] = result
            return sample
        groups = get_video_frame_groups(sample, self.frame_field, self.video_key, self.fused_ops[0])
        if not any(groups):
            metadata[self.tag_field_name] = result
            return sample
        model, processor = get_model(model_key=self.model_key, rank=rank, use_cuda=self.use_cuda())
        device = model.device
        for video_index, paths in enumerate(groups):
            for frame_index, path in enumerate(paths):
                try:
                    with Image.open(path) as source:
                        image = source.convert("RGB")
                except (OSError, ValueError) as error:
                    raise ValueError(f"Could not read video frame: {path}") from error
                for task in tasks:
                    inputs = processor(images=image, task_inputs=[task], return_tensors="pt").to(device)
                    with torch.no_grad():
                        outputs = model(**inputs)
                        prediction = getattr(processor, f"post_process_{task}_segmentation")(
                            outputs, target_sizes=[image.size[::-1]]
                        )[0]
                    if task == "semantic":
                        result["semantic_segmentation_map"].append(prediction.detach().cpu().tolist())
                    else:
                        result[f"{task}_segmentation_map"].append(prediction["segmentation"].detach().cpu().tolist())
                        result[f"{task}_segmentation_info"].append(prediction["segments_info"])
                    if self.if_save_visualization:
                        directory = os.path.join(
                            get_video_visualization_dir(self.save_visualization_dir, OP_NAME, paths, video_index), task
                        )
                        os.makedirs(directory, exist_ok=True)
                        visualization = (
                            self.visualization_semantic(prediction, model, image)
                            if task == "semantic"
                            else self.visualization_instance_or_panoptic(prediction, model, image)
                        )
                        cv2.imwrite(os.path.join(directory, f"vis_{frame_index}.jpg"), visualization)
                result["frame_paths"].append(path)
                result["video_indices"].append(video_index)
        metadata[self.tag_field_name] = result
        return sample

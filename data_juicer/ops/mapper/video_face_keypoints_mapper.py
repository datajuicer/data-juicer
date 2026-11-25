import os

import numpy as np
from pydantic import PositiveInt

from data_juicer.ops.load import load_ops
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

cv2 = LazyLoader("cv2", "opencv-contrib-python")

OP_NAME = "video_face_keypoints_mapper"

torch = LazyLoader("torch")


@TAGGING_OPS.register_module(OP_NAME)
@UNFORKABLE.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
@LOADED_VIDEOS.register_module(OP_NAME)
class VideoFaceKeypointsMapper(Mapper):
    """Detect face keypoints (98 points) on the video."""

    _accelerator = "cuda"

    def __init__(
        self,
        ldeq_model_path: str = "final.pth.tar",
        if_save_visualization: bool = False,
        save_visualization_dir: str = DATA_JUICER_ASSETS_CACHE,
        frame_num: PositiveInt = 3,
        duration: float = 0,
        frame_dir: str = DATA_JUICER_ASSETS_CACHE,
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param ldeq_model_path: The path to the LDEQ model.
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
        LazyLoader.check_packages(["insightface", "torchinfo"])

        self.model_key = prepare_model(model_type="face_keypoints_ldeq", model_path=ldeq_model_path)
        self.if_save_visualization = if_save_visualization
        self.save_visualization_dir = save_visualization_dir
        self.frame_field = MetaKeys.video_frames
        self.tag_field_name = MetaKeys.video_face_keypoints_tags
        self.frame_num = frame_num
        self.duration = duration
        self.frame_dir = frame_dir

        self.video_extract_frames_mapper_args = {
            "frame_sampling_method": "uniform",
            "frame_num": frame_num,
            "duration": duration,
            "frame_dir": frame_dir,
            "frame_field": MetaKeys.video_frames,
            "video_key": self.video_key,
            "legacy_split_by_text_token": False,
            "num_proc": None,  # Disable multiprocessing to avoid nested process pool issue
            "auto_op_parallelism": False,  # Disable auto parallelism to avoid nested process pool issue
        }
        self.fused_ops = load_ops([{"video_extract_frames_mapper": self.video_extract_frames_mapper_args}])

    def preprocess(self, face_crop):
        img = face_crop.transpose(2, 0, 1) / 255.0

        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        img = (img - mean) / std

        return torch.from_numpy(img).float().unsqueeze(0)

    def crop_and_pad(self, image, bbox, target_size=256, padding_ratio=0.05):

        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

        # 1. Side length of the square (the maximum of the width and height) and add padding.
        side = max(w, h)
        side = int(side * (1 + padding_ratio))

        # 2. Calculate the new coordinates
        new_x1 = center_x - side // 2
        new_y1 = center_y - side // 2
        new_x2 = new_x1 + side
        new_y2 = new_y1 + side

        # 3. Handling cases that exceed the original image boundaries
        img_h, img_w = image.shape[:2]

        pad_top = max(0, -new_y1)
        pad_bottom = max(0, new_y2 - img_h)
        pad_left = max(0, -new_x1)
        pad_right = max(0, new_x2 - img_w)

        crop_x1 = max(0, new_x1)
        crop_y1 = max(0, new_y1)
        crop_x2 = min(img_w, new_x2)
        crop_y2 = min(img_h, new_y2)

        crop = image[crop_y1:crop_y2, crop_x1:crop_x2]

        # 4. If it goes out of bounds, fill with black borders.
        if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
            crop = cv2.copyMakeBorder(
                crop, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )

        final_crop = cv2.resize(crop, (target_size, target_size))

        return final_crop, (new_x1, new_y1), side

    def draw_landmarks_on_image(self, image, landmarks_list, color=(0, 255, 0)):
        vis_img = image.copy()

        for kpts in landmarks_list:
            for i in range(kpts.shape[0]):
                x, y = int(kpts[i][0]), int(kpts[i][1])
                cv2.circle(vis_img, (x, y), 2, color, -1)

        return vis_img

    def _get_output_features(self, input_features):
        if input_features is None:
            return None
        fields = {
            "face_keypoints": nested_video_feature("float64", 4),
            "face_bboxes": nested_video_feature("int64", 3),
            "frame_paths": nested_video_feature("string"),
            "video_indices": nested_video_feature("int64"),
        }
        return get_video_output_features(input_features, self.tag_field_name, fields)

    def process_single(self, sample=None, rank=None):
        meta = sample.setdefault(Fields.meta, {})
        if self.tag_field_name in meta:
            return sample
        groups = get_video_frame_groups(sample, self.frame_field, self.video_key, self.fused_ops[0])
        tags = {"face_keypoints": [], "face_bboxes": [], "frame_paths": [], "video_indices": []}
        if not any(groups):
            meta[self.tag_field_name] = tags
            return sample

        ldeq_model, detector, train_args = get_model(model_key=self.model_key, rank=rank, use_cuda=self.use_cuda())
        device = (
            f"cuda:{rank % torch.cuda.device_count()}"
            if self.use_cuda() and rank is not None
            else ("cuda" if self.use_cuda() else "cpu")
        )
        for video_index, paths in enumerate(groups):
            if not paths:
                continue
            if self.if_save_visualization:
                visualization_dir = get_video_visualization_dir(
                    self.save_visualization_dir, OP_NAME, paths, video_index
                )
            for frame_index, path in enumerate(paths):
                img = cv2.imread(path)
                if img is None:
                    raise ValueError(f"Could not read video frame: {path}")
                keypoints, bboxes = [], []
                for face in detector.get(img):
                    bbox = face.bbox.astype(int)
                    bboxes.append(bbox.tolist())
                    crop, (new_x1, new_y1), side = self.crop_and_pad(img, bbox)
                    input_tensor = self.preprocess(crop).to(device)
                    with torch.no_grad():
                        output = ldeq_model(
                            input_tensor,
                            mode=train_args.model_mode,
                            args=train_args,
                            z0=torch.zeros(1, train_args.z_width, train_args.heatmap_size, train_args.heatmap_size).to(
                                device
                            ),
                        )
                    prediction = output["keypoints"][0].cpu().numpy()
                    keypoints.append(prediction * [side, side] + [new_x1, new_y1])
                tags["face_keypoints"].append([points.tolist() for points in keypoints])
                tags["face_bboxes"].append(bboxes)
                tags["frame_paths"].append(path)
                tags["video_indices"].append(video_index)
                if self.if_save_visualization:
                    visualization = self.draw_landmarks_on_image(img, keypoints)
                    cv2.imwrite(os.path.join(visualization_dir, f"vis_{frame_index}.jpg"), visualization)
        meta[self.tag_field_name] = tags
        return sample

import importlib
import os
import subprocess
import sys

from loguru import logger
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

OP_NAME = "video_animal_pose_mapper"


@TAGGING_OPS.register_module(OP_NAME)
@UNFORKABLE.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
@LOADED_VIDEOS.register_module(OP_NAME)
class VideoAnimalPoseMapper(Mapper):
    """Detect quadruped animal pose on the video."""

    _accelerator = "cuda"

    def __init__(
        self,
        vitpose_model_path: str = "apt36k.pth",
        vitpose_config: str = "configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/apt36k/ViTPose_huge_apt36k_256x192.py",
        yoloe_model_path: str = "yoloe-26x-seg.pt",
        animal_class: list = [],
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

        :param vitpose_model_path: The path to the ViTPose model.
        :param vitpose_config: Please select the appropriate model configuration.
        :param yoloe_model_path: The path to the YOLOE model.
        :param animal_class: Specifies the quadruped animal categories to be
            detected. If no value is input, the default list will be used.
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
        LazyLoader.check_packages(["ultralytics"])
        self._install_required_packages()

        vitpose_repo_path = os.path.join(DATA_JUICER_ASSETS_CACHE, "ViTPose")
        if not os.path.exists(vitpose_repo_path):
            subprocess.run(
                [
                    "git",
                    "clone",
                    "https://github.com/ViTAE-Transformer/ViTPose.git",
                    vitpose_repo_path,
                ],
                check=True,
            )

        try:
            importlib.import_module("mmpose")
        except Exception:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "chumpy", "--no-build-isolation", "--no-deps"], check=True
            )
            subprocess.run(["pip", "install", "-e", vitpose_repo_path], check=True)
        subprocess.run(["pip", "install", "numpy==1.26.4"], check=True)

        from mmpose.apis import inference_top_down_pose_model

        self.inference_top_down_pose_model = inference_top_down_pose_model

        self.model_key = prepare_model(
            model_type="vitpose_animal_pose", model_path=vitpose_model_path, vitpose_config=vitpose_config
        )
        self.yolo_model_key = prepare_model(model_type="yolo", model_path=yoloe_model_path)
        self.if_save_visualization = if_save_visualization
        self.save_visualization_dir = save_visualization_dir
        self.frame_field = MetaKeys.video_frames
        self.tag_field_name = MetaKeys.video_animal_pose_tags
        self.frame_num = frame_num
        self.duration = duration
        self.frame_dir = frame_dir

        self.skeleton = [
            [0, 2],
            [1, 2],
            [2, 3],
            [3, 5],
            [5, 6],
            [6, 7],
            [3, 8],
            [8, 9],
            [9, 10],
            [3, 4],
            [4, 11],
            [11, 12],
            [12, 13],
            [4, 14],
            [14, 15],
            [15, 16],
        ]

        if isinstance(animal_class, list) and len(animal_class) == 0:
            self.animal_class = [
                "bear",
                "cat",
                "cougar",
                "cow",
                "deer",
                "dog",
                "elephant",
                "goat",
                "hippo",
                "horse",
                "moose",
                "panther",
                "pig",
                "rabbit",
                "rhino",
                "sheep",
                "tiger",
                "wolf",
                "zebra",
            ]
        elif isinstance(animal_class, list):
            self.animal_class = animal_class
        else:
            raise ValueError("The 'animal_class' must be in list format.")

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

    def _install_required_packages(self):
        subprocess.run(["pip", "install", "numpy==1.26.4"], check=True)
        try:
            importlib.import_module("mim")
        except ImportError:
            logger.info("Installing openmim...")
            try:
                subprocess.run(["pip", "install", "openmim"], check=True)
            except Exception:
                raise ValueError(
                    "Failed to install openmim, please refer to the documentation at "
                    "https://github.com/open-mmlab/mim/blob/main/docs/en/installation.md for installation instructions."
                )

        try:
            importlib.import_module("mmcv")
        except ImportError:
            logger.info("Installing mmcv using mim...")
            try:
                subprocess.run(["mim", "install", "mmcv==1.3.9", "--no-build-isolation"], check=True)
            except Exception:
                raise ValueError(
                    "Failed to install mmcv, please refer to the documentation at "
                    "https://mmdetection.readthedocs.io/en/latest/get_started.html#installation for installation instructions."
                )

    def draw_pose(self, img, keypoints, scores, threshold=0.3):

        for i in range(len(keypoints)):
            x, y = int(keypoints[i][0]), int(keypoints[i][1])
            score = scores[i]
            if score > threshold:
                cv2.circle(img, (x, y), 5, (0, 255, 0), -1)

        for p1, p2 in self.skeleton:
            if scores[p1] > threshold and scores[p2] > threshold:
                cv2.line(
                    img,
                    (int(keypoints[p1][0]), int(keypoints[p1][1])),
                    (int(keypoints[p2][0]), int(keypoints[p2][1])),
                    (255, 0, 0),
                    2,
                )
        return img

    def _get_output_features(self, input_features):
        if input_features is None:
            return None
        fields = {
            "pose_list": nested_video_feature("float64", 4),
            "pose_score_list": nested_video_feature("float64", 3),
            "animal_bboxes": nested_video_feature("float64", 3),
            "frame_paths": nested_video_feature("string"),
            "video_indices": nested_video_feature("int64"),
        }
        return get_video_output_features(input_features, self.tag_field_name, fields)

    def process_single(self, sample=None, rank=None):
        meta = sample.setdefault(Fields.meta, {})
        if self.tag_field_name in meta:
            return sample
        groups = get_video_frame_groups(sample, self.frame_field, self.video_key, self.fused_ops[0])
        tags = {"pose_list": [], "pose_score_list": [], "animal_bboxes": [], "frame_paths": [], "video_indices": []}
        if not any(groups):
            meta[self.tag_field_name] = tags
            return sample

        pose_inferencer = get_model(model_key=self.model_key, rank=rank, use_cuda=self.use_cuda())
        yolo_model = get_model(model_key=self.yolo_model_key, rank=rank, use_cuda=self.use_cuda())
        yolo_model.set_classes(self.animal_class, yolo_model.get_text_pe(self.animal_class))

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
                detections = yolo_model.predict(img, verbose=False)[0]
                bboxes = []
                for box in detections.boxes:
                    coordinates = box.xyxy[0].cpu().numpy().tolist()
                    bboxes.append({"bbox": coordinates})
                poses, scores = [], []
                if bboxes:
                    pose_results, _ = self.inference_top_down_pose_model(pose_inferencer, img, bboxes, format="xyxy")
                    for result in pose_results:
                        keypoints = result["keypoints"][:, :2]
                        confidence = result["keypoints"][:, 2]
                        poses.append(keypoints.tolist())
                        scores.append(confidence.tolist())
                        if self.if_save_visualization:
                            bbox = result["bbox"]
                            cv2.rectangle(
                                img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2
                            )
                            img = self.draw_pose(img, keypoints, confidence)
                tags["pose_list"].append(poses)
                tags["pose_score_list"].append(scores)
                tags["animal_bboxes"].append([box["bbox"] for box in bboxes])
                tags["frame_paths"].append(path)
                tags["video_indices"].append(video_index)
                if self.if_save_visualization:
                    cv2.imwrite(os.path.join(visualization_dir, f"vis_{frame_index}.jpg"), img)
        meta[self.tag_field_name] = tags
        return sample

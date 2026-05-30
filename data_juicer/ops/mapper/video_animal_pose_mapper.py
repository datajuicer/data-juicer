import importlib
import os
import subprocess
import sys

import cv2
from loguru import logger
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
            "frame_key": MetaKeys.video_frames,
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

    def process_single(self, sample=None, rank=None):

        # check if it's generated already
        if self.tag_field_name in sample[Fields.meta]:
            return sample

        # there is no video in this sample
        if (self.video_key not in sample or not sample[self.video_key]) and self.frame_field not in sample:
            sample[Fields.meta][self.tag_field_name] = {"pose_list": [], "pose_score_list": [], "animal_bboxes": []}
            return sample

        pose_inferencer = get_model(model_key=self.model_key, rank=rank, use_cuda=self.use_cuda())
        yolo_model = get_model(model_key=self.yolo_model_key, rank=rank, use_cuda=self.use_cuda())
        yolo_model.set_classes(self.animal_class, yolo_model.get_text_pe(self.animal_class))

        if self.frame_field in sample:
            frames_path = sample[self.frame_field]
            video_name = frames_path[0].split("/")[-2]
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

        final_pose_list = []
        final_pose_score_list = []
        final_bboxes = []

        for temp_img_path_id, temp_img_path in enumerate(frames_path):
            img = cv2.imread(temp_img_path)

            temp_results = yolo_model.predict(img, verbose=False)[0]
            bboxes = []
            bboxes_only_num = []
            for box in temp_results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                bboxes.append({"bbox": [x1, y1, x2, y2]})
                bboxes_only_num.append([x1, y1, x2, y2])

            if not bboxes:
                final_pose_list.append([])
                final_pose_score_list.append([])
                final_bboxes.append([])
                continue

            pose_results, _ = self.inference_top_down_pose_model(pose_inferencer, img, bboxes, format="xyxy")

            temp_pose_list = []
            temp_score_list = []

            for res in pose_results:
                keypoints = res["keypoints"][:, :2]
                scores = res["keypoints"][:, 2]

                temp_pose_list.append(keypoints)
                temp_score_list.append(scores)

                if self.if_save_visualization:
                    cv2.rectangle(
                        img,
                        (int(res["bbox"][0]), int(res["bbox"][1])),
                        (int(res["bbox"][2]), int(res["bbox"][3])),
                        (255, 0, 0),
                        2,
                    )
                    img = self.draw_pose(img, keypoints, scores)

            if self.if_save_visualization:
                cv2.imwrite(
                    os.path.join(self.save_visualization_dir, video_name, f"vis_{str(temp_img_path_id)}.jpg"), img
                )

            final_pose_list.append(temp_pose_list)
            final_pose_score_list.append(temp_score_list)
            final_bboxes.append(bboxes_only_num)

        sample[Fields.meta][self.tag_field_name] = {}
        sample[Fields.meta][self.tag_field_name]["pose_list"] = final_pose_list
        sample[Fields.meta][self.tag_field_name]["pose_score_list"] = final_pose_score_list
        sample[Fields.meta][self.tag_field_name]["animal_bboxes"] = final_bboxes

        return sample

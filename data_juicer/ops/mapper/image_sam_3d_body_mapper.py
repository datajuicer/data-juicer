import importlib
import os
import subprocess
import sys

import numpy as np
from loguru import logger

from data_juicer.utils.cache_utils import DATA_JUICER_ASSETS_CACHE
from data_juicer.utils.constant import Fields
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.model_utils import get_model, prepare_model
from data_juicer.utils.ray_utils import is_ray_mode

from ..base_op import OPERATORS, TAGGING_OPS, Mapper

OP_NAME = "image_sam_3d_body_mapper"


cv2 = LazyLoader("cv2", "opencv-python")


@TAGGING_OPS.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
class ImageSAM3DBodyMapper(Mapper):
    """SAM 3D Body (3DB) is a promptable model for single-image full-body 3D human mesh recovery (HMR)."""

    _accelerator = "cuda"

    def __init__(
        self,
        checkpoint_path: str = "",
        detector_name: str = "vitdet",
        segmentor_name: str = "sam2",
        fov_name: str = "moge2",
        mhr_path: str = "",
        detector_path: str = "",
        segmentor_path: str = "",
        fov_path: str = "",
        bbox_thresh: float = 0.8,
        use_mask: bool = False,
        visualization_dir: str = None,
        tag_field_name: str = "sam_3d_body_data",
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param checkpoint_path: Path to SAM 3D Body model checkpoint.
        :param mhr_path: Path to MoHR/assets folder (or set SAM3D_mhr_path).
        :param detector_path: Path to human detection model folder (or set SAM3D_DETECTOR_PATH).
        :param segmentor_path: Path to human segmentation model folder (or set SAM3D_SEGMENTOR_PATH).
        :param fov_path: Path to fov estimation model folder (or set SAM3D_FOV_PATH).
        :param detector_name: Human detection model for demo (Default `vitdet`, add your favorite detector if needed).
        :param segmentor_name: Human segmentation model for demo (Default `sam2`, add your favorite segmentor if needed).
        :param fov_name: FOV estimation model for demo (Default `moge2`, add your favorite fov estimator if needed).
        :param bbox_thresh: Bounding box detection threshold.
        :param use_mask:Use mask-conditioned prediction (segmentation mask is automatically generated from bbox).
        :param visualization_dir: Directory to save visualization results. If None, no visualization will be saved.
        :param tag_field_name: Field name for storing the results.
        """

        super().__init__(*args, **kwargs)
        self.checkpoint_path = checkpoint_path
        self.mhr_path = mhr_path
        self.detector_path = detector_path
        self.segmentor_path = segmentor_path
        self.fov_path = fov_path
        self.detector_name = detector_name
        self.segmentor_name = segmentor_name
        self.fov_name = fov_name
        self.bbox_thresh = bbox_thresh
        self.use_mask = use_mask
        self.visualization_dir = visualization_dir

        self._install_required_packages()

        self.tag_field_name = tag_field_name

        sam_3d_body_repo_path = os.path.join(DATA_JUICER_ASSETS_CACHE, "sam-3d-body")
        if not os.path.exists(sam_3d_body_repo_path):
            logger.info("Cloning SAM 3D Body repo...")
            subprocess.run(
                [
                    "git",
                    "clone",
                    "https://github.com/facebookresearch/sam-3d-body.git",
                    sam_3d_body_repo_path,
                ],
                check=True,
            )
        self._sam_3d_body_repo_path = sam_3d_body_repo_path

        if self.num_proc > 1 or not is_ray_mode():
            logger.warning(
                "num_proc > 1 may not supported for SAM 3D Body in standalone mode "
                "Please set num_proc=1 or use ray mode."
            )

        self.model_key = prepare_model(
            model_type="sam_3d_body",
            checkpoint_path=checkpoint_path,
            mhr_path=mhr_path,
            detector_path=detector_path,
            segmentor_path=segmentor_path,
            fov_path=fov_path,
            detector_name=detector_name,
            segmentor_name=segmentor_name,
            fov_name=fov_name,
        )

    def _install_required_packages(self):
        LazyLoader.check_packages(
            [
                "pytorch-lightning",
                "pyrender",
                "opencv-python",
                "yacs",
                "scikit-image",
                "einops",
                "timm",
                "dill",
                "pandas",
                "rich",
                "hydra-core",
                "hydra-submitit-launcher",
                "hydra-colorlog",
                "pyrootutils",
                "webdataset",
                "chump",
                "networkx==3.2.1",
                "roma",
                "joblib",
                "seaborn",
                "wandb",
                "appdirs",
                "appnope",
                "ffmpeg",
                "cython",
                "jsonlines",
                "pytest",
                "xtcocotools",
                "loguru",
                "optree",
                "fvcore",
                "black",
                "pycocotools",
                "tensorboard",
                "huggingface_hub",
            ]
        )

        try:
            importlib.import_module("detectron2")
        except ImportError:
            logger.info("Installing detectron2...")
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "git+https://github.com/facebookresearch/detectron2.git@a1ce2f9",
                    "--no-build-isolation",
                    "--no-deps",
                ],
                check=True,
                capture_output=True,
            )

        if self.fov_name.lower().startswith("moge"):
            try:
                importlib.import_module("moge")
            except ImportError:
                logger.info("Installing MoGe...")
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "git+https://github.com/microsoft/MoGe.git"],
                    check=True,
                    capture_output=True,
                )

    def process_single(self, sample=None, rank=None):
        # check if it's generated already
        if self.tag_field_name in sample[Fields.meta]:
            return sample

        # there is no video in this sample
        if self.image_key not in sample or not sample[self.image_key]:
            sample[Fields.meta][self.tag_field_name] = []
            return sample

        estimator = get_model(model_key=self.model_key, rank=rank, use_cuda=self.use_cuda())

        outputs = []
        for image_path in sample[self.image_key]:
            output = estimator.process_one_image(
                image_path,
                bbox_thr=self.bbox_thresh,
                use_mask=self.use_mask,
            )
            outputs.append(output)

            if self.visualization_dir:
                try:
                    sys.path.insert(0, self._sam_3d_body_repo_path)

                    module_path = f"{self._sam_3d_body_repo_path}/tools/vis_utils.py"
                    spec = importlib.util.spec_from_file_location("vis_utils", module_path)
                    if spec is None:
                        raise ImportError(f"Could not load spec from {module_path}")
                    vis_utils = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(vis_utils)
                    img_name = os.path.basename(image_path)
                    os.makedirs(self.visualization_dir, exist_ok=True)
                    vis_path = os.path.join(self.visualization_dir, os.path.splitext(img_name)[0] + "_vis.jpg")
                    img = cv2.imread(image_path)
                    try:
                        rend_img = vis_utils.visualize_sample_together(img, output, estimator.faces)
                    except (ImportError, OSError) as e:
                        if "EGL" in str(e):
                            raise RuntimeError(
                                "Visualization requires EGL for offscreen rendering, but EGL "
                                "library was not found. To fix this:\n"
                                "  - On Ubuntu/Debian: apt-get install libegl1-mesa libegl1-mesa-dev\n"
                                "  - On headless servers: also install libgl1-mesa-dri\n"
                                "  - Or disable visualization by not setting visualization_dir"
                            ) from e
                        raise
                    cv2.imwrite(
                        vis_path,
                        rend_img.astype(np.uint8),
                    )
                finally:
                    if self._sam_3d_body_repo_path in sys.path:
                        sys.path.remove(self._sam_3d_body_repo_path)

        sample[Fields.meta][self.tag_field_name] = outputs

        return sample

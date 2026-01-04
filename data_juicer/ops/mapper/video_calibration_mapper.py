import argparse
import logging
import os
import subprocess
import sys
from typing import Optional

import numpy as np

from data_juicer.utils.cache_utils import DATA_JUICER_ASSETS_CACHE
from data_juicer.utils.constant import Fields
from data_juicer.utils.lazy_loader import LazyLoader

from ..base_op import OPERATORS, Mapper

logger = logging.getLogger(__name__)

torch = LazyLoader("torch")
cv2 = LazyLoader("cv2", "opencv-python")

OP_NAME = "video_calibration_mapper"


@OPERATORS.register_module(OP_NAME)
class VideoCalibrationMapper(Mapper):
    """
    Extract camera intrinsics from videos using DroidCalib.
    """

    _accelerator = "cuda"

    def __init__(
        self,
        weights_path: Optional[str] = None,
        image_size: list = [384, 512],
        stride: int = 2,
        max_frames: int = 300,
        buffer: int = 1024,
        beta: float = 0.3,
        filter_thresh: float = 2.4,
        warmup: int = 8,
        keyframe_thresh: float = 4.0,
        frontend_thresh: float = 16.0,
        frontend_window: int = 25,
        frontend_radius: int = 2,
        frontend_nms: int = 1,
        backend_thresh: float = 22.0,
        backend_radius: int = 2,
        backend_nms: int = 3,
        upsample: bool = False,
        disable_vis: bool = True,
        verbose: bool = False,
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param weights_path: Path to the model weights.
        :param image_size: Target image size [height, width].
        :param stride: Frame stride.
        :param max_frames: Maximum number of frames to process.
        :param buffer: Buffer size for Droid.
        :param beta: Weight for translation / rotation components of flow.
        :param filter_thresh: Motion threshold before considering new keyframe.
        :param warmup: Number of warmup frames.
        :param keyframe_thresh: Threshold to create a new keyframe.
        :param frontend_thresh: Add edges between frames within this distance.
        :param frontend_window: Frontend optimization window.
        :param frontend_radius: Force edges between frames within radius.
        :param frontend_nms: Non-maximal suppression of edges.
        :param backend_thresh: Backend threshold.
        :param backend_radius: Backend radius.
        :param backend_nms: Backend NMS.
        :param upsample: Whether to upsample.
        :param disable_vis: Whether to disable visualization.
        """
        super().__init__(*args, **kwargs)

        self.verbose = verbose
        self._deps_ready = False

        self.droid_calib_path = os.path.join(DATA_JUICER_ASSETS_CACHE, "DroidCalib")

        # Auto-download logic
        if not os.path.exists(self.droid_calib_path):
            if self.verbose:
                print(f"Downloading DroidCalib to {self.droid_calib_path}...")
            subprocess.run(
                ["git", "clone", "https://github.com/boschresearch/DroidCalib.git", self.droid_calib_path],
                check=True,
            )

        self.droid_slam_path = os.path.join(self.droid_calib_path, "droid_slam")

        # Dynamic path append (best-effort; also repeated at runtime in workers)
        if os.path.exists(self.droid_slam_path) and self.droid_slam_path not in sys.path:
            sys.path.append(self.droid_slam_path)

        self.weights_path = weights_path
        if self.weights_path is None:
            self.weights_path = os.path.join(self.droid_calib_path, "droidcalib.pth")

        self.image_size = image_size
        self.stride = stride
        self.max_frames = max_frames

        # Droid args
        self.droid_args = argparse.Namespace()
        self.droid_args.weights = self.weights_path
        self.droid_args.buffer = buffer
        self.droid_args.image_size = image_size
        self.droid_args.beta = beta
        self.droid_args.filter_thresh = filter_thresh
        self.droid_args.warmup = warmup
        self.droid_args.keyframe_thresh = keyframe_thresh
        self.droid_args.frontend_thresh = frontend_thresh
        self.droid_args.frontend_window = frontend_window
        self.droid_args.frontend_radius = frontend_radius
        self.droid_args.frontend_nms = frontend_nms
        self.droid_args.backend_thresh = backend_thresh
        self.droid_args.backend_radius = backend_radius
        self.droid_args.backend_nms = backend_nms
        self.droid_args.upsample = upsample
        self.droid_args.disable_vis = disable_vis
        self.droid_args.stereo = False
        self.droid_args.camera_model = "pinhole"  # Default to pinhole
        self.droid_args.opt_intr = True

    def _ensure_droidcalib_ready(self) -> bool:
        """Ensure DroidCalib is importable in the *current process*.

        This matters because `Dataset.map(num_proc>1)` may execute in child
        processes where `sys.path` changes from `__init__` are not present.
        """

        if self._deps_ready:
            return True

        if not os.path.exists(self.droid_calib_path):
            # In normal flow this should have been cloned in __init__.
            # If multiprocessing spawned before __init__ ran, try again.
            try:
                if self.verbose:
                    print(f"Downloading DroidCalib to {self.droid_calib_path}...")
                subprocess.run(
                    ["git", "clone", "https://github.com/boschresearch/DroidCalib.git", self.droid_calib_path],
                    check=True,
                )
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Failed to download DroidCalib: {e}")
                return False

        # Ensure droid_slam is on sys.path in this process
        self.droid_slam_path = os.path.join(self.droid_calib_path, "droid_slam")
        if os.path.exists(self.droid_slam_path) and self.droid_slam_path not in sys.path:
            sys.path.append(self.droid_slam_path)

        # Try import first
        try:
            # import droid  # noqa: F401

            self._deps_ready = True
            return True
        except Exception as e:
            if self.verbose:
                print(f"Warning: DroidCalib not importable yet: {e}. Trying to install extensions...")

        # Try to build/install extensions (idempotent in most cases)
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-v", "."],
                cwd=self.droid_calib_path,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            if self.verbose:
                print(
                    "Warning: Failed to install DroidCalib extensions automatically. "
                    "This usually means missing CUDA toolkit / build toolchain. "
                    f"Error: {e}"
                )
            return False

        # Try import again
        try:
            import droid  # noqa: F401

            self._deps_ready = True
            return True
        except Exception as e:
            if self.verbose:
                print(f"Warning: DroidCalib still not importable after install: {e}")
            return False

    def _image_stream(self, video_path):
        """
        Generator that yields (t, image, intrinsics, size_factor)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return

        # Initial calibration guess (center of image)
        w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # fx, fy, cx, cy
        calib = np.array([(w0 + h0) / 2, (w0 + h0) / 2, w0 / 2, h0 / 2])
        fx, fy, cx, cy = calib

        ht, wd = self.image_size  # Target size [h, w]

        t = 0
        frame_idx = 0

        while cap.isOpened():
            ret, image = cap.read()
            if not ret:
                break

            if frame_idx % self.stride != 0:
                frame_idx += 1
                continue

            if self.max_frames and t >= self.max_frames:
                break

            h0, w0, _ = image.shape

            # Resize logic from demo.py
            # h1 = int(h0 * np.sqrt((ht * wd) / (h0 * w0)))
            # w1 = int(w0 * np.sqrt((ht * wd) / (h0 * w0)))
            # Actually demo.py logic seems to try to maintain aspect ratio but target specific area?
            # Let's stick to demo.py logic
            ratio = np.sqrt((ht * wd) / (h0 * w0))
            h1 = int(h0 * ratio)
            w1 = int(w0 * ratio)

            image = cv2.resize(image, (w1, h1))
            image = image[: h1 - h1 % 8, : w1 - w1 % 8]  # Crop to be divisible by 8

            image_tensor = torch.as_tensor(image).permute(2, 0, 1)

            intrinsics = torch.as_tensor([fx, fy, cx, cy])

            # Adjust intrinsics for resize
            h_final, w_final = image.shape[:2]
            size_factor = [(w_final / w0), (h_final / h0)]
            intrinsics[0::2] *= size_factor[0]
            intrinsics[1::2] *= size_factor[1]

            yield t, image_tensor[None], intrinsics, size_factor

            t += 1
            frame_idx += 1

        cap.release()

    def _process_video_file(self, video_path):
        if not self._ensure_droidcalib_ready():
            return None

        try:
            from droid import Droid
        except Exception as e:
            if self.verbose:
                print(f"Warning: Failed to import Droid after ensuring deps: {e}")
            return None

        if not os.path.exists(video_path):
            return None

        # Let's create a generator
        stream = self._image_stream(video_path)

        droid = None
        sf = None  # size factor
        intr_est_list = None

        # try:
        for i in range(1):
            for t, image, intrinsics, size_factor in stream:
                if droid is None:
                    # Update args with actual image size
                    self.droid_args.image_size = [image.shape[2], image.shape[3]]
                    droid = Droid(self.droid_args)

                droid.track(t, image, intrinsics=intrinsics)
                sf = size_factor

            if droid is not None:
                # Terminate and get results
                # We need to pass the stream again for terminate?
                # demo.py: droid.terminate(image_stream(...))
                # It seems terminate does a final BA pass using the stream?
                # Let's recreate stream
                stream_second_pass = self._image_stream(video_path)
                traj_est, intr_est = droid.terminate(stream_second_pass)

                # Rescale intrinsics back to original resolution
                if sf:
                    intr_est = intr_est.copy()
                    intr_est[0:4:2] /= sf[0]
                    intr_est[1:4:2] /= sf[1]

                intr_est_list = intr_est.tolist()

            # except Exception as e:
            #     # Log error or just skip
            #     print(f"Error processing video {video_path}: {e}")
            # finally:
            # Cleanup
            if droid:
                del droid
            torch.cuda.empty_cache()

        return intr_est_list

    def process_single(self, sample, rank=None):
        video_paths = sample[self.video_key]
        if isinstance(video_paths, str):
            video_paths = [video_paths]

        intrinsics_results = []
        for video_path in video_paths:
            res = self._process_video_file(video_path)
            intrinsics_results.append(res)

        if Fields.meta not in sample:
            sample[Fields.meta] = {}

        sample[Fields.meta]["camera_intrinsics"] = intrinsics_results

        return sample

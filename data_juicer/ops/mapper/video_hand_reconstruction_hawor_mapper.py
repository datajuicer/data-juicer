import os
import subprocess
import tempfile

import numpy as np

from data_juicer.utils.cache_utils import DATA_JUICER_MODELS_CACHE
from data_juicer.utils.constant import CameraCalibrationKeys, Fields, MetaKeys
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.model_utils import get_model, prepare_model

from ..base_op import OPERATORS, Mapper
from ..op_fusion import LOADED_VIDEOS

OP_NAME = "video_hand_reconstruction_hawor_mapper"

cv2 = LazyLoader("cv2", "opencv-python")
ultralytics = LazyLoader("ultralytics")
torch = LazyLoader("torch")


@OPERATORS.register_module(OP_NAME)
@LOADED_VIDEOS.register_module(OP_NAME)
class VideoHandReconstructionHaworMapper(Mapper):
    """Use HaWoR and MoGe-2 for hand reconstruction."""

    _accelerator = "cuda"

    def __init__(
        self,
        hawor_model_path: str = "hawor.ckpt",
        hawor_config_path: str = "model_config.yaml",
        hawor_detector_path: str = "detector.pt",
        mano_right_path: str = "path_to_mano_right_pkl",
        mano_left_path: str = "path_to_mano_left_pkl",
        frame_field: str = MetaKeys.video_frames,
        camera_calibration_field: str = "camera_calibration",
        tag_field_name: str = MetaKeys.hand_reconstruction_hawor_tags,
        thresh: float = 0.2,
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param hawor_model_path: The path to 'hawor.ckpt'. for the HaWoR
            model.
        :param hawor_config_path: The path to 'model_config.yaml' for the
            HaWoR model.
        :param hawor_detector_path: The path to 'detector.pt' for the HaWoR
            model.
        :param mano_right_path: The path to 'MANO_RIGHT.pkl'. Users need to
            download this file from https://mano.is.tue.mpg.de/ and comply
            with the MANO license.
        :param mano_left_path: The path to 'MANO_LEFT.pkl'. Users need to
            download this file from https://mano.is.tue.mpg.de/ and comply
            with the MANO license. Used for accurate left-hand wrist
            offset computation (with shapedirs bug-fix).
        :param frame_field: The field name where the video frames are stored.
        :param camera_calibration_field: The field name where the camera calibration info is stored.
        :param tag_field_name: The field name to store the tags. It's
            "hand_reconstruction_hawor_tags" in default.
        :param thresh: The confidence threshold for hand detection. Default is 0.2.
        :param args: extra args
        :param kwargs: extra args

        """

        LazyLoader.check_packages(["lap", "pytorch_lightning", "yacs", "scikit-image", "timm", "omegaconf", "smplx"])
        LazyLoader.check_packages(
            ["chumpy@ git+https://github.com/mattloper/chumpy"], pip_args=["--no-build-isolation"]
        )

        super().__init__(*args, **kwargs)

        from data_juicer.ops.common.hawor_func import (
            interpolate_bboxes,
            parse_chunks,
            rotation_matrix_to_angle_axis,
        )

        self.interpolate_bboxes = interpolate_bboxes
        self.parse_chunks = parse_chunks
        self.rotation_matrix_to_angle_axis = rotation_matrix_to_angle_axis
        self.frame_field = frame_field
        self.hawor_detector_path = hawor_detector_path
        self.tag_field_name = tag_field_name
        self.thresh = thresh
        self.camera_calibration_field = camera_calibration_field

        self.model_key = prepare_model(
            model_type="hawor",
            hawor_model_path=hawor_model_path,
            hawor_config_path=hawor_config_path,
            mano_right_path=mano_right_path,
            mano_left_path=mano_left_path,
        )

        if not os.path.exists(hawor_detector_path):
            hawor_model_dir = os.path.join(DATA_JUICER_MODELS_CACHE, "HaWor")
            os.makedirs(hawor_model_dir, exist_ok=True)
            hawor_detector_path = os.path.join(hawor_model_dir, "detector.pt")
            subprocess.run(
                [
                    "wget",
                    "https://huggingface.co/ThunderVVV/HaWoR/resolve/main/external/detector.pt",
                    "-O",
                    hawor_detector_path,
                ],
                check=True,
            )
            self.hawor_detector_path = hawor_detector_path

        self.det_model_key = prepare_model(model_type="yolo", model_path=self.hawor_detector_path)

    def detect_track(self, imgfiles: list, hand_det_model, thresh: float = 0.5) -> tuple:
        """
        Detects and tracks hands across a sequence of images using YOLO.

        Args:
            imgfiles (list): List of image frames.
            hand_det_model (YOLO): The initialized YOLO hand detection model.
            thresh (float): Confidence threshold for detection.

        Returns:
            tuple: (list of boxes (unused in original logic), dict of tracks)
        """
        boxes_ = []
        tracks = {}

        with torch.no_grad(), torch.amp.autocast("cuda"):
            for t, img_cv2 in enumerate(imgfiles):
                results = hand_det_model.track(img_cv2, conf=thresh, persist=True, verbose=False)

                boxes = results[0].boxes.xyxy.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                handedness = results[0].boxes.cls.cpu().numpy()
                if results[0].boxes.id is not None:
                    track_id = results[0].boxes.id.cpu().numpy()
                else:
                    track_id = [-1] * len(boxes)

                boxes = np.hstack([boxes, confs[:, None]])

                find_right = False
                find_left = False

                for idx, box in enumerate(boxes):
                    if track_id[idx] == -1:
                        if handedness[[idx]] > 0:
                            id = int(10000)
                        else:
                            id = int(5000)
                    else:
                        id = track_id[idx]
                    subj = dict()
                    subj["frame"] = t
                    subj["det"] = True
                    subj["det_box"] = boxes[[idx]]
                    subj["det_handedness"] = handedness[[idx]]

                    if (not find_right and handedness[[idx]] > 0) or (not find_left and handedness[[idx]] == 0):
                        if id in tracks:
                            tracks[id].append(subj)
                        else:
                            tracks[id] = [subj]

                        if handedness[[idx]] > 0:
                            find_right = True
                        elif handedness[[idx]] == 0:
                            find_left = True

        return boxes_, tracks

    def hawor_motion_estimation(
        self,
        imgfiles: list,
        tracks: dict,
        model,
        img_focal: float,
        frame_file_paths: list,
        single_image: bool = False,
    ) -> dict:
        """
        Performs HAWOR 3D hand reconstruction on detected and tracked hand regions.

        Args:
            imgfiles (list): List of decoded image frames (numpy arrays).
            tracks (dict): Dictionary mapping track ID to a list of detection objects.
            model (HAWOR): The initialized HAWOR model.
            img_focal (float): Camera focal length.
            frame_file_paths (list): List of file paths readable by HaWoR
                (pre-materialized on disk if input was bytes).
            single_image (bool): Flag for single-image processing mode.

        Returns:
            dict: Reconstructed parameters ('left' and 'right' hand results).
        """

        left_results = {}
        right_results = {}

        tid = np.array([tr for tr in tracks])

        left_trk = []
        right_trk = []
        for k, idx in enumerate(tid):
            trk = tracks[idx]

            valid = np.array([t["det"] for t in trk])
            is_right = np.concatenate([t["det_handedness"] for t in trk])[valid]

            if len(is_right) == 0 or is_right.sum() / len(is_right) < 0.5:
                left_trk.extend(trk)
            else:
                right_trk.extend(trk)
        left_trk = sorted(left_trk, key=lambda x: x["frame"])
        right_trk = sorted(right_trk, key=lambda x: x["frame"])
        final_tracks = {0: left_trk, 1: right_trk}
        tid = [0, 1]

        img = imgfiles[0]
        img_center = [img.shape[1] / 2, img.shape[0] / 2]  # w/2, h/2
        H, W = img.shape[:2]

        for idx in tid:
            trk = final_tracks[idx]

            # interp bboxes
            valid = np.array([t["det"] for t in trk])
            if not single_image:
                if valid.sum() < 2:
                    continue
            else:
                if valid.sum() < 1:
                    continue
            boxes = np.concatenate([t["det_box"] for t in trk])
            non_zero_indices = np.where(np.any(boxes != 0, axis=1))[0]
            first_non_zero = non_zero_indices[0]
            last_non_zero = non_zero_indices[-1]
            boxes[first_non_zero : last_non_zero + 1] = self.interpolate_bboxes(
                boxes[first_non_zero : last_non_zero + 1]
            )
            valid[first_non_zero : last_non_zero + 1] = True

            boxes = boxes[first_non_zero : last_non_zero + 1]
            is_right = np.concatenate([t["det_handedness"] for t in trk])[valid]
            frame = np.array([t["frame"] for t in trk])[valid]

            if len(is_right) == 0 or is_right.sum() / len(is_right) < 0.5:
                is_right = np.zeros((len(boxes), 1))
            else:
                is_right = np.ones((len(boxes), 1))

            frame_chunks, boxes_chunks = self.parse_chunks(frame, boxes, min_len=1)

            if len(frame_chunks) == 0:
                continue

            for frame_ck, boxes_ck in zip(frame_chunks, boxes_chunks):
                img_ck = [frame_file_paths[i] for i in frame_ck]
                if is_right[0] > 0:
                    do_flip = False
                else:
                    do_flip = True

                results = model.inference(img_ck, boxes_ck, img_focal=img_focal, img_center=img_center, do_flip=do_flip)

                data_out = {
                    "init_root_orient": results["pred_rotmat"][None, :, 0],  # (B, T, 3, 3)
                    "init_hand_pose": results["pred_rotmat"][None, :, 1:],  # (B, T, 15, 3, 3)
                    "init_trans": results["pred_trans"][None, :, 0],  # (B, T, 3)
                    "init_betas": results["pred_shape"][None, :],  # (B, T, 10)
                }

                # Convert to axis-angle (HaWoR native format)
                init_root = self.rotation_matrix_to_angle_axis(data_out["init_root_orient"])  # (B, T, 3)
                init_hand_pose = self.rotation_matrix_to_angle_axis(data_out["init_hand_pose"])  # (B, T, 15, 3)

                # Flip Y/Z axis-angle components for left hand
                # (this is HaWoR's convention for run_mano_left)
                if do_flip:
                    init_root[..., 1] *= -1
                    init_root[..., 2] *= -1
                    init_hand_pose[..., 1] *= -1
                    init_hand_pose[..., 2] *= -1

                T = data_out["init_betas"].shape[1]
                betas_all = data_out["init_betas"][0, :T].cpu().numpy()  # (T, 10)
                orient_all = init_root[0, :T].cpu().numpy()  # (T, 3)
                hand_pose_all = init_hand_pose[0, :T].reshape(T, -1).cpu().numpy()  # (T, 45)
                transl_all = data_out["init_trans"][0, :T].cpu().numpy()  # (T, 3)

                s_frame = frame_ck[0]
                e_frame = frame_ck[-1]

                for frame_id in range(s_frame, e_frame + 1):
                    fi = frame_id - s_frame
                    result = {
                        "betas": betas_all[fi],
                        "global_orient": orient_all[fi],
                        "hand_pose": hand_pose_all[fi],
                        "transl": transl_all[fi],
                    }

                    if idx == 0:
                        left_results[frame_id] = result
                    else:
                        right_results[frame_id] = result

        reformat_results = {"left": left_results, "right": right_results}

        return reformat_results

    @staticmethod
    def _compute_mano_joints(mano_model, global_orient_list, hand_pose_list, betas_list, transl_list):
        """Compute MANO 21-joint positions in camera space via forward kinematics.

        Args:
            mano_model: MANO model (right or left hand), on GPU.
            global_orient_list: List of (3,) axis-angle per frame.
            hand_pose_list: List of (45,) axis-angle per frame.
            betas_list: List of (10,) shape params per frame.
            transl_list: List of (3,) translation per frame.

        Returns:
            numpy array of shape (T, 21, 3) — 21 joint positions in camera space.
        """
        import torch as _torch
        from hawor.utils.geometry import aa_to_rotmat

        T = len(global_orient_list)
        device = next(mano_model.parameters()).device

        # Stack into tensors: (T, ...)
        orient_aa = _torch.tensor(global_orient_list, dtype=_torch.float32)  # (T, 3)
        hand_aa = _torch.tensor(hand_pose_list, dtype=_torch.float32)  # (T, 45)
        betas = _torch.tensor(betas_list, dtype=_torch.float32)  # (T, 10)
        transl = _torch.tensor(transl_list, dtype=_torch.float32)  # (T, 3)

        # Convert axis-angle to rotation matrices
        orient_rotmat = aa_to_rotmat(orient_aa).view(T, 1, 3, 3)  # (T, 1, 3, 3)
        hand_rotmat = aa_to_rotmat(hand_aa.reshape(T * 15, 3)).view(T, 15, 3, 3)  # (T, 15, 3, 3)

        # MANO forward pass on GPU
        with _torch.no_grad():
            mano_out = mano_model(
                global_orient=orient_rotmat.to(device),
                hand_pose=hand_rotmat.to(device),
                betas=betas.to(device),
                transl=transl.to(device),
                pose2rot=False,
            )

        # mano_out.joints: (T, 21, 3) in camera space
        joints_cam = mano_out.joints.cpu().numpy()  # (T, 21, 3)
        return joints_cam

    @staticmethod
    def _decode_frames(raw_frames):
        """Decode raw frames (bytes or paths) to numpy arrays.

        Returns:
            images: list of decoded BGR numpy arrays (None entries skipped)
        """
        from loguru import logger as _logger

        images = []
        for i, frame in enumerate(raw_frames):
            if isinstance(frame, bytes):
                image_array = np.frombuffer(frame, dtype=np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            else:
                image = cv2.imread(frame)

            if image is None:
                _logger.warning(f"Frame {i} decode failed, skipping.")
                continue
            images.append(image)
        return images

    @staticmethod
    def _materialize_bytes_to_files(raw_frames, temp_dir):
        """Write byte-frames to disk once, return file paths.

        If frames are already file paths, returns them directly.
        This avoids repeated per-chunk temp dir creation and disk I/O.
        """
        if not raw_frames:
            return []
        if not isinstance(raw_frames[0], bytes):
            return raw_frames

        file_paths = []
        for i, frame_bytes in enumerate(raw_frames):
            file_path = os.path.join(temp_dir, f"frame_{i}.jpg")
            with open(file_path, "wb") as f:
                f.write(frame_bytes)
            file_paths.append(file_path)
        return file_paths

    def process_single(self, sample=None, rank=None):

        # check if it's generated already
        if self.tag_field_name in sample[Fields.meta]:
            return sample

        # there is no video in this sample
        if self.video_key not in sample or not sample[self.video_key]:
            return sample

        hawor_model, model_cfg, mano_right, mano_left = get_model(self.model_key, rank, self.use_cuda())
        hand_det_model = get_model(self.det_model_key, rank, self.use_cuda())

        videos_frames = sample[self.frame_field]
        sample[Fields.meta][self.tag_field_name] = []

        for video_idx in range(len(videos_frames)):
            cur_camera_calibration = sample[Fields.meta][self.camera_calibration_field][video_idx]
            all_fov_x = cur_camera_calibration.get(CameraCalibrationKeys.hfov, None)

            # If horizontal FoV is not directly available, compute from intrinsics.
            # K is in pixel coordinates: hfov = 2 * arctan(cx / fx),
            # where cx ≈ width/2 (principal point convention).
            if all_fov_x is None:
                intrinsics = cur_camera_calibration.get(CameraCalibrationKeys.intrinsics, None)
                if intrinsics is not None:
                    all_fov_x = [2 * np.arctan(k[0][2] / k[0][0]) for k in intrinsics]
                else:
                    raise ValueError(
                        f"The sample must include an '{CameraCalibrationKeys.hfov}' field or an '{CameraCalibrationKeys.intrinsics}' field in the camera calibration info to store the horizontal FoV for hand reconstruction."
                    )

            frames = videos_frames[video_idx]
            images = self._decode_frames(frames)

            N = len(images)
            if N == 0:
                from loguru import logger as _logger

                _logger.warning(f"Video {video_idx}: all frames decode failed, " "producing empty hand output.")
                empty_hand = {
                    "frame_ids": [],
                    "global_orient": [],
                    "hand_pose": [],
                    "betas": [],
                    "transl": [],
                    "joints_cam": None,
                }
                sample[Fields.meta][self.tag_field_name].append(
                    {
                        "fov_x": 0.0,
                        "img_focal": 0.0,
                        "left": dict(empty_hand),
                        "right": dict(empty_hand),
                    }
                )
                continue
            H, W = images[0].shape[:2]

            # Use median FoV across all frames
            fov_x = np.median(np.array(all_fov_x))
            img_focal = 0.5 * W / np.tan(0.5 * fov_x)

            _, tracks = self.detect_track(images, hand_det_model, thresh=self.thresh)

            with tempfile.TemporaryDirectory() as temp_dir:
                frame_file_paths = self._materialize_bytes_to_files(frames, temp_dir)

                recon_results = self.hawor_motion_estimation(
                    images,
                    tracks,
                    hawor_model,
                    img_focal,
                    frame_file_paths=frame_file_paths,
                    single_image=(N == 1),
                )

            # Collect per-hand results in structured format
            hand_output = {}
            for hand_type in ["left", "right"]:
                frame_ids = []
                global_orient_list = []
                hand_pose_list = []
                betas_list = []
                transl_list = []

                for img_idx in range(N):
                    if img_idx not in recon_results[hand_type]:
                        continue
                    result = recon_results[hand_type][img_idx]
                    frame_ids.append(img_idx)
                    global_orient_list.append(result["global_orient"].tolist())  # (3,)
                    hand_pose_list.append(result["hand_pose"].tolist())  # (45,)
                    betas_list.append(result["betas"].tolist())  # (10,)
                    transl_list.append(result["transl"].tolist())  # (3,)

                # Compute MANO 21-joint positions in camera space
                joints_cam = None
                T_valid = len(frame_ids)
                if T_valid > 0:
                    mano_model = mano_left if hand_type == "left" else mano_right
                    if mano_model is not None:
                        joints_cam = self._compute_mano_joints(
                            mano_model,
                            global_orient_list,
                            hand_pose_list,
                            betas_list,
                            transl_list,
                        )  # (T, 21, 3) numpy

                hand_output[hand_type] = {
                    "frame_ids": frame_ids,
                    "global_orient": global_orient_list,
                    "hand_pose": hand_pose_list,
                    "betas": betas_list,
                    "transl": transl_list,
                    "joints_cam": joints_cam.tolist() if joints_cam is not None else None,  # (T, 21, 3)
                }

            sample[Fields.meta][self.tag_field_name].append(
                {
                    "fov_x": float(fov_x),
                    "img_focal": float(img_focal),
                    "left": hand_output["left"],
                    "right": hand_output["right"],
                }
            )

        return sample

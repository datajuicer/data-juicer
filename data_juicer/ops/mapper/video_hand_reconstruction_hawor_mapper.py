import copy
import os
import subprocess
import sys

import numpy as np
from pydantic import PositiveInt

import data_juicer
from data_juicer.ops.load import load_ops
from data_juicer.utils.cache_utils import (
    DATA_JUICER_ASSETS_CACHE,
    DATA_JUICER_MODELS_CACHE,
)
from data_juicer.utils.constant import Fields, MetaKeys
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
        moge_model_path: str = "Ruicheng/moge-2-vitl",
        mano_right_path: str = "path_to_mano_right_pkl",
        frame_num: PositiveInt = 3,
        duration: float = 0,
        thresh: float = 0.2,
        tag_field_name: str = MetaKeys.hand_reconstruction_hawor_tags,
        frame_dir: str = DATA_JUICER_ASSETS_CACHE,
        if_output_moge_info: bool = False,
        moge_output_info_dir: str = DATA_JUICER_ASSETS_CACHE,
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
        :param moge_model_path: The path to the Moge-2 model.
        :param mano_right_path: The path to 'MANO_RIGHT.pkl'. Users need to
            download this file from https://mano.is.tue.mpg.de/ and comply
            with the MANO license.
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
        :param thresh: Confidence threshold for hand detection.
        :param tag_field_name: The field name to store the tags. It's
            "hand_reconstruction_hawor_tags" in default.
        :param frame_dir: Output directory to save extracted frames.
        :param if_output_moge_info: Whether to save the results from MoGe-2
             to an JSON file.
        :param moge_output_info_dir: Output directory for saving camera
            parameters.
        :param args: extra args
        :param kwargs: extra args

        """

        LazyLoader.check_packages(["lap", "pytorch_lightning", "yacs", "scikit-image", "timm", "omegaconf", "smplx"])
        LazyLoader.check_packages(
            ["chumpy@ git+https://github.com/mattloper/chumpy"], pip_args=["--no-build-isolation"]
        )

        super().__init__(*args, **kwargs)

        self.video_camera_calibration_static_moge_mapper_args = {
            "model_path": moge_model_path,
            "frame_num": frame_num,
            "duration": duration,
            "frame_dir": frame_dir,
            "if_output_points_info": False,
            "if_output_depth_info": False,
            "if_output_mask_info": False,
            "if_output_info": if_output_moge_info,
            "output_info_dir": moge_output_info_dir,
        }
        self.fused_ops = load_ops(
            [{"video_camera_calibration_static_moge_mapper": self.video_camera_calibration_static_moge_mapper_args}]
        )

        hawor_repo_path = os.path.join(DATA_JUICER_ASSETS_CACHE, "HaWoR")
        if not os.path.exists(hawor_repo_path):
            subprocess.run(["git", "clone", "https://github.com/ThunderVVV/HaWoR.git", hawor_repo_path], check=True)

        sys.path.append(hawor_repo_path)
        from hawor.utils.rotation import (
            angle_axis_to_rotation_matrix,
            rotation_matrix_to_angle_axis,
        )
        from lib.eval_utils.custom_utils import interpolate_bboxes
        from lib.pipeline.tools import parse_chunks

        self.interpolate_bboxes = interpolate_bboxes
        self.parse_chunks = parse_chunks
        self.angle_axis_to_rotation_matrix = angle_axis_to_rotation_matrix
        self.rotation_matrix_to_angle_axis = rotation_matrix_to_angle_axis

        self.model_key = prepare_model(
            model_type="hawor",
            hawor_model_path=hawor_model_path,
            hawor_config_path=hawor_config_path,
            mano_right_path=mano_right_path,
        )

        if not os.path.exists(hawor_detector_path):
            hawor_model_dir = os.path.join(DATA_JUICER_MODELS_CACHE, "HaWor")
            os.makedirs(hawor_model_dir, exist_ok=True)
            hawor_detector_path = os.path.join(hawor_model_dir, "detector.pt")
            subprocess.run(
                [
                    "wget",
                    "https://huggingface.co/ThunderVVV/HaWoR/resolve/main/external/detector.pt",
                    hawor_detector_path,
                ],
                check=True,
            )

        self.hawor_detector_path = hawor_detector_path
        self.frame_num = frame_num
        self.duration = duration
        self.tag_field_name = tag_field_name
        self.frame_dir = frame_dir
        self.thresh = thresh

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

        for t, img_cv2 in enumerate(imgfiles):

            with torch.no_grad():
                with torch.amp.autocast("cuda"):
                    results = hand_det_model.track(img_cv2, conf=thresh, persist=True, verbose=False)

                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    confs = results[0].boxes.conf.cpu().numpy()
                    handedness = results[0].boxes.cls.cpu().numpy()
                    if not results[0].boxes.id is None:
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
        img_paths: list,
        single_image: bool = False,
    ) -> dict:
        """
        Performs HAWOR 3D hand reconstruction on detected and tracked hand regions.

        Args:
            imgfiles (list): List of image frames.
            tracks (dict): Dictionary mapping track ID to a list of detection objects.
            model (HAWOR): The initialized HAWOR model.
            img_focal (float): Camera focal length.
            img_paths (list): List of images paths.
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

            if is_right.sum() / len(is_right) < 0.5:
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

            if is_right.sum() / len(is_right) < 0.5:
                is_right = np.zeros((len(boxes), 1))
            else:
                is_right = np.ones((len(boxes), 1))

            frame_chunks, boxes_chunks = self.parse_chunks(frame, boxes, min_len=1)

            if len(frame_chunks) == 0:
                continue

            for frame_ck, boxes_ck in zip(frame_chunks, boxes_chunks):
                img_ck = [img_paths[i] for i in frame_ck]
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

                # flip left hand
                init_root = self.rotation_matrix_to_angle_axis(data_out["init_root_orient"])
                init_hand_pose = self.rotation_matrix_to_angle_axis(data_out["init_hand_pose"])
                if do_flip:
                    init_root[..., 1] *= -1
                    init_root[..., 2] *= -1
                data_out["init_root_orient"] = self.angle_axis_to_rotation_matrix(init_root)
                data_out["init_hand_pose"] = self.angle_axis_to_rotation_matrix(init_hand_pose)

                s_frame = frame_ck[0]
                e_frame = frame_ck[-1]

                for frame_id in range(s_frame, e_frame + 1):
                    result = {}
                    result["beta"] = data_out["init_betas"][0, frame_id - s_frame].cpu().numpy()
                    result["hand_pose"] = data_out["init_hand_pose"][0, frame_id - s_frame].cpu().numpy()
                    result["global_orient"] = data_out["init_root_orient"][0, frame_id - s_frame].cpu().numpy()
                    result["transl"] = data_out["init_trans"][0, frame_id - s_frame].cpu().numpy()

                    if idx == 0:
                        left_results[frame_id] = result
                    else:
                        right_results[frame_id] = result

        reformat_results = {"left": left_results, "right": right_results}

        return reformat_results

    def process_single(self, sample=None, rank=None):

        # check if it's generated already
        if self.tag_field_name in sample[Fields.meta]:
            return sample

        # there is no video in this sample
        if self.video_key not in sample or not sample[self.video_key]:
            return []

        # --- 1. FoV Estimation (MoGe) ---
        ds_list = [{"videos": sample[self.video_key]}]

        dataset = data_juicer.core.data.NestedDataset.from_list(ds_list)
        if Fields.meta not in dataset.features:
            dataset = dataset.add_column(name=Fields.meta, column=[{}] * dataset.num_rows)
        dataset = dataset.map(self.fused_ops[0].process, num_proc=1, with_rank=True)
        res_list = dataset.to_list()

        all_fov_x = res_list[0][Fields.meta][MetaKeys.static_camera_calibration_moge_tags]["hfov_list"]

        temp_frame_name = os.path.splitext(os.path.basename(sample[self.video_key][0]))[0]
        frames_root = os.path.join(self.frame_dir, temp_frame_name)
        frame_names = os.listdir(frames_root)
        frames_path = sorted([os.path.join(frames_root, frame_name) for frame_name in frame_names])

        images = []
        for temp_frame_path in frames_path:
            images.append(cv2.imread(temp_frame_path))

        N = len(images)
        H, W = images[0].shape[:2]

        # Use median FoV across all frames
        fov_x = np.median(np.array(all_fov_x))
        img_focal = 0.5 * W / np.tan(0.5 * fov_x)

        # --- 2. Hand Pose and Translation Estimation (HaWoR) ---
        if rank is not None:
            torch.cuda.set_device(rank)
            device = f"cuda:{rank}" if self.use_cuda() else "cpu"
        else:
            device = "cuda" if self.use_cuda() else "cpu"

        hawor_model, model_cfg, mano_model = get_model(self.model_key, rank, self.use_cuda())
        hand_det_model = ultralytics.YOLO(self.hawor_detector_path).to(device)
        _, tracks = self.detect_track(images, hand_det_model, thresh=self.thresh)

        recon_results = self.hawor_motion_estimation(
            images, tracks, hawor_model, img_focal, single_image=(N == 1), img_paths=frames_path
        )
        del hand_det_model

        # --- 3. Re-calculate Global Translation (MANO Alignment) ---
        left_frame_id_list = []
        left_beta_list = []
        left_hand_pose_list = []
        left_global_orient_list = []
        left_transl_list = []

        right_frame_id_list = []
        right_beta_list = []
        right_hand_pose_list = []
        right_global_orient_list = []
        right_transl_list = []

        for img_idx in range(N):
            for hand_type in ["left", "right"]:
                if hand_type == "left":
                    if img_idx not in recon_results["left"]:
                        continue
                    result = recon_results["left"][img_idx]
                else:
                    if img_idx not in recon_results["right"]:
                        continue
                    result = recon_results["right"][img_idx]

                # Convert results to tensors
                betas = torch.from_numpy(result["beta"]).unsqueeze(0).to(device)
                hand_pose = torch.from_numpy(result["hand_pose"]).unsqueeze(0).to(device)
                transl = torch.from_numpy(result["transl"]).unsqueeze(0).to(device)

                # Forward pass through MANO model
                model_output = mano_model(betas=betas, hand_pose=hand_pose)
                verts_m = model_output.vertices[0]
                joints_m = model_output.joints[0]

                # Flip x-axis for left hand consistency
                if hand_type == "left":
                    verts_m[:, 0] = -1 * verts_m[:, 0]
                    joints_m[:, 0] = -1 * joints_m[:, 0]

                wrist = joints_m[0]

                # Calculate new translation
                transl_new = wrist + transl

                # Store results with the new translation
                result_new_transl = copy.deepcopy(result)
                result_new_transl["transl"] = transl_new[0].cpu().numpy()

                if hand_type == "left":
                    left_frame_id_list.append(img_idx)
                    left_beta_list.append(result_new_transl["beta"])
                    left_hand_pose_list.append(result_new_transl["hand_pose"])
                    left_global_orient_list.append(result_new_transl["global_orient"])
                    left_transl_list.append(result_new_transl["transl"])

                else:
                    right_frame_id_list.append(img_idx)
                    right_beta_list.append(result_new_transl["beta"])
                    right_hand_pose_list.append(result_new_transl["hand_pose"])
                    right_global_orient_list.append(result_new_transl["global_orient"])
                    right_transl_list.append(result_new_transl["transl"])

        sample[Fields.meta][self.tag_field_name] = {
            "fov_x": fov_x,
            "left_frame_id_list": left_frame_id_list,
            "left_beta_list": left_beta_list,
            "left_hand_pose_list": left_hand_pose_list,
            "left_global_orient_list": left_global_orient_list,
            "left_transl_list": left_transl_list,
            "right_frame_id_list": right_frame_id_list,
            "right_beta_list": right_beta_list,
            "right_hand_pose_list": right_hand_pose_list,
            "right_global_orient_list": right_global_orient_list,
            "right_transl_list": right_transl_list,
        }

        return sample

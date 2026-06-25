import hashlib

import cv2
import numpy as np
from loguru import logger

from data_juicer.utils.constant import CameraCalibrationKeys, Fields, MetaKeys
from data_juicer.utils.file_utils import load_numpy

from ..base_op import OPERATORS, Mapper

OP_NAME = "video_clip_reassembly_mapper"


@OPERATORS.register_module(OP_NAME)
class VideoClipReassemblyMapper(Mapper):
    """Reassemble hand-action results from overlapping video clips.

    When long videos are chopped into overlapping clips (e.g. 5 s with 2 s
    overlap via ``VideoSplitByDurationMapper``), each clip is processed
    independently through the 3-D motion labelling pipeline.  This operator
    merges the per-clip results back into **one unified result** per original
    video, including:

    * ``hand_action_tags`` — states, actions, valid_frame_ids, joints
    * ``video_camera_pose_tags`` — ``cam_c2w`` array
    * ``hand_reconstruction_hawor_tags`` — frame_ids converted to global
    * ``video_frames`` — per-clip frame path lists merged into one global list
    * ``camera_calibration_moge_tags`` — per-clip depth/intrinsics merged
    * ``clips`` — replaced with the original video path

    Clip global offsets are determined automatically by **pixel-matching**
    overlapping frames between consecutive clips, rather than assuming an
    ideal step size.  This handles ffmpeg keyframe-alignment drift that
    causes actual clip boundaries to differ from the nominal
    ``(split_duration - overlap_duration) * fps`` calculation.

    Reference (paper §3.1):
        "To enhance efficiency, we chop long videos into overlapping
        20-second clips in this stage and recompose their results."
    """

    def __init__(
        self,
        hand_action_field: str = MetaKeys.hand_action_tags,
        camera_pose_field: str = MetaKeys.video_camera_pose_tags,
        hand_reconstruction_field: str = (MetaKeys.hand_reconstruction_hawor_tags),
        frame_field: str = MetaKeys.video_frames,
        moge_field: str = MetaKeys.camera_calibration_moge_tags,
        clip_field: str = "clips",
        video_key: str = "videos",
        split_duration: float = None,
        overlap_duration: float = None,
        fps: float = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.hand_action_field = hand_action_field
        self.camera_pose_field = camera_pose_field
        self.hand_reconstruction_field = hand_reconstruction_field
        self.frame_field = frame_field
        self.moge_field = moge_field
        self.clip_field = clip_field
        self.video_key = video_key
        self.split_duration = split_duration
        self.overlap_duration = overlap_duration
        self.fps = fps

    # ------------------------------------------------------------------
    # Detect actual clip offsets via frame content matching
    # ------------------------------------------------------------------
    @staticmethod
    def _frame_hash(path: str) -> str:
        """Compute a fast content hash for a frame image file."""
        img = cv2.imread(path)
        if img is None:
            return None
        return hashlib.md5(img.tobytes()).hexdigest()

    @classmethod
    def _detect_clip_offsets(
        cls,
        per_clip_frames: list[list[str]],
        nominal_step: int = None,
    ) -> list[int]:
        """Determine the global frame offset for each clip.

        Compares the first frame of clip[i] against frames of clip[i-1]
        to find the actual overlap point.  Falls back to the nominal step
        if pixel matching fails.

        Returns:
            List of global offsets, one per clip.  offsets[0] is always 0.
        """
        n_clips = len(per_clip_frames)
        offsets = [0]

        for ci in range(1, n_clips):
            prev_frames = per_clip_frames[ci - 1]
            curr_frames = per_clip_frames[ci]

            if not curr_frames or not prev_frames:
                step = nominal_step or len(prev_frames)
                offsets.append(offsets[-1] + step)
                continue

            # Hash the first frame of the current clip
            h_curr_0 = cls._frame_hash(curr_frames[0])
            if h_curr_0 is None:
                step = nominal_step or len(prev_frames)
                offsets.append(offsets[-1] + step)
                continue

            # Search for a match in the previous clip
            # Start from a reasonable range around the nominal step
            search_start = max(0, (nominal_step or len(prev_frames)) - 30)
            search_end = min(len(prev_frames), (nominal_step or len(prev_frames)) + 30)

            found = False
            for j in range(search_start, search_end):
                h_prev = cls._frame_hash(prev_frames[j])
                if h_prev == h_curr_0:
                    # Verify with a second frame to avoid hash collision
                    if len(curr_frames) > 1 and j + 1 < len(prev_frames):
                        h_c1 = cls._frame_hash(curr_frames[1])
                        h_p1 = cls._frame_hash(prev_frames[j + 1])
                        if h_c1 != h_p1:
                            continue
                    offsets.append(offsets[-1] + j)
                    found = True
                    logger.debug(
                        f"Clip {ci}: detected offset {j} from clip {ci-1} " f"(global offset {offsets[-1]})",
                    )
                    break

            if not found:
                step = nominal_step or len(prev_frames)
                offsets.append(offsets[-1] + step)
                logger.warning(
                    f"Clip {ci}: frame matching failed, using nominal " f"step {step} (global offset {offsets[-1]})",
                )

        return offsets

    # ------------------------------------------------------------------
    # World-frame alignment between clips
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_alignment_transforms(
        cam_pose_list: list[dict],
        offsets: list[int],
        clip_lengths: list[int],
    ) -> list[np.ndarray]:
        """Compute 4x4 transforms to align each clip's world frame to clip 0.

        Uses cam_c2w matrices from the overlap region:
            T_0i = c2w_0[g] @ inv(c2w_i[local])
        where g is a global frame index present in both clips.

        Returns:
            List of (4, 4) transforms.  transforms[0] = identity.
        """
        from scipy.spatial.transform import Rotation

        n_clips = len(cam_pose_list)
        transforms = [np.eye(4, dtype=np.float64)]

        for ci in range(1, n_clips):
            cp_prev = cam_pose_list[ci - 1]
            cp_curr = cam_pose_list[ci]

            if not cp_prev or not isinstance(cp_prev, dict) or not cp_curr or not isinstance(cp_curr, dict):
                transforms.append(transforms[-1].copy())
                continue

            raw_prev = cp_prev.get(CameraCalibrationKeys.cam_c2w)
            raw_curr = cp_curr.get(CameraCalibrationKeys.cam_c2w)
            if raw_prev is None or raw_curr is None:
                transforms.append(transforms[-1].copy())
                continue

            c2w_prev = np.asarray(load_numpy(raw_prev), dtype=np.float64)
            c2w_curr = np.asarray(load_numpy(raw_curr), dtype=np.float64)

            # Overlap: clip_curr[k] corresponds to clip_prev[offsets[ci] - offsets[ci-1] + k]
            step_in_prev = offsets[ci] - offsets[ci - 1]
            overlap_len = clip_lengths[ci - 1] - step_in_prev

            if overlap_len <= 0:
                transforms.append(transforms[-1].copy())
                continue

            # Compute T for each overlap frame, then average
            Rs = []
            ts = []
            for k in range(min(overlap_len, len(c2w_curr))):
                prev_idx = step_in_prev + k
                if prev_idx >= len(c2w_prev):
                    break
                T_local = c2w_prev[prev_idx] @ np.linalg.inv(c2w_curr[k])
                Rs.append(T_local[:3, :3])
                ts.append(T_local[:3, 3])

            if not Rs:
                transforms.append(transforms[-1].copy())
                continue

            # Robust average: median translation, mean quaternion rotation
            t_median = np.median(np.array(ts), axis=0)

            quats = Rotation.from_matrix(np.array(Rs)).as_quat()
            for j in range(1, len(quats)):
                if np.dot(quats[j], quats[j - 1]) < 0:
                    quats[j] = -quats[j]
            mean_quat = np.mean(quats, axis=0)
            mean_quat /= np.linalg.norm(mean_quat)
            R_mean = Rotation.from_quat(mean_quat).as_matrix()

            # This gives T: prev_world -> curr_world
            # Chain with the accumulated transform to get clip_0 world
            T_prev_curr = np.eye(4, dtype=np.float64)
            T_prev_curr[:3, :3] = R_mean
            T_prev_curr[:3, 3] = t_median
            transforms.append(transforms[ci - 1] @ T_prev_curr)

            logger.debug(
                f"Clip {ci} alignment: rotation "
                f"{np.degrees(Rotation.from_matrix(R_mean).magnitude()):.1f}°, "
                f"translation {np.linalg.norm(t_median):.4f}m",
            )

        return transforms

    @staticmethod
    def _apply_transform_to_hand_data(
        hand_data: dict,
        T: np.ndarray,
    ) -> dict:
        """Transform a clip's hand states/joints from its local world frame
        to the target world frame using rigid transform T (4x4).
        """
        from scipy.spatial.transform import Rotation

        if not hand_data or not hand_data.get("states"):
            return hand_data

        R = T[:3, :3]
        t = T[:3, 3]
        R_rot = Rotation.from_matrix(R)

        states = np.asarray(hand_data["states"], dtype=np.float64)
        # Transform positions
        states[:, 0:3] = (R @ states[:, 0:3].T).T + t
        # Transform orientations
        orig_rots = Rotation.from_euler("xyz", states[:, 3:6], degrees=False)
        new_rots = R_rot * orig_rots
        states[:, 3:6] = new_rots.as_euler("xyz", degrees=False)

        result = dict(hand_data)
        result["states"] = states.tolist()

        # Transform joints_world
        jw = hand_data.get("joints_world")
        if jw and len(jw) > 0:
            jw_arr = np.asarray(jw, dtype=np.float64)
            # (T, 21, 3) -> transform each joint
            orig_shape = jw_arr.shape
            flat = jw_arr.reshape(-1, 3)
            flat_aligned = (R @ flat.T).T + t
            result["joints_world"] = flat_aligned.reshape(
                orig_shape,
            ).tolist()

        # Recompute actions from transformed states
        from data_juicer.ops.mapper.video_hand_motion_smooth_mapper import (
            _recompute_actions,
        )

        result["actions"] = _recompute_actions(states).tolist()

        return result

    @staticmethod
    def _apply_transform_to_c2w(
        c2w: np.ndarray,
        T: np.ndarray,
    ) -> np.ndarray:
        """Transform cam_c2w array from local world to target world frame."""
        # c2w maps camera -> local_world
        # T maps local_world -> target_world
        # new_c2w = T @ c2w
        return np.einsum("ij,njk->nik", T, c2w)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _empty_hand_result(hand_type: str) -> dict:
        return {
            "hand_type": hand_type,
            "states": [],
            "actions": [],
            "valid_frame_ids": [],
            "joints_cam": [],
            "joints_world": [],
        }

    @staticmethod
    def _recompute_actions(states: np.ndarray) -> np.ndarray:
        """Recompute 7-DoF actions from 8-dim states."""
        from scipy.spatial.transform import Rotation

        T = len(states)
        actions = np.zeros((T, 7), dtype=np.float32)
        for t in range(T - 1):
            actions[t, 0:3] = states[t + 1, 0:3] - states[t, 0:3]
            R_prev = Rotation.from_euler(
                "xyz",
                states[t, 3:6],
                degrees=False,
            )
            R_next = Rotation.from_euler(
                "xyz",
                states[t + 1, 3:6],
                degrees=False,
            )
            R_delta = R_next * R_prev.inv()
            actions[t, 3:6] = R_delta.as_euler("xyz", degrees=False)
            actions[t, 6] = states[t + 1, 7]
        if T > 0:
            actions[T - 1, 6] = states[T - 1, 7]
        return actions

    def _compute_nominal_step(self) -> int:
        """Compute the nominal step from constructor params (fallback)."""
        if self.split_duration and self.overlap_duration and self.fps:
            return int(
                (self.split_duration - self.overlap_duration) * self.fps,
            )
        return None

    def _blend_weight(
        self,
        clip_idx: int,
        local_fid: int,
        n_clips: int,
        clip_len: int,
        overlap_prev: int,
        overlap_next: int,
    ) -> float:
        """Compute the blending weight for a frame given its clip position.

        Args:
            overlap_prev: number of frames this clip overlaps with the
                previous clip (ramp-up at start).
            overlap_next: number of frames this clip overlaps with the
                next clip (ramp-down at end).
        """
        w = 1.0
        if clip_idx > 0 and overlap_prev > 0 and local_fid < overlap_prev:
            w = (local_fid + 1) / (overlap_prev + 1)
        if clip_idx < n_clips - 1 and overlap_next > 0 and local_fid >= clip_len - overlap_next:
            frames_from_end = clip_len - local_fid
            w_end = frames_from_end / (overlap_next + 1)
            w = min(w, w_end)
        return w

    # ------------------------------------------------------------------
    # video_frames merge
    # ------------------------------------------------------------------
    @staticmethod
    def _merge_video_frames(
        per_clip_frames: list[list[str]],
        offsets: list[int],
    ) -> list[str]:
        """Merge per-clip frame path lists into one global ordered list."""
        total_frames = 0
        for ci, clip_frames in enumerate(per_clip_frames):
            end = offsets[ci] + len(clip_frames)
            if end > total_frames:
                total_frames = end

        merged = [None] * total_frames
        for ci, clip_frames in enumerate(per_clip_frames):
            offset = offsets[ci]
            for local_fid, frame_path in enumerate(clip_frames):
                gfid = offset + local_fid
                if gfid < total_frames and merged[gfid] is None:
                    merged[gfid] = frame_path

        # Fill any remaining None slots
        for i in range(len(merged)):
            if merged[i] is None:
                for delta in range(1, len(merged)):
                    if i - delta >= 0 and merged[i - delta] is not None:
                        merged[i] = merged[i - delta]
                        break
                    if i + delta < len(merged) and merged[i + delta] is not None:
                        merged[i] = merged[i + delta]
                        break

        return merged

    # ------------------------------------------------------------------
    # moge calibration merge
    # ------------------------------------------------------------------
    @staticmethod
    def _merge_moge(
        moge_list: list[dict],
        offsets: list[int],
    ) -> dict:
        """Merge per-clip MoGe calibration results into one global result."""
        total_frames = 0
        for ci, m in enumerate(moge_list):
            if not m or not isinstance(m, dict):
                continue
            for k in ("depth", "hfov", "intrinsics", "vfov"):
                v = m.get(k)
                if isinstance(v, list) and len(v) > 0:
                    end = offsets[ci] + len(v)
                    if end > total_frames:
                        total_frames = end
                    break

        if total_frames == 0:
            return moge_list[0] if moge_list else {}

        per_frame_keys = set()
        scalar_fields = {}
        for m in moge_list:
            if not m or not isinstance(m, dict):
                continue
            for k, v in m.items():
                if isinstance(v, list) and len(v) > 1:
                    per_frame_keys.add(k)
                elif k not in scalar_fields:
                    scalar_fields[k] = v

        merged = dict(scalar_fields)
        for key in per_frame_keys:
            arr = [None] * total_frames
            for ci, m in enumerate(moge_list):
                if not m or not isinstance(m, dict):
                    continue
                vals = m.get(key)
                if not isinstance(vals, list):
                    continue
                offset = offsets[ci]
                for local_fid, val in enumerate(vals):
                    gfid = offset + local_fid
                    if gfid < total_frames and arr[gfid] is None:
                        arr[gfid] = val
            for i in range(len(arr)):
                if arr[i] is None:
                    for delta in range(1, len(arr)):
                        if i - delta >= 0 and arr[i - delta] is not None:
                            arr[i] = arr[i - delta]
                            break
                        if i + delta < len(arr) and arr[i + delta] is not None:
                            arr[i] = arr[i + delta]
                            break
            merged[key] = arr

        return merged

    # ------------------------------------------------------------------
    # hand action merge
    # ------------------------------------------------------------------
    def _merge_hand_across_clips(
        self,
        clips_hand_data: list,
        hand_type: str,
        n_clips: int,
        offsets: list[int],
        clip_lengths: list[int],
    ) -> dict:
        """Merge one hand's data across all clips into a single trajectory."""
        clip_entries = []
        for clip_idx, hand_data in enumerate(clips_hand_data):
            if not hand_data or not hand_data.get("states"):
                continue
            global_offset = offsets[clip_idx]
            local_ids = hand_data["valid_frame_ids"]
            global_ids = [fid + global_offset for fid in local_ids]

            jw = hand_data.get("joints_world")
            jc = hand_data.get("joints_cam")
            clip_entries.append(
                {
                    "clip_idx": clip_idx,
                    "local_ids": local_ids,
                    "global_ids": global_ids,
                    "states": np.asarray(hand_data["states"], dtype=np.float64),
                    "joints_world": (np.asarray(jw, dtype=np.float64) if jw and len(jw) > 0 else None),
                    "joints_cam": (np.asarray(jc, dtype=np.float64) if jc and len(jc) > 0 else None),
                }
            )

        if not clip_entries:
            return self._empty_hand_result(hand_type)

        if len(clip_entries) == 1:
            e = clip_entries[0]
            src = clips_hand_data[e["clip_idx"]]
            return {
                "hand_type": hand_type,
                "states": src["states"],
                "actions": src["actions"],
                "valid_frame_ids": e["global_ids"],
                "joints_cam": src.get("joints_cam", []),
                "joints_world": src.get("joints_world", []),
            }

        # Global frame range
        all_gids = []
        for e in clip_entries:
            all_gids.extend(e["global_ids"])
        min_fid = min(all_gids)
        max_fid = max(all_gids)
        n_total = max_fid - min_fid + 1

        state_sum = np.zeros((n_total, 8), dtype=np.float64)
        weight_sum = np.zeros(n_total, dtype=np.float64)
        has_jw = any(e["joints_world"] is not None for e in clip_entries)
        has_jc = any(e["joints_cam"] is not None for e in clip_entries)
        jw_sum = np.zeros((n_total, 21, 3), dtype=np.float64) if has_jw else None
        jc_sum = np.zeros((n_total, 21, 3), dtype=np.float64) if has_jc else None

        for entry in clip_entries:
            ci = entry["clip_idx"]
            clip_len = clip_lengths[ci]
            # Compute overlap with previous clip
            if ci > 0:
                prev_end = offsets[ci - 1] + clip_lengths[ci - 1]
                overlap_prev = max(0, prev_end - offsets[ci])
            else:
                overlap_prev = 0
            # Compute overlap with next clip
            if ci < n_clips - 1:
                next_offset = offsets[ci + 1]
                this_end = offsets[ci] + clip_len
                overlap_next = max(0, this_end - next_offset)
            else:
                overlap_next = 0

            for i, gfid in enumerate(entry["global_ids"]):
                local_fid = entry["local_ids"][i]
                idx = gfid - min_fid
                w = self._blend_weight(
                    ci,
                    local_fid,
                    n_clips,
                    clip_len,
                    overlap_prev,
                    overlap_next,
                )
                state_sum[idx] += entry["states"][i] * w
                weight_sum[idx] += w
                if has_jw and entry["joints_world"] is not None and i < len(entry["joints_world"]):
                    jw_sum[idx] += entry["joints_world"][i] * w
                if has_jc and entry["joints_cam"] is not None and i < len(entry["joints_cam"]):
                    jc_sum[idx] += entry["joints_cam"][i] * w

        valid_mask = weight_sum > 1e-8
        valid_idx = np.where(valid_mask)[0]
        if len(valid_idx) == 0:
            return self._empty_hand_result(hand_type)

        w_col = weight_sum[valid_idx, np.newaxis]
        merged_states = state_sum[valid_idx] / w_col
        merged_fids = (valid_idx + min_fid).tolist()

        merged_jw = None
        if has_jw:
            merged_jw = (jw_sum[valid_idx] / weight_sum[valid_idx, np.newaxis, np.newaxis]).tolist()

        merged_jc = None
        if has_jc:
            merged_jc = (jc_sum[valid_idx] / weight_sum[valid_idx, np.newaxis, np.newaxis]).tolist()

        actions = self._recompute_actions(merged_states)

        return {
            "hand_type": hand_type,
            "states": merged_states.astype(np.float32).tolist(),
            "actions": actions.tolist(),
            "valid_frame_ids": merged_fids,
            "joints_cam": merged_jc if merged_jc else [],
            "joints_world": merged_jw if merged_jw else [],
        }

    # ------------------------------------------------------------------
    # camera pose (cam_c2w) merge
    # ------------------------------------------------------------------
    def _merge_cam_c2w(
        self,
        cam_pose_list: list[dict],
        offsets: list[int],
        clip_lengths: list[int],
    ) -> dict:
        """Merge per-clip cam_c2w (N,4,4) arrays into a single global array."""
        n_clips = len(cam_pose_list)

        clip_c2ws: list[tuple[int, np.ndarray]] = []
        for ci, cp in enumerate(cam_pose_list):
            if not cp or not isinstance(cp, dict):
                continue
            raw = cp.get(CameraCalibrationKeys.cam_c2w)
            if raw is None:
                continue
            arr = np.asarray(load_numpy(raw), dtype=np.float64)
            clip_c2ws.append((ci, arr))

        if not clip_c2ws:
            return cam_pose_list[0] if cam_pose_list else {}

        max_global = 0
        for ci, arr in clip_c2ws:
            end = offsets[ci] + len(arr)
            if end > max_global:
                max_global = end

        c2w_sum = np.zeros((max_global, 4, 4), dtype=np.float64)
        w_sum = np.zeros(max_global, dtype=np.float64)

        for ci, arr in clip_c2ws:
            offset = offsets[ci]
            clip_len = clip_lengths[ci]
            # Compute overlap with previous clip
            if ci > 0:
                prev_end = offsets[ci - 1] + clip_lengths[ci - 1]
                overlap_prev = max(0, prev_end - offset)
            else:
                overlap_prev = 0
            # Compute overlap with next clip
            if ci < n_clips - 1:
                next_offset = offsets[ci + 1]
                this_end = offset + clip_len
                overlap_next = max(0, this_end - next_offset)
            else:
                overlap_next = 0

            for local_fid in range(len(arr)):
                gfid = offset + local_fid
                w = self._blend_weight(
                    ci,
                    local_fid,
                    n_clips,
                    clip_len,
                    overlap_prev,
                    overlap_next,
                )
                c2w_sum[gfid] += arr[local_fid] * w
                w_sum[gfid] += w

        valid = w_sum > 1e-8
        for gfid in range(max_global):
            if valid[gfid]:
                c2w_sum[gfid] /= w_sum[gfid]
            else:
                c2w_sum[gfid] = np.eye(4)

        merged: dict = {}
        for cp in cam_pose_list:
            if cp and isinstance(cp, dict):
                for k, v in cp.items():
                    if k != CameraCalibrationKeys.cam_c2w and k not in merged:
                        merged[k] = v
                break

        merged[CameraCalibrationKeys.cam_c2w] = c2w_sum.tolist()
        return merged

    # ------------------------------------------------------------------
    # hawor reconstruction merge
    # ------------------------------------------------------------------
    def _merge_hawor(
        self,
        hawor_list: list[dict],
        offsets: list[int],
    ) -> dict:
        """Merge per-clip HaWoR results: convert local frame_ids to global."""
        n_clips = len(hawor_list)

        merged: dict = {}
        for hw in hawor_list:
            if hw and isinstance(hw, dict):
                for k in ("fov_x", "img_focal"):
                    if k in hw and k not in merged:
                        merged[k] = hw[k]
                break

        hand_types: set[str] = set()
        for hw in hawor_list:
            if hw and isinstance(hw, dict):
                for k in ("left", "right"):
                    if k in hw and isinstance(hw[k], dict):
                        hand_types.add(k)

        for ht in sorted(hand_types):
            seen_global: set[int] = set()
            merged_fids: list[int] = []
            merged_transl: list = []
            merged_orient: list = []
            merged_pose: list = []
            merged_betas: list = []
            merged_joints_cam: list = []

            for ci in range(n_clips):
                hw = hawor_list[ci]
                if not hw or not isinstance(hw, dict):
                    continue
                hand = hw.get(ht, {})
                if not hand or not isinstance(hand, dict):
                    continue

                fids = hand.get("frame_ids", [])
                offset = offsets[ci]
                transl = hand.get("transl", [])
                orient = hand.get("global_orient", [])
                pose = hand.get("hand_pose", [])
                betas = hand.get("betas", [])
                jc = hand.get("joints_cam", None)

                for i, local_fid in enumerate(fids):
                    gfid = local_fid + offset
                    if gfid in seen_global:
                        continue
                    seen_global.add(gfid)
                    merged_fids.append(gfid)
                    if i < len(transl):
                        merged_transl.append(transl[i])
                    if i < len(orient):
                        merged_orient.append(orient[i])
                    if i < len(pose):
                        merged_pose.append(pose[i])
                    if i < len(betas):
                        merged_betas.append(betas[i])
                    if jc is not None and i < len(jc):
                        merged_joints_cam.append(jc[i])

            if merged_fids:
                order = np.argsort(merged_fids).tolist()
                merged_fids = [merged_fids[j] for j in order]
                merged_transl = [merged_transl[j] for j in order] if merged_transl else []
                merged_orient = [merged_orient[j] for j in order] if merged_orient else []
                merged_pose = [merged_pose[j] for j in order] if merged_pose else []
                merged_betas = [merged_betas[j] for j in order] if merged_betas else []
                merged_joints_cam = [merged_joints_cam[j] for j in order] if merged_joints_cam else None

            merged[ht] = {
                "frame_ids": merged_fids,
                "transl": merged_transl,
                "global_orient": merged_orient,
                "hand_pose": merged_pose,
                "betas": merged_betas,
            }
            if merged_joints_cam is not None:
                merged[ht]["joints_cam"] = merged_joints_cam

        return merged

    # ------------------------------------------------------------------
    # main entry
    # ------------------------------------------------------------------
    def process_single(self, sample=None, rank=None):
        if Fields.meta not in sample:
            return sample

        meta = sample[Fields.meta]
        hand_action_list = meta.get(self.hand_action_field)

        # --- detect actual clip offsets from frame content ---
        per_clip_frames = sample.get(self.frame_field)
        has_multi_clips = (
            per_clip_frames
            and isinstance(per_clip_frames, list)
            and len(per_clip_frames) > 1
            and isinstance(per_clip_frames[0], list)
        )

        if not has_multi_clips:
            return sample

        n_clips = len(per_clip_frames)
        clip_lengths = [len(cf) for cf in per_clip_frames]
        nominal_step = self._compute_nominal_step()

        offsets = self._detect_clip_offsets(per_clip_frames, nominal_step)
        total_frames = max(off + clen for off, clen in zip(offsets, clip_lengths))
        logger.info(
            f"Clip offsets: {offsets}, clip_lengths: {clip_lengths}, " f"total_frames: {total_frames}",
        )

        # --- merge video_frames ---
        try:
            merged_frames = self._merge_video_frames(
                per_clip_frames,
                offsets,
            )
            sample[self.frame_field] = [merged_frames]
            logger.debug(
                f"Merged {n_clips} clip frame lists into " f"{len(merged_frames)} global frames",
            )
        except Exception as e:
            logger.warning(f"video_frames reassembly failed: {e}")

        # --- merge moge ---
        moge_list = meta.get(self.moge_field)
        if moge_list and isinstance(moge_list, list) and len(moge_list) > 1:
            try:
                merged_moge = self._merge_moge(moge_list, offsets)
                meta[self.moge_field] = [merged_moge]
            except Exception as e:
                logger.warning(f"MoGe reassembly failed: {e}")

        # --- compute world-frame alignment transforms ---
        cam_pose_list = meta.get(self.camera_pose_field)
        align_transforms = None
        if cam_pose_list and len(cam_pose_list) > 1:
            try:
                align_transforms = self._compute_alignment_transforms(
                    cam_pose_list,
                    offsets,
                    clip_lengths,
                )
            except Exception as e:
                logger.warning(f"Alignment transform computation failed: {e}")

        # --- align hand actions to clip 0's world frame, then merge ---
        if hand_action_list and len(hand_action_list) > 1:
            if align_transforms:
                for ci in range(1, len(hand_action_list)):
                    T = align_transforms[ci]
                    if hand_action_list[ci] and not np.allclose(T, np.eye(4)):
                        for ht in hand_action_list[ci]:
                            try:
                                hand_action_list[ci][ht] = self._apply_transform_to_hand_data(
                                    hand_action_list[ci][ht],
                                    T,
                                )
                            except Exception as e:
                                logger.warning(
                                    f"Alignment failed clip {ci} {ht}: {e}",
                                )

        # --- merge hand actions ---
        if hand_action_list and len(hand_action_list) > 1:
            hand_types: set[str] = set()
            for clip_result in hand_action_list:
                if clip_result:
                    hand_types.update(clip_result.keys())

            merged_result: dict = {}
            for ht in sorted(hand_types):
                per_clip = [(cr.get(ht) if cr else None) for cr in hand_action_list]
                try:
                    merged_result[ht] = self._merge_hand_across_clips(
                        per_clip,
                        ht,
                        n_clips,
                        offsets,
                        clip_lengths,
                    )
                except Exception as e:
                    logger.warning(
                        f"Hand '{ht}' reassembly failed: {e}. " f"Falling back to first clip.",
                    )
                    first_valid = next(
                        (d for d in per_clip if d and d.get("states")),
                        None,
                    )
                    merged_result[ht] = first_valid if first_valid else self._empty_hand_result(ht)

            meta[self.hand_action_field] = [merged_result]

        # --- align cam_c2w to clip 0's world frame, then merge ---
        cam_pose_list = meta.get(self.camera_pose_field)
        if cam_pose_list and len(cam_pose_list) > 1:
            # Apply alignment transforms to each clip's c2w before merging
            if align_transforms:
                for ci in range(1, len(cam_pose_list)):
                    cp = cam_pose_list[ci]
                    if not cp or not isinstance(cp, dict):
                        continue
                    raw = cp.get(CameraCalibrationKeys.cam_c2w)
                    if raw is None:
                        continue
                    T = align_transforms[ci]
                    if np.allclose(T, np.eye(4)):
                        continue
                    try:
                        c2w_arr = np.asarray(
                            load_numpy(raw),
                            dtype=np.float64,
                        )
                        aligned = self._apply_transform_to_c2w(c2w_arr, T)
                        cam_pose_list[ci] = dict(cp)
                        cam_pose_list[ci][CameraCalibrationKeys.cam_c2w] = aligned.tolist()
                    except Exception as e:
                        logger.warning(
                            f"cam_c2w alignment failed clip {ci}: {e}",
                        )

            try:
                merged_cam = self._merge_cam_c2w(
                    cam_pose_list,
                    offsets,
                    clip_lengths,
                )
                meta[self.camera_pose_field] = [merged_cam]
            except Exception as e:
                logger.warning(f"cam_c2w reassembly failed: {e}")

        # --- merge hawor ---
        hawor_list = meta.get(self.hand_reconstruction_field)
        if hawor_list and len(hawor_list) > 1:
            try:
                merged_hawor = self._merge_hawor(hawor_list, offsets)
                meta[self.hand_reconstruction_field] = [merged_hawor]
            except Exception as e:
                logger.warning(f"HaWoR reassembly failed: {e}")

        # --- merge clips → original video ---
        clips = sample.get(self.clip_field)
        if clips and isinstance(clips, list) and len(clips) > 1:
            videos = sample.get(self.video_key)
            if videos and isinstance(videos, list) and len(videos) > 0:
                sample[self.clip_field] = videos
            else:
                sample[self.clip_field] = [clips[0]]

        return sample

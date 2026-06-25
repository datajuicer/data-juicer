import os

import cv2
import numpy as np
from loguru import logger

from data_juicer.utils.constant import Fields, MetaKeys

from ..base_op import OPERATORS, Mapper

OP_NAME = "video_trajectory_overlay_mapper"


@OPERATORS.register_module(OP_NAME)
class VideoTrajectoryOverlayMapper(Mapper):
    """Prepare VLM-ready frames by sampling and overlaying hand trajectories.

    Implements the visualization step from paper https://arxiv.org/pdf/2510.21571:

        "From each segment, we evenly sample 8 frames and highlight hand
        trajectories on each frame by projecting the world-space trajectory
        of the hand palm from the current frame to the end of the clip."

    For each atomic action segment (output of
    ``VideoAtomicActionSegmentMapper``), this operator:

    1. Evenly samples ``n_sample_frames`` frames from the segment.
    2. For each sampled frame, projects the **future** world-space wrist
       trajectory (from the current frame to the end of the segment) onto
       the image using camera intrinsics and cam_c2w.
    3. Draws the trajectory as a colored line with a dot at the current
       wrist position.
    4. Saves the overlay images and stores their paths in the segment.

    The output is written back into each segment dict under
    ``"overlay_frames"``, ready to be consumed by the VLM captioning
    operator.
    """

    # MANO joint index for palm center (middle finger MCP).
    # Paper §3.3: "trajectory of the hand palm"
    PALM_JOINT_INDEX = 9

    def __init__(
        self,
        segment_field: str = "atomic_action_segments",
        camera_pose_field: str = MetaKeys.video_camera_pose_tags,
        moge_field: str = MetaKeys.camera_calibration_moge_tags,
        frame_field: str = MetaKeys.video_frames,
        save_dir: str = None,
        n_sample_frames: int = 8,
        palm_joint_index: int = 9,
        dot_radius: int = 10,
        line_thickness: int = 4,
        trajectory_alpha: float = 0.7,
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param segment_field: Meta field storing atomic action segments.
        :param camera_pose_field: Meta field storing camera pose (cam_c2w).
        :param moge_field: Meta field storing MoGe calibration (for fov_x).
        :param frame_field: Field storing frame image paths.
        :param save_dir: Directory to save overlay images.  If None, uses
            a temp directory derived from the first frame path.
        :param n_sample_frames: Number of frames to evenly sample from
            each segment.
        :param palm_joint_index: MANO joint index for the palm position.
            Default 9 = middle finger MCP (palm center proxy).
            Joint 0 = wrist root.
        :param dot_radius: Radius of the dot at the current wrist position.
        :param line_thickness: Thickness of the trajectory line.
        :param trajectory_alpha: Alpha blending for the trajectory overlay.
        """
        super().__init__(*args, **kwargs)
        self.segment_field = segment_field
        self.camera_pose_field = camera_pose_field
        self.moge_field = moge_field
        self.frame_field = frame_field
        self.save_dir = save_dir
        self.n_sample_frames = n_sample_frames
        self.palm_joint_index = palm_joint_index
        self.dot_radius = dot_radius
        self.line_thickness = line_thickness
        self.trajectory_alpha = trajectory_alpha

    # ------------------------------------------------------------------
    # Projection helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _world_to_camera(
        pos_world: np.ndarray,
        cam_c2w: np.ndarray,
    ) -> np.ndarray:
        """Convert world position(s) to camera space.

        Args:
            pos_world: (..., 3) world positions.
            cam_c2w: (4, 4) camera-to-world transform.

        Returns:
            (..., 3) camera-space positions.
        """
        R = cam_c2w[:3, :3]
        t = cam_c2w[:3, 3]
        # cam = R^T @ (world - t)
        return (pos_world - t) @ R  # equivalent to (R.T @ (p - t).T).T

    @staticmethod
    def _project_to_2d(
        pos_cam: np.ndarray,
        width: int,
        height: int,
        K: np.ndarray = None,
        fov_x: float = None,
    ) -> np.ndarray:
        """Project camera-space positions to 2D pixel coords.

        Args:
            pos_cam: (..., 3) camera-space positions.
            width: image width.
            height: image height.
            K: (3, 3) intrinsics matrix (preferred). If provided, fov_x
                is ignored and fx, fy, cx, cy are taken from K directly.
            fov_x: horizontal field of view in radians (fallback when K
                is not available).

        Returns:
            (..., 2) pixel coordinates (u, v).
        """
        if K is not None:
            K = np.asarray(K, dtype=np.float64)
            fx = K[0, 0]
            fy = K[1, 1]
            cx = K[0, 2]
            cy = K[1, 2]
        elif fov_x is not None:
            fx = width / (2.0 * np.tan(fov_x / 2.0))
            fy = fx
            cx, cy = width / 2.0, height / 2.0
        else:
            raise ValueError("Either K or fov_x must be provided")

        z = pos_cam[..., 2]
        z_safe = np.where(np.abs(z) < 1e-6, 1e-6, z)
        u = fx * pos_cam[..., 0] / z_safe + cx
        v = fy * pos_cam[..., 1] / z_safe + cy
        return np.stack([u, v], axis=-1)

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _temporal_color(t: float) -> tuple:
        """Map normalized time t ∈ [0, 1] to BGR color along
        blue → green → red gradient.

        t=0 (current) → blue, t=0.5 → green, t=1 (future end) → red.
        """
        # BGR format
        if t < 0.5:
            # blue → green
            ratio = t / 0.5
            b = int(255 * (1 - ratio))
            g = int(255 * ratio)
            r = 0
        else:
            # green → red
            ratio = (t - 0.5) / 0.5
            b = 0
            g = int(255 * (1 - ratio))
            r = int(255 * ratio)
        return (b, g, r)

    def _draw_trajectory(
        self,
        frame: np.ndarray,
        points_2d: np.ndarray,
        current_idx: int = 0,
    ) -> np.ndarray:
        """Draw trajectory line with blue→green→red temporal gradient
        and a blue dot at the current palm position.

        Args:
            frame: BGR image to draw on (modified in place).
            points_2d: (N, 2) pixel coordinates of trajectory points.
            current_idx: index of the current frame's position in
                points_2d (drawn as a blue dot).
        """
        h, w = frame.shape[:2]
        overlay = frame.copy()

        # Filter out-of-frame and behind-camera points, keep original index
        valid = []
        valid_indices = []
        for i, pt in enumerate(points_2d):
            if 0 <= pt[0] < w and 0 <= pt[1] < h:
                valid.append((int(pt[0]), int(pt[1])))
                valid_indices.append(i)
            elif valid:
                # Keep trajectory continuous by clamping
                valid.append(
                    (
                        int(np.clip(pt[0], 0, w - 1)),
                        int(np.clip(pt[1], 0, h - 1)),
                    )
                )
                valid_indices.append(i)

        # Draw trajectory line with temporal color gradient
        n_pts = len(points_2d)
        if len(valid) >= 2:
            for i in range(len(valid) - 1):
                t = valid_indices[i] / max(n_pts - 1, 1)
                line_color = self._temporal_color(t)
                cv2.line(
                    overlay,
                    valid[i],
                    valid[i + 1],
                    line_color,
                    self.line_thickness,
                    lineType=cv2.LINE_AA,
                )

        # Draw current position as a blue dot
        if current_idx < len(points_2d):
            pt = points_2d[current_idx]
            if 0 <= pt[0] < w and 0 <= pt[1] < h:
                blue_bgr = (255, 100, 0)  # blue in BGR
                cv2.circle(
                    overlay,
                    (int(pt[0]), int(pt[1])),
                    self.dot_radius,
                    blue_bgr,
                    -1,
                    lineType=cv2.LINE_AA,
                )
                # White border for visibility
                cv2.circle(
                    overlay,
                    (int(pt[0]), int(pt[1])),
                    self.dot_radius + 1,
                    (255, 255, 255),
                    1,
                    lineType=cv2.LINE_AA,
                )

        # Alpha blend
        cv2.addWeighted(
            overlay,
            self.trajectory_alpha,
            frame,
            1 - self.trajectory_alpha,
            0,
            frame,
        )
        return frame

    # ------------------------------------------------------------------
    # Process one segment
    # ------------------------------------------------------------------
    def _process_segment(
        self,
        segment: dict,
        all_frames: list[str],
        cam_c2w_all: np.ndarray,
        save_dir: str,
        intrinsics_list: list = None,
        fov_x: float = None,
        file_prefix: str = "",
    ) -> dict:
        """Process a single segment: sample frames, overlay trajectory.

        Args:
            intrinsics_list: per-frame (3,3) intrinsics matrices from MoGe.
                If provided, used for accurate projection (preferred).
            fov_x: fallback horizontal FOV in radians when intrinsics_list
                is not available.
            file_prefix: prefix added to overlay filenames to avoid
                collisions when multiple videos share the same save_dir.

        Returns the segment dict with ``overlay_frames`` added.
        """
        hand_type = segment["hand_type"]
        valid_fids = segment["valid_frame_ids"]

        # Use joints_world for the palm trajectory (paper §3.3).
        # states[:, 0:3] is MANO's root transl, NOT the actual palm/wrist
        # position — there is a significant offset (~10cm) between them.
        joints_world = segment.get("joints_world")
        if joints_world and len(joints_world) > 0:
            jw_arr = np.asarray(joints_world, dtype=np.float64)
            palm_positions = jw_arr[:, self.palm_joint_index, :]
        else:
            # Fallback to states (less accurate)
            states = np.asarray(segment["states"], dtype=np.float64)
            palm_positions = states[:, 0:3]
            logger.debug(
                f"No joints_world for {hand_type} segment, " f"falling back to states[:, 0:3]",
            )

        n = len(palm_positions)

        if n < 2:
            segment["overlay_frames"] = []
            segment["sampled_frame_indices"] = []
            return segment

        # Evenly sample frame indices within the segment
        if n <= self.n_sample_frames:
            sample_indices = list(range(n))
        else:
            sample_indices = np.linspace(
                0,
                n - 1,
                self.n_sample_frames,
                dtype=int,
            ).tolist()

        seg_id = segment.get("segment_id", 0)
        overlay_paths = []

        for local_idx in sample_indices:
            fid = valid_fids[local_idx]
            if fid >= len(all_frames) or not all_frames[fid]:
                continue

            frame_path = all_frames[fid]
            frame = cv2.imread(frame_path)
            if frame is None:
                continue

            h, w = frame.shape[:2]

            # Get future trajectory: from current frame to end of segment
            future_positions = palm_positions[local_idx:]

            if fid >= len(cam_c2w_all):
                continue

            # Project all future world positions using the CURRENT frame's
            # camera (we are drawing on the current frame's image).
            cam = cam_c2w_all[fid]

            # Determine per-frame intrinsics for projection
            frame_K = None
            if intrinsics_list is not None and fid < len(intrinsics_list):
                frame_K = np.asarray(intrinsics_list[fid], dtype=np.float64)

            points_2d_list = []
            for j in range(len(future_positions)):
                pos_cam = self._world_to_camera(future_positions[j], cam)
                pt_2d = self._project_to_2d(
                    pos_cam,
                    w,
                    h,
                    K=frame_K,
                    fov_x=fov_x,
                )
                points_2d_list.append(pt_2d)

            if not points_2d_list:
                continue

            points_2d = np.array(points_2d_list)
            frame = self._draw_trajectory(frame, points_2d, 0)

            # Save overlay frame (prefix avoids collisions across videos)
            fname = (f"{file_prefix}_" if file_prefix else "") + f"seg{seg_id}_{hand_type}_f{fid:06d}_overlay.jpg"
            out_path = os.path.join(save_dir, fname)
            cv2.imwrite(out_path, frame)
            overlay_paths.append(out_path)

        segment["overlay_frames"] = overlay_paths
        segment["sampled_frame_indices"] = [valid_fids[i] for i in sample_indices if i < len(valid_fids)]
        return segment

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------
    def _sample_prefix(self, sample: dict) -> str:
        """Derive a short unique prefix from the sample's video path.

        Used to namespace overlay files so different videos sharing
        the same save_dir do not overwrite each other.
        """
        videos = sample.get(self.video_key, [])
        if videos:
            v = videos[0] if isinstance(videos, list) else videos
            return os.path.splitext(os.path.basename(v))[0]

        return "unknown"

    def process_single(self, sample=None, rank=None):
        if Fields.meta not in sample:
            return sample

        meta = sample[Fields.meta]
        segments = meta.get(self.segment_field)
        if not segments:
            return sample

        # Get frame paths
        frame_data = sample.get(self.frame_field, [])
        if not frame_data:
            return sample
        all_frames = (
            frame_data[0]
            if isinstance(frame_data, list) and frame_data and isinstance(frame_data[0], list)
            else frame_data
        )

        # Get cam_c2w
        cam_pose_list = meta.get(self.camera_pose_field, [])
        if not cam_pose_list:
            logger.warning("No camera pose data for trajectory overlay.")
            return sample

        from data_juicer.utils.constant import CameraCalibrationKeys
        from data_juicer.utils.file_utils import load_numpy

        cam_pose = cam_pose_list[0] if isinstance(cam_pose_list, list) else cam_pose_list
        raw_c2w = cam_pose.get(CameraCalibrationKeys.cam_c2w)
        if raw_c2w is None:
            logger.warning("No cam_c2w for trajectory overlay.")
            return sample
        cam_c2w_all = np.asarray(load_numpy(raw_c2w), dtype=np.float64)

        # Get camera intrinsics (prefer full K matrix, fallback to fov_x)
        intrinsics_list, fov_x = self._get_intrinsics(meta)
        if intrinsics_list is None and fov_x is None:
            logger.warning(
                "Cannot determine camera intrinsics, skipping overlay.",
            )
            return sample

        # Determine save directory
        save_dir = self.save_dir
        if save_dir is None and all_frames:
            save_dir = os.path.join(
                os.path.dirname(all_frames[0]),
                "trajectory_overlays",
            )
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        # Unique prefix to avoid filename collisions across videos
        prefix = self._sample_prefix(sample)

        # Process each segment
        for i, seg in enumerate(segments):
            try:
                segments[i] = self._process_segment(
                    seg,
                    all_frames,
                    cam_c2w_all,
                    save_dir,
                    intrinsics_list=intrinsics_list,
                    fov_x=fov_x,
                    file_prefix=prefix,
                )
            except Exception as e:
                logger.warning(
                    f"Trajectory overlay failed for segment {i}: {e}",
                )
                seg["overlay_frames"] = []
                seg["sampled_frame_indices"] = []

        meta[self.segment_field] = segments
        return sample

    def _get_intrinsics(self, meta: dict) -> tuple:
        """Extract camera intrinsics for projection.

        Returns:
            (intrinsics_list, fov_x): intrinsics_list is a per-frame list
            of (3,3) K matrices if available (preferred), otherwise None.
            fov_x is a scalar fallback FOV in radians.
            At least one of them will be non-None if calibration data exists.
        """
        from data_juicer.utils.constant import CameraCalibrationKeys

        intrinsics_list = None
        fov_x = None

        # Try MoGe calibration — prefer full intrinsics matrix K
        moge_list = meta.get(self.moge_field, [])
        if moge_list:
            moge = moge_list[0] if isinstance(moge_list, list) else moge_list
            if isinstance(moge, dict):
                # Prefer per-frame intrinsics K matrix
                K_list = moge.get(CameraCalibrationKeys.intrinsics)
                if K_list and isinstance(K_list, list) and len(K_list) > 0:
                    intrinsics_list = K_list

                # Also get hfov as fallback
                hfov = moge.get(CameraCalibrationKeys.hfov)
                if hfov is not None:
                    if isinstance(hfov, list) and hfov:
                        fov_x = float(np.median(hfov))
                    else:
                        fov_x = float(hfov)

        # Try HaWoR fov_x (HaWoR uses median of MoGe hfov, most consistent)
        if fov_x is None:
            hawor_field = MetaKeys.hand_reconstruction_hawor_tags
            hawor_list = meta.get(hawor_field, [])
            if hawor_list:
                hawor = hawor_list[0] if isinstance(hawor_list, list) else hawor_list
                if isinstance(hawor, dict):
                    hawor_fov = hawor.get("fov_x")
                    if hawor_fov is not None:
                        fov_x = float(hawor_fov)

        return intrinsics_list, fov_x

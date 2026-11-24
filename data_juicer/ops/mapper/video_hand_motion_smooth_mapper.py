import numpy as np
from loguru import logger

from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.lazy_loader import LazyLoader

from ..base_op import OPERATORS, Mapper

OP_NAME = "video_hand_motion_smooth_mapper"

scipy_interpolate = LazyLoader("scipy.interpolate", "scipy")


def _recompute_actions(states: np.ndarray) -> np.ndarray:
    """Compute 7-dim delta actions from consecutive 8-dim states.

    Args:
        states: (T, 8) array — [x, y, z, roll, pitch, yaw, pad, gripper].

    Returns:
        (T, 7) actions — [dx, dy, dz, droll, dpitch, dyaw, gripper].
    """
    from scipy.spatial.transform import Rotation

    T = len(states)
    actions = np.zeros((T, 7), dtype=np.float32)

    for t in range(T - 1):
        actions[t, 0:3] = states[t + 1, 0:3] - states[t, 0:3]
        R_prev = Rotation.from_euler("xyz", states[t, 3:6], degrees=False)
        R_next = Rotation.from_euler("xyz", states[t + 1, 3:6], degrees=False)
        R_delta = R_next * R_prev.inv()
        actions[t, 3:6] = R_delta.as_euler("xyz", degrees=False)
        actions[t, 6] = states[t + 1, 7]

    if T > 0:
        actions[T - 1, 6] = states[T - 1, 7]

    return actions


@OPERATORS.register_module(OP_NAME)
class VideoHandMotionSmoothMapper(Mapper):
    """Apply smoothing to world-space hand motions and remove outliers.

    Reads hand action results (states, actions, joints_world) produced by
    ``VideoHandActionComputeMapper`` and applies:

    1. **Extreme outlier replacement** — frames whose instantaneous wrist
       speed exceeds ``median + outlier_velocity_threshold * MAD`` are
       replaced by linear interpolation from neighbors (not deleted).
    2. **Savitzky-Golay smoothing** — positions are smoothed with a
       Savitzky-Golay filter that preserves motion peaks while removing
       high-frequency jitter.
    3. **Quaternion smoothing** — orientations are smoothed in quaternion
       space to avoid gimbal lock and discontinuities.
    4. **Action recomputation** — 7-DoF actions are re-derived from the
       smoothed states so they stay consistent.

    Reference (paper §3.1):
        "we apply spline smoothing to the world-space hand motions and remove
        outliers"
    """

    def __init__(
        self,
        hand_action_field: str = MetaKeys.hand_action_tags,
        savgol_window: int = 11,
        savgol_polyorder: int = 3,
        outlier_velocity_threshold: float = 5.0,
        min_frames_for_smoothing: int = 5,
        smooth_joints: bool = True,
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param hand_action_field: Meta field storing hand action results
            (output of VideoHandActionComputeMapper).
        :param savgol_window: Window length for Savitzky-Golay filter.
            Must be odd.  Larger = smoother but may lose fast motions.
        :param savgol_polyorder: Polynomial order for Savitzky-Golay filter.
            Must be less than savgol_window.
        :param outlier_velocity_threshold: Frames whose wrist speed exceeds
            ``median + threshold * MAD`` are replaced by interpolation.
            Higher = more conservative (fewer replacements).
        :param min_frames_for_smoothing: Minimum number of valid frames
            required to apply smoothing.
        :param smooth_joints: Whether to also smooth ``joints_world``
            (21-joint MANO skeleton in world space).
        """
        super().__init__(*args, **kwargs)
        self.hand_action_field = hand_action_field
        self.savgol_window = savgol_window
        self.savgol_polyorder = savgol_polyorder
        self.outlier_velocity_threshold = outlier_velocity_threshold
        self.min_frames_for_smoothing = min_frames_for_smoothing
        self.smooth_joints = smooth_joints

    # ------------------------------------------------------------------
    # Outlier replacement (interpolate, don't delete)
    # ------------------------------------------------------------------
    @staticmethod
    def _replace_outliers(
        positions: np.ndarray,
        threshold_mad: float,
    ) -> np.ndarray:
        """Replace extreme outlier frames by linear interpolation.

        Uses median + MAD (median absolute deviation) which is more robust
        than mean + std for heavy-tailed distributions.

        Returns a copy with outliers replaced — no frames are deleted.
        """
        n = len(positions)
        if n < 4:
            return positions.copy()

        result = positions.copy()
        velocities = np.diff(positions, axis=0)
        speed = np.linalg.norm(velocities, axis=1)

        median_speed = np.median(speed)
        mad = np.median(np.abs(speed - median_speed))
        if mad < 1e-8:
            return result

        limit = median_speed + threshold_mad * mad * 1.4826  # MAD→σ scale
        outlier_mask = speed > limit

        n_outliers = int(np.sum(outlier_mask))
        if n_outliers == 0:
            return result

        # Replace outlier destination frames by linear interpolation
        for i in range(len(outlier_mask)):
            if not outlier_mask[i]:
                continue
            target = i + 1  # destination frame of the jump

            # Find nearest good frames before and after
            prev_good = i
            while prev_good > 0 and (
                prev_good > i - 1
                and prev_good - 1 >= 0
                and prev_good - 1 < len(outlier_mask)
                and outlier_mask[prev_good - 1]
            ):
                prev_good -= 1

            next_good = target + 1
            while next_good < n and next_good - 1 < len(outlier_mask) and outlier_mask[next_good - 1]:
                next_good += 1

            if next_good >= n:
                next_good = n - 1
            if prev_good == target or next_good == target:
                continue

            # Linear interpolation
            alpha = (target - prev_good) / max(next_good - prev_good, 1)
            result[target] = (1 - alpha) * result[prev_good] + alpha * result[next_good]

        return result

    # ------------------------------------------------------------------
    # Savitzky-Golay smoothing
    # ------------------------------------------------------------------
    @staticmethod
    def _savgol_smooth(
        data: np.ndarray,
        window: int,
        polyorder: int,
    ) -> np.ndarray:
        """Apply Savitzky-Golay filter to each column of data."""
        from scipy.signal import savgol_filter

        n = len(data)
        # Ensure window is valid
        win = min(window, n)
        if win % 2 == 0:
            win -= 1
        if win < polyorder + 2:
            return data.copy()

        result = np.empty_like(data)
        if data.ndim == 1:
            result = savgol_filter(data, win, polyorder)
        else:
            for d in range(data.shape[1]):
                result[:, d] = savgol_filter(data[:, d], win, polyorder)
        return result

    # ------------------------------------------------------------------
    # Quaternion orientation smoothing
    # ------------------------------------------------------------------
    @staticmethod
    def _smooth_orientations(
        eulers: np.ndarray,
        window: int,
        polyorder: int,
    ) -> np.ndarray:
        """Smooth orientations in quaternion space with Savitzky-Golay."""
        from scipy.signal import savgol_filter
        from scipy.spatial.transform import Rotation

        n = len(eulers)
        win = min(window, n)
        if win % 2 == 0:
            win -= 1
        if win < polyorder + 2:
            return eulers.copy()

        try:
            rots = Rotation.from_euler("xyz", eulers, degrees=False)
            quats = rots.as_quat()  # (N, 4) — [x, y, z, w]

            # Ensure quaternion hemisphere continuity
            for i in range(1, len(quats)):
                if np.dot(quats[i], quats[i - 1]) < 0:
                    quats[i] = -quats[i]

            # Savgol on each component
            smoothed_quats = np.empty_like(quats)
            for d in range(4):
                smoothed_quats[:, d] = savgol_filter(
                    quats[:, d],
                    win,
                    polyorder,
                )

            # Re-normalize
            norms = np.linalg.norm(smoothed_quats, axis=1, keepdims=True)
            norms = np.clip(norms, 1e-8, None)
            smoothed_quats = smoothed_quats / norms

            return Rotation.from_quat(smoothed_quats).as_euler(
                "xyz",
                degrees=False,
            )
        except Exception:
            # Fallback: unwrap + savgol on Euler angles
            smoothed = np.empty_like(eulers)
            for d in range(3):
                unwrapped = np.unwrap(eulers[:, d])
                smoothed[:, d] = savgol_filter(unwrapped, win, polyorder)
            return smoothed

    # ------------------------------------------------------------------
    # Per-hand smoothing
    # ------------------------------------------------------------------
    def _smooth_hand_result(self, hand_result: dict) -> dict:
        """Smooth a single hand's trajectory and recompute actions."""
        states = np.asarray(hand_result["states"], dtype=np.float64)
        valid_frame_ids = hand_result["valid_frame_ids"]

        if len(states) < self.min_frames_for_smoothing:
            return hand_result

        positions = states[:, 0:3]
        eulers = states[:, 3:6]
        grippers = states[:, 7].copy()

        # --- Step 1: replace extreme outliers (don't delete!) ---
        positions = self._replace_outliers(
            positions,
            self.outlier_velocity_threshold,
        )

        # --- Step 2: Savitzky-Golay smoothing ---
        smoothed_pos = self._savgol_smooth(
            positions,
            self.savgol_window,
            self.savgol_polyorder,
        )
        smoothed_euler = self._smooth_orientations(
            eulers,
            self.savgol_window,
            self.savgol_polyorder,
        )

        # Reconstruct state matrix (same frame count as original)
        smoothed_states = np.zeros_like(states)
        smoothed_states[:, 0:3] = smoothed_pos
        smoothed_states[:, 3:6] = smoothed_euler
        smoothed_states[:, 6] = 0.0  # pad
        smoothed_states[:, 7] = grippers

        # --- Step 3: recompute actions ---
        actions = _recompute_actions(smoothed_states)

        result = {
            "hand_type": hand_result["hand_type"],
            "states": smoothed_states.astype(np.float32).tolist(),
            "actions": actions.tolist(),
            "valid_frame_ids": valid_frame_ids,  # unchanged! no frames removed
        }

        # --- Step 4: optionally smooth joints_world ---
        joints_world = hand_result.get("joints_world")
        if self.smooth_joints and joints_world and len(joints_world) > 0:
            joints_arr = np.asarray(joints_world, dtype=np.float64)
            if len(joints_arr) == len(states):
                smoothed_joints = np.empty_like(joints_arr)
                for j_idx in range(joints_arr.shape[1]):  # 21 joints
                    smoothed_joints[:, j_idx, :] = self._savgol_smooth(
                        joints_arr[:, j_idx, :],
                        self.savgol_window,
                        self.savgol_polyorder,
                    )
                result["joints_world"] = smoothed_joints.tolist()
            else:
                result["joints_world"] = joints_world
        else:
            result["joints_world"] = joints_world if joints_world else []

        # Pass through joints_cam unchanged
        result["joints_cam"] = hand_result.get("joints_cam", [])

        return result

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------
    def process_single(self, sample=None, rank=None):
        if Fields.meta not in sample:
            return sample

        hand_action_list = sample[Fields.meta].get(self.hand_action_field)
        if not hand_action_list:
            return sample

        smoothed_results = []
        for clip_result in hand_action_list:
            if not clip_result:
                smoothed_results.append(clip_result)
                continue
            smoothed_clip = {}
            for hand_type, hand_data in clip_result.items():
                if not hand_data or not hand_data.get("states"):
                    smoothed_clip[hand_type] = hand_data
                    continue
                try:
                    smoothed_clip[hand_type] = self._smooth_hand_result(
                        hand_data,
                    )
                except Exception as e:
                    logger.warning(
                        f"Smoothing failed for hand '{hand_type}': {e}. " f"Keeping original data.",
                    )
                    smoothed_clip[hand_type] = hand_data
            smoothed_results.append(smoothed_clip)

        sample[Fields.meta][self.hand_action_field] = smoothed_results
        return sample

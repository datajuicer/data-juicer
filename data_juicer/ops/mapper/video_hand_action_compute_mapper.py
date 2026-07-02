import numpy as np
from loguru import logger

from data_juicer.utils.constant import CameraCalibrationKeys, Fields, MetaKeys
from data_juicer.utils.file_utils import load_numpy
from data_juicer.utils.lazy_loader import LazyLoader

from ..base_op import OPERATORS, Mapper

OP_NAME = "video_hand_action_compute_mapper"

scipy_rotation = LazyLoader("scipy.spatial.transform", "scipy")


def _rotation_matrix_to_euler(R):
    """Convert 3x3 rotation matrix to Euler angles (roll, pitch, yaw).

    Uses scipy Rotation with 'xyz' extrinsic convention, consistent with
    LIBERO / Open X-Embodiment action space.
    """
    from scipy.spatial.transform import Rotation

    rot = Rotation.from_matrix(R)
    return rot.as_euler("xyz", degrees=False)  # (3,) [roll, pitch, yaw]


def _euler_to_rotation_matrix(euler):
    """Convert Euler angles (roll, pitch, yaw) to 3x3 rotation matrix."""
    from scipy.spatial.transform import Rotation

    return Rotation.from_euler("xyz", euler, degrees=False).as_matrix()


def _delta_rotation_euler(euler_prev, euler_next):
    """Compute relative rotation as Euler angles: R_delta = R_next @ R_prev^T."""
    from scipy.spatial.transform import Rotation

    R_prev = Rotation.from_euler("xyz", euler_prev, degrees=False)
    R_next = Rotation.from_euler("xyz", euler_next, degrees=False)
    R_delta = R_next * R_prev.inv()
    return R_delta.as_euler("xyz", degrees=False)


def _estimate_gripper_from_hand_pose(hand_pose):
    """Estimate gripper state from MANO hand_pose parameters.

    Uses the average rotation angle of all 15 finger joints to estimate
    whether the hand is open or closed.

    Accepts either:
      - (15, 3, 3) rotation matrices
      - (45,) axis-angle (3 values per joint)

    Returns:
        float: gripper state in [-1, 1]. 1 = open, -1 = closed.
    """
    if hand_pose is None or len(hand_pose) == 0:
        return 1.0  # default: open

    hand_pose = np.asarray(hand_pose, dtype=np.float64)

    # Convert axis-angle (45,) to per-joint angles
    if hand_pose.ndim == 1 and hand_pose.shape[0] == 45:
        # axis-angle: angle = norm of each 3-vector
        hand_pose = hand_pose.reshape(15, 3)
        angles = [np.linalg.norm(hand_pose[j]) for j in range(15)]
    elif hand_pose.ndim == 2 and hand_pose.shape == (15, 3):
        # Already (15, 3) axis-angle
        angles = [np.linalg.norm(hand_pose[j]) for j in range(15)]
    else:
        # (15, 3, 3) rotation matrices
        angles = []
        for j in range(hand_pose.shape[0]):
            R = hand_pose[j]
            trace_val = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
            angle = np.arccos(trace_val)
            angles.append(angle)

    avg_angle = np.mean(angles)

    # Thresholds calibrated from typical MANO hand poses:
    # - Fully open hand: avg angle ~0.1 rad
    # - Fully closed fist: avg angle ~0.8 rad
    open_threshold = 0.15
    close_threshold = 0.6

    if avg_angle <= open_threshold:
        return 1.0
    elif avg_angle >= close_threshold:
        return -1.0
    else:
        # Linear interpolation between open and closed
        t = (avg_angle - open_threshold) / (close_threshold - open_threshold)
        return 1.0 - 2.0 * t


@OPERATORS.register_module(OP_NAME)
class VideoHandActionComputeMapper(Mapper):
    """Compute 7-DoF actions and 8-dim states from hand reconstruction
    and camera pose results.

    Reads hand MANO parameters (from VideoHandReconstructionHaworMapper)
    and camera-to-world transforms (from VideoCameraPoseMegaSaMMapper),
    then produces per-frame state [x,y,z,roll,pitch,yaw,pad,gripper]
    and per-frame action [dx,dy,dz,droll,dpitch,dyaw,gripper] compatible
    with LIBERO / StarVLA LeRobot format.
    """

    def __init__(
        self,
        hand_reconstruction_field: str = MetaKeys.hand_reconstruction_hawor_tags,
        camera_pose_field: str = MetaKeys.video_camera_pose_tags,
        tag_field_name: str = MetaKeys.hand_action_tags,
        hand_type: str = "both",
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param hand_reconstruction_field: Meta field storing HaWoR hand
            reconstruction results.
        :param camera_pose_field: Meta field storing camera pose
            (cam_c2w) results.
        :param tag_field_name: Output field name in Fields.meta.
        :param hand_type: Which hand to compute actions for.
            'right', 'left', or 'both'. Default is 'both'.
        """
        super().__init__(*args, **kwargs)
        self.hand_reconstruction_field = hand_reconstruction_field
        self.camera_pose_field = camera_pose_field
        self.tag_field_name = tag_field_name
        self.hand_type = hand_type

    def _get_hand_data(self, hand_recon, hand_type):
        """Extract frame-indexed hand data for the specified hand type."""
        # Support both new structured format and legacy flat format
        hand = hand_recon.get(hand_type, {}) if isinstance(hand_recon, dict) else {}
        if not hand:
            return [], [], [], []
        frame_ids = hand.get("frame_ids", [])
        transl_list = hand.get("transl", [])
        orient_list = hand.get("global_orient", [])
        hand_pose_list = hand.get("hand_pose", [])

        return frame_ids, transl_list, orient_list, hand_pose_list

    def _compute_state_for_frame(self, transl, global_orient, hand_pose, cam_c2w):
        """Compute 8-dim state for a single frame.

        Transforms hand pose from camera space to world space.

        Args:
            transl: (3,) translation in camera space
            global_orient: (3,3) rotation matrix OR (3,) axis-angle
            hand_pose: hand pose parameters (rotation matrices or axis-angle)
            cam_c2w: (4,4) camera-to-world transform

        Returns:
            np.ndarray: (8,) [x, y, z, roll, pitch, yaw, pad, gripper]
        """
        from scipy.spatial.transform import Rotation

        transl = np.asarray(transl, dtype=np.float64)
        global_orient = np.asarray(global_orient, dtype=np.float64)
        cam_c2w = np.asarray(cam_c2w, dtype=np.float64)

        # Convert axis-angle (3,) to rotation matrix (3,3) if needed
        if global_orient.shape == (3,):
            global_orient = Rotation.from_rotvec(global_orient).as_matrix()

        # Transform position: camera → world
        R_c2w = cam_c2w[:3, :3]
        t_c2w = cam_c2w[:3, 3]
        pos_world = R_c2w @ transl + t_c2w

        # Transform orientation: camera → world
        orient_world = R_c2w @ global_orient
        euler = _rotation_matrix_to_euler(orient_world)  # [roll, pitch, yaw]

        # Estimate gripper state from finger articulation
        gripper = _estimate_gripper_from_hand_pose(hand_pose)

        state = np.array(
            [
                pos_world[0],
                pos_world[1],
                pos_world[2],
                euler[0],
                euler[1],
                euler[2],
                0.0,  # pad (consistent with LIBERO 8-dim state)
                gripper,
            ],
            dtype=np.float32,
        )
        return state

    def _compute_actions(self, states):
        """Compute 7-dim delta actions from consecutive states.

        action[t] = state[t+1] - state[t] for position,
                    delta_rotation for orientation,
                    gripper from state[t].

        The last frame's action is set to zeros with current gripper.

        Returns:
            np.ndarray: (T, 7) actions
        """
        T = len(states)
        actions = np.zeros((T, 7), dtype=np.float32)

        for t in range(T - 1):
            # Position delta
            actions[t, 0:3] = states[t + 1, 0:3] - states[t, 0:3]

            # Rotation delta (in Euler angles)
            euler_cur = states[t, 3:6]
            euler_next = states[t + 1, 3:6]
            actions[t, 3:6] = _delta_rotation_euler(euler_cur, euler_next)

            # Gripper: use next frame's gripper state
            actions[t, 6] = states[t + 1, 7]  # index 7 is gripper in state

        # Last frame: zero action with current gripper
        if T > 0:
            actions[T - 1, 6] = states[T - 1, 7]

        return actions

    @staticmethod
    def _transform_joints_cam_to_world(joints_cam, cam_c2w_all, frame_ids):
        """Transform MANO 21-joint positions from camera space to world space.

        Args:
            joints_cam: numpy array (T_all, 21, 3) or None — all valid frames'
                joints in camera space (from HaWoR).
            cam_c2w_all: numpy array (N, 4, 4) — camera-to-world transforms.
            frame_ids: list of frame indices that have valid hand data.

        Returns:
            joints_world: list of (21, 3) arrays for valid frames, or None.
        """
        if joints_cam is None:
            return None

        joints_cam = np.asarray(joints_cam, dtype=np.float64)
        joints_world_list = []

        for i, fid in enumerate(frame_ids):
            if fid >= len(cam_c2w_all) or i >= len(joints_cam):
                continue
            R_c2w = cam_c2w_all[fid, :3, :3]  # (3, 3)
            t_c2w = cam_c2w_all[fid, :3, 3]  # (3,)
            # joints_cam[i]: (21, 3) in camera space
            # world = R @ cam + t  (applied per joint)
            j_world = (R_c2w @ joints_cam[i].T).T + t_c2w  # (21, 3)
            joints_world_list.append(j_world.tolist())

        return joints_world_list

    def _compute_hand_actions(self, hand_recon, cam_c2w_all, hand_type, video_idx):
        """Compute states and actions for a single hand in a single video.

        Returns:
            dict with 'states', 'actions', 'valid_frame_ids', 'hand_type',
            'joints_cam', 'joints_world', or None if insufficient data.
        """
        frame_ids, transl_list, orient_list, hp_list = self._get_hand_data(hand_recon, hand_type)

        if len(frame_ids) < 2:
            logger.debug(f"Video {video_idx}: insufficient {hand_type} hand " f"frames ({len(frame_ids)}), skipping.")
            return None

        states = []
        valid_frame_ids = []
        for i, fid in enumerate(frame_ids):
            if fid >= len(cam_c2w_all):
                logger.debug(f"Frame {fid} exceeds cam_c2w length " f"{len(cam_c2w_all)}, skipping.")
                continue

            state = self._compute_state_for_frame(transl_list[i], orient_list[i], hp_list[i], cam_c2w_all[fid])
            states.append(state)
            valid_frame_ids.append(fid)

        if len(states) < 2:
            return None

        states = np.stack(states, axis=0)  # (T, 8)
        actions = self._compute_actions(states)  # (T, 7)

        # Transform MANO joints from camera space to world space
        hand_data = hand_recon[hand_type]
        joints_cam = hand_data.get("joints_cam", None)
        joints_world = self._transform_joints_cam_to_world(joints_cam, cam_c2w_all, frame_ids)

        # Also pass through joints_cam for valid frames only
        joints_cam_valid = None
        if joints_cam is not None:
            joints_cam = np.asarray(joints_cam, dtype=np.float64)
            joints_cam_valid = [
                joints_cam[i].tolist()
                for i, fid in enumerate(frame_ids)
                if fid < len(cam_c2w_all) and i < len(joints_cam)
            ]

        return {
            "states": states.tolist(),
            "actions": actions.tolist(),
            "valid_frame_ids": valid_frame_ids,
            "hand_type": hand_type,
            "joints_cam": joints_cam_valid,  # (T, 21, 3) camera space
            "joints_world": joints_world,  # (T, 21, 3) world space
        }

    def process_single(self, sample=None, rank=None):
        # Check if already processed
        if self.tag_field_name in sample.get(Fields.meta, {}):
            return sample

        if Fields.meta not in sample:
            sample[Fields.meta] = {}

        hand_recon_list = sample[Fields.meta].get(self.hand_reconstruction_field, [])
        camera_pose_list = sample[Fields.meta].get(self.camera_pose_field, [])

        if not hand_recon_list or not camera_pose_list:
            logger.warning(
                f"Missing hand reconstruction or camera pose data. "
                f"hand_recon={len(hand_recon_list)}, "
                f"camera_pose={len(camera_pose_list)}"
            )
            sample[Fields.meta][self.tag_field_name] = []
            return sample

        # Determine which hands to process
        if self.hand_type == "both":
            hand_types = ["right", "left"]
        else:
            hand_types = [self.hand_type]

        all_video_results = []

        if len(hand_recon_list) != len(camera_pose_list):
            logger.warning(
                f"hand_recon ({len(hand_recon_list)}) and camera_pose "
                f"({len(camera_pose_list)}) list length mismatch. "
                f"Processing min of both."
            )

        for video_idx in range(min(len(hand_recon_list), len(camera_pose_list))):
            hand_recon = hand_recon_list[video_idx]
            camera_pose = camera_pose_list[video_idx]

            cam_c2w_raw = camera_pose.get(CameraCalibrationKeys.cam_c2w, None) if camera_pose else None
            if cam_c2w_raw is None:
                logger.warning(f"Video {video_idx}: missing cam_c2w, skipping.")
                empty = {
                    ht: {
                        "states": [],
                        "actions": [],
                        "valid_frame_ids": [],
                        "hand_type": ht,
                        "joints_cam": [],
                        "joints_world": [],
                    }
                    for ht in hand_types
                }
                all_video_results.append(empty)
                continue

            cam_c2w_all = np.asarray(load_numpy(cam_c2w_raw), dtype=np.float64)

            video_result = {}
            for ht in hand_types:
                result = self._compute_hand_actions(hand_recon, cam_c2w_all, ht, video_idx)
                if result is None:
                    video_result[ht] = {
                        "states": [],
                        "actions": [],
                        "valid_frame_ids": [],
                        "hand_type": ht,
                        "joints_cam": [],
                        "joints_world": [],
                    }
                else:
                    video_result[ht] = result

            all_video_results.append(video_result)

        sample[Fields.meta][self.tag_field_name] = all_video_results
        return sample

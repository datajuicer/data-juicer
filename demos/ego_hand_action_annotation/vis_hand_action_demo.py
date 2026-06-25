#!/usr/bin/env python
"""Demo: Verify action annotations with 3D hand mesh + trajectory overlay.

Combines:
  - 3D MANO hand mesh wireframe rendering
  - Action trajectory verification

Usage:
    python vis_hand_action_demo.py \
        --data_path data.pkl \
        --save_dir ./vis_action_verify
"""

import argparse
import fractions
import os
import pickle
import json
import sys

import cv2
import numpy as np
import pyarrow.parquet as pq

import torch

from hawor_utils.common_utils import prepare_hawor_and_add_to_path
prepare_hawor_and_add_to_path()

from hawor_utils.patches.process import get_mano_faces, run_mano, run_mano_left

from data_juicer.utils.constant import Fields, MetaKeys


def load_image(image_input):
    if isinstance(image_input, (str, bytes)):
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise ValueError(f"Error: File not found at {image_input}")
            img = cv2.imread(image_input)
        else:
            nparr = np.frombuffer(image_input, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Failed to decode image.")

        return img

    else:
        raise TypeError("Input must be a file path (str) or image bytes (bytes).")


# ---------------------------------------------------------------
# MANO hand mesh (using HaWoR's run_mano / run_mano_left)
# ---------------------------------------------------------------

def compute_hand_mesh(hand_transl, hand_orient, hand_pose, hand_betas,
                      is_left=False):
    """Compute MANO hand mesh vertices, joints and faces.

    Uses HaWoR's run_mano (right) / run_mano_left (left) to ensure
    correct coordinate conventions.

    Args:
        hand_transl: list of (3,) translations
        hand_orient: list of (3,) axis-angle global orientations
        hand_pose: list of (45,) axis-angle hand poses
        hand_betas: list of (10,) shape parameters
        is_left: whether this is the left hand

    Returns:
        vertices: (T, V, 3) numpy array in camera space
        joints: (T, J, 3) numpy array in camera space
        faces: (F, 3) numpy array of face indices
    """
    transl = torch.tensor(hand_transl, dtype=torch.float32).unsqueeze(0)
    rot = torch.tensor(hand_orient, dtype=torch.float32).unsqueeze(0)
    pose = torch.tensor(hand_pose, dtype=torch.float32).unsqueeze(0)
    betas = torch.tensor(hand_betas, dtype=torch.float32).unsqueeze(0)

    mano_fn = run_mano_left if is_left else run_mano
    mano_out = mano_fn(transl, rot, pose, betas=betas)

    vertices = mano_out['vertices'][0].cpu().numpy()  # (T, V, 3)
    joints = mano_out['joints'][0].cpu().numpy()       # (T, J, 3)

    # Build faces
    faces_base = get_mano_faces()
    faces_new = np.array([
        [92, 38, 234], [234, 38, 239], [38, 122, 239], [239, 122, 279],
        [122, 118, 279], [279, 118, 215], [118, 117, 215], [215, 117, 214],
        [117, 119, 214], [214, 119, 121], [119, 120, 121], [121, 120, 78],
        [120, 108, 78], [78, 108, 79],
    ])
    faces = np.concatenate([faces_base, faces_new], axis=0)
    if is_left:
        faces = faces[:, [0, 2, 1]]

    return vertices, joints, faces


def project_points_to_2d(points_3d, fov_x, width, height):
    """Project batch of 3D points to 2D pixel coords."""
    fx = width / (2.0 * np.tan(fov_x / 2.0))
    cx, cy = width / 2.0, height / 2.0
    z_safe = np.where(np.abs(points_3d[..., 2]) < 1e-6, 1e-6, points_3d[..., 2])
    u = fx * points_3d[..., 0] / z_safe + cx
    v = fx * points_3d[..., 1] / z_safe + cy
    return np.stack([u, v], axis=-1)


def draw_mesh_filled(frame, verts_2d, faces, color, alpha=0.3):
    """Draw filled semi-transparent mesh on frame."""
    overlay = frame.copy()
    h, w = frame.shape[:2]
    for face in faces:
        pts = verts_2d[face].astype(np.int32)
        if np.any(pts[:, 0] < -w) or np.any(pts[:, 0] > 2 * w):
            continue
        if np.any(pts[:, 1] < -h) or np.any(pts[:, 1] > 2 * h):
            continue
        cv2.fillPoly(overlay, [pts], color=color, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_mesh_wireframe(frame, verts_2d, faces, color, alpha=0.4, thickness=1):
    """Draw mesh wireframe on frame."""
    overlay = frame.copy()
    h, w = frame.shape[:2]
    for face in faces:
        pts = verts_2d[face].astype(np.int32)
        if np.any(pts[:, 0] < -w) or np.any(pts[:, 0] > 2 * w):
            continue
        if np.any(pts[:, 1] < -h) or np.any(pts[:, 1] > 2 * h):
            continue
        cv2.polylines(overlay, [pts], isClosed=True, color=color,
                      thickness=thickness, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


# ---------------------------------------------------------------
# MANO joint skeleton visualization
# ---------------------------------------------------------------

# MANO 21-joint skeleton: wrist(0), index(1-4), middle(5-8),
# ring(9-12), pinky(13-16), thumb(17-20)
MANO_SKELETON_BONES = [
    # Index finger
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Middle finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Ring finger
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Pinky finger
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Thumb
    (0, 17), (17, 18), (18, 19), (19, 20),
]

# Per-finger colors (BGR) for distinct visualization
FINGER_COLORS = {
    "thumb":  (0, 255, 255),   # yellow
    "index":  (0, 0, 255),     # red
    "middle": (0, 255, 0),     # green
    "ring":   (255, 165, 0),   # orange-ish
    "pinky":  (255, 0, 255),   # magenta
}

BONE_FINGER_MAP = {
    0: "index", 1: "index", 2: "index", 3: "index",
    4: "middle", 5: "middle", 6: "middle", 7: "middle",
    8: "ring", 9: "ring", 10: "ring", 11: "ring",
    12: "pinky", 13: "pinky", 14: "pinky", 15: "pinky",
    16: "thumb", 17: "thumb", 18: "thumb", 19: "thumb",
}


def draw_joints(frame, joints_2d, bone_color_override=None,
                joint_radius=4, bone_thickness=2, alpha=0.8):
    """Draw MANO hand joints and skeleton bones on frame.

    Args:
        frame: BGR image (modified in-place).
        joints_2d: (J, 2) array of 2D joint positions.
        bone_color_override: if set, use this single color for all bones
            instead of per-finger colors.
        joint_radius: radius of joint circles.
        bone_thickness: line thickness for bones.
        alpha: blending alpha for the overlay.
    """
    overlay = frame.copy()
    h, w = frame.shape[:2]

    # Draw bones
    for bone_idx, (j1, j2) in enumerate(MANO_SKELETON_BONES):
        pt1 = joints_2d[j1].astype(np.int32)
        pt2 = joints_2d[j2].astype(np.int32)
        # Skip if far out of frame
        if (pt1[0] < -w or pt1[0] > 2 * w or pt1[1] < -h or pt1[1] > 2 * h
                or pt2[0] < -w or pt2[0] > 2 * w or pt2[1] < -h
                or pt2[1] > 2 * h):
            continue
        if bone_color_override is not None:
            color = bone_color_override
        else:
            finger = BONE_FINGER_MAP[bone_idx]
            color = FINGER_COLORS[finger]
        cv2.line(overlay, tuple(pt1), tuple(pt2), color,
                 bone_thickness, cv2.LINE_AA)

    # Draw joint circles
    for j in range(joints_2d.shape[0]):
        pt = joints_2d[j].astype(np.int32)
        if pt[0] < -w or pt[0] > 2 * w or pt[1] < -h or pt[1] > 2 * h:
            continue
        # Wrist joint is larger
        r = joint_radius + 2 if j == 0 else joint_radius
        cv2.circle(overlay, tuple(pt), r, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(overlay, tuple(pt), r, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


# ---------------------------------------------------------------
# Trajectory reconstruction from actions
# ---------------------------------------------------------------


def world_to_camera(pos_world, cam_c2w):
    """Convert world position to camera space using cam_c2w."""
    cam_c2w = np.asarray(cam_c2w, dtype=np.float64)
    R = cam_c2w[:3, :3]
    t = cam_c2w[:3, 3]
    pos_cam = R.T @ (pos_world - t)
    return pos_cam


def project_to_2d(pos_cam, fov_x, w, h):
    """Project camera-space position to 2D pixel coords."""
    fx = w / (2.0 * np.tan(fov_x / 2.0))
    cx, cy = w / 2.0, h / 2.0
    z = pos_cam[2] if abs(pos_cam[2]) > 1e-6 else 1e-6
    u = fx * pos_cam[0] / z + cx
    v = fx * pos_cam[1] / z + cy
    return np.array([u, v])


# ---------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------

def draw_trajectory(frame, points_2d, color, thickness=2, dot_radius=4):
    """Draw trajectory line with dots."""
    for i in range(1, len(points_2d)):
        pt1 = tuple(points_2d[i - 1].astype(int))
        pt2 = tuple(points_2d[i].astype(int))
        cv2.line(frame, pt1, pt2, color, thickness, cv2.LINE_AA)
    for i, pt in enumerate(points_2d):
        cv2.circle(frame, tuple(pt.astype(int)), dot_radius, color, -1, cv2.LINE_AA)


def draw_current_marker(frame, pt, color, label=""):
    """Draw a highlighted marker for current position."""
    pt_int = tuple(pt.astype(int))
    cv2.circle(frame, pt_int, 10, color, -1, cv2.LINE_AA)
    cv2.circle(frame, pt_int, 12, (255, 255, 255), 2, cv2.LINE_AA)
    if label:
        cv2.putText(frame, label, (pt_int[0] + 15, pt_int[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def draw_action_info(frame, hand_infos, frame_idx):
    """Draw per-hand state and action values on frame."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w = frame.shape[:2]
    x = 10
    y0 = h - 22 * (len(hand_infos) * 3 + 1) - 10

    lines = [(f"Frame {frame_idx}", (255, 255, 255))]
    for info in hand_infos:
        hand_label = "R" if info["hand"] == "right" else "L"
        state = info["state"]
        action = info["action"]
        grip_str = "OPEN" if state[7] > 0 else "CLOSED"
        grip_color = (0, 255, 0) if state[7] > 0 else (0, 0, 255)

        lines.append((
            f"[{hand_label}] Pos: [{state[0]:+.3f},{state[1]:+.3f},{state[2]:+.3f}]"
            f"  Rot: [{state[3]:+.2f},{state[4]:+.2f},{state[5]:+.2f}]",
            (255, 255, 255)))
        lines.append((
            f"[{hand_label}] dPos: [{action[0]:+.4f},{action[1]:+.4f},{action[2]:+.4f}]"
            f"  dRot: [{action[3]:+.3f},{action[4]:+.3f},{action[5]:+.3f}]",
            (200, 200, 200)))
        lines.append((f"[{hand_label}] Grip: {grip_str} ({state[7]:+.2f})", grip_color))

    yy = y0
    for text, color in lines:
        (tw, th), _ = cv2.getTextSize(text, font, 0.45, 1)
        cv2.rectangle(frame, (x - 2, yy - th - 2), (x + tw + 4, yy + 4),
                      (0, 0, 0), -1)
        cv2.putText(frame, text, (x, yy), font, 0.45, color, 1, cv2.LINE_AA)
        yy += 22


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Verify action annotations with 3D hand mesh + trajectory (both hands)",
    )
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to data (must have hand_action_tags)")
    parser.add_argument("--save_dir", type=str, default="./vis_action_verify")
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--video_idx", type=int, default=0)
    parser.add_argument("--fps", type=float, default=30)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Load data
    print(f"Loading: {args.data_path}")
    with open(args.data_path, "rb") as f:
        if args.data_path.endswith(('.json', '.jsonl')):
            samples = [json.loads(line) for line in f.readlines()]
        elif args.data_path.endswith('.pkl'):
            samples = pickle.load(f)
        elif args.data_path.endswith('.parquet'):
            table = pq.read_table(f)
            samples = table.to_pylist()

    tgt_sample = samples[args.sample_idx]
    meta = tgt_sample[Fields.meta]
    if isinstance(meta, bytes):
        meta = pickle.loads(meta)

    frames = tgt_sample[MetaKeys.video_frames][args.video_idx]

    assert MetaKeys.hand_action_tags in meta, "Need hand_action_tags"
    assert MetaKeys.hand_reconstruction_hawor_tags in meta, "Need hawor tags"
    assert MetaKeys.video_camera_pose_tags in meta, "Need camera pose tags"

    action_tags = meta[MetaKeys.hand_action_tags][args.video_idx]
    hawor = meta[MetaKeys.hand_reconstruction_hawor_tags][args.video_idx]
    cam_pose = meta[MetaKeys.video_camera_pose_tags][args.video_idx]

    from data_juicer.utils.file_utils import load_numpy
    cam_c2w_all = load_numpy(cam_pose["cam_c2w"])

    fov_x = hawor["fov_x"]

    # Detect data format: new {"right":{...}, "left":{...}} or old flat
    if "states" in action_tags:
        # Old flat format — wrap as single hand (assume right)
        ht = action_tags.get("hand_type", "right")
        action_tags = {ht: action_tags}

    img = load_image(frames[0])
    img_h, img_w = img.shape[:2]

    # ---- Process BOTH hands ----
    hand_colors = {
        "right": {"gt": (0, 255, 0), "recon": (0, 165, 255), "mesh": (180, 120, 200)},
        "left":  {"gt": (255, 255, 0), "recon": (255, 0, 255), "mesh": (200, 150, 50)},
    }

    hand_results = {}

    for hand_type in ["right", "left"]:
        # Read action data from pipeline output
        hand_action = action_tags.get(hand_type, {})
        states_raw = hand_action.get("states", [])
        actions_raw = hand_action.get("actions", [])
        valid_ids = hand_action.get("valid_frame_ids", [])

        if len(states_raw) < 2:
            print(f"\n  {hand_type} hand: no action data from pipeline, skipping")
            continue

        states = np.array(states_raw, dtype=np.float64)
        actions = np.array(actions_raw, dtype=np.float64)

        print(f"\n  === {hand_type.upper()} hand ===")
        is_left = (hand_type == "left")

        # MANO mesh from hawor data (support both new and legacy format)
        if hand_type in hawor and isinstance(hawor[hand_type], dict):
            hand = hawor[hand_type]
            frame_ids = hand.get("frame_ids", [])
            hand_transl = hand.get("transl", [])
            hand_orient = hand.get("global_orient", [])
            hand_pose = hand.get("hand_pose", [])
            hand_betas = hand.get("betas", [])
        else:
            prefix = f"{hand_type}_"
            frame_ids = hawor.get(f"{prefix}frame_id_list", [])
            hand_transl = hawor.get(f"{prefix}transl_list", [])
            hand_orient = hawor.get(f"{prefix}global_orient_list", [])
            hand_pose = hawor.get(f"{prefix}hand_pose_list", [])
            hand_betas = hawor.get(f"{prefix}beta_list", [])

        # Read pipeline's joints_cam from action_tags (21 MANO joints)
        joints_cam_raw = hand_action.get("joints_cam", None)
        if joints_cam_raw is not None and len(joints_cam_raw) > 0:
            joints_cam = np.array(joints_cam_raw, dtype=np.float64)
            joints_cam_id_map = {fid: i for i, fid in enumerate(valid_ids)}
            print(f"    joints_cam (from action_tags): {joints_cam.shape}")
        else:
            joints_cam = None
            joints_cam_id_map = {}

        # Backup mesh_joints, used when joints_cam is not provided in the data.
        if len(frame_ids) >= 2:
            mesh_verts, mesh_joints, mesh_faces = compute_hand_mesh(
                hand_transl, hand_orient, hand_pose, hand_betas,
                is_left=is_left,
            )
            print(f"    Mesh: vertices {mesh_verts.shape}")
        else:
            mesh_verts, mesh_joints, mesh_faces, frame_ids = (
                None, None, None, [])

        T = len(states)
        print(f"    States: {states.shape}, Actions: {actions.shape}, Valid: {len(valid_ids)}")

        mesh_id_map = {fid: i for i, fid in enumerate(frame_ids)}

        # Project action states to 2D at actual wrist joint position.
        # Convert world-space states back to camera space, then shift
        # from MANO transl to wrist joint so trajectory sits on the hand.
        wrist_2d = []
        for t in range(T):
            fid = valid_ids[t]
            pos_cam = world_to_camera(states[t, :3], cam_c2w_all[fid])
            if joints_cam is not None and fid in joints_cam_id_map:
                jidx = joints_cam_id_map[fid]
                wrist_cam = joints_cam[jidx, 0, :]
                # Find matching hawor transl for this frame
                if fid in mesh_id_map:
                    midx = mesh_id_map[fid]
                    pos_cam = pos_cam + (wrist_cam - np.asarray(hand_transl[midx]))
            # Backup mesh_joints, used when joints_cam is not provided in the data.
            elif mesh_joints is not None and fid in mesh_id_map:
                midx = mesh_id_map[fid]
                wrist_cam = mesh_joints[midx, 0, :]
                pos_cam = pos_cam + (wrist_cam - np.asarray(hand_transl[midx]))
            wrist_2d.append(project_to_2d(pos_cam, fov_x, img_w, img_h))
        wrist_2d = np.array(wrist_2d)

        hand_results[hand_type] = {
            "states": states,
            "actions": actions,
            "valid_ids": valid_ids,
            "wrist_2d": wrist_2d,
            "mesh_verts": mesh_verts,
            "mesh_joints": mesh_joints,
            "mesh_faces": mesh_faces,
            "mesh_id_map": mesh_id_map,
            "frame_ids": frame_ids,
            "joints_cam": joints_cam,
            "joints_cam_id_map": joints_cam_id_map,
        }

    if not hand_results:
        print("No valid hand data found!")
        return

    # Collect all valid frame ids across both hands
    all_valid_fids = sorted(set().union(
        *(set(r["valid_ids"]) for r in hand_results.values())))

    # ---- Render video frames ----
    print(f"\nRendering {len(all_valid_fids)} frames...")
    frames_dir = os.path.join(args.save_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    output_frames = []

    for frame_seq, fid in enumerate(all_valid_fids):
        frame = load_image(frames[fid])
        if frame is None:
            continue

        canvas = frame.copy()
        hand_infos = []

        for hand_type, res in hand_results.items():
            colors = hand_colors[hand_type]

            # -- Draw 3D hand mesh --
            if res["mesh_verts"] is not None and fid in res["mesh_id_map"]:
                midx = res["mesh_id_map"][fid]
                verts_2d = project_points_to_2d(
                    res["mesh_verts"][midx], fov_x, img_w, img_h)
                draw_mesh_filled(canvas, verts_2d, res["mesh_faces"],
                                 colors["mesh"], alpha=0.25)
                draw_mesh_wireframe(canvas, verts_2d, res["mesh_faces"],
                                    colors["mesh"], alpha=0.5, thickness=1)

            # -- Draw 21 MANO joints from pipeline's joints_cam --
            if res["joints_cam"] is not None and fid in res["joints_cam_id_map"]:
                jidx = res["joints_cam_id_map"][fid]
                joints_2d = project_points_to_2d(
                    res["joints_cam"][jidx], fov_x, img_w, img_h)
                draw_joints(canvas, joints_2d,
                            joint_radius=4, bone_thickness=2,
                            alpha=0.85)
            # Backup mesh_joints, used when joints_cam is not provided in the data.
            elif res["mesh_joints"] is not None:
                joints_2d = project_points_to_2d(
                    res["mesh_joints"][midx], fov_x, img_w, img_h)
                draw_joints(canvas, joints_2d,
                            joint_radius=4, bone_thickness=2,
                            alpha=0.85)

            # -- Draw wrist trajectory --
            if fid not in set(res["valid_ids"]):
                continue

            t_idx = res["valid_ids"].index(fid)
            trail = res["wrist_2d"][:t_idx + 1]

            if len(trail) >= 2:
                draw_trajectory(canvas, trail, colors["gt"],
                                thickness=3, dot_radius=4)

            label_prefix = "R" if hand_type == "right" else "L"
            draw_current_marker(canvas, res["wrist_2d"][t_idx],
                                colors["gt"], label_prefix)

            hand_infos.append({
                "hand": hand_type,
                "state": res["states"][t_idx],
                "action": res["actions"][t_idx],
            })

        # Draw action state info
        if hand_infos:
            draw_action_info(canvas, hand_infos, fid)

        out_path = os.path.join(frames_dir, f"verify_{frame_seq:04d}.jpg")
        cv2.imwrite(out_path, canvas)
        output_frames.append(canvas)

    # ---- Save video ----
    if output_frames:
        import av
        video_path = os.path.join(args.save_dir, "action_verify.mp4")
        out_h, out_w = output_frames[0].shape[:2]
        fps_frac = fractions.Fraction(args.fps).limit_denominator(10000)

        container = av.open(video_path, mode="w")
        stream = container.add_stream("libx264", rate=fps_frac)
        stream.width = out_w
        stream.height = out_h
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18", "preset": "medium"}

        for f_bgr in output_frames:
            f_rgb = cv2.cvtColor(f_bgr, cv2.COLOR_BGR2RGB)
            av_frame = av.VideoFrame.from_ndarray(f_rgb, format="rgb24")
            for pkt in stream.encode(av_frame):
                container.mux(pkt)
        for pkt in stream.encode():
            container.mux(pkt)
        container.close()

        size_mb = os.path.getsize(video_path) / (1024 * 1024)
        print(f"\nVideo: {video_path} ({size_mb:.1f} MB)")

    print("Done!")


if __name__ == "__main__":
    main()

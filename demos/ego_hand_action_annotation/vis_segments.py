"""Visualize the results of VideoAtomicActionSegmentMapper and VideoTrajectoryOverlayMapper.

Features:
1. Save each atomic action segment as an independent video
2. Concatenate trajectory overlay results for easy checking
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


def write_video_ffmpeg(frames_bgr: list[np.ndarray], out_path: str, fps: float = 30.0):
    """Use ffmpeg pipe to write H.264 mp4, good compatibility."""
    if not frames_bgr:
        return
    import shutil
    import tempfile

    h, w = frames_bgr[0].shape[:2]

    # write to local /tmp (to avoid NFS not supporting faststart seek)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(tmp_fd)

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{w}x{h}",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "fast",
        "-crf", "23",
        "-movflags", "+faststart",
        tmp_path,
    ]
    raw_data = b"".join(f.tobytes() for f in frames_bgr)
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    _, stderr = proc.communicate(input=raw_data)
    if proc.returncode != 0:
        print(f"  ffmpeg 写入失败 (rc={proc.returncode}): {out_path}")
        print(f"  stderr: {stderr.decode()[-200:]}")
        os.unlink(tmp_path)
        return

    shutil.move(tmp_path, out_path)


def load_result(json_path: str) -> dict:
    with open(json_path) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────
# 1. atomic action segments → save as independent videos
# ─────────────────────────────────────────────────────────────
def export_atomic_segments_as_videos(
    data: dict,
    output_dir: str,
    fps: float = 30.0,
):
    """Save each atomic action segment as an independent video.
    
    Output directory structure:
        output_dir/
            left/
                seg0_f000-f040.mp4
                seg1_f041-f067.mp4
                ...
            right/
                seg0_f000-f028.mp4
                seg1_f029-f060.mp4
                ...
    """
    meta = data.get("__dj__meta__", {})
    segments = meta.get("atomic_action_segments", [])
    if not segments:
        print("Not found atomic_action_segments")
        return

    # get frame paths
    frames = data.get("video_frames", [])
    if frames and isinstance(frames[0], list):
        frames = frames[0]
    if not frames:
        print("Not found video_frames")
        return

    # separate by hand type
    for hand_type in ("left", "right"):
        hand_segs = [s for s in segments if s["hand_type"] == hand_type]
        if not hand_segs:
            continue

        hand_dir = os.path.join(output_dir, hand_type)
        os.makedirs(hand_dir, exist_ok=True)

        for seg in hand_segs:
            seg_id = seg["segment_id"]
            start = seg["start_frame"]
            end = seg["end_frame"]
            valid_fids = seg.get("valid_frame_ids", list(range(start, end + 1)))

            out_name = f"seg{seg_id}_f{start:03d}-f{end:03d}.mp4"
            out_path = os.path.join(hand_dir, out_name)

            collected_frames = []
            for fid in valid_fids:
                if fid >= len(frames):
                    continue
                img = cv2.imread(frames[fid])
                if img is None:
                    continue

                # put text label on the frame
                label = f"{hand_type.upper()} seg{seg_id} frame={fid}"
                cv2.putText(
                    img, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
                )
                collected_frames.append(img)

            if not collected_frames:
                print(f"  Skip {hand_type} seg{seg_id}: no valid frames")
                continue

            write_video_ffmpeg(collected_frames, out_path, fps)
            n_written = len(collected_frames)
            print(f"  [{hand_type}] seg{seg_id}: frames {start}-{end} "
                  f"({n_written} frames) → {out_path}")


# ─────────────────────────────────────────────────────────────
# 2. Trajectory overlay visualization → make grid image and video
#    - each segment's 8 overlay frames to a grid image
#    - all segment's overlay frames to a video
# ─────────────────────────────────────────────────────────────
def make_overlay_grid(
    data: dict,
    output_dir: str,
    grid_cols: int = 4,
    thumb_w: int = 480,
):
    """Make a 2xN grid image for each segment's overlay frames.
    """
    meta = data.get("__dj__meta__", {})
    segments = meta.get("atomic_action_segments", [])
    if not segments:
        print("Not found atomic_action_segments")
        return

    os.makedirs(output_dir, exist_ok=True)

    for seg in segments:
        hand_type = seg["hand_type"]
        seg_id = seg["segment_id"]
        overlay_paths = seg.get("overlay_frames", [])
        if not overlay_paths:
            continue

        # read all overlay frames
        imgs = []
        for p in overlay_paths:
            img = cv2.imread(p)
            if img is not None:
                imgs.append(img)
        if not imgs:
            continue

        # resize to uniform thumbnail size
        thumb_h = int(thumb_w * imgs[0].shape[0] / imgs[0].shape[1])
        thumbs = [cv2.resize(im, (thumb_w, thumb_h)) for im in imgs]

        # add frame index label
        sampled_indices = seg.get("sampled_frame_indices", [])
        for i, th in enumerate(thumbs):
            fid_label = f"f{sampled_indices[i]}" if i < len(sampled_indices) else f"#{i}"
            cv2.putText(
                th, fid_label, (5, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
            )

        # make grid
        n = len(thumbs)
        rows = (n + grid_cols - 1) // grid_cols
        # pad with blank
        while len(thumbs) < rows * grid_cols:
            thumbs.append(np.zeros_like(thumbs[0]))

        grid_rows = []
        for r in range(rows):
            row_imgs = thumbs[r * grid_cols: (r + 1) * grid_cols]
            grid_rows.append(np.hstack(row_imgs))
        grid = np.vstack(grid_rows)

        # add title bar on top
        title_h = 40
        title_bar = np.zeros((title_h, grid.shape[1], 3), dtype=np.uint8)
        title = (f"{hand_type.upper()} seg{seg_id} "
                 f"frames {seg['start_frame']}-{seg['end_frame']}")
        cv2.putText(
            title_bar, title, (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
        )
        grid = np.vstack([title_bar, grid])

        out_path = os.path.join(
            output_dir,
            f"grid_{hand_type}_seg{seg_id}_f{seg['start_frame']:03d}-f{seg['end_frame']:03d}.jpg",
        )
        cv2.imwrite(out_path, grid)
        print(f"  Grid: {out_path}")


def make_overlay_video(
    data: dict,
    output_path: str,
    fps: float = 2.0,
):
    """Make a video from all segment's overlay frames in sequence.

    Each segment's frames are separated by a separator frame.
    """
    meta = data.get("__dj__meta__", {})
    segments = meta.get("atomic_action_segments", [])
    if not segments:
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # determine video size first
    ref_img = None
    for seg in segments:
        for p in seg.get("overlay_frames", []):
            ref_img = cv2.imread(p)
            if ref_img is not None:
                break
        if ref_img is not None:
            break
    if ref_img is None:
        return

    h, w = ref_img.shape[:2]
    collected_frames = []

    for seg in segments:
        hand_type = seg["hand_type"]
        seg_id = seg["segment_id"]
        overlay_paths = seg.get("overlay_frames", [])
        sampled_indices = seg.get("sampled_frame_indices", [])

        # make a separator frame
        sep = np.zeros((h, w, 3), dtype=np.uint8)
        text = (f"{hand_type.upper()} Seg {seg_id}  "
                f"Frames {seg['start_frame']}-{seg['end_frame']}")
        cv2.putText(
            sep, text, (w // 6, h // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3,
        )
        collected_frames.append(sep)

        for i, p in enumerate(overlay_paths):
            img = cv2.imread(p)
            if img is None:
                continue
            img = cv2.resize(img, (w, h))
            fid_label = f"f{sampled_indices[i]}" if i < len(sampled_indices) else f"#{i}"
            label = f"{hand_type} seg{seg_id} {fid_label}"
            cv2.putText(
                img, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
            )
            collected_frames.append(img)

    write_video_ffmpeg(collected_frames, output_path, fps)
    print(f"  Overlay video: {output_path}")


# ─────────────────────────────────────────────────────────────
# 3. Segmentation timeline plot: plot speed curve + segmentation points
# ─────────────────────────────────────────────────────────────
def plot_segmentation_timeline(
    data: dict,
    output_path: str,
):
    """Plot left and right hand speed curves + atomic action segmentation points."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("Matplotlib is not installed, skipping timeline plot")
        return

    meta = data.get("__dj__meta__", {})
    hand_action = meta.get("hand_action_tags", [])
    segments = meta.get("atomic_action_segments", [])

    if not hand_action:
        return

    merged = hand_action[0] if isinstance(hand_action, list) else hand_action

    fig, axes = plt.subplots(2, 1, figsize=(18, 8), sharex=True)

    colors_seg = plt.cm.tab10.colors

    for ax_idx, hand_type in enumerate(("left", "right")):
        ax = axes[ax_idx]
        hdata = merged.get(hand_type, {})
        states = hdata.get("states", [])
        if not states:
            ax.set_title(f"{hand_type.upper()} hand - no data")
            continue

        states_arr = np.array(states, dtype=np.float64)
        positions = states_arr[:, 0:3]
        speed = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        speed = np.concatenate([[0.0], speed])

        ax.plot(speed, color="gray", alpha=0.5, label="raw speed")

        # Savitzky-Golay 平滑
        try:
            from scipy.signal import savgol_filter
            win = min(11, len(speed))
            if win % 2 == 0:
                win -= 1
            if win >= 3:
                smooth = savgol_filter(speed, win, polyorder=2)
                ax.plot(smooth, color="blue", linewidth=1.5, label="smooth speed")
        except Exception:
            pass

        # 画每个 segment 的区间
        hand_segs = [s for s in segments if s["hand_type"] == hand_type]
        for i, seg in enumerate(hand_segs):
            c = colors_seg[i % len(colors_seg)]
            ax.axvspan(
                seg["start_frame"], seg["end_frame"],
                alpha=0.15, color=c,
            )
            mid = (seg["start_frame"] + seg["end_frame"]) / 2
            ax.annotate(
                f"seg{seg['segment_id']}",
                (mid, ax.get_ylim()[1] * 0.9 if ax.get_ylim()[1] > 0 else 0.1),
                ha="center", fontsize=9, color=c, fontweight="bold",
            )
            # 切分线
            ax.axvline(seg["start_frame"], color=c, linestyle="--", alpha=0.5)
            ax.axvline(seg["end_frame"], color=c, linestyle="--", alpha=0.5)

        ax.set_title(f"{hand_type.upper()} Hand — Speed & Segments")
        ax.set_ylabel("Speed (m/frame)")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[1].set_xlabel("Frame")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Timeline: {output_path}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    data_dir = sys.argv[1]

    if len(sys.argv) < 2:
        print("Usage: python visualize_segments.py <data_dir>")
        return

    vis_dir = os.path.join(data_dir, "visualization")
    os.makedirs(vis_dir, exist_ok=True)

    # find all json results
    json_files = sorted(Path(data_dir).glob("*.json"))
    if not json_files:
        print(f"No json files found in {data_dir}")
        return

    for jf in json_files:
        print(f"\n{'='*60}")
        print(f"Processing: {jf.name}")
        print(f"{'='*60}")
        data = load_result(str(jf))

        sample_id = jf.stem
        sample_vis = os.path.join(vis_dir, sample_id)

        # 1. Atomic action segments → short videos
        print("\n[1] Export atomic action segments as videos (left and right hand separated)...")
        seg_video_dir = os.path.join(sample_vis, "atomic_segments")
        export_atomic_segments_as_videos(data, seg_video_dir, fps=30.0)

        # 2. Trajectory overlay grids
        print("\n[2] Generate trajectory overlay grids...")
        grid_dir = os.path.join(sample_vis, "overlay_grids")
        make_overlay_grid(data, grid_dir)

        # 3. Trajectory overlay video
        print("\n[3] Generate trajectory overlay continuous video...")
        overlay_vid = os.path.join(sample_vis, "overlay_all.mp4")
        make_overlay_video(data, overlay_vid, fps=2.0)

        # 4. Segmentation timeline plot
        print("\n[4] Generate segmentation timeline plot...")
        timeline_path = os.path.join(sample_vis, "segmentation_timeline.png")
        plot_segmentation_timeline(data, timeline_path)

    print(f"\nAll visualization results saved in: {vis_dir}")


if __name__ == "__main__":
    main()

import json
import os
import shutil
import subprocess
import uuid

import cv2
import numpy as np
from loguru import logger

from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.lazy_loader import LazyLoader

from ..base_op import OPERATORS, Mapper

OP_NAME = "export_to_lerobot_mapper"

pa = LazyLoader("pyarrow", "pyarrow")
pd = LazyLoader("pandas", "pandas")

DEFAULT_CHUNKS_SIZE = 1000


@OPERATORS.register_module(OP_NAME)
class ExportToLeRobotMapper(Mapper):
    """Export processed video data to LeRobot v2.0 dataset format (LIBERO-style).

    Designed for Ray distributed execution: each actor writes files
    independently using UUID-based names (no cross-process coordination).
    After all actors finish, call `finalize_dataset()` once to assign
    sequential episode indices, rename files, and generate metadata.

    Processing phase (parallel, per actor):
      staging/
      ├── data/{uuid}.parquet
      ├── videos/{uuid}.mp4
      └── meta/episodes_{uuid}.jsonl

    After finalize_dataset() (single-threaded):
      dataset_dir/
      ├── data/chunk-{NNN}/episode_XXXXXX.parquet
      ├── videos/chunk-{NNN}/observation.images.image/episode_XXXXXX.mp4
      └── meta/
          ├── info.json
          ├── tasks.jsonl
          ├── episodes.jsonl
          └── modality.json
    """

    def __init__(
        self,
        output_dir: str = "./lerobot_output",
        hand_action_field: str = "hand_action_tags",
        fps: int = 10,
        robot_type: str = "egodex_hand",
        chunks_size: int = DEFAULT_CHUNKS_SIZE,
        segment_field: str = None,
        frame_field: str = MetaKeys.video_frames,
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param output_dir: Root directory for the LeRobot dataset output.
        :param hand_action_field: Meta field with action/state data.
            Used in whole-video mode (segment_field=None).
        :param fps: Frames per second for the dataset.
        :param robot_type: Robot type identifier for info.json.
        :param chunks_size: Max episodes per chunk directory (default 1000).
        :param segment_field: Meta field storing atomic action segments.
            When set, each segment becomes a separate episode with its
            own caption as task description. When None (default), falls
            back to whole-video export via hand_action_field.
        :param frame_field: Sample field with extracted frame image paths.
            Used in segment mode to create per-segment videos.
        """
        super().__init__(*args, **kwargs)
        self.output_dir = output_dir
        self.hand_action_field = hand_action_field
        self.fps = fps
        self.robot_type = robot_type
        self.chunks_size = chunks_size
        self.segment_field = segment_field
        self.frame_field = frame_field

        # Staging directories for parallel-safe writes
        self.staging_data_dir = os.path.join(output_dir, "staging", "data")
        self.staging_video_dir = os.path.join(output_dir, "staging", "videos")
        self.staging_meta_dir = os.path.join(output_dir, "staging", "meta")
        self.meta_dir = os.path.join(output_dir, "meta")
        os.makedirs(self.staging_data_dir, exist_ok=True)
        os.makedirs(self.staging_video_dir, exist_ok=True)
        os.makedirs(self.staging_meta_dir, exist_ok=True)
        os.makedirs(self.meta_dir, exist_ok=True)

    def _stage_video(self, video_source, ep_uuid):
        """Copy or write video to staging with UUID name.

        :param video_source: Video file path (str) or video bytes (bytes).
        :param ep_uuid: Unique episode identifier.
        """
        if isinstance(video_source, bytes):
            dst = os.path.join(self.staging_video_dir, f"{ep_uuid}.mp4")
            if not os.path.exists(dst):
                with open(dst, "wb") as f:
                    f.write(video_source)
        else:
            ext = os.path.splitext(video_source)[1] or ".mp4"
            dst = os.path.join(self.staging_video_dir, f"{ep_uuid}{ext}")
            if not os.path.exists(dst):
                shutil.copy2(video_source, dst)
        return dst

    def _stage_parquet(self, states, actions, ep_uuid, valid_frame_ids=None):
        """Write parquet to staging with UUID name.

        episode_index, index, task_index are placeholders — they will
        be rewritten by finalize_dataset().

        :param valid_frame_ids: Original video frame indices corresponding
            to each state/action row. Used as frame_index so that LeRobot
            can align parquet rows with video frames. Falls back to
            sequential 0..T-1 when not provided.
        """
        T = len(states)
        states_arr = np.array(states, dtype=np.float32)
        actions_arr = np.array(actions, dtype=np.float32)

        rows = []
        for t in range(T):
            frame_id = valid_frame_ids[t] if valid_frame_ids else t
            rows.append(
                {
                    "observation.state": states_arr[t].tolist(),
                    "action": actions_arr[t].tolist(),
                    "timestamp": float(frame_id) / self.fps,
                    "frame_index": frame_id,
                    "episode_index": 0,  # placeholder
                    "index": t,  # placeholder
                    "task_index": 0,  # placeholder
                    "next.done": t == T - 1,
                }
            )

        df = pd.DataFrame(rows)
        path = os.path.join(self.staging_data_dir, f"{ep_uuid}.parquet")
        table = pa.Table.from_pandas(df)
        pa.parquet.write_table(table, path)

        return path, T

    def _stage_episode_meta(self, ep_uuid, num_frames, task_desc, video_path):
        """Write per-episode metadata to a UUID-named jsonl fragment.

        Each actor writes its own file — no cross-process contention.
        """
        meta_path = os.path.join(self.staging_meta_dir, f"{ep_uuid}.jsonl")
        if video_path and isinstance(video_path, str):
            video_ext = os.path.splitext(video_path)[1] or ".mp4"
        else:
            video_ext = ".mp4"
        entry = {
            "uuid": ep_uuid,
            "length": num_frames,
            "task": task_desc,
            "video_ext": video_ext,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    @staticmethod
    def _encode_frames_to_video(frame_paths, output_path, fps=30):
        """Encode a sequence of frame images into an H.264 mp4 video.

        :param frame_paths: List of image file paths.
        :param output_path: Destination mp4 path.
        :param fps: Output video frame rate.
        :return: output_path on success, None on failure.
        """
        if not frame_paths:
            return None

        # Read all frames and collect raw bytes upfront
        first = cv2.imread(frame_paths[0])
        if first is None:
            return None
        h, w = first.shape[:2]

        raw_chunks = [first.tobytes()]
        for p in frame_paths[1:]:
            img = cv2.imread(p)
            if img is None:
                continue
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h))
            raw_chunks.append(img.tobytes())
        raw_data = b"".join(raw_chunks)

        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{w}x{h}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "fast",
            "-crf",
            "23",
            "-movflags",
            "frag_keyframe+empty_moov",
            output_path,
        ]
        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            _, stderr = proc.communicate(input=raw_data)
            if proc.returncode != 0:
                logger.warning(f"ffmpeg encode failed (rc={proc.returncode}): " f"{stderr.decode()[-300:]}")
                return None
        except Exception as e:
            logger.warning(f"ffmpeg encode error: {e}")
            return None

        return output_path if os.path.exists(output_path) else None

    def _get_frame_paths(self, sample):
        """Get flat list of frame image paths from sample."""
        frame_data = sample.get(self.frame_field, [])
        if not frame_data:
            # Also check inside meta
            frame_data = sample.get(Fields.meta, {}).get(self.frame_field, [])
        # Unwrap nested list from reassembly: [[frames]] → [frames]
        if isinstance(frame_data, list) and frame_data and isinstance(frame_data[0], list):
            frame_data = frame_data[0]
        return frame_data

    def process_single(self, sample=None, rank=None):
        if Fields.meta not in sample:
            return sample

        if self.segment_field:
            return self._process_segments(sample)
        return self._process_whole_video(sample)

    def _process_segments(self, sample):
        """Per-segment export: each atomic action segment → one episode.

        Each segment's VLM caption becomes the episode's task description.
        A segment video is created from the extracted frame images.
        """
        meta = sample.get(Fields.meta, {})
        segments = meta.get(self.segment_field, [])
        if not segments:
            logger.warning("No segments found, skipping export.")
            sample[Fields.meta]["lerobot_export"] = []
            return sample

        all_frames = self._get_frame_paths(sample)
        exported_episodes = []

        for seg in segments:
            states = seg.get("states", [])
            actions = seg.get("actions", [])

            if len(states) < 2:
                continue

            # Pad actions with zeros if missing (e.g. last segment)
            if not actions or len(actions) < len(states):
                actions = [[0.0] * 7] * len(states)

            # Task description from VLM caption
            caption = seg.get("caption", {})
            if isinstance(caption, dict):
                task_desc = caption.get("action", "")
            else:
                task_desc = str(caption) if caption else ""

            # Skip segments explicitly marked as no action
            if task_desc == "N/A":
                continue

            # Fallback description when caption is empty
            if not task_desc:
                hand = seg.get("hand_type", "hand")
                task_desc = f"{hand} hand action"

            # Frame range and valid IDs
            start = seg.get("start_frame", 0)
            end = seg.get("end_frame", len(all_frames) - 1)
            valid_fids = seg.get("valid_frame_ids", list(range(start, end + 1)))
            # Convert to segment-relative frame indices (0-based)
            seg_relative_fids = [fid - start for fid in valid_fids]

            ep_uuid = uuid.uuid4().hex

            # Stage parquet
            parquet_path, num_frames = self._stage_parquet(states, actions, ep_uuid, seg_relative_fids)

            # Create segment video from frame images
            video_dst = None
            if all_frames:
                seg_frame_paths = [
                    all_frames[fid]
                    for fid in range(start, min(end + 1, len(all_frames)))
                    if fid < len(all_frames) and all_frames[fid]
                ]
                if seg_frame_paths:
                    video_path = os.path.join(self.staging_video_dir, f"{ep_uuid}.mp4")
                    video_dst = self._encode_frames_to_video(seg_frame_paths, video_path, self.fps)

            # Stage metadata
            self._stage_episode_meta(ep_uuid, num_frames, task_desc, video_dst)

            exported_episodes.append(
                {
                    "uuid": ep_uuid,
                    "parquet_path": parquet_path,
                    "video_path": video_dst,
                    "num_frames": num_frames,
                    "segment_id": seg.get("segment_id", -1),
                    "hand_type": seg.get("hand_type", "unknown"),
                }
            )

        sample[Fields.meta]["lerobot_export"] = exported_episodes
        return sample

    def _process_whole_video(self, sample):
        """Original whole-video export: one video → one episode."""
        action_data_list = sample[Fields.meta].get(self.hand_action_field, [])
        if not action_data_list:
            logger.warning("No hand action data found, skipping export.")
            return sample

        # Get task description from text field
        task_desc = sample.get(self.text_key, "")
        if not task_desc:
            task_desc = "manipulate object"

        # Get video sources (paths or bytes)
        video_sources = sample.get(self.video_key, [])

        # Track export results
        exported_episodes = []

        for video_idx, video_action_data in enumerate(action_data_list):
            # Support both old format (flat dict) and new format
            # (dict keyed by hand_type).
            if "states" in video_action_data:
                action_data = video_action_data
            else:
                action_data = {}
                for ht in ["right", "left"]:
                    hand_entry = video_action_data.get(ht, {})
                    if hand_entry.get("states", []):
                        action_data = hand_entry
                        break

            states = action_data.get("states", [])
            actions = action_data.get("actions", [])
            valid_frame_ids = action_data.get("valid_frame_ids", None)

            if len(states) < 2:
                continue

            # Generate a unique ID for this episode — no coordination
            ep_uuid = uuid.uuid4().hex

            # Write parquet to staging (use valid_frame_ids as frame_index)
            parquet_path, num_frames = self._stage_parquet(states, actions, ep_uuid, valid_frame_ids)

            # Copy/write video to staging (supports both path and bytes)
            video_dst = None
            if video_idx < len(video_sources):
                video_dst = self._stage_video(video_sources[video_idx], ep_uuid)

            # Write episode metadata fragment
            self._stage_episode_meta(ep_uuid, num_frames, task_desc, video_dst)

            exported_episodes.append(
                {
                    "uuid": ep_uuid,
                    "parquet_path": parquet_path,
                    "video_path": video_dst,
                    "num_frames": num_frames,
                }
            )

        sample[Fields.meta]["lerobot_export"] = exported_episodes
        return sample

    @staticmethod
    def _write_modality_json(meta_dir):
        """Write modality.json following StarVLA LIBERO convention."""
        modality = {
            "state": {
                "x": {"start": 0, "end": 1},
                "y": {"start": 1, "end": 2},
                "z": {"start": 2, "end": 3},
                "roll": {"start": 3, "end": 4},
                "pitch": {"start": 4, "end": 5},
                "yaw": {"start": 5, "end": 6},
                "pad": {"start": 6, "end": 7},
                "gripper": {"start": 7, "end": 8},
            },
            "action": {
                "x": {"start": 0, "end": 1},
                "y": {"start": 1, "end": 2},
                "z": {"start": 2, "end": 3},
                "roll": {"start": 3, "end": 4},
                "pitch": {"start": 4, "end": 5},
                "yaw": {"start": 5, "end": 6},
                "gripper": {"start": 6, "end": 7},
            },
            "video": {
                "primary_image": {
                    "original_key": "observation.images.image",
                },
            },
            "annotation": {
                "human.action.task_description": {
                    "original_key": "task_index",
                },
            },
        }
        path = os.path.join(meta_dir, "modality.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(modality, f, indent=4)

    @staticmethod
    def _probe_video_resolution(video_base_dir):
        """Probe the first video file to get resolution and codec info."""
        if not os.path.exists(video_base_dir):
            raise ValueError(f"Video directory {video_base_dir} does not exist.")

        # Find the first video file
        video_path = None
        for root, _dirs, files in os.walk(video_base_dir):
            for f in sorted(files):
                if f.endswith((".mp4", ".avi", ".mkv")):
                    video_path = os.path.join(root, f)
                    break
            if video_path:
                break

        if not video_path:
            raise ValueError("No video files found.")

        defaults = {
            "width": 0,
            "height": 0,
            "channels": 3,
            "codec": "av1",
            "pix_fmt": "yuv420p",
        }

        try:
            import cv2

            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                defaults["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                defaults["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                ret, frame = cap.read()
                if ret and frame is not None:
                    defaults["channels"] = frame.shape[2] if frame.ndim == 3 else 1
                cap.release()
        except Exception:
            pass

        try:
            import subprocess

            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "quiet",
                    "-print_format",
                    "json",
                    "-show_streams",
                    "-select_streams",
                    "v:0",
                    video_path,
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                import json as _json

                probe = _json.loads(result.stdout)
                if probe.get("streams"):
                    stream = probe["streams"][0]
                    defaults["codec"] = stream.get("codec_name", defaults["codec"])
                    pix_fmt = stream.get("pix_fmt", defaults["pix_fmt"])
                    defaults["pix_fmt"] = pix_fmt
                    # Infer channels from pix_fmt
                    if "gray" in pix_fmt:
                        defaults["channels"] = 1
                    elif "a" in pix_fmt and pix_fmt not in ("yuv420p", "yuvj420p"):
                        defaults["channels"] = 4
                    else:
                        defaults["channels"] = 3
        except Exception:
            pass

        return defaults

    @staticmethod
    def finalize_dataset(output_dir, fps=10, robot_type="egodex_hand", chunks_size=DEFAULT_CHUNKS_SIZE):
        """Merge staged files into final LeRobot dataset structure.

        Must be called ONCE after all Ray actors have finished.
        This is single-threaded — no concurrency issues.

        Steps:
          1. Collect all episode metadata fragments from staging
          2. Sort by UUID for deterministic ordering
          3. Assign sequential episode_index (0, 1, 2, ...)
          4. Rewrite parquet files with correct episode_index / index
          5. Move video files to chunk directories
          6. Write episodes.jsonl, tasks.jsonl, info.json
          7. Clean up staging directory
        """
        staging_dir = os.path.join(output_dir, "staging")
        staging_data = os.path.join(staging_dir, "data")
        staging_video = os.path.join(staging_dir, "videos")
        staging_meta = os.path.join(staging_dir, "meta")
        meta_dir = os.path.join(output_dir, "meta")

        ExportToLeRobotMapper._write_modality_json(meta_dir)

        # 1. Collect all episode metadata fragments
        episodes = []
        if os.path.exists(staging_meta):
            for fname in sorted(os.listdir(staging_meta)):
                if not fname.endswith(".jsonl"):
                    continue
                fpath = os.path.join(staging_meta, fname)
                with open(fpath, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            episodes.append(json.loads(line))

        if not episodes:
            logger.warning("No staged episodes found. Nothing to finalize.")
            return

        # 2. Sort for deterministic ordering
        episodes.sort(key=lambda e: e["uuid"])

        # 3. Assign sequential episode_index, build task index
        task_to_index = {}
        global_frame_offset = 0

        for ep_idx, ep in enumerate(episodes):
            ep["episode_index"] = ep_idx
            ep["global_frame_offset"] = global_frame_offset
            global_frame_offset += ep["length"]

            task = ep["task"]
            if task not in task_to_index:
                task_to_index[task] = len(task_to_index)
            ep["task_index"] = task_to_index[task]

        total_episodes = len(episodes)
        total_frames = global_frame_offset
        total_chunks = max(1, (total_episodes + chunks_size - 1) // chunks_size)

        # 4. Create chunk directories
        for chunk_idx in range(total_chunks):
            chunk_name = f"chunk-{chunk_idx:03d}"
            os.makedirs(os.path.join(output_dir, "data", chunk_name), exist_ok=True)
            os.makedirs(os.path.join(output_dir, "videos", chunk_name, "observation.images.image"), exist_ok=True)

        # 5. Process each episode: rewrite parquet, move video
        total_videos = 0
        for ep in episodes:
            ep_uuid = ep["uuid"]
            ep_idx = ep["episode_index"]
            chunk_name = f"chunk-{ep_idx // chunks_size:03d}"

            # Rewrite parquet with correct indices
            src_parquet = os.path.join(staging_data, f"{ep_uuid}.parquet")
            if os.path.exists(src_parquet):
                table = pa.parquet.read_table(src_parquet)
                df = table.to_pandas()

                df["episode_index"] = ep_idx
                df["task_index"] = ep["task_index"]
                df["index"] = ep["global_frame_offset"] + df["frame_index"].values

                dst_parquet = os.path.join(output_dir, "data", chunk_name, f"episode_{ep_idx:06d}.parquet")
                out_table = pa.Table.from_pandas(df)
                pa.parquet.write_table(out_table, dst_parquet)

            # Move video file
            video_ext = ep.get("video_ext", ".mp4")
            src_video = os.path.join(staging_video, f"{ep_uuid}{video_ext}")
            if os.path.exists(src_video):
                dst_video = os.path.join(
                    output_dir, "videos", chunk_name, "observation.images.image", f"episode_{ep_idx:06d}{video_ext}"
                )
                shutil.move(src_video, dst_video)
                total_videos += 1

        # 6. Write episodes.jsonl
        episodes_path = os.path.join(meta_dir, "episodes.jsonl")
        with open(episodes_path, "w", encoding="utf-8") as f:
            for ep in episodes:
                entry = {
                    "episode_index": ep["episode_index"],
                    "length": ep["length"],
                    "task": ep["task"],
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        # 7. Write tasks.jsonl
        tasks_path = os.path.join(meta_dir, "tasks.jsonl")
        with open(tasks_path, "w", encoding="utf-8") as f:
            for task, idx in sorted(task_to_index.items(), key=lambda x: x[1]):
                f.write(json.dumps({"task_index": idx, "task": task}, ensure_ascii=False) + "\n")

        # 8. Probe video resolution
        video_base_dir = os.path.join(output_dir, "videos")
        video_info = ExportToLeRobotMapper._probe_video_resolution(video_base_dir)

        # 9. Write info.json with features
        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": [8],
            },
            "action": {
                "dtype": "float32",
                "shape": [7],
            },
            "observation.images.image": {
                "dtype": "video",
                "shape": [
                    video_info["height"],
                    video_info["width"],
                    video_info["channels"],
                ],
                "names": ["height", "width", "channels"],
                "info": {
                    "video.height": video_info["height"],
                    "video.width": video_info["width"],
                    "video.channels": video_info["channels"],
                    "video.codec": video_info["codec"],
                    "video.pix_fmt": video_info["pix_fmt"],
                    "video.is_depth_map": False,
                    "video.fps": fps,
                    "has_audio": False,
                },
            },
            "timestamp": {
                "dtype": "float32",
                "shape": [1],
                "names": None,
            },
            "frame_index": {
                "dtype": "int64",
                "shape": [1],
                "names": None,
            },
            "episode_index": {
                "dtype": "int64",
                "shape": [1],
                "names": None,
            },
            "index": {
                "dtype": "int64",
                "shape": [1],
                "names": None,
            },
            "task_index": {
                "dtype": "int64",
                "shape": [1],
                "names": None,
            },
        }

        info = {
            "codebase_version": "v2.0",
            "robot_type": robot_type,
            "total_episodes": total_episodes,
            "total_frames": total_frames,
            "total_tasks": len(task_to_index),
            "total_videos": total_videos,
            "total_chunks": total_chunks,
            "chunks_size": chunks_size,
            "fps": fps,
            "splits": {"train": f"0:{total_episodes}"},
            "data_path": "data/chunk-{episode_chunk:03d}/" "episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/" "{video_key}/episode_{episode_index:06d}.mp4",
            "features": features,
        }

        info_path = os.path.join(meta_dir, "info.json")
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)

        # 10. Clean up staging directory
        shutil.rmtree(staging_dir, ignore_errors=True)

        logger.info(
            f"LeRobot dataset finalized: {total_episodes} episodes, "
            f"{total_frames} frames, {len(task_to_index)} tasks, "
            f"{total_chunks} chunks"
        )

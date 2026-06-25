import os
import shutil
import tempfile
import unittest

import cv2
import numpy as np

from data_juicer.ops.base_op import Fields
from data_juicer.ops.mapper.video_clip_reassembly_mapper import \
    VideoClipReassemblyMapper
from data_juicer.utils.constant import CameraCalibrationKeys, MetaKeys
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class VideoClipReassemblyMapperTest(DataJuicerTestCaseBase):
    """Tests for VideoClipReassemblyMapper."""

    def setUp(self):
        super().setUp()
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        super().tearDown()
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _create_frames(self, n_frames, prefix="clip0"):
        """Create unique dummy frames with reproducible content."""
        clip_dir = os.path.join(self.tmp_dir, prefix)
        os.makedirs(clip_dir, exist_ok=True)
        paths = []
        for i in range(n_frames):
            # Deterministic content based on global frame id
            img = np.full((100, 100, 3), fill_value=(i * 7) % 256,
                          dtype=np.uint8)
            path = os.path.join(clip_dir, f"frame_{i:04d}.jpg")
            cv2.imwrite(path, img)
            paths.append(path)
        return paths

    def _create_overlapping_clips(
        self, total_frames=30, clip_len=15, overlap=5,
    ):
        """Create overlapping clip frame lists with shared frame images.

        Returns (per_clip_frames, all_frame_paths).
        """
        step = clip_len - overlap
        all_dir = os.path.join(self.tmp_dir, "all_frames")
        os.makedirs(all_dir, exist_ok=True)

        # Create all unique frames
        all_paths = []
        for i in range(total_frames):
            img = np.full((100, 100, 3), fill_value=(i * 7) % 256,
                          dtype=np.uint8)
            path = os.path.join(all_dir, f"frame_{i:04d}.jpg")
            cv2.imwrite(path, img)
            all_paths.append(path)

        # Build per-clip frame lists (with real overlapping files)
        per_clip = []
        offset = 0
        while offset < total_frames:
            end = min(offset + clip_len, total_frames)
            clip_frames = []
            clip_dir = os.path.join(
                self.tmp_dir, f"clip_{len(per_clip)}")
            os.makedirs(clip_dir, exist_ok=True)
            for local_i, global_i in enumerate(range(offset, end)):
                # Copy the global frame so pixel matching works
                src = all_paths[global_i]
                dst = os.path.join(clip_dir, f"frame_{local_i:04d}.jpg")
                img = cv2.imread(src)
                cv2.imwrite(dst, img)
                clip_frames.append(dst)
            per_clip.append(clip_frames)
            offset += step
            if end >= total_frames:
                break

        return per_clip, all_paths

    @staticmethod
    def _make_hand_data(n_frames, hand_type="right", offset=0.0):
        states = np.zeros((n_frames, 8), dtype=np.float32)
        for i in range(n_frames):
            states[i, 0] = offset + i * 0.01
        return {
            "hand_type": hand_type,
            "states": states.tolist(),
            "actions": np.zeros((n_frames, 7)).tolist(),
            "valid_frame_ids": list(range(n_frames)),
            "joints_world": [],
            "joints_cam": [],
        }

    @staticmethod
    def _make_cam_pose(n_frames):
        c2w = np.array([np.eye(4) for _ in range(n_frames)])
        return {CameraCalibrationKeys.cam_c2w: c2w.tolist()}

    # ------------------------------------------------------------------
    # _merge_video_frames
    # ------------------------------------------------------------------
    def test_merge_video_frames_no_overlap(self):
        frames_a = ["a0", "a1", "a2"]
        frames_b = ["b0", "b1", "b2"]
        merged = VideoClipReassemblyMapper._merge_video_frames(
            [frames_a, frames_b], [0, 3])
        self.assertEqual(merged, ["a0", "a1", "a2", "b0", "b1", "b2"])

    def test_merge_video_frames_with_overlap(self):
        frames_a = ["a0", "a1", "a2", "a3", "a4"]
        frames_b = ["b0", "b1", "b2", "b3", "b4"]
        # Clip B starts at global offset 3 → overlaps with a3, a4
        merged = VideoClipReassemblyMapper._merge_video_frames(
            [frames_a, frames_b], [0, 3])
        self.assertEqual(len(merged), 8)  # 0..7
        # First clip fills 0-4, second fills 5-7 (3+0=3 already filled)
        self.assertEqual(merged[0], "a0")
        self.assertEqual(merged[3], "a3")  # first clip wins
        self.assertEqual(merged[5], "b2")  # only from clip B

    # ------------------------------------------------------------------
    # _detect_clip_offsets
    # ------------------------------------------------------------------
    def test_detect_clip_offsets_with_matching(self):
        """Pixel matching should detect the correct overlap offset."""
        per_clip, _ = self._create_overlapping_clips(
            total_frames=30, clip_len=15, overlap=5)
        offsets = VideoClipReassemblyMapper._detect_clip_offsets(
            per_clip, nominal_step=10)
        # First offset is always 0
        self.assertEqual(offsets[0], 0)
        # With clip_len=15, overlap=5 → step=10
        if len(offsets) > 1:
            self.assertEqual(offsets[1], 10)

    def test_detect_clip_offsets_single_clip(self):
        frames = self._create_frames(10, "only_clip")
        offsets = VideoClipReassemblyMapper._detect_clip_offsets(
            [frames], nominal_step=10)
        self.assertEqual(offsets, [0])

    # ------------------------------------------------------------------
    # _blend_weight
    # ------------------------------------------------------------------
    def test_blend_weight_no_overlap(self):
        op = VideoClipReassemblyMapper()
        w = op._blend_weight(
            clip_idx=0, local_fid=5, n_clips=1,
            clip_len=10, overlap_prev=0, overlap_next=0)
        self.assertAlmostEqual(w, 1.0)

    def test_blend_weight_overlap_ramp(self):
        op = VideoClipReassemblyMapper()
        # First frame of overlap region with previous clip
        w = op._blend_weight(
            clip_idx=1, local_fid=0, n_clips=3,
            clip_len=20, overlap_prev=5, overlap_next=5)
        self.assertLess(w, 1.0)
        self.assertGreater(w, 0.0)

    # ------------------------------------------------------------------
    # _merge_moge
    # ------------------------------------------------------------------
    def test_merge_moge_basic(self):
        moge_a = {"depth": ["d0", "d1", "d2"], "hfov": [1.0, 1.0, 1.0]}
        moge_b = {"depth": ["d3", "d4", "d5"], "hfov": [1.0, 1.0, 1.0]}
        merged = VideoClipReassemblyMapper._merge_moge(
            [moge_a, moge_b], [0, 3])
        self.assertEqual(len(merged["depth"]), 6)

    # ------------------------------------------------------------------
    # end-to-end with single clip → passthrough
    # ------------------------------------------------------------------
    def test_single_clip_passthrough(self):
        """Single clip (non-nested frames) → no reassembly needed."""
        frames = self._create_frames(10, "single")
        sample = {
            "videos": ["video.mp4"],
            "clips": ["video.mp4"],
            "video_frames": frames,  # not nested
            Fields.meta: {
                MetaKeys.hand_action_tags: [
                    {"right": self._make_hand_data(10)},
                ],
                MetaKeys.video_camera_pose_tags: [
                    self._make_cam_pose(10),
                ],
                MetaKeys.camera_calibration_moge_tags: [
                    {"hfov": [1.0] * 10},
                ],
            },
        }
        op = VideoClipReassemblyMapper(
            split_duration=5.0, overlap_duration=2.0, fps=10.0)
        result = op.process_single(sample)
        # Should be unchanged
        self.assertEqual(len(result["video_frames"]), 10)

    # ------------------------------------------------------------------
    # end-to-end with multiple clips
    # ------------------------------------------------------------------
    def test_multi_clip_reassembly(self):
        """Two overlapping clips should be merged correctly."""
        per_clip, _ = self._create_overlapping_clips(
            total_frames=25, clip_len=15, overlap=5)
        n_clips = len(per_clip)

        # Build per-clip hand action data
        hand_actions = []
        cam_poses = []
        moge_list = []
        for ci in range(n_clips):
            n = len(per_clip[ci])
            hand_actions.append({"right": self._make_hand_data(n)})
            cam_poses.append(self._make_cam_pose(n))
            moge_list.append({"depth": [f"d{ci}_{j}" for j in range(n)],
                              "hfov": [1.0] * n})

        sample = {
            "videos": ["video.mp4"],
            "clips": ["clip0.mp4", "clip1.mp4"],
            "video_frames": per_clip,
            Fields.meta: {
                MetaKeys.hand_action_tags: hand_actions,
                MetaKeys.video_camera_pose_tags: cam_poses,
                MetaKeys.camera_calibration_moge_tags: moge_list,
            },
        }

        op = VideoClipReassemblyMapper(
            split_duration=1.5, overlap_duration=0.5, fps=10.0)
        result = op.process_single(sample)

        meta = result[Fields.meta]
        # hand_action_tags should be merged into single entry
        ha = meta[MetaKeys.hand_action_tags]
        self.assertEqual(len(ha), 1)
        merged_right = ha[0]["right"]
        # Merged trajectory should cover more frames than a single clip
        self.assertGreater(len(merged_right["states"]),
                           len(per_clip[0]))

        # video_frames should be merged
        merged_frames = result["video_frames"]
        self.assertIsInstance(merged_frames, list)

    # ------------------------------------------------------------------
    # _empty_hand_result
    # ------------------------------------------------------------------
    def test_empty_hand_result(self):
        r = VideoClipReassemblyMapper._empty_hand_result("left")
        self.assertEqual(r["hand_type"], "left")
        self.assertEqual(r["states"], [])
        self.assertEqual(r["valid_frame_ids"], [])

    # ------------------------------------------------------------------
    # nominal step computation
    # ------------------------------------------------------------------
    def test_compute_nominal_step(self):
        op = VideoClipReassemblyMapper(
            split_duration=5.0, overlap_duration=2.0, fps=30.0)
        self.assertEqual(op._compute_nominal_step(), 90)

    def test_compute_nominal_step_none(self):
        op = VideoClipReassemblyMapper()
        self.assertIsNone(op._compute_nominal_step())

    def test_no_meta_passthrough(self):
        sample = {"text": "hello"}
        op = VideoClipReassemblyMapper()
        result = op.process_single(sample)
        self.assertEqual(result, sample)


if __name__ == "__main__":
    unittest.main()

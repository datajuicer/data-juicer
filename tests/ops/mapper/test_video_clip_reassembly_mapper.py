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
    def _write_frame_image(self, path, value_or_array):
        if isinstance(value_or_array, np.ndarray):
            img = value_or_array
        else:
            img = np.full((100, 100, 3), fill_value=value_or_array, dtype=np.uint8)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, img)
        return path

    def _create_frames(self, n_frames, prefix="clip0"):
        clip_dir = os.path.join(self.tmp_dir, prefix)
        paths = []
        for i in range(n_frames):
            path = os.path.join(clip_dir, f"frame_{i:04d}.jpg")
            self._write_frame_image(path, (i * 7) % 256)
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

        # Create all unique frames.
        all_paths = []
        for i in range(total_frames):
            path = os.path.join(all_dir, f"frame_{i:04d}.jpg")
            self._write_frame_image(path, (i * 7) % 256)
            all_paths.append(path)

        # Build per-clip frame lists (with real overlapping files).
        per_clip = []
        offset = 0
        while offset < total_frames:
            end = min(offset + clip_len, total_frames)
            clip_frames = []
            clip_dir = os.path.join(self.tmp_dir, f"clip_{len(per_clip)}")
            for local_i, global_i in enumerate(range(offset, end)):
                src = all_paths[global_i]
                dst = os.path.join(clip_dir, f"frame_{local_i:04d}.jpg")
                self._write_frame_image(dst, cv2.imread(src))
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

    def test_detect_offsets_fallbacks(self):
        self.assertIsNone(
            VideoClipReassemblyMapper._frame_hash(
                os.path.join(self.tmp_dir, "missing.jpg")))

        offsets = VideoClipReassemblyMapper._detect_clip_offsets(
            [["a0", "a1"], []], nominal_step=7)
        self.assertEqual(offsets, [0, 7])

        offsets = VideoClipReassemblyMapper._detect_clip_offsets(
            [["a0", "a1"], ["missing"]], nominal_step=5)
        self.assertEqual(offsets, [0, 5])

        prev = self._create_frames(3, "prev")
        curr_dir = os.path.join(self.tmp_dir, "curr")
        first = os.path.join(curr_dir, "frame_0000.jpg")
        second = os.path.join(curr_dir, "frame_0001.jpg")
        self._write_frame_image(first, cv2.imread(prev[1]))
        self._write_frame_image(second, np.full((100, 100, 3), 222, dtype=np.uint8))

        offsets = VideoClipReassemblyMapper._detect_clip_offsets(
            [prev, [first, second]], nominal_step=2)
        self.assertEqual(offsets, [0, 2])

    def test_compute_alignment_transform_branches(self):
        eye = np.eye(4).tolist()
        self.assertEqual(
            len(VideoClipReassemblyMapper._compute_alignment_transforms(
                [{}, {CameraCalibrationKeys.cam_c2w: [eye]}], [0, 1], [2, 1])),
            2,
        )
        self.assertEqual(
            len(VideoClipReassemblyMapper._compute_alignment_transforms(
                [{CameraCalibrationKeys.cam_c2w: [eye]}, {}], [0, 1], [2, 1])),
            2,
        )
        self.assertEqual(
            len(VideoClipReassemblyMapper._compute_alignment_transforms(
                [{CameraCalibrationKeys.cam_c2w: [eye]},
                 {CameraCalibrationKeys.cam_c2w: [eye]}],
                [0, 3],
                [2, 1])),
            2,
        )

        prev0 = np.eye(4)
        prev1 = np.eye(4)
        prev1[:3, 3] = [3.0, 0.0, 0.0]
        curr0 = np.eye(4)
        transforms = VideoClipReassemblyMapper._compute_alignment_transforms(
            [
                {CameraCalibrationKeys.cam_c2w: [prev0.tolist(),
                                                 prev1.tolist()]},
                {CameraCalibrationKeys.cam_c2w: [curr0.tolist()]},
            ],
            [0, 1],
            [2, 1],
        )
        self.assertEqual(len(transforms), 2)
        self.assertTrue(np.allclose(transforms[1][:3, 3], [3.0, 0.0, 0.0]))

    def test_apply_transform_to_hand_data_and_c2w(self):
        hand = self._make_hand_data(2)
        hand["joints_world"] = np.zeros((2, 21, 3), dtype=np.float32).tolist()
        T = np.eye(4)
        T[:3, 3] = [1.0, 2.0, 3.0]

        out = VideoClipReassemblyMapper._apply_transform_to_hand_data(hand, T)

        self.assertAlmostEqual(out["states"][0][0], 1.0)
        self.assertAlmostEqual(out["states"][0][1], 2.0)
        self.assertAlmostEqual(out["states"][0][2], 3.0)
        self.assertEqual(out["joints_world"][0][0], [1.0, 2.0, 3.0])
        self.assertEqual(len(out["actions"]), 2)
        self.assertEqual(
            VideoClipReassemblyMapper._apply_transform_to_hand_data({}, T),
            {},
        )

        c2w = np.array([np.eye(4), np.eye(4)])
        aligned = VideoClipReassemblyMapper._apply_transform_to_c2w(c2w, T)
        self.assertTrue(np.allclose(aligned[:, :3, 3], [[1, 2, 3], [1, 2, 3]]))

    def test_merge_video_frames_and_moge_fill_gaps(self):
        frames = VideoClipReassemblyMapper._merge_video_frames(
            [["a0"], ["b3"]], [0, 3])
        self.assertEqual(frames, ["a0", "a0", "a0", "b3"])

        self.assertEqual(VideoClipReassemblyMapper._merge_moge([], []), {})
        self.assertEqual(
            VideoClipReassemblyMapper._merge_moge([None, {"meta": "x"}], [0, 0]),
            None,
        )

        merged = VideoClipReassemblyMapper._merge_moge(
            [
                {"depth": ["d0", "d1"], "hfov": 90},
                None,
                {"depth": ["d4", "d5"], "intrinsics": ["i4", "i5"]},
                {"depth": "bad", "vfov": ["v6", "v7"]},
            ],
            [0, 2, 4, 6],
        )
        self.assertEqual(merged["hfov"], 90)
        self.assertEqual(merged["depth"][:6], ["d0", "d1", "d1", "d1", "d4", "d5"])
        self.assertEqual(merged["depth"][6:], ["d5", "d5"])
        self.assertEqual(merged["vfov"][-2:], ["v6", "v7"])

    def test_merge_hand_variants(self):
        op = VideoClipReassemblyMapper()
        empty = op._merge_hand_across_clips([None, {}], "right", 2, [0, 1], [1, 1])
        self.assertEqual(empty["hand_type"], "right")
        self.assertEqual(empty["states"], [])

        single = self._make_hand_data(2)
        merged_single = op._merge_hand_across_clips(
            [single, None], "right", 2, [5, 10], [2, 2])
        self.assertEqual(merged_single["valid_frame_ids"], [5, 6])
        self.assertEqual(merged_single["states"], single["states"])

        hand_a = self._make_hand_data(2, offset=0.0)
        hand_b = self._make_hand_data(2, offset=1.0)
        hand_a["joints_world"] = np.ones((2, 21, 3)).tolist()
        hand_b["joints_world"] = (np.ones((2, 21, 3)) * 3).tolist()
        hand_a["joints_cam"] = np.ones((2, 21, 3)).tolist()
        hand_b["joints_cam"] = (np.ones((2, 21, 3)) * 5).tolist()

        merged = op._merge_hand_across_clips(
            [hand_a, hand_b], "right", 2, [0, 1], [2, 2])
        self.assertEqual(merged["valid_frame_ids"], [0, 1, 2])
        self.assertEqual(len(merged["joints_world"]), 3)
        self.assertEqual(len(merged["joints_cam"]), 3)

    def test_merge_cam_c2w_edge_cases(self):
        op = VideoClipReassemblyMapper()
        empty = op._merge_cam_c2w([None, {}], [0, 1], [1, 1])
        self.assertIsNone(empty)

        eye = np.eye(4).tolist()
        merged = op._merge_cam_c2w(
            [
                {"source": "first", CameraCalibrationKeys.cam_c2w: [eye]},
                {"ignored": "x"},
                {"source": "later", CameraCalibrationKeys.cam_c2w: [eye]},
            ],
            [0, 1, 3],
            [1, 1, 1],
        )
        self.assertEqual(merged["source"], "first")
        self.assertEqual(len(merged[CameraCalibrationKeys.cam_c2w]), 4)
        self.assertEqual(merged[CameraCalibrationKeys.cam_c2w][1], np.eye(4).tolist())

    def test_merge_hawor_deduplicates_frames(self):
        op = VideoClipReassemblyMapper()
        merged = op._merge_hawor(
            [
                {
                    "fov_x": 55,
                    "right": {
                        "frame_ids": [2, 0],
                        "transl": ["t2", "t0"],
                        "global_orient": ["o2", "o0"],
                        "hand_pose": ["p2", "p0"],
                        "betas": ["b2", "b0"],
                        "joints_cam": ["j2", "j0"],
                    },
                },
                None,
                {
                    "right": {
                        "frame_ids": [0, 1],
                        "transl": ["dup", "t4"],
                        "global_orient": ["dup", "o4"],
                        "hand_pose": ["dup", "p4"],
                        "betas": ["dup", "b4"],
                        "joints_cam": ["dupj", "j4"],
                    },
                    "left": {"frame_ids": [], "transl": []},
                },
            ],
            [0, 1, 2],
        )
        self.assertEqual(merged["fov_x"], 55)
        self.assertEqual(merged["right"]["frame_ids"], [0, 2, 3])
        self.assertEqual(merged["right"]["transl"], ["t0", "t2", "t4"])
        self.assertEqual(merged["right"]["joints_cam"], ["j0", "j2", "j4"])
        self.assertEqual(merged["left"]["frame_ids"], [])

    def test_process_single_falls_back_on_merge_failure(self):
        class RaisingMapper(VideoClipReassemblyMapper):
            def _merge_video_frames(self, *args, **kwargs):
                raise RuntimeError("frames")

            def _merge_moge(self, *args, **kwargs):
                raise RuntimeError("moge")

            def _compute_alignment_transforms(self, *args, **kwargs):
                raise RuntimeError("align")

            def _merge_hand_across_clips(self, *args, **kwargs):
                raise RuntimeError("hand")

            def _merge_cam_c2w(self, *args, **kwargs):
                raise RuntimeError("cam")

            def _merge_hawor(self, *args, **kwargs):
                raise RuntimeError("hawor")

        eye = np.eye(4).tolist()
        sample = {
            "clips": ["clip0.mp4", "clip1.mp4"],
            "video_frames": [["missing0"], ["missing1"]],
            Fields.meta: {
                MetaKeys.camera_calibration_moge_tags: [{"depth": [1]},
                                                       {"depth": [2]}],
                MetaKeys.video_camera_pose_tags: [
                    {CameraCalibrationKeys.cam_c2w: [eye]},
                    {CameraCalibrationKeys.cam_c2w: [eye]},
                ],
                MetaKeys.hand_action_tags: [
                    {"right": self._make_hand_data(1)},
                    {"right": self._make_hand_data(1)},
                ],
                MetaKeys.hand_reconstruction_hawor_tags: [{"right": {}},
                                                         {"right": {}}],
            },
        }

        out = RaisingMapper(split_duration=1.0, overlap_duration=0.0, fps=1.0).process_single(sample)

        self.assertEqual(out["clips"], ["clip0.mp4"])
        self.assertEqual(out[Fields.meta][MetaKeys.hand_action_tags][0]["right"]["hand_type"], "right")

    def test_process_single_aligns_despite_hand_failure(self):
        class AligningMapper(VideoClipReassemblyMapper):
            def _detect_clip_offsets(self, *args, **kwargs):
                return [0, 1]

            def _compute_alignment_transforms(self, *args, **kwargs):
                T = np.eye(4)
                T[:3, 3] = [1.0, 0.0, 0.0]
                return [np.eye(4), T]

            def _apply_transform_to_hand_data(self, *args, **kwargs):
                raise RuntimeError("hand alignment")

        eye = np.eye(4).tolist()
        sample = {
            "videos": ["video.mp4"],
            "clips": ["clip0.mp4", "clip1.mp4"],
            "video_frames": [["a0", "a1"], ["b0", "b1"]],
            Fields.meta: {
                MetaKeys.video_camera_pose_tags: [
                    {CameraCalibrationKeys.cam_c2w: [eye, eye], "source": "cam"},
                    {CameraCalibrationKeys.cam_c2w: [eye, eye]},
                ],
                MetaKeys.hand_action_tags: [
                    {"right": self._make_hand_data(2)},
                    {"right": self._make_hand_data(2)},
                ],
            },
        }

        out = AligningMapper().process_single(sample)

        self.assertEqual(out["clips"], ["video.mp4"])
        cam = out[Fields.meta][MetaKeys.video_camera_pose_tags][0]
        self.assertEqual(len(cam[CameraCalibrationKeys.cam_c2w]), 3)
        self.assertEqual(out[Fields.meta][MetaKeys.hand_action_tags][0]["right"]["hand_type"], "right")


if __name__ == "__main__":
    unittest.main()

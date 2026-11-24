import os
import shutil
import tempfile
import unittest

import cv2
import numpy as np

from data_juicer.ops.base_op import Fields
from data_juicer.ops.mapper.video_trajectory_overlay_mapper import \
    VideoTrajectoryOverlayMapper
from data_juicer.utils.constant import CameraCalibrationKeys, MetaKeys
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class VideoTrajectoryOverlayMapperTest(DataJuicerTestCaseBase):
    """Tests for VideoTrajectoryOverlayMapper."""

    def setUp(self):
        super().setUp()
        self.tmp_dir = tempfile.mkdtemp()
        self.frames_dir = os.path.join(self.tmp_dir, "frames")
        self.overlay_dir = os.path.join(self.tmp_dir, "overlays")
        os.makedirs(self.frames_dir, exist_ok=True)

    def tearDown(self):
        super().tearDown()
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _create_dummy_frames(self, n_frames=30, width=640, height=480):
        """Create dummy frame images and return their paths."""
        paths = []
        for i in range(n_frames):
            img = np.random.randint(0, 255, (height, width, 3),
                                    dtype=np.uint8)
            path = os.path.join(self.frames_dir, f"frame_{i:04d}.jpg")
            cv2.imwrite(path, img)
            paths.append(path)
        return paths

    def _make_sample(self, n_frames=30, n_segments=2):
        """Build a sample with dummy segments and camera data."""
        frame_paths = self._create_dummy_frames(n_frames)

        # Identity camera poses
        cam_c2w = [np.eye(4).tolist() for _ in range(n_frames)]

        # Simple intrinsics K matrix (fx=320, fy=320, cx=320, cy=240)
        K = [[320.0, 0, 320.0], [0, 320.0, 240.0], [0, 0, 1]]
        intrinsics_list = [K for _ in range(n_frames)]

        # Simple segments
        frames_per_seg = n_frames // n_segments
        segments = []
        for s in range(n_segments):
            start = s * frames_per_seg
            end = min((s + 1) * frames_per_seg - 1, n_frames - 1)
            n = end - start + 1
            states = np.zeros((n, 8), dtype=np.float32)
            # Build joints_world: (n, 21, 3) — use joint 9 as palm
            joints_world = np.zeros((n, 21, 3), dtype=np.float32)
            # Linear motion for palm (joint 9)
            for i in range(n):
                palm_pos = np.array([
                    (start + i) * 0.01,
                    (start + i) * 0.005,
                    1.0,  # z=1 so projection works
                ])
                joints_world[i, 9] = palm_pos
                # Wrist (joint 0) slightly offset from palm
                joints_world[i, 0] = palm_pos + np.array([-0.02, 0.01, 0])
                states[i, 0:3] = palm_pos  # states for fallback
            segments.append({
                "hand_type": "right" if s % 2 == 0 else "left",
                "segment_id": s,
                "start_frame": start,
                "end_frame": end,
                "states": states.tolist(),
                "actions": [],
                "valid_frame_ids": list(range(start, end + 1)),
                "joints_world": joints_world.tolist(),
            })

        return {
            "video_frames": frame_paths,
            Fields.meta: {
                "atomic_action_segments": segments,
                MetaKeys.video_camera_pose_tags: [{
                    CameraCalibrationKeys.cam_c2w: cam_c2w,
                }],
                MetaKeys.camera_calibration_moge_tags: [{
                    CameraCalibrationKeys.hfov: [1.0],
                    CameraCalibrationKeys.intrinsics: intrinsics_list,
                }],
            },
        }

    # ------------------------------------------------------------------
    # projection helpers
    # ------------------------------------------------------------------
    def test_world_to_camera_identity(self):
        """Identity c2w should return the same position."""
        pos = np.array([1.0, 2.0, 3.0])
        c2w = np.eye(4)
        result = VideoTrajectoryOverlayMapper._world_to_camera(pos, c2w)
        np.testing.assert_allclose(result, pos, atol=1e-10)

    def test_world_to_camera_translation(self):
        """Translation-only c2w should shift the position."""
        pos = np.array([5.0, 5.0, 5.0])
        c2w = np.eye(4)
        c2w[:3, 3] = [1.0, 2.0, 3.0]
        result = VideoTrajectoryOverlayMapper._world_to_camera(pos, c2w)
        # cam = (world - t) @ R = (pos - t)
        expected = np.array([4.0, 3.0, 2.0])
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_project_to_2d_center_fov(self):
        """A point on the optical axis should project to the image center
        when using fov_x fallback."""
        pos_cam = np.array([0.0, 0.0, 1.0])
        fov_x = np.pi / 2  # 90 degrees
        w, h = 640, 480
        result = VideoTrajectoryOverlayMapper._project_to_2d(
            pos_cam, w, h, fov_x=fov_x)
        np.testing.assert_allclose(result, [320.0, 240.0], atol=1e-10)

    def test_project_to_2d_center_K(self):
        """A point on the optical axis should project to (cx, cy)
        when using a K intrinsics matrix."""
        pos_cam = np.array([0.0, 0.0, 1.0])
        K = np.array([[320.0, 0, 320.0],
                      [0, 320.0, 240.0],
                      [0, 0, 1.0]])
        w, h = 640, 480
        result = VideoTrajectoryOverlayMapper._project_to_2d(
            pos_cam, w, h, K=K)
        np.testing.assert_allclose(result, [320.0, 240.0], atol=1e-10)

    def test_project_to_2d_batch(self):
        """Batch projection should work for multiple points."""
        pos_cam = np.array([
            [0.0, 0.0, 1.0],
            [0.1, 0.0, 1.0],
        ])
        result = VideoTrajectoryOverlayMapper._project_to_2d(
            pos_cam, 640, 480, fov_x=np.pi / 2)
        self.assertEqual(result.shape, (2, 2))

    def test_project_to_2d_batch_K(self):
        """Batch projection with K matrix should work for multiple points."""
        pos_cam = np.array([
            [0.0, 0.0, 1.0],
            [0.1, 0.0, 1.0],
        ])
        K = np.array([[320.0, 0, 320.0],
                      [0, 320.0, 240.0],
                      [0, 0, 1.0]])
        result = VideoTrajectoryOverlayMapper._project_to_2d(
            pos_cam, 640, 480, K=K)
        self.assertEqual(result.shape, (2, 2))
        # First point on optical axis → center
        np.testing.assert_allclose(result[0], [320.0, 240.0], atol=1e-10)
        # Second point offset in x → u > cx
        self.assertGreater(result[1, 0], 320.0)

    # ------------------------------------------------------------------
    # temporal color gradient
    # ------------------------------------------------------------------
    def test_temporal_color_blue_at_start(self):
        """t=0 should give blue (BGR: high B, low G, low R)."""
        b, g, r = VideoTrajectoryOverlayMapper._temporal_color(0.0)
        self.assertEqual(b, 255)
        self.assertEqual(g, 0)
        self.assertEqual(r, 0)

    def test_temporal_color_green_at_mid(self):
        """t=0.5 should give green (BGR: low B, high G, low R)."""
        b, g, r = VideoTrajectoryOverlayMapper._temporal_color(0.5)
        self.assertEqual(b, 0)
        self.assertEqual(g, 255)
        self.assertEqual(r, 0)

    def test_temporal_color_red_at_end(self):
        """t=1.0 should give red (BGR: low B, low G, high R)."""
        b, g, r = VideoTrajectoryOverlayMapper._temporal_color(1.0)
        self.assertEqual(b, 0)
        self.assertEqual(g, 0)
        self.assertEqual(r, 255)

    def test_temporal_color_gradient_monotonic(self):
        """Blue should decrease and red should increase over time."""
        colors = [VideoTrajectoryOverlayMapper._temporal_color(t)
                  for t in np.linspace(0, 1, 11)]
        blues = [c[0] for c in colors]
        reds = [c[2] for c in colors]
        # Blue starts high and ends low
        self.assertGreater(blues[0], blues[-1])
        # Red starts low and ends high
        self.assertLess(reds[0], reds[-1])

    # ------------------------------------------------------------------
    # draw trajectory
    # ------------------------------------------------------------------
    def test_draw_trajectory_returns_image(self):
        """_draw_trajectory should return an image of the same shape."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        points = np.array([[100, 200], [150, 250], [200, 300]],
                          dtype=np.float64)
        op = VideoTrajectoryOverlayMapper()
        result = op._draw_trajectory(frame, points, current_idx=0)
        self.assertEqual(result.shape, frame.shape)
        # Should have drawn something (not all zeros)
        self.assertGreater(np.sum(result), 0)

    def test_draw_trajectory_out_of_bounds(self):
        """Points outside frame should not crash."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        points = np.array([[-100, -100], [700, 500], [320, 240]],
                          dtype=np.float64)
        op = VideoTrajectoryOverlayMapper()
        result = op._draw_trajectory(frame, points, current_idx=2)
        self.assertEqual(result.shape, frame.shape)

    # ------------------------------------------------------------------
    # end-to-end process_single
    # ------------------------------------------------------------------
    def test_process_single_creates_overlays(self):
        """process_single should create overlay images for each segment."""
        sample = self._make_sample(n_frames=30, n_segments=2)
        op = VideoTrajectoryOverlayMapper(
            segment_field="atomic_action_segments",
            save_dir=self.overlay_dir,
            n_sample_frames=4,
        )
        result = op.process_single(sample)
        segments = result[Fields.meta]["atomic_action_segments"]

        for seg in segments:
            overlay_paths = seg.get("overlay_frames", [])
            self.assertGreater(len(overlay_paths), 0,
                               f"No overlays for {seg['hand_type']} "
                               f"seg{seg['segment_id']}")
            for p in overlay_paths:
                self.assertTrue(os.path.exists(p),
                                f"Overlay file not found: {p}")

            sampled = seg.get("sampled_frame_indices", [])
            self.assertEqual(len(sampled), len(overlay_paths))

    def test_process_single_no_segments(self):
        """No segments → sample unchanged."""
        sample = {
            "video_frames": [],
            Fields.meta: {"atomic_action_segments": []},
        }
        op = VideoTrajectoryOverlayMapper()
        result = op.process_single(sample)
        self.assertEqual(result[Fields.meta]["atomic_action_segments"], [])

    def test_process_single_no_camera(self):
        """Missing camera data → segments get empty overlay_frames."""
        frame_paths = self._create_dummy_frames(10)
        sample = {
            "video_frames": frame_paths,
            Fields.meta: {
                "atomic_action_segments": [{
                    "hand_type": "right",
                    "segment_id": 0,
                    "start_frame": 0,
                    "end_frame": 9,
                    "states": np.zeros((10, 8)).tolist(),
                    "valid_frame_ids": list(range(10)),
                }],
                MetaKeys.video_camera_pose_tags: [],
            },
        }
        op = VideoTrajectoryOverlayMapper(save_dir=self.overlay_dir)
        result = op.process_single(sample)
        # Should not crash, just skip
        self.assertIn("atomic_action_segments", result[Fields.meta])

    def test_n_sample_frames_respected(self):
        """Number of overlay frames should match n_sample_frames."""
        sample = self._make_sample(n_frames=30, n_segments=1)
        n_sample = 4
        op = VideoTrajectoryOverlayMapper(
            segment_field="atomic_action_segments",
            save_dir=self.overlay_dir,
            n_sample_frames=n_sample,
        )
        result = op.process_single(sample)
        seg = result[Fields.meta]["atomic_action_segments"][0]
        self.assertEqual(len(seg["overlay_frames"]), n_sample)

    def test_no_meta_passthrough(self):
        """Sample without meta should pass through."""
        sample = {"text": "test"}
        op = VideoTrajectoryOverlayMapper()
        result = op.process_single(sample)
        self.assertEqual(result, sample)


if __name__ == "__main__":
    unittest.main()

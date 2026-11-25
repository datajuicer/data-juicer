import unittest

import numpy as np

from data_juicer.ops.base_op import Fields
from data_juicer.ops.mapper.video_hand_motion_smooth_mapper import (
    VideoHandMotionSmoothMapper,
    _recompute_actions,
)
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class RecomputeActionsTest(DataJuicerTestCaseBase):
    """Tests for the _recompute_actions helper."""

    def test_linear_motion(self):
        """Linear translation should produce constant dx."""
        n = 10
        states = np.zeros((n, 8), dtype=np.float64)
        for i in range(n):
            states[i, 0] = i * 0.1  # x increases linearly
        actions = _recompute_actions(states)
        self.assertEqual(actions.shape, (n, 7))
        for t in range(n - 1):
            self.assertAlmostEqual(actions[t, 0], 0.1, places=5)
            self.assertAlmostEqual(actions[t, 1], 0.0, places=5)
            self.assertAlmostEqual(actions[t, 2], 0.0, places=5)

    def test_gripper_passthrough(self):
        """Gripper value from next state should appear in action."""
        states = np.zeros((5, 8), dtype=np.float64)
        states[1, 7] = 0.5
        states[2, 7] = 1.0
        actions = _recompute_actions(states)
        self.assertAlmostEqual(actions[0, 6], 0.5)
        self.assertAlmostEqual(actions[1, 6], 1.0)

    def test_single_frame(self):
        states = np.zeros((1, 8), dtype=np.float64)
        states[0, 7] = 0.3
        actions = _recompute_actions(states)
        self.assertEqual(actions.shape, (1, 7))
        self.assertAlmostEqual(actions[0, 6], 0.3)


class VideoHandMotionSmoothMapperTest(DataJuicerTestCaseBase):
    """Tests for VideoHandMotionSmoothMapper."""

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _make_hand_data(positions, hand_type="right"):
        """Build hand data dict from (N, 3) positions."""
        n = len(positions)
        states = np.zeros((n, 8), dtype=np.float32)
        states[:, 0:3] = positions
        return {
            "hand_type": hand_type,
            "states": states.tolist(),
            "actions": np.zeros((n, 7), dtype=np.float32).tolist(),
            "valid_frame_ids": list(range(n)),
            "joints_world": [],
            "joints_cam": [],
        }

    def _make_sample(self, right_data=None, left_data=None):
        clip = {}
        if right_data is not None:
            clip["right"] = right_data
        if left_data is not None:
            clip["left"] = left_data
        return {
            Fields.meta: {
                "hand_action_tags": [clip],
            }
        }

    # ------------------------------------------------------------------
    # outlier replacement
    # ------------------------------------------------------------------
    def test_replace_outliers_no_outliers(self):
        """Smooth trajectory should be unchanged."""
        positions = np.column_stack([
            np.linspace(0, 1, 20),
            np.zeros(20),
            np.zeros(20),
        ])
        result = VideoHandMotionSmoothMapper._replace_outliers(
            positions, threshold_mad=5.0)
        np.testing.assert_allclose(result, positions, atol=1e-10)

    def test_replace_outliers_with_spike(self):
        """A single spike should be interpolated away."""
        np.random.seed(42)
        n = 20
        positions = np.column_stack([
            np.linspace(0, 1, n) + np.random.normal(0, 0.02, n),
            np.random.normal(0, 0.02, n),
            np.random.normal(0, 0.02, n),
        ])
        # Insert a huge spike at frame 10
        positions[10] = [100.0, 100.0, 100.0]
        result = VideoHandMotionSmoothMapper._replace_outliers(
            positions, threshold_mad=3.0)
        # The spike should be reduced from the original (100, 100, 100)
        original_dist = np.linalg.norm(positions[10] - positions[9])
        result_dist = np.linalg.norm(result[10] - result[9])
        self.assertLess(result_dist, original_dist)

    def test_replace_outliers_short_trajectory(self):
        """Very short trajectories should be returned as-is."""
        positions = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]],
                             dtype=np.float64)
        result = VideoHandMotionSmoothMapper._replace_outliers(
            positions, threshold_mad=3.0)
        np.testing.assert_array_equal(result, positions)

    # ------------------------------------------------------------------
    # Savitzky-Golay smoothing
    # ------------------------------------------------------------------
    def test_savgol_smooth_reduces_noise(self):
        """Smoothed noisy signal should have lower variance than raw."""
        np.random.seed(42)
        n = 50
        clean = np.linspace(0, 1, n)
        noisy = clean + np.random.normal(0, 0.1, n)
        smoothed = VideoHandMotionSmoothMapper._savgol_smooth(
            noisy, window=11, polyorder=3)
        residual_raw = np.std(noisy - clean)
        residual_smooth = np.std(smoothed - clean)
        self.assertLess(residual_smooth, residual_raw)

    def test_savgol_smooth_short(self):
        """Data shorter than the window should not crash."""
        data = np.array([1.0, 2.0, 3.0])
        result = VideoHandMotionSmoothMapper._savgol_smooth(
            data, window=11, polyorder=3)
        self.assertEqual(len(result), 3)

    def test_savgol_smooth_2d(self):
        """Should smooth each column independently."""
        np.random.seed(0)
        data = np.random.randn(30, 3)
        result = VideoHandMotionSmoothMapper._savgol_smooth(
            data, window=7, polyorder=2)
        self.assertEqual(result.shape, data.shape)

    # ------------------------------------------------------------------
    # orientation smoothing
    # ------------------------------------------------------------------
    def test_smooth_orientations_preserves_shape(self):
        n = 30
        eulers = np.random.randn(n, 3) * 0.5
        result = VideoHandMotionSmoothMapper._smooth_orientations(
            eulers, window=7, polyorder=2)
        self.assertEqual(result.shape, (n, 3))

    # ------------------------------------------------------------------
    # end-to-end process_single
    # ------------------------------------------------------------------
    def test_process_single_smooths(self):
        """Smoothing should modify states while preserving structure."""
        np.random.seed(42)
        n = 30
        positions = np.column_stack([
            np.linspace(0, 1, n) + np.random.normal(0, 0.05, n),
            np.zeros(n),
            np.zeros(n),
        ])
        hand_data = self._make_hand_data(positions, "right")
        sample = self._make_sample(right_data=hand_data)

        op = VideoHandMotionSmoothMapper(
            hand_action_field="hand_action_tags",
            savgol_window=7,
            savgol_polyorder=2,
            min_frames_for_smoothing=5,
        )
        result = op.process_single(sample)

        smoothed = result[Fields.meta]["hand_action_tags"][0]["right"]
        self.assertEqual(len(smoothed["states"]), n)
        self.assertEqual(len(smoothed["actions"]), n)
        self.assertEqual(len(smoothed["valid_frame_ids"]), n)
        # States should be different from original (smoothed)
        orig_states = np.array(hand_data["states"])
        new_states = np.array(smoothed["states"])
        self.assertFalse(np.allclose(orig_states, new_states, atol=1e-6))

    def test_process_single_preserves_frame_count(self):
        """Smoothing should NOT change the number of frames."""
        n = 25
        positions = np.column_stack([
            np.linspace(0, 1, n),
            np.zeros(n),
            np.zeros(n),
        ])
        hand_data = self._make_hand_data(positions, "right")
        sample = self._make_sample(right_data=hand_data)

        op = VideoHandMotionSmoothMapper(min_frames_for_smoothing=5)
        result = op.process_single(sample)

        smoothed = result[Fields.meta]["hand_action_tags"][0]["right"]
        self.assertEqual(len(smoothed["states"]), n)
        self.assertEqual(len(smoothed["valid_frame_ids"]), n)

    def test_process_single_too_few_frames(self):
        """Fewer frames than threshold → keep original."""
        positions = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64)
        hand_data = self._make_hand_data(positions, "right")
        sample = self._make_sample(right_data=hand_data)

        op = VideoHandMotionSmoothMapper(min_frames_for_smoothing=10)
        result = op.process_single(sample)

        smoothed = result[Fields.meta]["hand_action_tags"][0]["right"]
        np.testing.assert_array_equal(
            np.array(smoothed["states"]),
            np.array(hand_data["states"]),
        )

    def test_no_meta_passthrough(self):
        sample = {"text": "hello"}
        op = VideoHandMotionSmoothMapper()
        result = op.process_single(sample)
        self.assertEqual(result, sample)

    def test_empty_hand_action(self):
        sample = {Fields.meta: {"hand_action_tags": []}}
        op = VideoHandMotionSmoothMapper()
        result = op.process_single(sample)
        self.assertEqual(
            result[Fields.meta]["hand_action_tags"], [])

    def test_smooth_joints_world(self):
        """joints_world should also be smoothed when enabled."""
        np.random.seed(0)
        n = 30
        positions = np.column_stack([
            np.linspace(0, 1, n),
            np.zeros(n), np.zeros(n),
        ])
        hand_data = self._make_hand_data(positions, "right")
        # Add noisy joints_world: (n, 21, 3)
        joints = np.random.randn(n, 21, 3) * 0.1
        hand_data["joints_world"] = joints.tolist()
        sample = self._make_sample(right_data=hand_data)

        op = VideoHandMotionSmoothMapper(
            smooth_joints=True,
            min_frames_for_smoothing=5,
        )
        result = op.process_single(sample)
        smoothed = result[Fields.meta]["hand_action_tags"][0]["right"]
        self.assertEqual(len(smoothed["joints_world"]), n)
        # Should be different (smoothed)
        self.assertFalse(np.allclose(
            np.array(smoothed["joints_world"]),
            joints, atol=1e-6,
        ))


if __name__ == "__main__":
    unittest.main()

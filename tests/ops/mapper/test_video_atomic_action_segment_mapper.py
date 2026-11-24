import unittest

import numpy as np

from data_juicer.ops.base_op import Fields
from data_juicer.ops.mapper.video_atomic_action_segment_mapper import \
    VideoAtomicActionSegmentMapper
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class VideoAtomicActionSegmentMapperTest(DataJuicerTestCaseBase):
    """Tests for VideoAtomicActionSegmentMapper."""

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _make_states(positions, n_dims=8):
        """Build an (N, 8) state array from (N, 3) positions."""
        n = len(positions)
        states = np.zeros((n, n_dims), dtype=np.float32)
        states[:, 0:3] = positions
        return states.tolist()

    def _make_sample(self, right_states=None, left_states=None):
        hand_data = {}
        if right_states is not None:
            hand_data["right"] = {
                "hand_type": "right",
                "states": right_states,
                "actions": [],
                "valid_frame_ids": list(range(len(right_states))),
                "joints_world": [],
                "joints_cam": [],
            }
        if left_states is not None:
            hand_data["left"] = {
                "hand_type": "left",
                "states": left_states,
                "actions": [],
                "valid_frame_ids": list(range(len(left_states))),
                "joints_world": [],
                "joints_cam": [],
            }
        return {
            Fields.meta: {
                "hand_action_tags": [hand_data],
            }
        }

    # ------------------------------------------------------------------
    # speed / minima helpers
    # ------------------------------------------------------------------
    def test_compute_speed_basic(self):
        """Speed should be the norm of consecutive position differences."""
        positions = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [1, 1, 1],
        ], dtype=np.float64)
        speed = VideoAtomicActionSegmentMapper._compute_speed(positions)
        self.assertEqual(len(speed), 4)
        self.assertAlmostEqual(speed[0], 0.0)
        self.assertAlmostEqual(speed[1], 1.0)
        self.assertAlmostEqual(speed[2], 1.0)
        self.assertAlmostEqual(speed[3], 1.0)

    def test_compute_speed_single_frame(self):
        positions = np.array([[1, 2, 3]], dtype=np.float64)
        speed = VideoAtomicActionSegmentMapper._compute_speed(positions)
        self.assertEqual(len(speed), 1)
        self.assertAlmostEqual(speed[0], 0.0)

    def test_find_local_minima(self):
        # Create a speed profile with clear minima
        # Pattern: high - low - high - low - high
        speed = np.array([5, 2, 1, 2, 5, 3, 0.5, 3, 5], dtype=np.float64)
        minima = VideoAtomicActionSegmentMapper._find_local_minima(
            speed, half_window=2)
        # Frame 2 (speed=1) and frame 6 (speed=0.5) should be minima
        self.assertIn(2, minima)
        self.assertIn(6, minima)

    def test_find_local_minima_flat(self):
        """Flat speed profile → every frame is a local minimum."""
        speed = np.array([1.0] * 10, dtype=np.float64)
        minima = VideoAtomicActionSegmentMapper._find_local_minima(
            speed, half_window=3)
        # All interior frames should be minima (<=)
        self.assertTrue(len(minima) > 0)

    # ------------------------------------------------------------------
    # segment merging / splitting
    # ------------------------------------------------------------------
    def test_merge_short_segments(self):
        op = VideoAtomicActionSegmentMapper(min_segment_frames=10)
        # Cut points produce segments of length 5, 5, 90 → second cut
        # should be merged because 10-5=5 < 10
        cut_points = [5, 10]
        result = op._merge_short_segments(cut_points, n_frames=100)
        # The second cut point should be removed (too close to first)
        self.assertNotIn(10, result)
        # Result should have fewer cut points than original
        self.assertLess(len(result), len(cut_points))

    def test_split_long_segments(self):
        op = VideoAtomicActionSegmentMapper(
            min_segment_frames=5,
            max_segment_frames=20,
        )
        # One segment of 50 frames should be split
        speed = np.concatenate([
            np.linspace(5, 1, 25),
            np.linspace(1, 5, 25),
        ])
        cut_points = op._split_long_segments([], speed, n_frames=50)
        self.assertTrue(len(cut_points) > 0)

    # ------------------------------------------------------------------
    # end-to-end process_single
    # ------------------------------------------------------------------
    def test_simple_segmentation(self):
        """Two distinct motion bursts should produce at least 2 segments."""
        n = 60
        positions = np.zeros((n, 3), dtype=np.float64)
        # First motion: frames 0-20 move right
        for i in range(20):
            positions[i] = [i * 0.05, 0, 0]
        # Pause: frames 20-40 stay still
        for i in range(20, 40):
            positions[i] = [1.0, 0, 0]
        # Second motion: frames 40-60 move up
        for i in range(40, 60):
            positions[i] = [1.0, (i - 40) * 0.05, 0]

        states = self._make_states(positions)
        sample = self._make_sample(right_states=states)

        op = VideoAtomicActionSegmentMapper(
            hand_action_field="hand_action_tags",
            segment_field="atomic_action_segments",
            speed_smooth_window=5,
            min_window=5,
            min_segment_frames=5,
            max_segment_frames=300,
            hand_type="right",
        )
        result = op.process_single(sample)
        segments = result[Fields.meta]["atomic_action_segments"]

        self.assertGreaterEqual(len(segments), 2)
        # All segments should be for right hand
        for seg in segments:
            self.assertEqual(seg["hand_type"], "right")
            self.assertIn("start_frame", seg)
            self.assertIn("end_frame", seg)
            self.assertIn("states", seg)
            self.assertIn("valid_frame_ids", seg)
            self.assertGreater(len(seg["states"]), 1)

    def test_both_hands(self):
        """With hand_type='both', both hands should be segmented."""
        n = 40
        pos_r = np.column_stack([np.linspace(0, 1, n),
                                 np.zeros(n), np.zeros(n)])
        pos_l = np.column_stack([np.zeros(n),
                                 np.linspace(0, 1, n), np.zeros(n)])
        sample = self._make_sample(
            right_states=self._make_states(pos_r),
            left_states=self._make_states(pos_l),
        )
        op = VideoAtomicActionSegmentMapper(
            hand_action_field="hand_action_tags",
            segment_field="segs",
            hand_type="both",
            min_segment_frames=5,
        )
        result = op.process_single(sample)
        segments = result[Fields.meta]["segs"]

        hand_types_present = {s["hand_type"] for s in segments}
        self.assertIn("right", hand_types_present)
        self.assertIn("left", hand_types_present)

    def test_too_few_frames_skip(self):
        """Fewer frames than min_segment_frames → no segments."""
        states = self._make_states(np.zeros((3, 3)))
        sample = self._make_sample(right_states=states)

        op = VideoAtomicActionSegmentMapper(
            hand_action_field="hand_action_tags",
            segment_field="segs",
            min_segment_frames=8,
            hand_type="right",
        )
        result = op.process_single(sample)
        segments = result[Fields.meta]["segs"]
        self.assertEqual(len(segments), 0)

    def test_no_meta(self):
        """Sample without meta should pass through unchanged."""
        sample = {"text": "hello"}
        op = VideoAtomicActionSegmentMapper()
        result = op.process_single(sample)
        self.assertEqual(result, sample)

    def test_empty_hand_data(self):
        """Empty hand_action_tags → no segments."""
        sample = {Fields.meta: {"hand_action_tags": []}}
        op = VideoAtomicActionSegmentMapper(
            hand_action_field="hand_action_tags",
            segment_field="segs",
        )
        result = op.process_single(sample)
        self.assertNotIn("segs", result[Fields.meta])

    def test_segments_sorted_by_start_frame(self):
        """All segments should be sorted by start_frame."""
        n = 100
        positions = np.zeros((n, 3), dtype=np.float64)
        for i in range(n):
            positions[i] = [np.sin(i * 0.2) * 0.5, np.cos(i * 0.3) * 0.3, 0]
        sample = self._make_sample(
            right_states=self._make_states(positions),
        )
        op = VideoAtomicActionSegmentMapper(
            hand_action_field="hand_action_tags",
            segment_field="segs",
            hand_type="right",
            min_segment_frames=5,
            min_window=5,
        )
        result = op.process_single(sample)
        segments = result[Fields.meta]["segs"]
        starts = [s["start_frame"] for s in segments]
        self.assertEqual(starts, sorted(starts))

    def test_segment_coverage(self):
        """Segments should collectively cover all frames without gaps."""
        n = 50
        positions = np.zeros((n, 3), dtype=np.float64)
        for i in range(n):
            positions[i] = [i * 0.02, 0, 0]
        sample = self._make_sample(
            right_states=self._make_states(positions),
        )
        op = VideoAtomicActionSegmentMapper(
            hand_action_field="hand_action_tags",
            segment_field="segs",
            hand_type="right",
            min_segment_frames=3,
            min_window=3,
        )
        result = op.process_single(sample)
        segments = result[Fields.meta]["segs"]

        if len(segments) >= 2:
            # Verify no gaps between consecutive segments
            for i in range(len(segments) - 1):
                self.assertEqual(
                    segments[i]["end_frame"],
                    segments[i + 1]["start_frame"] - 1,
                    "Gap between consecutive segments",
                )


if __name__ == "__main__":
    unittest.main()

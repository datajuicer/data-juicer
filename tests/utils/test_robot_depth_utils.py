import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from data_juicer.utils import robot_depth_utils as depth


class RobotDepthUtilsTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_resize_intrinsics_preserves_input(self):
        intrinsics = np.array([[600, 0, 320], [0, 400, 240], [0, 0, 1]], dtype=float)
        resized = depth.resize_intrinsics(intrinsics, (640, 480), (120, 320))
        np.testing.assert_allclose(resized, [[300, 0, 160], [0, 100, 60], [0, 0, 1]])
        np.testing.assert_allclose(intrinsics, [[600, 0, 320], [0, 400, 240], [0, 0, 1]])

    def test_invalid_camera_intrinsics_are_rejected(self):
        for matrix in (np.eye(2), np.zeros((3, 3)), np.full((3, 3), np.nan)):
            with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                depth.resize_intrinsics(matrix, (640, 480), (120, 320))

    def test_depth_units_distinguish_calibrated_and_relative_predictions(self):
        self.assertEqual(depth.depth_units("da3metric-large", True, False), "focal_normalized")
        self.assertEqual(depth.depth_units("da3metric-large", False, True), "metres")
        self.assertEqual(depth.depth_units("da3mono-large", False, False), "relative")
        self.assertEqual(depth.depth_units("da3-large", True, False), "metres")

    def test_save_predictions_and_track_sparse_frame_timelines(self):
        prediction = SimpleNamespace(depth=np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), is_metric=False)
        entries = [
            {"sample_index": i, "source_frame_index": index, "timestamp_seconds": index / 10}
            for i, index in enumerate([10, 30])
        ]
        records = depth.save_prediction_batch(
            prediction, entries, self.root / "depth", (4, 4), "da3metric-large", [[1200, 0, 2], [0, 1200, 2], [0, 0, 1]]
        )
        for record in records:
            record["path"] = str(Path(record["path"]).relative_to(self.root))
        result = depth.tracked_depth(
            self.root,
            {"views": {"left": {"frames": records}}},
            "left",
            {"frame_indices": [10, 30], "image_size": {"width": 4, "height": 4}},
            {"object": {0: [0, 0, 2, 2], 1: [2, 2, 4, 4]}, "gripper": {}},
        )
        self.assertEqual([r["source_frame_index"] for r in result], [10, 30])
        self.assertEqual([r["object"] for r in result], [2.0, 16.0])
        self.assertTrue(all(r["units"] == "metres" and r["gripper"] is None for r in result))
        view = {"frame_indices": [10, 31], "image_size": {"width": 4, "height": 4}}
        with self.assertRaisesRegex(ValueError, "frame indices"):
            depth.tracked_depth(self.root, {"views": {"left": {"frames": records}}}, "left", view, {})

    def test_sample_depth_handles_boundaries_and_invalid_values(self):
        values = np.array([[1.0, np.nan], [0.0, 4.0]])
        self.assertEqual(depth.sample_depth(values, [0, 0]), 1)
        self.assertEqual(depth.sample_depth(values, [1000, 1000]), 4)
        for point in ([1000, 0], [0, 1000], [-1, 0], [1001, 0]):
            with self.subTest(point=point):
                self.assertIsNone(depth.sample_depth(values, point))

    def test_mismatched_prediction_count_is_rejected(self):
        prediction = SimpleNamespace(depth=np.ones((2, 2, 2)), is_metric=False)
        with self.assertRaisesRegex(ValueError, "one depth map"):
            depth.save_prediction_batch(prediction, [], self.root, (4, 4), "da3mono-large")


if __name__ == "__main__":
    unittest.main()

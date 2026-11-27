import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from data_juicer.utils import robot_utils as robot


class RobotUtilsTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_instruction_skips_placeholders(self):
        path = self.root / "metadata.parquet"
        pq.write_table(
            pa.table(
                {
                    "other_information.language_instruction_2": [" -1 "],
                    "other_information.language_instruction_3": [" pick up the cup "],
                    "other_information.language_instruction_1": ["fallback"],
                }
            ),
            path,
        )
        self.assertEqual(robot.read_instruction(path), ("pick up the cup", "other_information.language_instruction_3"))

    def test_instruction_supports_custom_columns(self):
        path = self.root / "metadata.parquet"
        pq.write_table(pa.table({"unused": ["ignore"], "task": ["open the drawer"]}), path)
        self.assertEqual(robot.read_instruction(path, ["missing", "task"]), ("open the drawer", "task"))

    def test_empty_metadata_has_clear_error(self):
        path = self.root / "empty.parquet"
        pq.write_table(pa.table({"instruction": pa.array([], type=pa.string())}), path)
        with self.assertRaisesRegex(ValueError, "Empty parquet"):
            robot.read_instruction(path)

    def test_missing_instruction_is_rejected(self):
        path = self.root / "metadata.parquet"
        pq.write_table(pa.table({"other_information.language_instruction_2": [None]}), path)
        with self.assertRaises(ValueError):
            robot.read_instruction(path)

    def test_gripper_anchor_filters_episode_and_preserves_source_indices(self):
        path = self.root / "states.parquet"
        indices = [101, 104, 108, 115, 119, 123, 128, 134, 140]
        pq.write_table(
            pa.table(
                {
                    "frame_index": [9, *reversed(indices)],
                    "episode_index": [0, *([7] * len(indices))],
                    "observation.state.gripper_position": [[0.0, 0.0], *([[1.0, 0.0]] * len(indices))],
                }
            ),
            path,
        )
        output = self.root / "ir" / "phase_anchor.json"
        result = robot.extract_gripper_anchor(path, 7, output)
        self.assertEqual(result["state_frame_count"], len(indices))
        self.assertEqual((result["state_start"], result["state_end"]), (101, 140))
        self.assertIn(result["minimum_smoothed_gripper_frame"], indices)
        self.assertEqual(robot.read_json(output), result)

    def test_gripper_anchor_rejects_short_episode(self):
        path = self.root / "states.parquet"
        pq.write_table(
            pa.table(
                {
                    "frame_index": [0, 1, 2],
                    "episode_index": [7] * 3,
                    "observation.state.gripper_position": [1.0, 0.0, 1.0],
                }
            ),
            path,
        )
        with self.assertRaisesRegex(ValueError, "at least four"):
            robot.extract_gripper_anchor(path, 7, self.root / "anchor.json")

    def test_episode_ids_cannot_escape_artifact_directory(self):
        for name in ("", ".", "..", "../outside", "/outside"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                robot.episode_root({"source": "droid", "episode_id": name}, self.root)
        sample = {"source": "droid", "episode_id": "episode-7"}
        path = robot.episode_root(sample, self.root)
        self.assertEqual(path, self.root / "droid" / "episode-7")
        self.assertEqual(sample["episode_dir"], str(path))

    def test_overlap_and_mask_bounding_box(self):
        self.assertEqual(robot.overlap_ratio([0, 0, 10, 10], [5, 0, 15, 10]), 0.5)
        self.assertEqual(robot.overlap_ratio([0, 0, 10, 10], [10, 0, 20, 10]), 0.0)
        mask = np.zeros((5, 7), dtype=bool)
        mask[1:4, 2:6] = True
        self.assertEqual(robot.mask_to_bbox(np, mask), [2.0, 1.0, 6.0, 4.0])
        self.assertIsNone(robot.mask_to_bbox(np, np.zeros((5, 7))))

    def test_decode_two_camera_videos_preserves_all_source_frames(self):
        import cv2

        raw = self.root / "raw"
        raw.mkdir()
        for camera in ("left", "right"):
            writer = cv2.VideoWriter(str(raw / f"{camera}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (32, 32))
            if not writer.isOpened():
                self.skipTest("OpenCV MPEG-4 encoder unavailable")
            try:
                for i in range(4):
                    writer.write(np.full((32, 32, 3), i * 40, dtype=np.uint8))
            finally:
                writer.release()
        views = robot.decode_camera_views(raw, self.root / "frames")
        self.assertEqual(set(views), {"left", "right"})
        for view in views.values():
            self.assertEqual(view["frame_indices"], [0, 1, 2, 3])
            self.assertEqual(view["source_frame_count"], 4)
            self.assertEqual(view["source_fps"], 10)
            self.assertEqual(len(view["images"]), 4)
            self.assertTrue(all(Path(p).is_file() for p in view["images"]))
            self.assertTrue(Path(view["sampled_video"]).is_file())

    def test_decode_rejects_missing_camera_views(self):
        with self.assertRaisesRegex(ValueError, "at least two"):
            robot.decode_camera_views(self.root, self.root / "frames")


if __name__ == "__main__":
    unittest.main()

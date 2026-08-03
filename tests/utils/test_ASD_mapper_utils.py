import os
import unittest

import numpy as np

from data_juicer.utils.ASD_mapper_utils import (
    bb_intersection_over_union,
    detect_and_mark_anomalies,
    find_human_bounding_box,
    find_max_intersection_and_remaining_dicts,
    get_faces_array,
    get_video_array_cv2,
    scene_detect,
    track_shot,
    update_negative_ones,
)
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

cv2 = LazyLoader("cv2", "opencv-contrib-python")


def _scenedetect_available():
    try:
        import scenedetect  # noqa: F401

        return True
    except ImportError:
        return False


def _cv2_available():
    try:
        import cv2  # noqa: F401

        return True
    except ImportError:
        return False


_SCENEDETECT_AVAILABLE = _scenedetect_available()
_CV2_AVAILABLE = _cv2_available()

DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "ops", "data")
VIDEO_PATH = os.path.join(DATA_DIR, "video1.mp4")


class BBIntersectionOverUnionTest(DataJuicerTestCaseBase):
    """Test IOU computation between two boxes (x1, y1, x2, y2)."""

    def test_identical_boxes(self):
        box = [0, 0, 10, 10]
        self.assertAlmostEqual(bb_intersection_over_union(box, box), 1.0)

    def test_no_overlap(self):
        self.assertAlmostEqual(bb_intersection_over_union([0, 0, 10, 10], [20, 20, 30, 30]), 0.0)

    def test_partial_overlap(self):
        # boxA and boxB are 10x10, overlap region is 5x5 = 25.
        # union = 100 + 100 - 25 = 175 -> iou = 25 / 175
        iou = bb_intersection_over_union([0, 0, 10, 10], [5, 5, 15, 15])
        self.assertAlmostEqual(iou, 25.0 / 175.0)

    def test_eval_col_uses_box_a_area(self):
        # With evalCol=True the denominator is boxAArea only.
        iou = bb_intersection_over_union([0, 0, 10, 10], [5, 5, 15, 15], evalCol=True)
        self.assertAlmostEqual(iou, 25.0 / 100.0)


class FindHumanBoundingBoxTest(DataJuicerTestCaseBase):
    """Test matching a face bbox to the enclosing human body bbox."""

    def test_no_candidate_returns_empty(self):
        # The single human bbox does not enclose the face bbox.
        self.assertEqual(find_human_bounding_box([0, 0, 5, 5], [[10, 10, 20, 20]]), ())

    def test_single_enclosing_candidate(self):
        face = [10, 10, 20, 20]
        human = [0, 0, 100, 100]
        self.assertEqual(find_human_bounding_box(face, [human]), human)

    def test_picks_closest_and_smallest(self):
        face = [40, 10, 60, 30]  # center_x = 50
        # both enclose the face; the first is centered closer to the face and smaller
        human_close = [0, 0, 100, 200]  # center_x = 50
        human_far = [30, 0, 200, 200]  # center_x = 115, larger
        result = find_human_bounding_box(face, [human_close, human_far])
        self.assertEqual(result, human_close)


class UpdateNegativeOnesTest(DataJuicerTestCaseBase):
    """Test filling -1 placeholders by neighbor interpolation."""

    def test_no_negatives_unchanged(self):
        self.assertEqual(update_negative_ones([1, 2, 3]), [1, 2, 3])

    def test_interior_negative_averaged(self):
        self.assertEqual(update_negative_ones([2, -1, 4]), [2, 3, 4])

    def test_leading_negative_uses_right(self):
        self.assertEqual(update_negative_ones([-1, 5, 6]), [5, 5, 6])

    def test_trailing_negative_uses_left(self):
        self.assertEqual(update_negative_ones([7, 8, -1]), [7, 8, 8])

    def test_consecutive_negatives(self):
        # processed left-to-right, so the already-updated left neighbor is
        # reused: idx1 = (2 + 8) / 2 = 5.0, idx2 = (5.0 + 8) / 2 = 6.5
        self.assertEqual(update_negative_ones([2, -1, -1, 8]), [2, 5.0, 6.5, 8])

    def test_all_negatives_raises(self):
        with self.assertRaises(ValueError):
            update_negative_ones([-1, -1])


class DetectAndMarkAnomaliesTest(DataJuicerTestCaseBase):
    """Test sliding-window anomaly marking (outliers set to -1)."""

    def test_no_anomaly_when_smooth(self):
        data = [10, 10, 10, 10, 10]
        result = detect_and_mark_anomalies(data)
        self.assertNotIn(-1, result.tolist())

    def test_outlier_marked_negative(self):
        data = [10, 10, 10, 1000, 10, 10, 10]
        result = detect_and_mark_anomalies(data)
        self.assertEqual(result[3], -1)

    def test_non_positive_values_ignored(self):
        # values <= 0 are never evaluated as anomalies
        data = [0, 0, 0, 0]
        result = detect_and_mark_anomalies(data)
        self.assertEqual(result.tolist(), data)


class TrackShotTest(DataJuicerTestCaseBase):
    """Test face tracking across frames using IOU association."""

    def test_short_track_dropped(self):
        # only a couple of frames -> below minTrack -> no track produced
        scene_faces = [
            [{"frame": 0, "bbox": [0, 0, 50, 50]}],
            [{"frame": 1, "bbox": [1, 1, 51, 51]}],
        ]
        self.assertEqual(track_shot(scene_faces), [])

    def test_continuous_track_kept(self):
        # 15 frames of a steadily moving 50x50 face -> one interpolated track
        scene_faces = []
        for i in range(15):
            scene_faces.append([{"frame": i, "bbox": [i, i, i + 50, i + 50]}])
        tracks = track_shot(scene_faces)
        self.assertEqual(len(tracks), 1)
        self.assertIn("frame", tracks[0])
        self.assertIn("bbox", tracks[0])
        # interpolated frames cover the full [0, 14] range
        self.assertEqual(len(tracks[0]["frame"]), 15)


class FindMaxIntersectionTest(DataJuicerTestCaseBase):
    """Test selecting the largest group of dicts sharing common frames."""

    def test_empty_input(self):
        self.assertEqual(find_max_intersection_and_remaining_dicts([]), ([], []))

    def test_picks_largest_shared_group(self):
        dicts = [
            {"track": {"frame": [1, 2, 3]}},
            {"track": {"frame": [2, 3, 4]}},
            {"track": {"frame": [100, 101]}},
        ]
        max_combo, remaining = find_max_intersection_and_remaining_dicts(dicts)
        # the first two share frames 2 and 3; the third is isolated
        self.assertEqual(len(max_combo), 2)
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0]["track"]["frame"], [100, 101])


class GetFacesArrayTest(DataJuicerTestCaseBase):
    """Test cropping a padded face patch from a frame."""

    def test_output_is_square_patch(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        s, x, y = 20, 50, 50
        face = get_faces_array(frame, s, x, y)
        # the padded crop should be a non-empty 3-channel patch
        self.assertEqual(face.ndim, 3)
        self.assertEqual(face.shape[2], 3)
        self.assertGreater(face.shape[0], 0)
        self.assertGreater(face.shape[1], 0)


@unittest.skipUnless(_CV2_AVAILABLE, "opencv not installed")
class GetVideoArrayCv2Test(DataJuicerTestCaseBase):
    """Test decoding a video into a numpy frame array."""

    def test_reads_frames(self):
        arr = get_video_array_cv2(VIDEO_PATH)
        self.assertIsNotNone(arr)
        # (num_frames, H, W, C)
        self.assertEqual(arr.ndim, 4)
        self.assertGreater(arr.shape[0], 0)
        self.assertEqual(arr.shape[3], 3)

    def test_missing_file_returns_none(self):
        self.assertIsNone(get_video_array_cv2(os.path.join(DATA_DIR, "does_not_exist.mp4")))


@unittest.skipUnless(_SCENEDETECT_AVAILABLE, "scenedetect not installed")
class SceneDetectTest(DataJuicerTestCaseBase):
    """Test scene_detect: the whole video is returned as a single shot.

    Opens the video via scenedetect.open_video and reads its
    base_timecode / duration. Note that open_video returns a VideoStream
    that is not a context manager, so scene_detect must not use `with`.
    """

    def test_single_shot_covers_whole_video(self):
        scenes = scene_detect(VIDEO_PATH)
        self.assertEqual(len(scenes), 1)
        start, end = scenes[0]
        # shot starts at frame 0 and ends at the total number of frames
        self.assertEqual(start.frame_num, 0)
        self.assertGreater(end.frame_num, start.frame_num)

    def test_duration_matches_get_video_array(self):
        if not _CV2_AVAILABLE:
            self.skipTest("opencv not installed")
        scenes = scene_detect(VIDEO_PATH)
        end = scenes[0][1]
        arr = get_video_array_cv2(VIDEO_PATH)
        # scenedetect frame count should be within one frame of cv2 decode
        self.assertLessEqual(abs(end.frame_num - arr.shape[0]), 1)

    def test_repeated_calls_are_stable(self):
        # Repeated calls should succeed and return a consistent single shot.
        first = scene_detect(VIDEO_PATH)
        for _ in range(5):
            scenes = scene_detect(VIDEO_PATH)
            self.assertEqual(len(scenes), 1)
            self.assertEqual(scenes[0][1].frame_num, first[0][1].frame_num)


if __name__ == "__main__":
    unittest.main()

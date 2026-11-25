import copy
import unittest
from pathlib import Path
from unittest.mock import Mock

from datasets import Features, List, Value

from data_juicer.utils.constant import Fields
from data_juicer.utils.video_perception_utils import (
    get_video_frame_groups,
    get_video_output_features,
    nested_video_feature,
)


class VideoPerceptionUtilsTest(unittest.TestCase):
    def test_flat_pathlike_sequence_preserves_subset_and_order(self):
        extractor = Mock()
        paths = [Path("clip/frame_10.png"), Path("clip/frame_2.png")]
        groups = get_video_frame_groups({"frames": paths}, "frames", "clips", extractor)
        self.assertEqual(groups, [["clip/frame_10.png", "clip/frame_2.png"]])
        extractor.process_single.assert_not_called()

    def test_invalid_mixed_groups_raise_actionable_error(self):
        with self.assertRaisesRegex(ValueError, "grouped by video"):
            get_video_frame_groups({"frames": [["a.png"], "b.png"]}, "frames", "clips", Mock())

    def test_output_schema_keeps_existing_meta_fields_without_mutation(self):
        features = Features({"text": Value("string"), Fields.meta: {"label": Value("string")}})
        original = copy.deepcopy(features)
        result = get_video_output_features(features, "pose", {"values": nested_video_feature("float64", 2)})
        self.assertEqual(features, original)
        self.assertEqual(result[Fields.meta]["label"], Value("string"))
        self.assertEqual(result[Fields.meta]["pose"]["values"], List(List(Value("float64"))))

    def test_output_schema_preserves_existing_calibration_fields(self):
        features = Features({Fields.meta: {"calibration": List({"existing": Value("string")})}})
        result = get_video_output_features(features, "calibration", List({"hfov": nested_video_feature("float64")}))
        self.assertIn("existing", result[Fields.meta]["calibration"].feature)
        self.assertIn("hfov", result[Fields.meta]["calibration"].feature)

    def test_non_hf_input_does_not_request_hf_features(self):
        self.assertIsNone(get_video_output_features(None, "pose", {}))


if __name__ == "__main__":
    unittest.main()

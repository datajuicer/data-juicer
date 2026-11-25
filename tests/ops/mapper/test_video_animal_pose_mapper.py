import os
import shutil
import sys
import tempfile
import unittest
from types import ModuleType
from unittest.mock import Mock, patch

import numpy as np

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.video_animal_pose_mapper import VideoAnimalPoseMapper
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


@unittest.skip(
    "The current code can automatically configure the environment, but after the initial setup (including mmpose), the user need to re-run the command for it to work properly."
)
class VideoAnimalPoseMapperTest(DataJuicerTestCaseBase):
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data")
    vid13_path = os.path.join(data_path, "video_animal_pose.mp4")
    vid13_frames_dir = os.path.join(data_path, "video13_frames")
    vid13_frames_path = []
    for x in os.listdir(vid13_frames_dir):
        vid13_frames_path.append(os.path.join(vid13_frames_dir, x))

    tgt_list = [
        {
            "frame_length": 7,
            "frame3_pose_list_shape": [3, 17, 2],
            "frame3_pose_score_list_shape": [3, 17],
            "frame3_animal_bboxes_shape": [3, 4],
        },
        {
            "frame_length": 7,
            "frame3_pose_list_shape": [3, 17, 2],
            "frame3_pose_score_list_shape": [3, 17],
            "frame3_animal_bboxes_shape": [3, 4],
        },
    ]

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory().name
        super().setUp()

    def tearDown(self):
        super().tearDown()
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

    def test(self):
        ds_list = [{"videos": [self.vid13_path]}, {"videos": [self.vid13_path]}]

        op = VideoAnimalPoseMapper(
            vitpose_model_path="apt36k.pth",
            vitpose_config="configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/apt36k/ViTPose_huge_apt36k_256x192.py",
            yoloe_model_path="yoloe-26x-seg.pt",
            if_save_visualization=True,
            save_visualization_dir=os.path.join(self.tmp_dir, "animal_pose_vis1"),
            frame_num=1,
            duration=1,
            frame_dir=os.path.join(self.tmp_dir, "animal_pose_test"),
        )

        dataset = Dataset.from_list(ds_list)
        if Fields.meta not in dataset.features:
            dataset = dataset.add_column(name=Fields.meta, column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.process, num_proc=2, with_rank=True)
        res_list = dataset.to_list()

        for sample, target in zip(res_list, self.tgt_list):
            self.assertEqual(
                len(sample[Fields.meta][MetaKeys.video_animal_pose_tags]["pose_list"]), target["frame_length"]
            )
            self.assertEqual(
                list(np.array(sample[Fields.meta][MetaKeys.video_animal_pose_tags]["pose_list"][3]).shape),
                target["frame3_pose_list_shape"],
            )
            self.assertEqual(
                list(np.array(sample[Fields.meta][MetaKeys.video_animal_pose_tags]["pose_score_list"][3]).shape),
                target["frame3_pose_score_list_shape"],
            )
            self.assertEqual(
                list(np.array(sample[Fields.meta][MetaKeys.video_animal_pose_tags]["animal_bboxes"][3]).shape),
                target["frame3_animal_bboxes_shape"],
            )

    def test_from_extracted_frames(self):
        ds_list = [{MetaKeys.video_frames: self.vid13_frames_path}, {MetaKeys.video_frames: self.vid13_frames_path}]

        op = VideoAnimalPoseMapper(
            vitpose_model_path="apt36k.pth",
            vitpose_config="configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/apt36k/ViTPose_huge_apt36k_256x192.py",
            yoloe_model_path="yoloe-26x-seg.pt",
            if_save_visualization=True,
            save_visualization_dir=os.path.join(self.tmp_dir, "animal_pose_vis2"),
        )

        dataset = Dataset.from_list(ds_list)
        if Fields.meta not in dataset.features:
            dataset = dataset.add_column(name=Fields.meta, column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.process, num_proc=1, with_rank=True)
        res_list = dataset.to_list()

        for sample, target in zip(res_list, self.tgt_list):
            self.assertEqual(
                len(sample[Fields.meta][MetaKeys.video_animal_pose_tags]["pose_list"]), target["frame_length"]
            )
            self.assertEqual(
                list(np.array(sample[Fields.meta][MetaKeys.video_animal_pose_tags]["pose_list"][3]).shape),
                target["frame3_pose_list_shape"],
            )
            self.assertEqual(
                list(np.array(sample[Fields.meta][MetaKeys.video_animal_pose_tags]["pose_score_list"][3]).shape),
                target["frame3_pose_score_list_shape"],
            )
            self.assertEqual(
                list(np.array(sample[Fields.meta][MetaKeys.video_animal_pose_tags]["animal_bboxes"][3]).shape),
                target["frame3_animal_bboxes_shape"],
            )


class VideoAnimalPoseConfigurationTest(unittest.TestCase):
    def construct(self, **kwargs):
        from data_juicer.ops.mapper import video_animal_pose_mapper as module

        modules = {name: ModuleType(name) for name in ["mmpose", "mmpose.apis"]}
        modules["mmpose.apis"].inference_top_down_pose_model = Mock()
        with (
            patch.dict(sys.modules, modules),
            patch.object(module.LazyLoader, "check_packages"),
            patch.object(module.VideoAnimalPoseMapper, "_install_required_packages"),
            patch.object(module.subprocess, "run"),
            patch.object(module, "prepare_model", return_value="offline"),
            patch.object(module, "load_ops") as load,
        ):
            operator = VideoAnimalPoseMapper(accelerator="cpu", num_proc=1, auto_op_parallelism=False, **kwargs)
        return operator, load

    def test_custom_animal_classes(self):
        operator, _ = self.construct(animal_class=["cat", "dog"])
        self.assertEqual(operator.animal_class, ["cat", "dog"])

    def test_default_categories_are_separate(self):
        operator, _ = self.construct()
        self.assertIn("panther", operator.animal_class)
        self.assertIn("pig", operator.animal_class)
        self.assertNotIn("pantherpig", operator.animal_class)

    def test_custom_video_key_is_passed_to_frame_extractor(self):
        operator, load = self.construct(video_key="clips")
        args = load.call_args.args[0][0]["video_extract_frames_mapper"]
        self.assertEqual(args["video_key"], "clips")
        self.assertEqual(args["frame_field"], operator.frame_field)
        self.assertFalse(args["legacy_split_by_text_token"])


if __name__ == "__main__":
    unittest.main()

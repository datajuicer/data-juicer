import os
import unittest
import numpy as np
import tempfile
import shutil

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.video_animal_pose_mapper import \
    VideoAnimalPoseMapper
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

@unittest.skip('The current code can automatically configure the environment, but after the initial setup (including mmpose), the user need to re-run the command for it to work properly.')
class VideoAnimalPoseMapperTest(DataJuicerTestCaseBase):
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')
    vid13_path = os.path.join(data_path, 'video13.mp4')
    vid13_frames_dir = os.path.join(data_path, 'video13_frames')
    vid13_frames_path = []
    for x in os.listdir(vid13_frames_dir):
        vid13_frames_path.append(os.path.join(vid13_frames_dir, x))


    tgt_list = [{
        "frame_length": 7,
        "frame3_pose_list_shape": [3, 17, 2],
        "frame3_pose_score_list_shape": [3, 17],
        "frame3_animal_bboxes_shape": [3, 4],
    }, {
        "frame_length": 7,
        "frame3_pose_list_shape": [3, 17, 2],
        "frame3_pose_score_list_shape": [3, 17],
        "frame3_animal_bboxes_shape": [3, 4],
    }]


    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory().name
        super().setUp()

    def tearDown(self):
        super().tearDown()
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)


    def test(self):
        ds_list = [{
            'videos': [self.vid13_path]
        },  {
            'videos': [self.vid13_path]
        }]

        op = VideoAnimalPoseMapper(
            vitpose_model_path="apt36k.pth",
            vitpose_config="configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/apt36k/ViTPose_huge_apt36k_256x192.py",
            yoloe_model_path="yoloe-26x-seg.pt",
            if_save_visualization=True,
            save_visualization_dir=os.path.join(self.tmp_dir, "animal_pose_vis1"),
            frame_num=1,
            duration=1,
            frame_dir=os.path.join(self.tmp_dir, "animal_pose_test")
        )

        dataset = Dataset.from_list(ds_list)
        if Fields.meta not in dataset.features:
            dataset = dataset.add_column(name=Fields.meta,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.process, num_proc=2, with_rank=True)
        res_list = dataset.to_list()

        for sample, target in zip(res_list, self.tgt_list):
            self.assertEqual(len(sample[Fields.meta][MetaKeys.video_animal_pose_tags]["pose_list"]), target["frame_length"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.video_animal_pose_tags]["pose_list"][3]).shape), target["frame3_pose_list_shape"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.video_animal_pose_tags]["pose_score_list"][3]).shape), target["frame3_pose_score_list_shape"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.video_animal_pose_tags]["animal_bboxes"][3]).shape), target["frame3_animal_bboxes_shape"])


    def test_from_extracted_frames(self):
        ds_list = [{
            MetaKeys.video_frames: self.vid13_frames_path
        },  {
            MetaKeys.video_frames: self.vid13_frames_path
        }]

        op = VideoAnimalPoseMapper(
            vitpose_model_path="apt36k.pth",
            vitpose_config="configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/apt36k/ViTPose_huge_apt36k_256x192.py",
            yoloe_model_path="yoloe-26x-seg.pt",
            if_save_visualization=True,
            save_visualization_dir=os.path.join(self.tmp_dir, "animal_pose_vis2"),
        )

        dataset = Dataset.from_list(ds_list)
        if Fields.meta not in dataset.features:
            dataset = dataset.add_column(name=Fields.meta,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.process, num_proc=1, with_rank=True)
        res_list = dataset.to_list()

        for sample, target in zip(res_list, self.tgt_list):
            self.assertEqual(len(sample[Fields.meta][MetaKeys.video_animal_pose_tags]["pose_list"]), target["frame_length"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.video_animal_pose_tags]["pose_list"][3]).shape), target["frame3_pose_list_shape"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.video_animal_pose_tags]["pose_score_list"][3]).shape), target["frame3_pose_score_list_shape"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.video_animal_pose_tags]["animal_bboxes"][3]).shape), target["frame3_animal_bboxes_shape"])


if __name__ == '__main__':
    unittest.main()
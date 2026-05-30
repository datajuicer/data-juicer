import os
import unittest
import numpy as np
import tempfile
import shutil

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.video_face_keypoints_mapper import \
    VideoFaceKeypointsMapper
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class VideoFaceKeypointsMapperTest(DataJuicerTestCaseBase):
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')
    vid3_path = os.path.join(data_path, 'video3.mp4')
    vid4_path = os.path.join(data_path, 'video4.mp4')
    vid3_frames_dir = os.path.join(data_path, 'video3_frames')
    vid4_frames_dir = os.path.join(data_path, 'video4_frames')
    vid3_frames_path = []
    vid4_frames_path = []
    for x in os.listdir(vid3_frames_dir):
        vid3_frames_path.append(os.path.join(vid3_frames_dir, x))
    for x in os.listdir(vid4_frames_dir):
        vid4_frames_path.append(os.path.join(vid4_frames_dir, x))


    tgt_list = [{
        "keypoints_list_shape": [98, 2],
        "face_bboxes_shape": [4]
    }, {
        "keypoints_list_shape": [98, 2],
        "face_bboxes_shape": [4]
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
            'videos': [self.vid3_path]
        },  {
            'videos': [self.vid4_path]
        }]

        op = VideoFaceKeypointsMapper(
            ldeq_model_path="final.pth.tar",
            if_save_visualization=True,
            save_visualization_dir=os.path.join(self.tmp_dir, "facekeypoints_vis"),
            frame_num = 1,
            duration = 3,
            frame_dir = os.path.join(self.tmp_dir, "facekeypoints_test")
        )

        dataset = Dataset.from_list(ds_list)
        if Fields.meta not in dataset.features:
            dataset = dataset.add_column(name=Fields.meta,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.process, num_proc=2, with_rank=True)
        res_list = dataset.to_list()

        for sample, target in zip(res_list, self.tgt_list):
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.video_face_keypoints_tags]["face_keypoints"][0]).shape[1:]), target["keypoints_list_shape"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.video_face_keypoints_tags]["face_bboxes"][0]).shape[1:]), target["face_bboxes_shape"])


    def test_from_extracted_frames(self):

        ds_list = [{
            MetaKeys.video_frames: self.vid3_frames_path,
        },  {
            MetaKeys.video_frames: self.vid4_frames_path,
        }]

        op = VideoFaceKeypointsMapper(
            ldeq_model_path="final.pth.tar",
            if_save_visualization=True,
            save_visualization_dir=os.path.join(self.tmp_dir, "facekeypoints_vis")
        )

        dataset = Dataset.from_list(ds_list)
        if Fields.meta not in dataset.features:
            dataset = dataset.add_column(name=Fields.meta,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.process, num_proc=1, with_rank=True)
        res_list = dataset.to_list()

        for sample, target in zip(res_list, self.tgt_list):
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.video_face_keypoints_tags]["face_keypoints"][0]).shape[1:]), target["keypoints_list_shape"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.video_face_keypoints_tags]["face_bboxes"][0]).shape[1:]), target["face_bboxes_shape"])

if __name__ == '__main__':
    unittest.main()
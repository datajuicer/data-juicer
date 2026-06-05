import os
import unittest
import numpy as np
import tempfile
import shutil

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.video_human_3d_pose_mapper import \
    VideoHuman3DPoseMapper
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


@unittest.skip("Please refer to the Human3R repository to set up the environment and download the model.")
class VideoHuman3DPoseMapperTest(DataJuicerTestCaseBase):
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data')
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
        "valid_frame_length": 16,
        "camera_pose_shape": [7],
        "conf_shape": [512, 288],
        "smpl_shape_shape": [2, 10],
        "smpl_transl_shape": [2, 3],
        "smpl_rotmat_shape": [2, 53, 3, 3],
        "smpl_expression_shape": [2, 10],
        "smpl_id_shape": [2],
        "smpl_loc_shape": [2, 2]
    }, {
        "valid_frame_length": 5,
        "camera_pose_shape": [7],
        "conf_shape": [384, 512],
        "smpl_shape_shape": [2, 10],
        "smpl_transl_shape": [2, 3],
        "smpl_rotmat_shape": [2, 53, 3, 3],
        "smpl_expression_shape": [2, 10],
        "smpl_id_shape": [2],
        "smpl_loc_shape": [2, 2]
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

        op = VideoHuman3DPoseMapper(
            model_path="human3r_896L.pth",
            frame_num = 1,
            duration = 3,
            frame_dir = self.tmp_dir
        )

        dataset = Dataset.from_list(ds_list)
        if Fields.meta not in dataset.features:
            dataset = dataset.add_column(name=Fields.meta,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.process, num_proc=2, with_rank=True)
        res_list = dataset.to_list()

        for sample, target in zip(res_list, self.tgt_list):
            self.assertEqual(len(sample[Fields.meta][MetaKeys.video_human_3d_pose_tags]["valid_frame_list"]), target["valid_frame_length"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.video_human_3d_pose_tags]["camera_pose"][0]).shape), target["camera_pose_shape"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.video_human_3d_pose_tags]["conf"][0]).shape), target["conf_shape"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.video_human_3d_pose_tags]["smpl_shape"][0]).shape), target["smpl_shape_shape"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.video_human_3d_pose_tags]["smpl_transl"][0]).shape), target["smpl_transl_shape"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.video_human_3d_pose_tags]["smpl_rotmat"][0]).shape), target["smpl_rotmat_shape"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.video_human_3d_pose_tags]["smpl_expression"][0]).shape), target["smpl_expression_shape"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.video_human_3d_pose_tags]["smpl_id"][0]).shape), target["smpl_id_shape"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.video_human_3d_pose_tags]["smpl_loc"][0]).shape), target["smpl_loc_shape"])


    def test_from_extracted_frames(self):

        ds_list = [{
            MetaKeys.video_frames: self.vid3_frames_path,
        },  {
            MetaKeys.video_frames: self.vid4_frames_path,
        }]

        op = VideoHuman3DPoseMapper(
            model_path="human3r_896L.pth",
        )

        dataset = Dataset.from_list(ds_list)
        if Fields.meta not in dataset.features:
            dataset = dataset.add_column(name=Fields.meta,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.process, num_proc=1, with_rank=True)
        res_list = dataset.to_list()

        for sample, target in zip(res_list, self.tgt_list):
            self.assertEqual(len(sample[Fields.meta][MetaKeys.video_human_3d_pose_tags]["valid_frame_list"]), target["valid_frame_length"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.video_human_3d_pose_tags]["camera_pose"][0]).shape), target["camera_pose_shape"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.video_human_3d_pose_tags]["conf"][0]).shape), target["conf_shape"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.video_human_3d_pose_tags]["smpl_shape"][0]).shape), target["smpl_shape_shape"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.video_human_3d_pose_tags]["smpl_transl"][0]).shape), target["smpl_transl_shape"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.video_human_3d_pose_tags]["smpl_rotmat"][0]).shape), target["smpl_rotmat_shape"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.video_human_3d_pose_tags]["smpl_expression"][0]).shape), target["smpl_expression_shape"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.video_human_3d_pose_tags]["smpl_id"][0]).shape), target["smpl_id_shape"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.video_human_3d_pose_tags]["smpl_loc"][0]).shape), target["smpl_loc_shape"])

if __name__ == '__main__':
    unittest.main()
import os
import unittest
import numpy as np
import tempfile
import shutil

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.video_whole_body_pose_estimation_mapper import VideoWholeBodyPoseEstimationMapper
from data_juicer.utils.mm_utils import SpecialTokens
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class VideoWholeBodyPoseEstimationMapperTest(DataJuicerTestCaseBase):
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

    ds_list = [{
        'videos': [vid3_path]
    },  {
        'videos': [vid4_path]
    }]

    ds_from_frames_list = [{
        MetaKeys.video_frames: vid3_frames_path,
    },  {
        MetaKeys.video_frames: vid4_frames_path,
    }]

    tgt_list = [{
        "body_keypoints_shape": [2, 18, 2],
        "foot_keypoints_shape": [2, 6, 2],
        "faces_keypoints_shape": [2, 68, 2],
        "hands_keypoints_shape": [4, 21, 2],
        "bbox_shape": [2, 4]
    }, {
        "body_keypoints_shape": [2, 18, 2],
        "foot_keypoints_shape": [2, 6, 2],
        "faces_keypoints_shape": [2, 68, 2],
        "hands_keypoints_shape": [4, 21, 2],
        "bbox_shape": [2, 4]
    }]

    tgt_from_frames_list = [{
        "body_keypoints_shape": [2, 18, 2],
        "foot_keypoints_shape": [2, 6, 2],
        "faces_keypoints_shape": [2, 68, 2],
        "hands_keypoints_shape": [4, 21, 2],
        "bbox_shape": [2, 4]
    }, {
        "body_keypoints_shape": [2, 18, 2],
        "foot_keypoints_shape": [2, 6, 2],
        "faces_keypoints_shape": [2, 68, 2],
        "hands_keypoints_shape": [4, 21, 2],
        "bbox_shape": [2, 4]
    }]

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory().name
        super().setUp()

    def tearDown(self):
        super().tearDown()
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

    def test(self):

        op = VideoWholeBodyPoseEstimationMapper(
            onnx_det_model="yolox_l.onnx",
            onnx_pose_model="dw-ll_ucoco_384.onnx",
            frame_num=1,
            duration=1,
            tag_field_name=MetaKeys.pose_estimation_tags,
            frame_dir=os.path.join(self.tmp_dir, "dwpose_test1"),
            if_save_visualization=True,
            save_visualization_dir=os.path.join(self.tmp_dir, "dwpose_vis1")
        )
        dataset = Dataset.from_list(self.ds_list)
        if Fields.meta not in dataset.features:
            dataset = dataset.add_column(name=Fields.meta,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.process, num_proc=1, with_rank=True)
        res_list = dataset.to_list()

        for sample, target in zip(res_list, self.tgt_list):
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.pose_estimation_tags]["body_keypoints"][2]).shape), target["body_keypoints_shape"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.pose_estimation_tags]["foot_keypoints"][2]).shape), target["foot_keypoints_shape"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.pose_estimation_tags]["faces_keypoints"][2]).shape), target["faces_keypoints_shape"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.pose_estimation_tags]["hands_keypoints"][2]).shape), target["hands_keypoints_shape"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.pose_estimation_tags]["bbox_results_list"][2]).shape), target["bbox_shape"])


    def test_mul_proc(self):

        op = VideoWholeBodyPoseEstimationMapper(
            onnx_det_model="yolox_l.onnx",
            onnx_pose_model="dw-ll_ucoco_384.onnx",
            frame_num=1,
            duration=1,
            tag_field_name=MetaKeys.pose_estimation_tags,
            frame_dir=os.path.join(self.tmp_dir, "dwpose_test2"),
            if_save_visualization=True,
            save_visualization_dir=os.path.join(self.tmp_dir, "dwpose_vis2")
        )
        dataset = Dataset.from_list(self.ds_list)
        if Fields.meta not in dataset.features:
            dataset = dataset.add_column(name=Fields.meta,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.process, num_proc=2, with_rank=True)
        res_list = dataset.to_list()

        for sample, target in zip(res_list, self.tgt_list):
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.pose_estimation_tags]["body_keypoints"][2]).shape), target["body_keypoints_shape"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.pose_estimation_tags]["foot_keypoints"][2]).shape), target["foot_keypoints_shape"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.pose_estimation_tags]["faces_keypoints"][2]).shape), target["faces_keypoints_shape"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.pose_estimation_tags]["hands_keypoints"][2]).shape), target["hands_keypoints_shape"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.pose_estimation_tags]["bbox_results_list"][2]).shape), target["bbox_shape"])


    def test_from_extracted_frames(self):

        op = VideoWholeBodyPoseEstimationMapper(
            onnx_det_model="yolox_l.onnx",
            onnx_pose_model="dw-ll_ucoco_384.onnx",
            tag_field_name=MetaKeys.pose_estimation_tags,
            if_save_visualization=True,
            save_visualization_dir=os.path.join(self.tmp_dir, "dwpose_vis3")
        )
        dataset = Dataset.from_list(self.ds_from_frames_list)
        if Fields.meta not in dataset.features:
            dataset = dataset.add_column(name=Fields.meta,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.process, num_proc=1, with_rank=True)
        res_list = dataset.to_list()

        for sample, target in zip(res_list, self.tgt_from_frames_list):
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.pose_estimation_tags]["body_keypoints"][1]).shape), target["body_keypoints_shape"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.pose_estimation_tags]["foot_keypoints"][1]).shape), target["foot_keypoints_shape"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.pose_estimation_tags]["faces_keypoints"][1]).shape), target["faces_keypoints_shape"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.pose_estimation_tags]["hands_keypoints"][1]).shape), target["hands_keypoints_shape"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.pose_estimation_tags]["bbox_results_list"][1]).shape), target["bbox_shape"])


if __name__ == '__main__':
    unittest.main()
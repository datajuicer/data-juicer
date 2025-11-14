import os
import unittest
import numpy as np

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.video_whole_body_pose_estimation_mapper import VideoWholeBodyPoseEstimationMapper
from data_juicer.utils.mm_utils import SpecialTokens
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase
from data_juicer.utils.cache_utils import DATA_JUICER_ASSETS_CACHE


class VideoWholeBodyPoseEstimationMapperTest(DataJuicerTestCaseBase):
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')
    vid3_path = os.path.join(data_path, 'video3.mp4')
    vid4_path = os.path.join(data_path, 'video4.mp4')

    ds_list = [{
        'videos': [vid3_path]
    },  {
        'videos': [vid4_path]
    }]

    tgt_list = [{
        "body_keypoints_shape": [2, 18, 2],
        "foot_keypoints_shape": [2, 6, 2],
        "faces_keypoints_shape": [2, 68, 2],
        "hands_keypoints_shape": [4, 21, 2],
        "bbox_results_list_length": 49,
        "bbox_shape": [2, 4]
    }, {
        "body_keypoints_shape": [2, 18, 2],
        "foot_keypoints_shape": [2, 6, 2],
        "faces_keypoints_shape": [2, 68, 2],
        "hands_keypoints_shape": [4, 21, 2],
        "bbox_results_list_length": 22,
        "bbox_shape": [2, 4]
    }]

    def test(self):

        op = VideoWholeBodyPoseEstimationMapper(
            onnx_det_model="yolox_l.onnx",
            onnx_pose_model="dw-ll_ucoco_384.onnx",
            frame_num=1,
            duration=1,
            tag_field_name=MetaKeys.pose_estimation_tags,
            frame_dir=DATA_JUICER_ASSETS_CACHE,
            if_save_visualization=True,
            save_visualization_dir=DATA_JUICER_ASSETS_CACHE
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
            self.assertEqual(len(sample[Fields.meta][MetaKeys.pose_estimation_tags]["bbox_results_list"]), target["bbox_results_list_length"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.pose_estimation_tags]["bbox_results_list"][2]).shape), target["bbox_shape"])


    def test_mul_proc(self):

        op = VideoWholeBodyPoseEstimationMapper(
            onnx_det_model="yolox_l.onnx",
            onnx_pose_model="dw-ll_ucoco_384.onnx",
            frame_num=1,
            duration=1,
            tag_field_name=MetaKeys.pose_estimation_tags,
            frame_dir=DATA_JUICER_ASSETS_CACHE,
            if_save_visualization=True,
            save_visualization_dir=DATA_JUICER_ASSETS_CACHE
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
            self.assertEqual(len(sample[Fields.meta][MetaKeys.pose_estimation_tags]["bbox_results_list"]), target["bbox_results_list_length"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.pose_estimation_tags]["bbox_results_list"][2]).shape), target["bbox_shape"])


if __name__ == '__main__':
    unittest.main()
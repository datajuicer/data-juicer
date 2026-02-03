import os
import unittest
import numpy as np

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.video_hand_reconstruction_hawor_mapper import VideoHandReconstructionHaworMapper
from data_juicer.utils.mm_utils import SpecialTokens
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase
from data_juicer.utils.cache_utils import DATA_JUICER_ASSETS_CACHE


@unittest.skip('Users need to download MANO_RIGHT.pkl.')
class VideoHandReconstructionHaworMapperTest(DataJuicerTestCaseBase):
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
        "fov_x": 0.7572688730116571,
        "left_frame_id_list": [2, 7, 8, 9, 10, 28, 33, 34, 36, 38, 39, 43, 44, 45, 46, 47, 48],
        "left_beta_list_shape": (17, 10),
        "left_hand_pose_list_shape": (17, 15, 3, 3),
        "left_global_orient_list_shape": (17, 3, 3),
        "left_transl_list_shape": (17, 3),
        "right_frame_id_list": [1, 2, 3, 4, 8, 9, 11, 12, 13, 14, 16, 17, 19, 20, 22, 23, 24, 29, 30, 33, 34, 36, 37, 40, 41, 42, 43, 44, 45, 47, 48],
        "right_beta_list_shape": (31, 10),
        "right_hand_pose_list_shape": (31, 15, 3, 3),
        "right_global_orient_list_shape": (31, 3, 3),
        "right_transl_list_shape": (31, 3),
    }, {
        "fov_x": 0.6575318204118722,
        "left_frame_id_list": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 17, 18, 19, 20, 21],
        "left_beta_list_shape": (17, 10),
        "left_hand_pose_list_shape": (17, 15, 3, 3),
        "left_global_orient_list_shape": (17, 3, 3),
        "left_transl_list_shape": (17, 3),
        "right_frame_id_list": [0, 3, 8, 16],
        "right_beta_list_shape": (4, 10),
        "right_hand_pose_list_shape": (4, 15, 3, 3),
        "right_global_orient_list_shape": (4, 3, 3),
        "right_transl_list_shape": (4, 3),
    }]

    def test(self):

        op = VideoHandReconstructionHaworMapper(
            hawor_model_path="hawor.ckpt",
            hawor_config_path="model_config.yaml",
            hawor_detector_path="detector.pt",
            moge_model_path="Ruicheng/moge-2-vitl",
            mano_right_path="path_to_mano_right_pkl",
            frame_num=1,
            duration=1,
            thresh=0.2,
            frame_dir=DATA_JUICER_ASSETS_CACHE,
            moge_output_info_dir=DATA_JUICER_ASSETS_CACHE,
        )
        dataset = Dataset.from_list(self.ds_list)
        if Fields.meta not in dataset.features:
            dataset = dataset.add_column(name=Fields.meta,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.process, num_proc=1, with_rank=True)
        res_list = dataset.to_list()

        for sample, target in zip(res_list, self.tgt_list):
            self.assertEqual(abs(sample[Fields.meta][MetaKeys.hand_reconstruction_hawor_tags]["fov_x"] - target["fov_x"]) < 0.01, True)
            self.assertEqual(np.array(sample[Fields.meta][MetaKeys.hand_reconstruction_hawor_tags]["left_beta_list"]).shape[1:], target["left_beta_list_shape"][1:])
            self.assertEqual(np.array(sample[Fields.meta][MetaKeys.hand_reconstruction_hawor_tags]["left_hand_pose_list"]).shape[1:], target["left_hand_pose_list_shape"][1:])
            self.assertEqual(np.array(sample[Fields.meta][MetaKeys.hand_reconstruction_hawor_tags]["left_global_orient_list"]).shape[1:], target["left_global_orient_list_shape"][1:])
            self.assertEqual(np.array(sample[Fields.meta][MetaKeys.hand_reconstruction_hawor_tags]["left_transl_list"]).shape[1:], target["left_transl_list_shape"][1:])
            self.assertEqual(np.array(sample[Fields.meta][MetaKeys.hand_reconstruction_hawor_tags]["right_beta_list"]).shape[1:], target["right_beta_list_shape"][1:])
            self.assertEqual(np.array(sample[Fields.meta][MetaKeys.hand_reconstruction_hawor_tags]["right_hand_pose_list"]).shape[1:], target["right_hand_pose_list_shape"][1:])
            self.assertEqual(np.array(sample[Fields.meta][MetaKeys.hand_reconstruction_hawor_tags]["right_global_orient_list"]).shape[1:], target["right_global_orient_list_shape"][1:])
            self.assertEqual(np.array(sample[Fields.meta][MetaKeys.hand_reconstruction_hawor_tags]["right_transl_list"]).shape[1:], target["right_transl_list_shape"][1:])

    
    def test_mul_proc(self):

        op = VideoHandReconstructionHaworMapper(
            hawor_model_path="hawor.ckpt",
            hawor_config_path="model_config.yaml",
            hawor_detector_path="detector.pt",
            moge_model_path="Ruicheng/moge-2-vitl",
            mano_right_path="path_to_mano_right_pkl",
            frame_num=1,
            duration=1,
            thresh=0.2,
            frame_dir=DATA_JUICER_ASSETS_CACHE,
            moge_output_info_dir=DATA_JUICER_ASSETS_CACHE,
        )
        dataset = Dataset.from_list(self.ds_list)
        if Fields.meta not in dataset.features:
            dataset = dataset.add_column(name=Fields.meta,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.process, num_proc=2, with_rank=True)
        res_list = dataset.to_list()

        for sample, target in zip(res_list, self.tgt_list):
            self.assertEqual(abs(sample[Fields.meta][MetaKeys.hand_reconstruction_hawor_tags]["fov_x"] - target["fov_x"]) < 0.01, True)
            self.assertEqual(np.array(sample[Fields.meta][MetaKeys.hand_reconstruction_hawor_tags]["left_beta_list"]).shape[1:], target["left_beta_list_shape"][1:])
            self.assertEqual(np.array(sample[Fields.meta][MetaKeys.hand_reconstruction_hawor_tags]["left_hand_pose_list"]).shape[1:], target["left_hand_pose_list_shape"][1:])
            self.assertEqual(np.array(sample[Fields.meta][MetaKeys.hand_reconstruction_hawor_tags]["left_global_orient_list"]).shape[1:], target["left_global_orient_list_shape"][1:])
            self.assertEqual(np.array(sample[Fields.meta][MetaKeys.hand_reconstruction_hawor_tags]["left_transl_list"]).shape[1:], target["left_transl_list_shape"][1:])
            self.assertEqual(np.array(sample[Fields.meta][MetaKeys.hand_reconstruction_hawor_tags]["right_beta_list"]).shape[1:], target["right_beta_list_shape"][1:])
            self.assertEqual(np.array(sample[Fields.meta][MetaKeys.hand_reconstruction_hawor_tags]["right_hand_pose_list"]).shape[1:], target["right_hand_pose_list_shape"][1:])
            self.assertEqual(np.array(sample[Fields.meta][MetaKeys.hand_reconstruction_hawor_tags]["right_global_orient_list"]).shape[1:], target["right_global_orient_list_shape"][1:])
            self.assertEqual(np.array(sample[Fields.meta][MetaKeys.hand_reconstruction_hawor_tags]["right_transl_list"]).shape[1:], target["right_transl_list_shape"][1:])


if __name__ == '__main__':
    unittest.main()
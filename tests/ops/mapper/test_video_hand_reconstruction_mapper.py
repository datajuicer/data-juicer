import os
import unittest
import numpy as np

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.video_hand_reconstruction_mapper import VideoHandReconstructionMapper
from data_juicer.utils.mm_utils import SpecialTokens
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase
from data_juicer.utils.cache_utils import DATA_JUICER_ASSETS_CACHE


class VideoHandReconstructionMapperTest(DataJuicerTestCaseBase):
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
        "frame_nums": 49,
        "vertices_shape": [2, 778, 3],
        "camera_translation_shape": [2, 3],
        "if_right_hand_shape": [2],
        "joints_shape": [2, 21, 3],
        "keypoints_shape": [2, 778, 2]
    }, {
        "frame_nums": 22,
        "vertices_shape": [1, 778, 3],
        "camera_translation_shape": [1, 3],
        "if_right_hand_shape": [1],
        "joints_shape": [1, 21, 3],
        "keypoints_shape": [1, 778, 2]
    }]


    def test(self):

        op = VideoHandReconstructionMapper(
            wilor_model_path="wilor_final.ckpt",
            wilor_model_config="model_config.yaml",
            detector_model_path="detector.pt",
            mano_right_path="path_to_mano_right_pkl",
            frame_num=1,
            duration=1,
            frame_dir=DATA_JUICER_ASSETS_CACHE,
            if_save_visualization=True,
            save_visualization_dir=DATA_JUICER_ASSETS_CACHE,
            if_save_mesh=True,
            save_mesh_dir=DATA_JUICER_ASSETS_CACHE,
        )
        dataset = Dataset.from_list(self.ds_list)
        if Fields.meta not in dataset.features:
            dataset = dataset.add_column(name=Fields.meta,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.process, num_proc=1, with_rank=True)
        res_list = dataset.to_list()

        for sample, target in zip(res_list, self.tgt_list):
            self.assertEqual(len(sample[Fields.meta][MetaKeys.hand_reconstruction_tags]["vertices"]), target["frame_nums"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.hand_reconstruction_tags]["vertices"][10]).shape), target["vertices_shape"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.hand_reconstruction_tags]["camera_translation"][10]).shape), target["camera_translation_shape"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.hand_reconstruction_tags]["if_right_hand"][10]).shape), target["if_right_hand_shape"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.hand_reconstruction_tags]["joints"][10]).shape), target["joints_shape"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.hand_reconstruction_tags]["keypoints"][10]).shape), target["keypoints_shape"])


    def test_mul_proc(self):

        op = VideoHandReconstructionMapper(
            wilor_model_path="wilor_final.ckpt",
            wilor_model_config="model_config.yaml",
            detector_model_path="detector.pt",
            mano_right_path="path_to_mano_right_pkl",
            frame_num=1,
            duration=1,
            frame_dir=DATA_JUICER_ASSETS_CACHE,
            if_save_visualization=True,
            save_visualization_dir=DATA_JUICER_ASSETS_CACHE,
            if_save_mesh=True,
            save_mesh_dir=DATA_JUICER_ASSETS_CACHE,
        )
        dataset = Dataset.from_list(self.ds_list)
        if Fields.meta not in dataset.features:
            dataset = dataset.add_column(name=Fields.meta,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.process, num_proc=2, with_rank=True)
        res_list = dataset.to_list()

        for sample, target in zip(res_list, self.tgt_list):
            self.assertEqual(len(sample[Fields.meta][MetaKeys.hand_reconstruction_tags]["vertices"]), target["frame_nums"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.hand_reconstruction_tags]["vertices"][10]).shape), target["vertices_shape"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.hand_reconstruction_tags]["camera_translation"][10]).shape), target["camera_translation_shape"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.hand_reconstruction_tags]["if_right_hand"][10]).shape), target["if_right_hand_shape"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.hand_reconstruction_tags]["joints"][10]).shape), target["joints_shape"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.hand_reconstruction_tags]["keypoints"][10]).shape), target["keypoints_shape"])


if __name__ == '__main__':
    unittest.main()
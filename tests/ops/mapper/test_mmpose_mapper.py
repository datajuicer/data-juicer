import os
import shutil
import unittest
import tempfile
import numpy as np

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.mmpose_mapper import MMPoseMapper
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.mm_utils import SpecialTokens
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class MMPoseMapperTest(DataJuicerTestCaseBase):

    def setUp(self):
        super().setUp()

        self.tmp_dir = tempfile.TemporaryDirectory().name
        os.makedirs(self.tmp_dir, exist_ok=True)

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.tmp_dir)

    def test_mmpose_mapper(self):
        data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data')
        img1 = os.path.join(data_path, 'img3.jpg')
        img2 = os.path.join(data_path, 'img8.jpg')

        base_dir = '/mnt/workspace/mmdeploy/'
        deploy_cfg = base_dir + 'configs/mmpose/pose-detection_onnxruntime_static.py'
        model_cfg = base_dir + 'td-hm_hrnet-w32_8xb64-210e_coco-256x192.py'
        backend_model = [base_dir + 'mmdeploy_models/mmpose/ort/end2end.onnx']

        ds_list = [{
            'text': f'{SpecialTokens.image}a photo',
            'images': [img2]
        }, {
            'text': f'{SpecialTokens.image}a photo, a women with an umbrella',
            'images': [img1]
        }]
        dataset = Dataset.from_list(ds_list)

        visualization_dir=os.path.join(self.tmp_dir, 'vis_outs')
        op = MMPoseMapper(
            deploy_cfg=deploy_cfg,
            model_cfg=model_cfg, 
            model_files=backend_model,
            visualization_dir=visualization_dir
        )

        dataset = dataset.map(op.process, with_rank=True)
        dataset_list = dataset.to_list()
        
        for out in dataset_list:
            pose_info = out[Fields.meta][MetaKeys.pose_info][0]
            self.assertEqual(np.array(pose_info['bbox_scores']), (1, ))
            self.assertEqual(np.array(pose_info['bboxes']).shape, (1, 4))
            self.assertEqual(np.array(pose_info['keypoint_names']).shape, (17, ))
            self.assertEqual(np.array(pose_info['keypoint_scores'][0]).shape, (17, ))
            self.assertEqual(np.array(pose_info['keypoints'][0]).shape, (17, 2))


if __name__ == '__main__':
    unittest.main()

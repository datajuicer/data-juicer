import os
import glob
import shutil
import unittest
import subprocess
import tempfile
import numpy as np

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.image_mmpose_mapper import ImageMMPoseMapper
from data_juicer.utils.cache_utils import DATA_JUICER_CACHE_HOME
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.mm_utils import SpecialTokens
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

mmengine = LazyLoader('mmengine')


class ImageMMPoseMapperTest(DataJuicerTestCaseBase):

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

        mmlab_home = os.path.join(DATA_JUICER_CACHE_HOME, 'mmlab')
        mmdeploy_home = os.path.join(mmlab_home, 'mmdeploy')
        os.makedirs(mmlab_home, exist_ok=True)

        deploy_cfg = os.path.join(mmdeploy_home, 'configs/mmpose/pose-detection_onnxruntime_static.py')
        model_cfg = os.path.join(mmlab_home, 'td-hm_hrnet-w32_8xb64-210e_coco-256x192.py')
        torch_model = glob.glob(os.path.join(mmlab_home, 'td-hm_hrnet-w32_8xb64-210e_coco-256x192-*.pth'))
        if len(torch_model) >= 1:
            torch_model = torch_model[0]
        out_deploy_dir = f"{mmdeploy_home}/mmdeploy_models/mmpose/ort"
        backend_model = os.path.join(out_deploy_dir, 'end2end.onnx')

        # clone mmpose codes
        if not os.path.exists(mmdeploy_home):
            res = subprocess.call(
                f"git clone https://github.com/open-mmlab/mmdeploy.git {mmdeploy_home}",
                shell=True
            )
            if res != 0:
                raise RuntimeError('Failed to clone mmdeploy.')
        
        # download mmpose model and config
        if not os.path.exists(model_cfg) or not os.path.exists(torch_model):
            res = subprocess.call(
                f"mim download mmpose --config td-hm_hrnet-w32_8xb64-210e_coco-256x192 --dest {mmlab_home}",
                shell=True
            )
            if res != 0:
                raise RuntimeError('Failed to download mmpose model.')
        
        # convert mmpose model to onnx
        if not os.path.exists(backend_model):
            import mmengine
            cmd = f"python {mmdeploy_home}/tools/deploy.py {deploy_cfg} {model_cfg} {torch_model} " + \
                f"{mmdeploy_home}/demo/resources/human-pose.jpg --work-dir {out_deploy_dir}"
            res = subprocess.call(
                cmd,
                shell=True
            )
            if res != 0:
                raise RuntimeError(f'Convert mmpose model failed with cmd: {cmd}.')

        ds_list = [{
            'text': f'{SpecialTokens.image}a photo',
            'images': [img2],
            Fields.meta: {}
        }, {
            'text': f'{SpecialTokens.image}a photo, a women with an umbrella',
            'images': [img1],
            Fields.meta: {}
        }]
        dataset = Dataset.from_list(ds_list)

        visualization_dir=os.path.join(self.tmp_dir, 'vis_outs')
        op = ImageMMPoseMapper(
            deploy_cfg=deploy_cfg,
            model_cfg=model_cfg, 
            model_files=backend_model,
            visualization_dir=visualization_dir
        )

        dataset = dataset.map(op.process, with_rank=True)
        dataset_list = dataset.to_list()
        
        for out in dataset_list:
            pose_info = out[Fields.meta][MetaKeys.pose_info][0]
            self.assertEqual(np.array(pose_info['bbox_scores']).shape, (1, ))
            self.assertEqual(np.array(pose_info['bboxes']).shape, (1, 4))
            self.assertEqual(np.array(pose_info['keypoint_names']).shape, (17, ))
            self.assertEqual(np.array(pose_info['keypoint_scores'][0]).shape, (17, ))
            self.assertEqual(np.array(pose_info['keypoints'][0]).shape, (17, 2))


if __name__ == '__main__':
    unittest.main()

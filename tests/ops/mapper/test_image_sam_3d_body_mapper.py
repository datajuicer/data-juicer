import os
import unittest
import numpy as np
import tempfile
import shutil

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.image_sam_3d_body_mapper import \
    ImageSAM3DBodyMapper
from data_juicer.utils.constant import Fields
from data_juicer.utils.unittest_utils import TEST_TAG, DataJuicerTestCaseBase


def _is_egl_available():
    """Check if EGL is available for offscreen rendering."""
    try:
        from OpenGL.platform import ctypesloader
        ctypesloader.loadLibrary(None, 'EGL')
        return True
    except (ImportError, OSError, TypeError):
        return False


EGL_AVAILABLE = _is_egl_available()


class ImageSAM3DBodyMapperTest(DataJuicerTestCaseBase):
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')
    img_path1 = os.path.join(data_path, 'img7.jpg')
    img_path2 = os.path.join(data_path, 'img8.jpg')

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory().name
        super().setUp()

    def tearDown(self):
        super().tearDown()
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

    def _run_test(self, visualization_dir=None, ray_mode=False, num_proc=1):
        ds_list = [{
            'images': [self.img_path1]
        },  {
            'images': [self.img_path2]
        }]

        op_kwargs = dict(
            visualization_dir=visualization_dir,
            num_proc=num_proc,
        )
        op = ImageSAM3DBodyMapper(**op_kwargs)

        if ray_mode:
            import ray
            from ray.data import ActorPoolStrategy
            ds = ray.data.from_items(ds_list)
            ds = ds.add_column(Fields.meta, lambda df: [{}])
            ds = ds.map(
                        op.__class__,
                        fn_constructor_kwargs=op_kwargs,
                        num_cpus=1,
                        num_gpus=1,
                        compute=ActorPoolStrategy(size=num_proc),
                    )
            res_list = ds.take_all()
        else:
            dataset = Dataset.from_list(ds_list)
            if Fields.meta not in dataset.features:
                dataset = dataset.add_column(name=Fields.meta,
                                            column=[{}] * dataset.num_rows)
            dataset = dataset.map(op.process, num_proc=num_proc, with_rank=True)
            res_list = dataset.to_list()

        if visualization_dir:
            self.assertEqual(sorted(os.listdir(visualization_dir)), ['img7_vis.jpg', 'img8_vis.jpg'])

        for sample in res_list:
            body_outputs = sample[Fields.meta]["sam_3d_body_data"][0]  # only one image data

            if sample['images'][0] == self.img_path1:
                self.assertEqual(len(body_outputs), 1)
            else:
                self.assertEqual(len(body_outputs), 4)
            for data in body_outputs:
                keys = list(data.keys())
                self.assertListEqual(sorted(keys), sorted(['bbox', 'body_pose_params', 'expr_params', 'focal_length', 'global_rot', 
                    'hand_pose_params', 'lhand_bbox', 'mask', 'pred_cam_t', 'pred_global_rots', 'pred_joint_coords', 'pred_keypoints_2d', 
                    'pred_keypoints_3d', 'pred_pose_raw', 'pred_vertices', 'rhand_bbox', 'scale_params', 'shape_params']))
                
                self.assertEqual(np.array(data["bbox"]).shape, (4,))
                self.assertEqual(np.array(data["body_pose_params"]).shape, (133,))
                self.assertEqual(np.array(data["expr_params"]).shape, (72,))
                self.assertTrue(isinstance(data["focal_length"], (float, np.float32)))  # ray returns np.float32
                self.assertEqual(np.array(data["global_rot"]).shape, (3,))
                self.assertEqual(np.array(data["hand_pose_params"]).shape, (108,))
                self.assertEqual(np.array(data["lhand_bbox"]).shape, (4,))
                self.assertEqual(data["mask"], None)
                self.assertEqual(np.array(data["pred_cam_t"]).shape, (3,))
                self.assertEqual(np.array(data["pred_global_rots"]).shape, (127, 3, 3))
                self.assertEqual(np.array(data["pred_joint_coords"]).shape, (127, 3))
                self.assertEqual(np.array(data["pred_keypoints_2d"]).shape, (70, 2))
                self.assertEqual(np.array(data["pred_keypoints_3d"]).shape, (70, 3))
                self.assertEqual(np.array(data["pred_pose_raw"]).shape, (266,))
                self.assertEqual(np.array(data["pred_vertices"]).shape, (18439, 3))
                self.assertEqual(np.array(data["rhand_bbox"]).shape, (4,))
                self.assertEqual(np.array(data["scale_params"]).shape, (28,))
                self.assertEqual(np.array(data["shape_params"]).shape, (45,))

    def test(self):
        self._run_test()
    
    @unittest.skip('not support multi-processing')
    def test_multi_process(self):
        """The source code (sam-3d-body) does not support multi-processing for the HF dataset. 
        You need to modify the `.cuda, to("cuda")` part in the `sam_3d_body/models/meta_arch/sam3d_body.py` file, 
        changing it to, for example, `.to(batch["img"].device)`, to match the `device` in the original model.
        """
        self._run_test(num_proc=2)

    @unittest.skipUnless(EGL_AVAILABLE, 'EGL not available for visualization')
    def test_vis(self):
        self._run_test(visualization_dir=self.tmp_dir, num_proc=1)

    @TEST_TAG('ray')
    @unittest.skipUnless(EGL_AVAILABLE, 'EGL not available for visualization')
    def test_ray(self):
        self._run_test(visualization_dir=self.tmp_dir, ray_mode=True, num_proc=2)

       
if __name__ == '__main__':
    unittest.main()

import os
import shutil
import unittest

import numpy as np

from data_juicer.core.data import NestedDataset as Dataset

from data_juicer.ops.mapper.image_face_blur_mapper import ImageFaceBlurMapper
from data_juicer.utils.mm_utils import load_image
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class ImageFaceBlurMapperTest(DataJuicerTestCaseBase):

    maxDiff = None

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')
    img1_path = os.path.join(data_path, 'cat.jpg')  # no face
    img2_path = os.path.join(data_path, 'lena.jpg')  # [[228, 228, 377, 377]]
    img3_path = os.path.join(data_path,
                             'lena-face.jpg')  # [[29, 29, 178, 178]]

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.chk_path = os.path.join(cls.data_path, cls.__name__)
        shutil.rmtree(cls.chk_path, ignore_errors=True)
        os.makedirs(cls.chk_path)

    def _run_helper(self, op, source_list, np=1):
        dataset = Dataset.from_list(source_list)
        dataset = dataset.map(op.process, num_proc=np)
        res_list = dataset.to_list()
        for source, res in zip(source_list, res_list):
            self.assertEqual(len(source[op.image_key]), len(res[op.image_key]))
            # for manual check
            for path in res[op.image_key]:
                basename = os.path.basename(path)
                dst = f'{self.chk_path}/{op.blur_type}:{op.radius}_np:{np}_{basename}'
                shutil.copy(path, dst)

    def test_gaussian(self):
        ds_list = [{
            'images': [self.img1_path]
        }, {
            'images': [self.img2_path]
        }, {
            'images': [self.img3_path]
        }]
        op = ImageFaceBlurMapper(blur_type='gaussian')
        self._run_helper(op, ds_list)

    def test_gaussian_radius(self):
        ds_list = [{
            'images': [self.img1_path]
        }, {
            'images': [self.img2_path]
        }, {
            'images': [self.img3_path]
        }]
        op = ImageFaceBlurMapper(blur_type='gaussian', radius=10)
        self._run_helper(op, ds_list)

    def test_box(self):
        ds_list = [{
            'images': [self.img2_path]
        }, {
            'images': [self.img3_path]
        }, {
            'images': [self.img1_path]
        }]
        op = ImageFaceBlurMapper(blur_type='box')
        self._run_helper(op, ds_list)

    def test_box_radius(self):
        ds_list = [{
            'images': [self.img2_path]
        }, {
            'images': [self.img3_path]
        }, {
            'images': [self.img1_path]
        }]
        op = ImageFaceBlurMapper(blur_type='box', radius=10)
        self._run_helper(op, ds_list)

    def test_mean(self):
        ds_list = [{
            'images': [self.img2_path]
        }, {
            'images': [self.img3_path]
        }, {
            'images': [self.img1_path]
        }]
        op = ImageFaceBlurMapper(blur_type='mean')
        self._run_helper(op, ds_list)

    def test_gaussian_radius_parallel(self):
        import multiprocess as mp
        mp.set_start_method('forkserver', force=True)
    
        ds_list = [{
            'images': [self.img1_path]
        }, {
            'images': [self.img2_path]
        }, {
            'images': [self.img3_path]
        }]
        op = ImageFaceBlurMapper(blur_type='gaussian', radius=10)
        self._run_helper(op, ds_list, np=3)

    def test_image_bytes(self):
        # the images are only in the bytes field, e.g. after download_file_mapper
        img_bytes = []
        for path in (self.img3_path, self.img1_path):
            with open(path, 'rb') as f:
                img_bytes.append(f.read())
        op = ImageFaceBlurMapper()
        res = op.process_single({
            'images': ['https://example.com/lena-face.jpg', 'https://example.com/cat.jpg'],
            'image_bytes': list(img_bytes),
        })
        face, no_face = (np.array(load_image(b)) for b in res['image_bytes'])
        self.assertEqual(face.shape, np.array(load_image(img_bytes[0])).shape)
        self.assertFalse((face == np.array(load_image(img_bytes[0]))).all())
        np.testing.assert_array_equal(no_face, np.array(load_image(img_bytes[1])))

if __name__ == '__main__':
    unittest.main()

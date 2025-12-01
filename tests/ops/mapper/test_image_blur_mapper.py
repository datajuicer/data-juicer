import os
import unittest
import numpy as np

from PIL import ImageFilter

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.image_blur_mapper import ImageBlurMapper
from data_juicer.utils.mm_utils import load_image
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

class ImageBlurMapperTest(DataJuicerTestCaseBase):

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')
    img1_path = os.path.join(data_path, 'img1.png')
    img2_path = os.path.join(data_path, 'img2.jpg')
    img3_path = os.path.join(data_path, 'img3.jpg')

    def _get_blur_kernel(self, blur_type='gaussian', radius=2):
        if blur_type == 'mean':
            return ImageFilter.BLUR
        elif blur_type == 'box':
            return ImageFilter.BoxBlur(radius)
        else:
            return ImageFilter.GaussianBlur(radius)

    def _run_image_blur_mapper(self, op, source_list, blur_kernel):
        dataset = Dataset.from_list(source_list)
        dataset = dataset.map(op.process)
        res_list = dataset.to_list()
        for source, res in zip(source_list, res_list):
            for s_path, r_path in zip(source[op.image_key], res[op.image_key]):
                s_img = load_image(s_path).convert('RGB').filter(blur_kernel)
                t_path = 'temp4test' + os.path.splitext(s_path)[-1]
                s_img.save(t_path)
                t_img = np.array(load_image(t_path))
                r_img = np.array(load_image(r_path))
                os.remove(t_path)
                np.testing.assert_array_equal(t_img, r_img)

    def test(self):
        ds_list = [{
            'images': [self.img1_path]
        }, {
            'images': [self.img2_path]
        }, {
            'images': [self.img3_path]
        }]
        op = ImageBlurMapper(p=1, blur_type='gaussian', radius=2)
        blur_kernel = self._get_blur_kernel('gaussian', 2)
        self._run_image_blur_mapper(op, ds_list, blur_kernel)

    def test_blur_type(self):
        ds_list = [{
            'images': [self.img2_path]
        }, {
            'images': [self.img3_path]
        }, {
            'images': [self.img1_path]
        }]
        op = ImageBlurMapper(p=1, blur_type='box', radius=2)
        blur_kernel = self._get_blur_kernel('box', 2)
        self._run_image_blur_mapper(op, ds_list, blur_kernel)

    def test_radius(self):
        ds_list = [{
            'images': [self.img3_path]
        }, {
            'images': [self.img2_path]
        }, {
            'images': [self.img1_path]
        }]
        op = ImageBlurMapper(p=1, blur_type='gaussian', radius=5)
        blur_kernel = self._get_blur_kernel('gaussian', 5)
        self._run_image_blur_mapper(op, ds_list, blur_kernel)

    def test_multi_img(self):
        ds_list = [{
            'images': [self.img1_path, self.img2_path, self.img3_path]
        }, {
            'images': [self.img2_path]
        }, {
            'images': [self.img3_path, self.img1_path]
        }]
        op = ImageBlurMapper(p=1, blur_type='gaussian', radius=2)
        blur_kernel = self._get_blur_kernel('gaussian', 2)
        self._run_image_blur_mapper(op, ds_list, blur_kernel)


if __name__ == '__main__':
    unittest.main()

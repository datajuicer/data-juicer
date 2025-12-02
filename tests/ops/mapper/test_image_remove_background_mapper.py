import os
import unittest

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.image_remove_background_mapper import ImageRemoveBackgroundMapper
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase
from data_juicer.utils.constant import Fields


class ImageRemoveBackgroundMapperTest(DataJuicerTestCaseBase):

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')
    img1_path = os.path.join(data_path, 'img1.png')
    img2_path = os.path.join(data_path, 'img2.jpg')
    img3_path = os.path.join(data_path, 'img3.jpg')
    img4_path = os.path.join(data_path, 'img4.png')
    img5_path = os.path.join(data_path, 'img5.jpg')
    img6_path = os.path.join(data_path, 'img6.jpg')

    def _run_mapper(self, op, source_list):
        dataset = Dataset.from_list(source_list)
        dataset = dataset.map(op.process)
        res_list = dataset.to_list()
        for source, res in zip(source_list, res_list):
            for src_path, res_path in zip(source[op.image_key], res[op.image_key]):
                self.assertNotEqual(src_path, res_path)
            self.assertIn(Fields.source_file, res)

    def test_single_image(self):
        ds_list = [{
            'images': [self.img1_path]
        }, {
            'images': [self.img2_path]
        }, {
            'images': [self.img3_path]
        }]
        op = ImageRemoveBackgroundMapper()
        self._run_mapper(op, ds_list)

    def test_multiple_images(self):
        ds_list = [{
            'images': [self.img1_path, self.img4_path]
        }, {
            'images': [self.img2_path, self.img5_path]
        }, {
            'images': [self.img3_path, self.img6_path]
        }]
        op = ImageRemoveBackgroundMapper()
        self._run_mapper(op, ds_list)


if __name__ == '__main__':
    unittest.main()

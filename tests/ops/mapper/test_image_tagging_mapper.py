# flake8: noqa: E501
import os
import unittest

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.image_tagging_mapper import ImageTaggingMapper
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

class ImageTaggingMapperTest(DataJuicerTestCaseBase):
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')
    img1_path = os.path.join(data_path, 'img1.png')
    img2_path = os.path.join(data_path, 'img2.jpg')
    img3_path = os.path.join(data_path, 'img3.jpg')


    def _run_image_tagging_mapper(self,
                                  op,
                                  source_list,
                                  target_list,
                                  num_proc=1):
        dataset = Dataset.from_list(source_list)
        if Fields.meta not in dataset.features:
            dataset = dataset.add_column(name=Fields.meta,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.process, num_proc=num_proc, with_rank=True)
        res_list = dataset.to_list()
        self.assertEqual(res_list, target_list)

    def test(self):
        ds_list = [{
            'images': [self.img1_path]
        }, {
            'images': [self.img2_path]
        }, {
            'images': [self.img3_path]
        }]
        tgt_list = [{
            'images': [self.img1_path],
            Fields.meta: {
                MetaKeys.image_tags: [[
                    'bed', 'bedcover', 'bedroom', 'bedding', 'lamp', 'ceiling',
                    'chair', 'pillar', 'comfort', 'side table', 'floor',
                    'hardwood floor', 'headboard', 'linen', 'mattress',
                    'nightstand', 'picture frame', 'pillow', 'room', 'wall lamp',
                    'stool', 'white', 'window', 'wood floor']]},
        }, {
            'images': [self.img2_path],
            Fields.meta: {
                MetaKeys.image_tags: [[
                    'advertisement', 'back', 'bus', 'car', 'city bus',
                    'city street', 'curb', 'decker bus', 'drive', 'license plate',
                    'road', 'street scene', 'tour bus', 'travel', 'white']]},
        }, {
            'images': [self.img3_path],
            Fields.meta: {
                MetaKeys.image_tags: [[
                    'alley', 'black', 'building', 'catch', 'person', 'pavement',
                    'photo', 'rain', 'road', 'umbrella', 'walk', 'woman']]},
        }]
        op = ImageTaggingMapper()
        self._run_image_tagging_mapper(op, ds_list, tgt_list)

    def test_no_images(self):
        ds_list = [{
            'images': []
        }, {
            'images': [self.img2_path]
        }]
        tgt_list = [{
            'images': [],
            Fields.meta: {
                MetaKeys.image_tags: [[]]},
        }, {
            'images': [self.img2_path],
            Fields.meta: {
                MetaKeys.image_tags: [[
                    'advertisement', 'back', 'bus', 'car', 'city bus',
                    'city street', 'curb', 'decker bus', 'drive', 'license plate',
                    'road', 'street scene', 'tour bus', 'travel', 'white']]},
        }]
        op = ImageTaggingMapper()
        self._run_image_tagging_mapper(op, ds_list, tgt_list)

    def test_specified_tag_field_name(self):
        tag_field_name = 'my_tags'

        ds_list = [{
            'images': [self.img1_path]
        }, {
            'images': [self.img2_path]
        }, {
            'images': [self.img3_path]
        }]
        tgt_list = [{
            'images': [self.img1_path],
            Fields.meta: {
                tag_field_name: [[
                    'bed', 'bedcover', 'bedroom', 'bedding', 'lamp', 'ceiling',
                    'chair', 'pillar', 'comfort', 'side table', 'floor',
                    'hardwood floor', 'headboard', 'linen', 'mattress',
                    'nightstand', 'picture frame', 'pillow', 'room', 'wall lamp',
                    'stool', 'white', 'window', 'wood floor']]},
        }, {
            'images': [self.img2_path],
            Fields.meta: {
                tag_field_name: [[
                    'advertisement', 'back', 'bus', 'car', 'city bus',
                    'city street', 'curb', 'decker bus', 'drive', 'license plate',
                    'road', 'street scene', 'tour bus', 'travel', 'white']]},
        }, {
            'images': [self.img3_path],
            Fields.meta: {
                tag_field_name: [[
                    'alley', 'black', 'building', 'catch', 'person', 'pavement',
                    'photo', 'rain', 'road', 'umbrella', 'walk', 'woman']]},
        }]
        op = ImageTaggingMapper(tag_field_name=tag_field_name)
        self._run_image_tagging_mapper(op, ds_list, tgt_list)

    def test_multi_process(self):
        # WARNING: current parallel tests only work in spawn method
        import multiprocess
        original_method = multiprocess.get_start_method()
        multiprocess.set_start_method('spawn', force=True)
        # WARNING: current parallel tests only work in spawn method
        ds_list = [{
            'images': [self.img1_path]
        }, {
            'images': [self.img2_path]
        }, {
            'images': [self.img3_path]
        }]
        tgt_list = [{
            'images': [self.img1_path],
            Fields.meta: {
                MetaKeys.image_tags: [[
                    'bed', 'bedcover', 'bedroom', 'bedding', 'lamp', 'ceiling',
                    'chair', 'pillar', 'comfort', 'side table', 'floor',
                    'hardwood floor', 'headboard', 'linen', 'mattress',
                    'nightstand', 'picture frame', 'pillow', 'room', 'wall lamp',
                    'stool', 'white', 'window', 'wood floor']]},
        }, {
            'images': [self.img2_path],
            Fields.meta: {
                MetaKeys.image_tags: [[
                    'advertisement', 'back', 'bus', 'car', 'city bus',
                    'city street', 'curb', 'decker bus', 'drive', 'license plate',
                    'road', 'street scene', 'tour bus', 'travel', 'white']]},
        }, {
            'images': [self.img3_path],
            Fields.meta: {
                MetaKeys.image_tags: [[
                    'alley', 'black', 'building', 'catch', 'person', 'pavement',
                    'photo', 'rain', 'road', 'umbrella', 'walk', 'woman']]},
        }]
        op = ImageTaggingMapper()
        self._run_image_tagging_mapper(op,
                                       ds_list,
                                       tgt_list,
                                       num_proc=2)
        # WARNING: current parallel tests only work in spawn method
        multiprocess.set_start_method(original_method, force=True)
        # WARNING: current parallel tests only work in spawn method


if __name__ == '__main__':
    unittest.main()

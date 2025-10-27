# flake8: noqa: E501
import os
import unittest

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase
from data_juicer.ops.mapper.image_tagging_vlm_mapper import ImageTaggingVLMMapper


class ImageTaggingVLMMapperTest(DataJuicerTestCaseBase):
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')
    img1_path = os.path.join(data_path, 'img1.png')
    img2_path = os.path.join(data_path, 'img2.jpg')
    img3_path = os.path.join(data_path, 'img3.jpg')

    def _run_image_tagging_vlm_mapper(self,
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
                   'bed', 'pillow', 'mattress', 'bedroom', 'home', 'comfort', 'sleeping', 'house']]},
        }, {
            'images': [self.img2_path],
            Fields.meta: {
                MetaKeys.image_tags: [[
                    'bus', 'advertisement', 'street scene', 'city life', 'tour bus', 
                    'travel service', 'urban environment', 'local transportation', 'daylight']]},
        }, {
            'images': [self.img3_path],
            Fields.meta: {
                MetaKeys.image_tags: [[
                    'urban', 'alley', 'rainy-day', 'building', 'person', 'umbrella', 'sidewalk', 
                    'black-and-white', '/blackandwhite', 'shoe', 'clothing', 'architecture', 
                    'trees', 'reflection', 'exploration']]},
        }]
        op = ImageTaggingVLMMapper(
            api_or_hf_model="Qwen2.5-VL-7B-Instruct",
            is_api_model=False,
            model_params={"tensor_parallel_size": 1}
        )
        self._run_image_tagging_vlm_mapper(op, ds_list, tgt_list)

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
                    'bus', 'advertisement', 'street scene', 'city life', 'tour bus', 
                    'travel service', 'urban environment', 'local transportation', 'daylight']]},
        }]
        op = ImageTaggingVLMMapper(
            api_or_hf_model="Qwen2.5-VL-7B-Instruct",
            is_api_model=False,
            model_params={"tensor_parallel_size": 1}
        )
        self._run_image_tagging_vlm_mapper(op, ds_list, tgt_list)

    def test_api_model(self):

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
                    'bus', 'advertisement', 'street scene', 'city life', 'tour bus', 
                    'travel service', 'urban environment', 'local transportation', 'daylight']]},
        }]

        op = ImageTaggingVLMMapper(
            api_or_hf_model='qwen2.5-vl-3b-instruct',
            is_api_model=True,
        )
        self._run_image_tagging_vlm_mapper(op, ds_list, tgt_list)

    def test_specify_tag_fieldl(self):
        tag_field_name = 'my_tags'

        ds_list = [{
            'images': []
        }, {
            'images': [self.img2_path]
        }]
        tgt_list = [{
            'images': [],
            Fields.meta: {
                tag_field_name: [[]]},
        }, {
            'images': [self.img2_path],
            Fields.meta: {
                tag_field_name: [[
                    'bus', 'advertisement', 'street scene', 'city life', 'tour bus', 
                    'travel service', 'urban environment', 'local transportation', 'daylight']]},
        }]

        op = ImageTaggingVLMMapper(
            api_or_hf_model="Qwen2.5-VL-7B-Instruct",
            is_api_model=False,
            tag_field_name=tag_field_name,
            model_params={"tensor_parallel_size": 1}
        )
        self._run_image_tagging_vlm_mapper(op, ds_list, tgt_list)


if __name__ == '__main__':
    unittest.main()

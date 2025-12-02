import unittest
import os

from data_juicer.ops.mapper.imgdiff_difference_caption_generator_mapper import Difference_Caption_Generator_Mapper
from data_juicer.core import NestedDataset
from data_juicer.ops.base_op import OP
from data_juicer.ops.load import load_ops
from data_juicer.ops.op_fusion import fuse_operators
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase
from data_juicer.utils.constant import Fields, InterVars, MetaKeys


class Difference_Caption_Generator_MapperTest(DataJuicerTestCaseBase):

    def setUp(self) -> None:
        super().setUp()

        data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')
        img_pair_1 = os.path.join(data_path, 'img_pair_1.jpg')
        img_pair_2 = os.path.join(data_path, 'img_pair_2.jpg')

        self.dataset = NestedDataset.from_list([
            {"image_path1" : img_pair_1, "image_path2" : img_pair_2, Fields.meta: {MetaKeys.bbox_tag: [[430.9378662109375, 549.3763427734375, 283.80419921875, 131.1605224609375], [452.3218688964844, 544.7178955078125, 300.14129638671875, 137.62643432617188]]}},
            {"image_path1" : img_pair_1, "image_path2" : img_pair_2, Fields.meta: {MetaKeys.bbox_tag: [[230.9378662109375, 149.3763427734375, 83.80419921875, 31.1605224609375]]}},
            {"image_path1" : img_pair_1, "image_path2" : img_pair_2, Fields.meta: {MetaKeys.bbox_tag: [[430.9378662109375, 549.3763427734375, 283.80419921875, 131.1605224609375], [452.3218688964844, 544.7178955078125, 300.14129638671875, 137.62643432617188]]}},
        ])


    def test_regular_config(self):

        mllm_mapper_args = {
            "max_new_tokens": 1024,
            "temperature": 0.2,
            "num_beams": 1,
            "hf_model": 'llava-hf/llava-v1.6-vicuna-7b-hf'
        }
        image_text_matching_filter_args = {
            'min_score': 0.4,
            'max_score': 1.0,
            'hf_blip': 'Salesforce/blip-itm-base-coco',
            "num_proc": 1
        }
        text_pair_similarity_filter_args = {
            'min_score': 0,
            'max_score': 0.8,
            'hf_clip': 'openai/clip-vit-base-patch32',
            'text_key_second': "target_text",
            "num_proc": 1
        }

        op = Difference_Caption_Generator_Mapper(
                mllm_mapper_args,
                image_text_matching_filter_args,
                text_pair_similarity_filter_args)

        res = self.dataset.map(op.process, num_proc=1, with_rank=True)
        print(res)
        print(res.to_list())



if __name__ == '__main__':
    unittest.main()
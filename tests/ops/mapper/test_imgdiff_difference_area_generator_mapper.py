import unittest
import os 

from data_juicer.core import NestedDataset
from data_juicer.ops.base_op import OP
from data_juicer.ops.load import load_ops
from data_juicer.ops.op_fusion import fuse_operators
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase
from data_juicer.ops.mapper.imgdiff_difference_area_generator_mapper import Difference_Area_Generator_Mapper


class Difference_Area_Generator_FusedOPTest(DataJuicerTestCaseBase):

    def setUp(self) -> None:
        super().setUp()

        data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')
        img_pair_1 = os.path.join(data_path, 'img_pair_1.jpg')
        img_pair_2 = os.path.join(data_path, 'img_pair_2.jpg')
        noise_img = os.path.join(data_path, 'lena.jpg')

        self.dataset = NestedDataset.from_list([
            {"caption1" : "A plane in the sky flying alone.", "caption2" : "A boat in the ocean sailing alone.", "image_path1" : img_pair_1, "image_path2" : img_pair_2},
            {"caption1" : "A plane in the sky flying alone.", "caption2" : "A boat in the ocean sailing alone.", "image_path1" : img_pair_1, "image_path2" : noise_img},
            {"caption1" : "A plane in the sky flying alone.", "caption2" : "A boat in the ocean sailing alone.", "image_path1" : img_pair_1, "image_path2" : img_pair_2},
        ])


    def test_regular_config(self):

        image_pair_similarity_filter_args = {
            "min_score_1": 0.9,
            "max_score_1": 0.99,
            "min_score_2": 0,
            "max_score_2": 0.85,
            "hf_clip": 'openai/clip-vit-base-patch32',
            "num_proc": 1
        }
        image_segment_mapper_args = {
            'imgsz': 1024,
            'conf': 0.05,
            'iou': 0.5,
            'model_path': 'FastSAM-x.pt',
        }
        image_text_matching_filter_args = {
            'min_score': 0.4,
            'max_score': 1.0,
            'hf_blip': 'Salesforce/blip-itm-base-coco',
            "num_proc": 1
        }

        op = Difference_Area_Generator_Mapper(
                image_pair_similarity_filter_args,
                image_segment_mapper_args,
                image_text_matching_filter_args)

        res = self.dataset.map(op.process, num_proc=1, with_rank=True)
        print(res)
        print(res.to_list())



if __name__ == '__main__':
    unittest.main()
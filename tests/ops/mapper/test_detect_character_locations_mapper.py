import unittest
from data_juicer.core import NestedDataset
from data_juicer.ops.mapper.detect_character_locations_mapper import DetectCharacterLocationsMapper
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase
import os


class DetectCharacterLocationsMapperTest(DataJuicerTestCaseBase):
    def setUp(self) -> None:
        super().setUp()

        data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')
        img1 = os.path.join(data_path, 'img9.jpg')
        img2 = os.path.join(data_path, 'img10.jpg')

        self.dataset = NestedDataset.from_list([
            {"main_character_list": ['beige and white puppy standing with its right paw resting against the white barrier', 'black and white puppy sitting on its hind legs to the right', 'beige puppy sitting on its hind legs to the left', 'light cream colored puppy sitting on its hind legs directly behind the standing puppy'], "images" : [img1]},
            {"main_character_list": ['stuffed dog with blue hard hat', 'stuffed penguin with blue hard hat', 'real cat with blue hard hat'], "images" : [img2]},
        ])

    def test_regular_config(self):
        mllm_mapper_args = {
            "max_new_tokens": 1024,
            "temperature": 0.2,
            "num_beams": 1,
            "hf_model": 'llava-hf/llava-v1.6-vicuna-7b-hf'
        }

        image_text_matching_filter_args = {
            'min_score': 0,
            'max_score': 1.0,
            'hf_blip': 'Salesforce/blip-itm-base-coco',
            "num_proc": 1
        }

        op = DetectCharacterLocationsMapper(
            mllm_mapper_args=mllm_mapper_args,
            image_text_matching_filter_args=image_text_matching_filter_args,
            yoloe_path="yoloe-11l-seg.pt",
            iou_threshold=0.7,
            matching_score_threshold=0.4)

        res = self.dataset.map(op.process, num_proc=1, with_rank=True)
        print(res)
        print(res.to_list())



if __name__ == '__main__':
    unittest.main()
import unittest
from data_juicer.core import NestedDataset
from data_juicer.ops.mapper.detect_main_character_mapper import DetectMainCharacterMapper
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase
import os

class DetectMainCharacterMapperTest(DataJuicerTestCaseBase):
    def setUp(self) -> None:
        super().setUp()

        data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')
        img1 = os.path.join(data_path, 'img9.jpg')
        img2 = os.path.join(data_path, 'img10.jpg')

        self.dataset = NestedDataset.from_list([
            {"text": "An overhead view of four labradoodle puppies, three puppies are sitting and one puppy is standing with its right paw resting against the white barrier at the bottom of the image. The puppies are on a light blue rug placed on a black floor. The puppy standing is beige and white, there is a black and white puppy sitting on its hind legs to the right, and two the left is another beige puppy sitting on its hind legs as well. Directly behind the standing puppy is another light cream colored puppy sitting on its hind legs. The three puppies in the front are looking up, the puppy behind them is looking toward the bottom right corner of the image. There is a blue plush toy in the bottom right corner of the image underneath the black puppy. The rug the puppies are on is not laying completely flat on the ground, its unintentionally folded up in some areas and folded over itself in the top right corner of the image.", "images" : [img1]},
            {"text": "An indoor angled down medium close-up front view of a real sized stuffed dog with white and black colored fur wearing a blue hard hat with a light on it. A couple inches to the right of the dog is a real sized black and white penguin that is also wearing a blue hard hat with a light on it. The dog is sitting, and is facing slightly towards the right while looking to its right with its mouth slightly open, showing its pink tongue. The dog and penguin are placed on a gray and white carpet, and placed against a white drawer that has a large gray cushion on top of it. Behind the gray cushion is a transparent window showing green trees on the outside.", "images" : [img2]},
        ])

        
    def test_regular_config(self):

        mllm_mapper_args = {
            "max_new_tokens": 1024,
            "temperature": 0.2,
            "num_beams": 1,
            "hf_model": 'llava-hf/llava-v1.6-vicuna-7b-hf'
        }

        op = DetectMainCharacterMapper(
            mllm_mapper_args=mllm_mapper_args,
            filter_min_character_num=1)

        res = self.dataset.map(op.process, num_proc=1, with_rank=True)
        print(res)
        print(res.to_list())



if __name__ == '__main__':
    unittest.main()
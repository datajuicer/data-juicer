import os
import unittest

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.video_object_segmenting_mapper import \
    VideoObjectSegmentingMapper
from data_juicer.utils.mm_utils import SpecialTokens
from data_juicer.utils.constant import Fields
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class VideoObjectSegmentingMapperTest(DataJuicerTestCaseBase):
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')
    vid3_path = os.path.join(data_path, 'video3.mp4')
    vid4_path = os.path.join(data_path, 'video4.mp4')

    def test(self):
        ds_list = [{
            'main_character_list': ["glasses", "a woman", "a window"],
            'videos': [self.vid4_path]
        },  {
            'main_character_list': ["a laptop"],
            'videos': [self.vid3_path]
        }]
        
        op = VideoObjectSegmentingMapper(
            sam2_hf_model="facebook/sam2.1-hiera-tiny",
            yoloe_path="yoloe-11l-seg.pt",
            yoloe_conf=0.2,
            torch_dtype="bf16",
            if_binarize=True,
            if_save_visualization=False,
        )
        dataset = Dataset.from_list(ds_list)
        if Fields.meta not in dataset.features:
            dataset = dataset.add_column(name=Fields.meta,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.process, num_proc=1, with_rank=True)
        res_list = dataset.to_list()


if __name__ == '__main__':
    unittest.main()
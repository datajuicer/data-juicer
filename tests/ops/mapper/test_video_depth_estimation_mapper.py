import os
import unittest

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.video_depth_estimation_mapper import \
    VideoDepthEstimationMapper
from data_juicer.utils.mm_utils import SpecialTokens
from data_juicer.utils.constant import Fields
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase
from data_juicer.utils.cache_utils import DATA_JUICER_ASSETS_CACHE

class VideoDepthEstimationMapperTest(DataJuicerTestCaseBase):
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')
    vid3_path = os.path.join(data_path, 'video3.mp4')
    vid4_path = os.path.join(data_path, 'video4.mp4')

    def test(self):
        ds_list = [{
            'videos': [self.vid4_path]
        },  {
            'videos': [self.vid3_path]
        }]

        op = VideoDepthEstimationMapper(
            video_depth_model_path="metric_video_depth_anything_vits.pth",
            point_cloud_dir_for_metric=DATA_JUICER_ASSETS_CACHE,
            max_res=1280,
            torch_dtype="fp16",
            if_save_visualization=True,
            save_visualization_dir=DATA_JUICER_ASSETS_CACHE,
            grayscale=False,
        )

        dataset = Dataset.from_list(ds_list)
        if Fields.meta not in dataset.features:
            dataset = dataset.add_column(name=Fields.meta,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.process, num_proc=1, with_rank=True)
        res_list = dataset.to_list()

if __name__ == '__main__':
    unittest.main()

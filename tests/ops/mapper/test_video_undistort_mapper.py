import os
import unittest
import numpy as np

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.video_undistort_mapper import VideoUndistortMapper
from data_juicer.utils.mm_utils import SpecialTokens
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase
from data_juicer.utils.cache_utils import DATA_JUICER_ASSETS_CACHE

class VideoUndistortMapperTest(DataJuicerTestCaseBase):
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')
    vid3_path = os.path.join(data_path, 'video3.mp4')
    vid12_path = os.path.join(data_path, 'video12.mp4')

    def _run_and_assert(self, output_video_dir, num_proc):
        ds_list = [{
            'videos': [self.vid3_path],
            'intrinsics': [[465.4728460758426, 0, 181.0], [0, 465.4728460758426, 320.0], [0, 0, 1]],
            'distortion_coefficients': None,
            'xi': 0.203957462310791,
            'rotation_matrix': None,
            'intrinsics_new': None
        },  {
            'videos': [self.vid12_path],
            'intrinsics': [[1227.3657989501953, 0, 960.0], [0, 1227.3657989501953, 540.0], [0, 0, 1]],
            'distortion_coefficients': None,
            'xi': 0.33518279,
            'rotation_matrix': None,
            'intrinsics_new': None
        }]

        tgt_key_names = ["new_video_path"]

        op = VideoUndistortMapper(
            output_video_dir=output_video_dir
        )
        dataset = Dataset.from_list(ds_list)
        if Fields.meta not in dataset.features:
            dataset = dataset.add_column(name=Fields.meta,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.process, num_proc=num_proc, with_rank=True)
        res_list = dataset.to_list()

        for sample in res_list:
            self.assertEqual(list(sample[Fields.meta][MetaKeys.video_undistortion_tags].keys()), tgt_key_names)


    def test(self):
        self._run_and_assert(output_video_dir=os.path.join(DATA_JUICER_ASSETS_CACHE, "output_video1"), num_proc=1)

    def test_mul_proc(self):
        self._run_and_assert(output_video_dir=os.path.join(DATA_JUICER_ASSETS_CACHE, "output_video2"), num_proc=2)


if __name__ == '__main__':
    unittest.main()
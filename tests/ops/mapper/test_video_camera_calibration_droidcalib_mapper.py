import os
import unittest
import numpy as np

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.video_camera_calibration_droidcalib_mapper import VideoCameraCalibrationDroidCalibMapper
from data_juicer.utils.constant import Fields, MetaKeys, CameraCalibrationKeys
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase
from data_juicer.utils.cache_utils import DATA_JUICER_ASSETS_CACHE


@unittest.skip(
    'Requires CUDA and DroidCalib compiled extensions. '
    'Run manually with GPU available.'
)
class VideoCameraCalibrationDroidCalibMapperTest(DataJuicerTestCaseBase):
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')
    vid3_path = os.path.join(data_path, 'video3.mp4')
    vid12_path = os.path.join(data_path, 'video12.mp4')

    def _run_and_assert(self, num_proc):
        ds_list = [{
            'videos': [self.vid3_path]
        }, {
            'videos': [self.vid12_path]
        }]

        op = VideoCameraCalibrationDroidCalibMapper(
            tag_field_name=MetaKeys.camera_calibration_droidcalib_tags,
        )
        dataset = Dataset.from_list(ds_list)
        if Fields.meta not in dataset.features:
            dataset = dataset.add_column(name=Fields.meta,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.process, num_proc=num_proc, with_rank=True)
        res_list = dataset.to_list()

        for sample in res_list:
            tag_list = sample[Fields.meta][MetaKeys.camera_calibration_droidcalib_tags]
            self.assertIsInstance(tag_list, list)
            self.assertGreater(len(tag_list), 0)

            for video_result in tag_list:
                self.assertIn(CameraCalibrationKeys.intrinsics, video_result)
                intrinsics = np.array(video_result[CameraCalibrationKeys.intrinsics])
                self.assertEqual(intrinsics.shape, (3, 3))

    def test(self):
        self._run_and_assert(num_proc=1)

    def test_mul_proc(self):
        self._run_and_assert(num_proc=2)


if __name__ == '__main__':
    unittest.main()

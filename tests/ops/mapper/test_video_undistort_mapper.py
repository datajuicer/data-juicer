import os
import unittest
import numpy as np
import shutil
import tempfile

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.video_undistort_mapper import VideoUndistortMapper
from data_juicer.utils.constant import Fields, MetaKeys, CameraCalibrationKeys
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class VideoUndistortMapperTest(DataJuicerTestCaseBase):
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')
    vid3_path = os.path.join(data_path, 'video3.mp4')
    vid12_path = os.path.join(data_path, 'video12.mp4')
    temp_dir = tempfile.TemporaryDirectory().name

    def tearDown(self) -> None:
        super().tearDown()

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _run_and_assert(self, output_video_dir, num_proc):
        ds_list = [{
            'videos': [self.vid3_path],
            Fields.meta: {
                'camera_calibration': [{
                    CameraCalibrationKeys.intrinsics: [[465.4728460758426, 0, 181.0], [0, 465.4728460758426, 320.0], [0, 0, 1]],
                    CameraCalibrationKeys.xi: [0.203957462310791],
                    CameraCalibrationKeys.dist_coeffs: None,
                    CameraCalibrationKeys.rectify_R: None,
                    CameraCalibrationKeys.new_intrinsics: None,
                }],
            }
        },  {
            'videos': [self.vid12_path],
            Fields.meta: {
                'camera_calibration': [{
                    CameraCalibrationKeys.intrinsics: [[1227.3657989501953, 0, 960.0], [0, 1227.3657989501953, 540.0], [0, 0, 1]],
                    CameraCalibrationKeys.xi: [0.33518279],
                    CameraCalibrationKeys.dist_coeffs: None,
                    CameraCalibrationKeys.rectify_R: None,
                    CameraCalibrationKeys.new_intrinsics: None,
                }],
            }
        }]

        op = VideoUndistortMapper(
            output_video_dir=output_video_dir,
            camera_calibration_field='camera_calibration',
        )
        dataset = Dataset.from_list(ds_list)
        dataset = dataset.map(op.process, num_proc=num_proc, with_rank=True)
        res_list = dataset.to_list()

        for sample in res_list:
            tag_list = sample[MetaKeys.undistorted_video]
            self.assertIsInstance(tag_list, list)
            self.assertGreater(len(tag_list), 0)


    def test(self):
        self._run_and_assert(output_video_dir=os.path.join(self.temp_dir, "output_video1"), num_proc=1)

    def test_mul_proc(self):
        self._run_and_assert(output_video_dir=os.path.join(self.temp_dir, "output_video2"), num_proc=2)


if __name__ == '__main__':
    unittest.main()

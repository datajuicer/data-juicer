import os
import unittest
import numpy as np

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.video_camera_calibration_deepcalib_mapper import VideoCameraCalibrationDeepcalibMapper
from data_juicer.ops.mapper.video_extract_frames_mapper import VideoExtractFramesMapper
from data_juicer.utils.mm_utils import SpecialTokens
from data_juicer.utils.constant import Fields, MetaKeys, CameraCalibrationKeys
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase
from data_juicer.utils.cache_utils import DATA_JUICER_ASSETS_CACHE


class VideoCameraCalibrationDeepcalibMapperTest(DataJuicerTestCaseBase):
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')
    vid3_path = os.path.join(data_path, 'video3.mp4')
    vid4_path = os.path.join(data_path, 'video4.mp4')
    vid12_path = os.path.join(data_path, 'video12.mp4')

    def _run_and_assert(self, num_proc):
        ds_list = [{
            'videos': [self.vid3_path]
        },  {
            'videos': [self.vid4_path]
        },  {
            'videos': [self.vid12_path]
        }]

        tgt_list = [{"frame_names_shape": [6],
            "intrinsics_list_shape": [6, 3, 3],
            "xi_list_shape": [6],
            "hfov_list_shape": [6],
            "vfov_list_shape": [6]},
            {"frame_names_shape": [5],
            "intrinsics_list_shape": [5, 3, 3],
            "xi_list_shape": [5],
            "hfov_list_shape": [5],
            "vfov_list_shape": [5]},
            {"frame_names_shape": [1],
            "intrinsics_list_shape": [1, 3, 3],
            "xi_list_shape": [1],
            "hfov_list_shape": [1],
            "vfov_list_shape": [1]}]

        # Step 1: Extract frames from videos
        extract_op = VideoExtractFramesMapper(
            frame_sampling_method='all_keyframes',
            output_format='bytes',
            legacy_split_by_text_token=False,
        )
        dataset = Dataset.from_list(ds_list)
        if Fields.meta not in dataset.features:
            dataset = dataset.add_column(name=Fields.meta,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(extract_op.process, num_proc=num_proc, batched=True, batch_size=1)

        # Step 2: Run camera calibration
        op = VideoCameraCalibrationDeepcalibMapper(
            model_path="weights_10_0.02.h5",
        )
        dataset = dataset.map(op.process, num_proc=num_proc)
        res_list = dataset.to_list()

        for sample, target in zip(res_list, tgt_list):
            self.assertEqual(
                list(np.array(sample[Fields.meta][MetaKeys.camera_calibration_deepcalib_tags][0][CameraCalibrationKeys.intrinsics]).shape),
                target["intrinsics_list_shape"])
            self.assertEqual(
                list(np.array(sample[Fields.meta][MetaKeys.camera_calibration_deepcalib_tags][0][CameraCalibrationKeys.xi]).shape),
                target["xi_list_shape"])
            self.assertEqual(
                list(np.array(sample[Fields.meta][MetaKeys.camera_calibration_deepcalib_tags][0][CameraCalibrationKeys.hfov]).shape),
                target["hfov_list_shape"])
            self.assertEqual(
                list(np.array(sample[Fields.meta][MetaKeys.camera_calibration_deepcalib_tags][0][CameraCalibrationKeys.vfov]).shape),
                target["vfov_list_shape"])

    def test(self):
        self._run_and_assert(num_proc=1)

    def test_mul_proc(self):
        self._run_and_assert(num_proc=2)


if __name__ == '__main__':
    unittest.main()

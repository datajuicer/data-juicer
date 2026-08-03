import os
import unittest
import numpy as np

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper import VideoExtractFramesMapper, VideoCameraCalibrationMogeMapper
from data_juicer.utils.mm_utils import SpecialTokens
from data_juicer.utils.constant import Fields, MetaKeys, CameraCalibrationKeys
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase
from data_juicer.utils.cache_utils import DATA_JUICER_ASSETS_CACHE


class VideoCameraCalibrationMogeMapperTest(DataJuicerTestCaseBase):
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')
    vid3_path = os.path.join(data_path, 'video3.mp4')
    vid4_path = os.path.join(data_path, 'video4.mp4')
    vid12_path = os.path.join(data_path, 'video12.mp4')

    def test_default(self):
        ds_list = [{
            'videos': [self.vid3_path]
        },  {
            'videos': [self.vid4_path]
        },  {
            'videos': [self.vid12_path]
        }]

        num_frames_vid3 = 6
        num_frames_vid4 = 5
        num_frames_vid12 = 1
        tgt_list = [{"frame_names_shape": [num_frames_vid3],
            "intrinsics_list_shape": [num_frames_vid3, 3, 3],
            "hfov_list_shape": [num_frames_vid3],
            "vfov_list_shape": [num_frames_vid3],
            "points_list_shape": [num_frames_vid3, 640, 362, 3],
            "depth_list_shape": [num_frames_vid3, 640, 362],
            "mask_list_shape": [num_frames_vid3, 640, 362]},
            {"frame_names_shape": [num_frames_vid4],
            "intrinsics_list_shape": [num_frames_vid4, 3, 3],
            "hfov_list_shape": [num_frames_vid4],
            "vfov_list_shape": [num_frames_vid4],
            "points_list_shape": [num_frames_vid4, 360, 480, 3],
            "depth_list_shape": [num_frames_vid4, 360, 480],
            "mask_list_shape": [num_frames_vid4, 360, 480]},
            {"frame_names_shape": [num_frames_vid12],
            "intrinsics_list_shape": [num_frames_vid12, 3, 3],
            "hfov_list_shape": [num_frames_vid12],
            "vfov_list_shape": [num_frames_vid12],
            "points_list_shape": [num_frames_vid12, 1080, 1920, 3],
            "depth_list_shape": [num_frames_vid12, 1080, 1920],
            "mask_list_shape": [num_frames_vid12, 1080, 1920]}]


        # Step 1: Extract frames from videos
        extract_op = VideoExtractFramesMapper(
            frame_sampling_method='all_keyframes',
            output_format='bytes',
            legacy_split_by_text_token=False,
            frame_field=MetaKeys.video_frames,
        )
        dataset = Dataset.from_list(ds_list)
        if Fields.meta not in dataset.features:
            dataset = dataset.add_column(name=Fields.meta,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(extract_op.process, batched=True, batch_size=1)

        # Step 2: Run camera calibration
        op = VideoCameraCalibrationMogeMapper(
            model_path="Ruicheng/moge-2-vitl",
            frame_field=MetaKeys.video_frames,
            tag_field_name=MetaKeys.camera_calibration_moge_tags,
        )

        dataset = dataset.map(op.process)
        res_list = dataset.to_list()

        for sample, target in zip(res_list, tgt_list):
            self.assertEqual(
                list(np.array(sample[Fields.meta][MetaKeys.camera_calibration_moge_tags][0][CameraCalibrationKeys.intrinsics]).shape),
                target["intrinsics_list_shape"])
            self.assertEqual(
                list(np.array(sample[Fields.meta][MetaKeys.camera_calibration_moge_tags][0][CameraCalibrationKeys.hfov]).shape),
                target["hfov_list_shape"])
            self.assertEqual(
                list(np.array(sample[Fields.meta][MetaKeys.camera_calibration_moge_tags][0][CameraCalibrationKeys.vfov]).shape),
                target["vfov_list_shape"])
            self.assertEqual(
                list(np.array(sample[Fields.meta][MetaKeys.camera_calibration_moge_tags][0][CameraCalibrationKeys.points]).shape),
                target["points_list_shape"])
            self.assertEqual(
                list(np.array(sample[Fields.meta][MetaKeys.camera_calibration_moge_tags][0][CameraCalibrationKeys.depth]).shape),
                target["depth_list_shape"])
            self.assertEqual(
                list(np.array(sample[Fields.meta][MetaKeys.camera_calibration_moge_tags][0][CameraCalibrationKeys.mask]).shape),
                target["mask_list_shape"])


if __name__ == '__main__':
    unittest.main()
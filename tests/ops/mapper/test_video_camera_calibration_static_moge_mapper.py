import os
import unittest
import numpy as np

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.video_camera_calibration_static_moge_mapper import VideoCameraCalibrationStaticMogeMapper
from data_juicer.utils.mm_utils import SpecialTokens
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase
from data_juicer.utils.cache_utils import DATA_JUICER_ASSETS_CACHE


class VideoCameraCalibrationStaticMogeMapperTest(DataJuicerTestCaseBase):
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

        tgt_list = [{"frame_names_shape": [49],
            "intrinsics_list_shape": [49, 3, 3],
            "hfov_list_shape": [49],
            "vfov_list_shape": [49],
            "points_list_shape": [49, 640, 362, 3],
            "depth_list_shape": [49, 640, 362],
            "mask_list_shape": [49, 640, 362]},
            {"frame_names_shape": [22],
            "intrinsics_list_shape": [22, 3, 3],
            "hfov_list_shape": [22],
            "vfov_list_shape": [22],
            "points_list_shape": [22, 360, 480, 3],
            "depth_list_shape": [22, 360, 480],
            "mask_list_shape": [22, 360, 480]},
            {"frame_names_shape": [3],
            "intrinsics_list_shape": [3, 3, 3],
            "hfov_list_shape": [3],
            "vfov_list_shape": [3],
            "points_list_shape": [3, 1080, 1920, 3],
            "depth_list_shape": [3, 1080, 1920],
            "mask_list_shape": [3, 1080, 1920]}]

        op = VideoCameraCalibrationStaticMogeMapper(
            model_path="Ruicheng/moge-2-vitl",
            frame_num=1,
            duration=1,
            frame_dir=DATA_JUICER_ASSETS_CACHE,
            if_output_info=True,
            output_info_dir=DATA_JUICER_ASSETS_CACHE,
            if_output_points_info=True,
            if_output_depth_info=True,
            if_output_mask_info=True,
        )

        dataset = Dataset.from_list(ds_list)
        if Fields.meta not in dataset.features:
            dataset = dataset.add_column(name=Fields.meta,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.process, num_proc=num_proc, with_rank=True)
        res_list = dataset.to_list()


        for sample, target in zip(res_list, tgt_list):
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.static_camera_calibration_moge_tags]["frame_names"]).shape), target["frame_names_shape"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.static_camera_calibration_moge_tags]["intrinsics_list"]).shape), target["intrinsics_list_shape"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.static_camera_calibration_moge_tags]["hfov_list"]).shape), target["hfov_list_shape"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.static_camera_calibration_moge_tags]["vfov_list"]).shape), target["vfov_list_shape"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.static_camera_calibration_moge_tags]["points_list"]).shape), target["points_list_shape"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.static_camera_calibration_moge_tags]["depth_list"]).shape), target["depth_list_shape"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.static_camera_calibration_moge_tags]["mask_list"]).shape), target["mask_list_shape"])


    def test(self):
        self._run_and_assert(num_proc=1)

    def test_mul_proc(self):
        self._run_and_assert(num_proc=2)


if __name__ == '__main__':
    unittest.main()
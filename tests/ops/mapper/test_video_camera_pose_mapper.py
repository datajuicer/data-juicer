import os
import unittest
import numpy as np

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.video_camera_pose_mapper import VideoCameraPoseMapper
from data_juicer.utils.mm_utils import SpecialTokens
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase
from data_juicer.utils.cache_utils import DATA_JUICER_ASSETS_CACHE



class VideoCameraPoseMapperTest(DataJuicerTestCaseBase):
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')
    vid3_path = os.path.join(data_path, 'video3.mp4')
    vid11_path = os.path.join(data_path, 'video11.mp4')
    vid12_path = os.path.join(data_path, 'video12.mp4')


    def _run_and_assert(self, num_proc):
        ds_list = [{
            'videos': [self.vid3_path]
        },  {
            'videos': [self.vid11_path]
        },  {
            'videos': [self.vid12_path]
        }]

        tgt_list = [{"images_shape": [49, 584, 328, 3],
            "depths_shape": [49, 584, 328],
            "intrinsic_shape": [3, 3],
            "cam_c2w_shape": [49, 4, 4]},
            {"images_shape": [11, 328, 584, 3],
            "depths_shape": [11, 328, 584],
            "intrinsic_shape": [3, 3],
            "cam_c2w_shape": [11, 4, 4]},
            {"images_shape": [3, 328, 584, 3],
            "depths_shape": [3, 328, 584],
            "intrinsic_shape": [3, 3],
            "cam_c2w_shape": [3, 4, 4]}]

        op = VideoCameraPoseMapper(
            moge_model_path="Ruicheng/moge-2-vitl",
            frame_num=1,
            duration=1,
            frame_dir=DATA_JUICER_ASSETS_CACHE,
            if_output_moge_info=False,
            moge_output_info_dir=DATA_JUICER_ASSETS_CACHE,
            if_save_info=True,
            output_info_dir=DATA_JUICER_ASSETS_CACHE,
        )

        dataset = Dataset.from_list(ds_list)
        if Fields.meta not in dataset.features:
            dataset = dataset.add_column(name=Fields.meta,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.process, num_proc=num_proc, with_rank=True)
        res_list = dataset.to_list()

        for sample, target in zip(res_list, tgt_list):
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.video_camera_pose_tags]["images"]).shape), target["images_shape"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.video_camera_pose_tags]["depths"]).shape), target["depths_shape"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.video_camera_pose_tags]["intrinsic"]).shape), target["intrinsic_shape"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.video_camera_pose_tags]["cam_c2w"]).shape), target["cam_c2w_shape"])


    def test(self):
        self._run_and_assert(num_proc=1)

    def test_mul_proc(self):
        self._run_and_assert(num_proc=2)


if __name__ == '__main__':
    unittest.main()
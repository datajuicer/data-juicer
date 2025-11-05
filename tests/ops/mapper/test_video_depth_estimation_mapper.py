import os
import unittest
import numpy as np

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.video_depth_estimation_mapper import \
    VideoDepthEstimationMapper
from data_juicer.utils.mm_utils import SpecialTokens
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase
from data_juicer.utils.cache_utils import DATA_JUICER_ASSETS_CACHE

class VideoDepthEstimationMapperTest(DataJuicerTestCaseBase):
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')
    vid3_path = os.path.join(data_path, 'video3.mp4')
    vid4_path = os.path.join(data_path, 'video4.mp4')

    tgt_list = [{
        "depth_data": [673, 360, 480],
        "fps": 30.0
    }, {
        "depth_data": [1190, 640, 362],
        "fps": 24.0
    }]

    def test(self):
        ds_list = [{
            'videos': [self.vid4_path]
        },  {
            'videos': [self.vid3_path]
        }]

        op = VideoDepthEstimationMapper(
            video_depth_model_path="video_depth_anything_vits.pth",
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

        for sample, target in zip(res_list, self.tgt_list):
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.video_depth_tags]["depth_data"]).shape), target["depth_data"])
            self.assertEqual(sample[Fields.meta][MetaKeys.video_depth_tags]["fps"], target["fps"])

    
    def test_metric(self):
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

        for sample, target in zip(res_list, self.tgt_list):
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.video_depth_tags]["depth_data"]).shape), target["depth_data"])
            self.assertEqual(sample[Fields.meta][MetaKeys.video_depth_tags]["fps"], target["fps"])


    def test_mul_proc(self):
        ds_list = [{
            'videos': [self.vid4_path]
        },  {
            'videos': [self.vid3_path]
        }]

        op = VideoDepthEstimationMapper(
            video_depth_model_path="video_depth_anything_vits.pth",
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
        dataset = dataset.map(op.process, num_proc=2, with_rank=True)
        res_list = dataset.to_list()

        for sample, target in zip(res_list, self.tgt_list):
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.video_depth_tags]["depth_data"]).shape), target["depth_data"])
            self.assertEqual(sample[Fields.meta][MetaKeys.video_depth_tags]["fps"], target["fps"])


    def test_metric_mul_proc(self):
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
        dataset = dataset.map(op.process, num_proc=2, with_rank=True)
        res_list = dataset.to_list()

        for sample, target in zip(res_list, self.tgt_list):
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.video_depth_tags]["depth_data"]).shape), target["depth_data"])
            self.assertEqual(sample[Fields.meta][MetaKeys.video_depth_tags]["fps"], target["fps"])


if __name__ == '__main__':
    unittest.main()

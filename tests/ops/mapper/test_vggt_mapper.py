import os
import unittest

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.vggt_mapper import VggtMapper
from data_juicer.utils.mm_utils import SpecialTokens
from data_juicer.utils.constant import Fields
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase
from data_juicer.utils.cache_utils import DATA_JUICER_ASSETS_CACHE


class VggtMapperTest(DataJuicerTestCaseBase):
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')
    vid11_path = os.path.join(data_path, 'video11.mp4')
    vid10_path = os.path.join(data_path, 'video10.mp4')

    def test(self):
        ds_list = [{
            'query_points': [[320.0, 200.0], [500.72, 100.94]],
            'videos': [self.vid11_path]
        },  {
            'query_points': [[50.72, 100.94]],
            'videos': [self.vid10_path]
        }]
        
        op = VggtMapper(
            vggt_model_path="facebook/VGGT-1B",
            frame_num=2,
            duration=2,
            frame_dir=DATA_JUICER_ASSETS_CACHE,
            if_output_camera_parameters=True,
            if_output_depth_maps=True,
            if_output_point_maps_from_projection=True,
            if_output_point_maps_from_unprojection=True,
            if_output_point_tracks=True
        )
        dataset = Dataset.from_list(ds_list)
        if Fields.meta not in dataset.features:
            dataset = dataset.add_column(name=Fields.meta,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.process, num_proc=1, with_rank=True)
        res_list = dataset.to_list()


    def test_mul_proc(self):
        ds_list = [{
            'query_points': [[320.0, 200.0], [500.72, 100.94]],
            'videos': [self.vid11_path]
        },  {
            'query_points': [[50.72, 100.94]],
            'videos': [self.vid10_path]
        }]
        
        op = VggtMapper(
            vggt_model_path="facebook/VGGT-1B",
            frame_num=2,
            duration=2,
            frame_dir=DATA_JUICER_ASSETS_CACHE,
            if_output_camera_parameters=True,
            if_output_depth_maps=True,
            if_output_point_maps_from_projection=True,
            if_output_point_maps_from_unprojection=True,
            if_output_point_tracks=True
        )
        dataset = Dataset.from_list(ds_list)
        if Fields.meta not in dataset.features:
            dataset = dataset.add_column(name=Fields.meta,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.process, num_proc=2, with_rank=True)
        res_list = dataset.to_list()


    def test_point_maps_from_unprojection(self):
        ds_list = [{
            'query_points': [],
            'videos': [self.vid11_path]
        },  {
            'query_points': [],
            'videos': [self.vid10_path]
        }]
        
        op = VggtMapper(
            vggt_model_path="facebook/VGGT-1B",
            frame_num=2,
            duration=2,
            frame_dir=DATA_JUICER_ASSETS_CACHE,
            if_output_camera_parameters=False,
            if_output_depth_maps=False,
            if_output_point_maps_from_projection=False,
            if_output_point_maps_from_unprojection=True,
            if_output_point_tracks=False
        )
        dataset = Dataset.from_list(ds_list)
        if Fields.meta not in dataset.features:
            dataset = dataset.add_column(name=Fields.meta,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.process, num_proc=1, with_rank=True)
        res_list = dataset.to_list()


if __name__ == '__main__':
    unittest.main()
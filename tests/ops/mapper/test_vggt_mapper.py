import os
import unittest
import numpy as np

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.vggt_mapper import VggtMapper
from data_juicer.utils.mm_utils import SpecialTokens
from data_juicer.utils.constant import Fields, MetaKeys
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

        tgt_list = [{"camera_parameters_extrinsic": [1, 10, 3, 4],
            "camera_parameters_intrinsic": [1, 10, 3, 3],
            "depth_maps_depth_maps": [1, 10, 294, 518, 1],
            "depth_maps_depth_conf": [1, 10, 294, 518],
            "point_maps_from_projection_point_map": [1, 10, 294, 518, 3],
            "point_maps_from_projection_point_conf": [1, 10, 294, 518],
            "point_maps_from_unprojection_point_maps_from_unprojection": [10, 294, 518, 3],
            "point_tracks_track_list": [1, 10, 2, 2],
            "point_tracks_vis_score": [1, 10, 2],
            "point_tracks_conf_score": [1, 10, 2]},
            {"camera_parameters_extrinsic": [1, 18, 3, 4],
            "camera_parameters_intrinsic": [1, 18, 3, 3],
            "depth_maps_depth_maps": [1, 18, 392, 518, 1],
            "depth_maps_depth_conf": [1, 18, 392, 518],
            "point_maps_from_projection_point_map": [1, 18, 392, 518, 3],
            "point_maps_from_projection_point_conf": [1, 18, 392, 518],
            "point_maps_from_unprojection_point_maps_from_unprojection": [18, 392, 518, 3],
            "point_tracks_track_list": [1, 18, 1, 2],
            "point_tracks_vis_score": [1, 18, 1],
            "point_tracks_conf_score": [1, 18, 1]}]

        for sample, target in zip(res_list, tgt_list):
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.vggt_tags]["camera_parameters"]["extrinsic"]).shape), target["camera_parameters_extrinsic"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.vggt_tags]["camera_parameters"]["intrinsic"]).shape), target["camera_parameters_intrinsic"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.vggt_tags]["depth_maps"]["depth_map"]).shape), target["depth_maps_depth_maps"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.vggt_tags]["depth_maps"]["depth_conf"]).shape), target["depth_maps_depth_conf"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.vggt_tags]["point_maps_from_projection"]["point_map"]).shape), target["point_maps_from_projection_point_map"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.vggt_tags]["point_maps_from_projection"]["point_conf"]).shape), target["point_maps_from_projection_point_conf"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.vggt_tags]["point_maps_from_unprojection"]["point_maps_from_unprojection"]).shape), target["point_maps_from_unprojection_point_maps_from_unprojection"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.vggt_tags]["point_tracks"]["track_list"][3]).shape), target["point_tracks_track_list"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.vggt_tags]["point_tracks"]["vis_score"]).shape), target["point_tracks_vis_score"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.vggt_tags]["point_tracks"]["conf_score"]).shape), target["point_tracks_conf_score"])


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

        tgt_list = [{"camera_parameters_extrinsic": [1, 10, 3, 4],
            "camera_parameters_intrinsic": [1, 10, 3, 3],
            "depth_maps_depth_maps": [1, 10, 294, 518, 1],
            "depth_maps_depth_conf": [1, 10, 294, 518],
            "point_maps_from_projection_point_map": [1, 10, 294, 518, 3],
            "point_maps_from_projection_point_conf": [1, 10, 294, 518],
            "point_maps_from_unprojection_point_maps_from_unprojection": [10, 294, 518, 3],
            "point_tracks_track_list": [1, 10, 2, 2],
            "point_tracks_vis_score": [1, 10, 2],
            "point_tracks_conf_score": [1, 10, 2]},
            {"camera_parameters_extrinsic": [1, 18, 3, 4],
            "camera_parameters_intrinsic": [1, 18, 3, 3],
            "depth_maps_depth_maps": [1, 18, 392, 518, 1],
            "depth_maps_depth_conf": [1, 18, 392, 518],
            "point_maps_from_projection_point_map": [1, 18, 392, 518, 3],
            "point_maps_from_projection_point_conf": [1, 18, 392, 518],
            "point_maps_from_unprojection_point_maps_from_unprojection": [18, 392, 518, 3],
            "point_tracks_track_list": [1, 18, 1, 2],
            "point_tracks_vis_score": [1, 18, 1],
            "point_tracks_conf_score": [1, 18, 1]}]

        for sample, target in zip(res_list, tgt_list):
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.vggt_tags]["camera_parameters"]["extrinsic"]).shape), target["camera_parameters_extrinsic"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.vggt_tags]["camera_parameters"]["intrinsic"]).shape), target["camera_parameters_intrinsic"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.vggt_tags]["depth_maps"]["depth_map"]).shape), target["depth_maps_depth_maps"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.vggt_tags]["depth_maps"]["depth_conf"]).shape), target["depth_maps_depth_conf"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.vggt_tags]["point_maps_from_projection"]["point_map"]).shape), target["point_maps_from_projection_point_map"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.vggt_tags]["point_maps_from_projection"]["point_conf"]).shape), target["point_maps_from_projection_point_conf"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.vggt_tags]["point_maps_from_unprojection"]["point_maps_from_unprojection"]).shape), target["point_maps_from_unprojection_point_maps_from_unprojection"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.vggt_tags]["point_tracks"]["track_list"][3]).shape), target["point_tracks_track_list"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.vggt_tags]["point_tracks"]["vis_score"]).shape), target["point_tracks_vis_score"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.vggt_tags]["point_tracks"]["conf_score"]).shape), target["point_tracks_conf_score"])


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

        tgt_list = [{"camera_parameters_extrinsic": [1, 10, 3, 4],
            "camera_parameters_intrinsic": [1, 10, 3, 3],
            "depth_maps_depth_maps": [1, 10, 294, 518, 1],
            "depth_maps_depth_conf": [1, 10, 294, 518],
            "point_maps_from_unprojection_point_maps_from_unprojection": [10, 294, 518, 3]},
            {"camera_parameters_extrinsic": [1, 18, 3, 4],
            "camera_parameters_intrinsic": [1, 18, 3, 3],
            "depth_maps_depth_maps": [1, 18, 392, 518, 1],
            "depth_maps_depth_conf": [1, 18, 392, 518],
            "point_maps_from_unprojection_point_maps_from_unprojection": [18, 392, 518, 3]}]

        for sample, target in zip(res_list, tgt_list):
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.vggt_tags]["camera_parameters"]["extrinsic"]).shape), target["camera_parameters_extrinsic"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.vggt_tags]["camera_parameters"]["intrinsic"]).shape), target["camera_parameters_intrinsic"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.vggt_tags]["depth_maps"]["depth_map"]).shape), target["depth_maps_depth_maps"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.vggt_tags]["depth_maps"]["depth_conf"]).shape), target["depth_maps_depth_conf"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.vggt_tags]["point_maps_from_unprojection"]["point_maps_from_unprojection"]).shape), target["point_maps_from_unprojection_point_maps_from_unprojection"])


if __name__ == '__main__':
    unittest.main()
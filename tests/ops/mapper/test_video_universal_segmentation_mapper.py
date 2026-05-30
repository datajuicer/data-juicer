import os
import unittest
import numpy as np
import tempfile
import shutil

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.video_universal_segmentation_mapper import \
    VideoUniversalSegmentationMapper
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class VideoUniversalSegmentationMapperTest(DataJuicerTestCaseBase):
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')
    vid10_path = os.path.join(data_path, 'video10.mp4')
    vid3_path = os.path.join(data_path, 'video3.mp4')
    vid10_frames_dir = os.path.join(data_path, 'video10_frames')
    vid3_frames_dir = os.path.join(data_path, 'video3_frames')
    vid10_frames_path = []
    vid3_frames_path = []
    for x in os.listdir(vid10_frames_dir):
        vid10_frames_path.append(os.path.join(vid10_frames_dir, x))
    for x in os.listdir(vid3_frames_dir):
        vid3_frames_path.append(os.path.join(vid3_frames_dir, x))


    tgt_list = [{
        "frame_length": 49,
        "semantic_segmentation_map_shape": [49, 640, 362],
        "instance_segmentation_map_shape": [49, 640, 362],
        "instance_segmentation_info_length": 49,
        "instance_segmentation_info_keys": sorted(['id', 'label_id', 'was_fused', 'score']),
        "panoptic_segmentation_map_shape": [49, 640, 362],
        "panoptic_segmentation_info_length": 49,
        "panoptic_segmentation_info_keys": sorted(['id', 'label_id', 'was_fused', 'score']),
    }, {
        "frame_length": 19,
        "semantic_segmentation_map_shape": [19, 756, 1008],
        "instance_segmentation_map_shape": [19, 756, 1008],
        "instance_segmentation_info_length": 19,
        "instance_segmentation_info_keys": sorted(['id', 'label_id', 'was_fused', 'score']),
        "panoptic_segmentation_map_shape": [19, 756, 1008],
        "panoptic_segmentation_info_length": 19,
        "panoptic_segmentation_info_keys": sorted(['id', 'label_id', 'was_fused', 'score']),
    }]


    tgt_list_from_extracted_frames = [{
        "frame_length": 16,
        "semantic_segmentation_map_shape": [16, 640, 362],
        "instance_segmentation_map_shape": [16, 640, 362],
        "instance_segmentation_info_length": 16,
        "instance_segmentation_info_keys": sorted(['id', 'label_id', 'was_fused', 'score']),
        "panoptic_segmentation_map_shape": [16, 640, 362],
        "panoptic_segmentation_info_length": 16,
        "panoptic_segmentation_info_keys": sorted(['id', 'label_id', 'was_fused', 'score']),
    }, {
        "frame_length": 19,
        "semantic_segmentation_map_shape": [19, 756, 1008],
        "instance_segmentation_map_shape": [19, 756, 1008],
        "instance_segmentation_info_length": 19,
        "instance_segmentation_info_keys": sorted(['id', 'label_id', 'was_fused', 'score']),
        "panoptic_segmentation_map_shape": [19, 756, 1008],
        "panoptic_segmentation_info_length": 19,
        "panoptic_segmentation_info_keys": sorted(['id', 'label_id', 'was_fused', 'score']),
    }]

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory().name
        super().setUp()

    def tearDown(self):
        super().tearDown()
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)


    def test(self):
        ds_list = [{
            'videos': [self.vid3_path]
        },  {
            'videos': [self.vid10_path]
        }]

        op = VideoUniversalSegmentationMapper(
            model_path="shi-labs/oneformer_ade20k_swin_large",
            if_output_semantic_segmentation=True,
            if_output_instance_segmentation=True,
            if_output_panoptic_segmentation=True,
            if_save_visualization=True,
            save_visualization_dir=os.path.join(self.tmp_dir, "universal_segmentation_vis1"),
            frame_num=1,
            duration=1,
            frame_dir=os.path.join(self.tmp_dir, "universal_segmentation_test")
        )

        dataset = Dataset.from_list(ds_list)
        if Fields.meta not in dataset.features:
            dataset = dataset.add_column(name=Fields.meta,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.process, num_proc=2, with_rank=True)
        res_list = dataset.to_list()

        for sample, target in zip(res_list, self.tgt_list):

            self.assertEqual(len(sample[Fields.meta][MetaKeys.video_universal_segmentation_tags]["semantic_segmentation_map"]), target["frame_length"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.video_universal_segmentation_tags]["semantic_segmentation_map"]).shape), target["semantic_segmentation_map_shape"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.video_universal_segmentation_tags]["instance_segmentation_map"]).shape), target["instance_segmentation_map_shape"])
            self.assertEqual(len(sample[Fields.meta][MetaKeys.video_universal_segmentation_tags]["instance_segmentation_info"]), target["instance_segmentation_info_length"])
            self.assertEqual(sorted(list(sample[Fields.meta][MetaKeys.video_universal_segmentation_tags]["instance_segmentation_info"][0][0].keys())), target["instance_segmentation_info_keys"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.video_universal_segmentation_tags]["panoptic_segmentation_map"]).shape), target["panoptic_segmentation_map_shape"])
            self.assertEqual(len(sample[Fields.meta][MetaKeys.video_universal_segmentation_tags]["panoptic_segmentation_info"]), target["panoptic_segmentation_info_length"])
            self.assertEqual(sorted(list(sample[Fields.meta][MetaKeys.video_universal_segmentation_tags]["panoptic_segmentation_info"][0][0].keys())), target["panoptic_segmentation_info_keys"])
           


    def test_from_extracted_frames(self):

        ds_list = [{
            MetaKeys.video_frames: self.vid3_frames_path,
        },  {
            MetaKeys.video_frames: self.vid10_frames_path,
        }]

        op = VideoUniversalSegmentationMapper(
            model_path="shi-labs/oneformer_ade20k_swin_large",
            if_output_semantic_segmentation=True,
            if_output_instance_segmentation=True,
            if_output_panoptic_segmentation=True,
            if_save_visualization=True,
            save_visualization_dir=os.path.join(self.tmp_dir, "universal_segmentation_vis2"),
        )

        dataset = Dataset.from_list(ds_list)
        if Fields.meta not in dataset.features:
            dataset = dataset.add_column(name=Fields.meta,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.process, num_proc=1, with_rank=True)
        res_list = dataset.to_list()

        for sample, target in zip(res_list, self.tgt_list_from_extracted_frames):

            self.assertEqual(len(sample[Fields.meta][MetaKeys.video_universal_segmentation_tags]["semantic_segmentation_map"]), target["frame_length"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.video_universal_segmentation_tags]["semantic_segmentation_map"]).shape), target["semantic_segmentation_map_shape"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.video_universal_segmentation_tags]["instance_segmentation_map"]).shape), target["instance_segmentation_map_shape"])
            self.assertEqual(len(sample[Fields.meta][MetaKeys.video_universal_segmentation_tags]["instance_segmentation_info"]), target["instance_segmentation_info_length"])
            self.assertEqual(sorted(list(sample[Fields.meta][MetaKeys.video_universal_segmentation_tags]["instance_segmentation_info"][0][0].keys())), target["instance_segmentation_info_keys"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.video_universal_segmentation_tags]["panoptic_segmentation_map"]).shape), target["panoptic_segmentation_map_shape"])
            self.assertEqual(len(sample[Fields.meta][MetaKeys.video_universal_segmentation_tags]["panoptic_segmentation_info"]), target["panoptic_segmentation_info_length"])
            self.assertEqual(sorted(list(sample[Fields.meta][MetaKeys.video_universal_segmentation_tags]["panoptic_segmentation_info"][0][0].keys())), target["panoptic_segmentation_info_keys"])
           

if __name__ == '__main__':
    unittest.main()
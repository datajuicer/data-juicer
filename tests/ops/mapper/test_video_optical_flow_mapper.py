import os
import unittest
import numpy as np
import tempfile
import shutil

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.video_optical_flow_mapper import \
    VideoOpticalFlowMapper
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class VideoOpticalFlowMapperTest(DataJuicerTestCaseBase):
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')
    vid10_path = os.path.join(data_path, 'video10.mp4')
    vid11_path = os.path.join(data_path, 'video11.mp4')
    vid10_frames_dir = os.path.join(data_path, 'video10_frames')
    vid11_frames_dir = os.path.join(data_path, 'video11_frames')
    vid10_frames_path = []
    vid11_frames_path = []
    for x in os.listdir(vid10_frames_dir):
        vid10_frames_path.append(os.path.join(vid10_frames_dir, x))
    for x in os.listdir(vid11_frames_dir):
        vid11_frames_path.append(os.path.join(vid11_frames_dir, x))


    tgt_list = [{
        "length": 18,
        "pred_norm_shape": [18, 2, 520, 960],
    }, {
        "length": 10,
        "pred_norm_shape": [10, 2, 520, 960],
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
            'videos': [self.vid10_path]
        },  {
            'videos': [self.vid11_path]
        }]

        op = VideoOpticalFlowMapper(
            if_save_visualization=True,
            save_visualization_dir=os.path.join(self.tmp_dir, "optical_flow_vis1"),
            frame_num=1,
            duration=1,
            frame_dir=os.path.join(self.tmp_dir, "optical_flow_test")
        )

        dataset = Dataset.from_list(ds_list)
        if Fields.meta not in dataset.features:
            dataset = dataset.add_column(name=Fields.meta,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.process, num_proc=2, with_rank=True)
        res_list = dataset.to_list()

        for sample, target in zip(res_list, self.tgt_list):
            
            self.assertEqual(len(sample[Fields.meta][MetaKeys.video_optical_flow_tags]["pred_flow"]), target["length"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.video_optical_flow_tags]["pred_flow"]).shape), target["pred_norm_shape"])
            

    def test_from_extracted_frames(self):

        ds_list = [{
            MetaKeys.video_frames: self.vid10_frames_path,
        },  {
            MetaKeys.video_frames: self.vid11_frames_path,
        }]

        op = VideoOpticalFlowMapper(
            if_save_visualization=True,
            save_visualization_dir=os.path.join(self.tmp_dir, "optical_flow_vis2"),
        )

        dataset = Dataset.from_list(ds_list)
        if Fields.meta not in dataset.features:
            dataset = dataset.add_column(name=Fields.meta,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.process, num_proc=1, with_rank=True)
        res_list = dataset.to_list()

        for sample, target in zip(res_list, self.tgt_list):

            self.assertEqual(len(sample[Fields.meta][MetaKeys.video_optical_flow_tags]["pred_flow"]), target["length"])
            self.assertEqual(list(np.array(sample[Fields.meta][MetaKeys.video_optical_flow_tags]["pred_flow"]).shape), target["pred_norm_shape"])
            

if __name__ == '__main__':
    unittest.main()
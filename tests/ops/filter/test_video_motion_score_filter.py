import os
import unittest
import numpy as np

from datasets import Dataset

from data_juicer.ops.filter.video_motion_score_filter import \
    VideoMotionScoreFilter
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class VideoMotionScoreFilterTest(DataJuicerTestCaseBase):

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')
    vid1_path = os.path.join(data_path, 'video1.mp4')  # 1.869317
    vid2_path = os.path.join(data_path, 'video2.mp4')  # 3.52111
    vid3_path = os.path.join(data_path, 'video3.mp4')  # 1.1731424

    def _run_helper(self, op, source_list, target_list, np=1):
        dataset = Dataset.from_list(source_list)
        if Fields.stats not in dataset.features:
            dataset = dataset.add_column(name=Fields.stats,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.compute_stats, num_proc=np)
        dataset = dataset.filter(op.process, num_proc=np)
        dataset = dataset.select_columns(column_names=[op.video_key])
        res_list = dataset.to_list()
        self.assertEqual(res_list, target_list)

    def test_default(self):
        ds_list = [{
            'videos': [self.vid1_path]
        }, {
            'videos': [self.vid2_path]
        }, {
            'videos': [self.vid3_path]
        }]
        tgt_list = [{
            'videos': [self.vid1_path]
        }, {
            'videos': [self.vid2_path]
        }, {
            'videos': [self.vid3_path]
        }]
        op = VideoMotionScoreFilter()
        self._run_helper(op, ds_list, tgt_list)

    def test_downscale(self):
        ds_list = [{
            'videos': [self.vid1_path]
        }, {
            'videos': [self.vid2_path]
        }, {
            'videos': [self.vid3_path]
        }]
        tgt_list = [{
            'videos': [self.vid1_path]
        }, {
            'videos': [self.vid2_path]
        }]
        op = VideoMotionScoreFilter(min_score=1.0, size=120)
        self._run_helper(op, ds_list, tgt_list)

    def test_downscale_max(self):
        ds_list = [{
            'videos': [self.vid1_path]
        }, {
            'videos': [self.vid2_path]
        }, {
            'videos': [self.vid3_path]
        }]
        tgt_list = [{
            'videos': [self.vid1_path]
        }, {
            'videos': [self.vid2_path]
        }]
        op = VideoMotionScoreFilter(min_score=1.0, size=120, max_size=160)
        self._run_helper(op, ds_list, tgt_list)

    def test_downscale_relative(self):
        ds_list = [{
            'videos': [self.vid1_path]
        }, {
            'videos': [self.vid2_path]
        }, {
            'videos': [self.vid3_path]
        }]
        tgt_list = [{
            'videos': [self.vid1_path]
        }, {
            'videos': [self.vid2_path]
        }]
        op = VideoMotionScoreFilter(min_score=0.005, size=(120, 160), relative=True)
        self._run_helper(op, ds_list, tgt_list)

    def test_high(self):
        ds_list = [{
            'videos': [self.vid1_path]
        }, {
            'videos': [self.vid2_path]
        }, {
            'videos': [self.vid3_path]
        }]
        tgt_list = [{'videos': [self.vid2_path]}]
        op = VideoMotionScoreFilter(min_score=3.0)
        self._run_helper(op, ds_list, tgt_list)

    def test_low(self):
        ds_list = [{
            'videos': [self.vid1_path]
        }, {
            'videos': [self.vid2_path]
        }, {
            'videos': [self.vid3_path]
        }]
        tgt_list = [{'videos': [self.vid3_path]}]
        op = VideoMotionScoreFilter(min_score=0.0, max_score=1.50)
        self._run_helper(op, ds_list, tgt_list)

    def test_middle(self):
        ds_list = [{
            'videos': [self.vid1_path]
        }, {
            'videos': [self.vid2_path]
        }, {
            'videos': [self.vid3_path]
        }]
        tgt_list = [{'videos': [self.vid1_path]}]
        op = VideoMotionScoreFilter(min_score=1.5, max_score=3.0)
        self._run_helper(op, ds_list, tgt_list)

    def test_any(self):
        ds_list = [{
            'videos': [self.vid1_path, self.vid2_path]
        }, {
            'videos': [self.vid2_path, self.vid3_path]
        }, {
            'videos': [self.vid1_path, self.vid3_path]
        }]
        tgt_list = [{
            'videos': [self.vid1_path, self.vid2_path]
        }, {
            'videos': [self.vid1_path, self.vid3_path]
        }]
        op = VideoMotionScoreFilter(min_score=1.5,
                                    max_score=3.0,
                                    any_or_all='any')
        self._run_helper(op, ds_list, tgt_list)

    def test_all(self):
        ds_list = [{
            'videos': [self.vid1_path, self.vid2_path]
        }, {
            'videos': [self.vid2_path, self.vid3_path]
        }, {
            'videos': [self.vid1_path, self.vid3_path]
        }]
        tgt_list = []
        op = VideoMotionScoreFilter(min_score=1.5,
                                    max_score=3.0,
                                    any_or_all='all')
        self._run_helper(op, ds_list, tgt_list)

    def test_parallel(self):
        import multiprocess as mp
        mp.set_start_method('forkserver', force=True)

        ds_list = [{
            'videos': [self.vid1_path]
        }, {
            'videos': [self.vid2_path]
        }, {
            'videos': [self.vid3_path]
        }]
        tgt_list = [{'videos': [self.vid1_path]}]
        op = VideoMotionScoreFilter(min_score=1.5, max_score=3.0)
        self._run_helper(op, ds_list, tgt_list, np=2)

    def test_output_optical_flow(self):
        ds_list = [{
            'videos': [self.vid1_path]
        }, {
            'videos': [self.vid2_path]
        }, {
            'videos': [self.vid3_path]
        }]
        tgt_list = [{
            'videos': [self.vid1_path]
        }, {
            'videos': [self.vid2_path]
        }, {
            'videos': [self.vid3_path]
        }]
        op = VideoMotionScoreFilter(if_output_optical_flow=True)
        dataset = Dataset.from_list(ds_list)
        if Fields.stats not in dataset.features:
            dataset = dataset.add_column(name=Fields.stats,
                                         column=[{}] * dataset.num_rows)
        if Fields.meta not in dataset.features:
            dataset = dataset.add_column(name=Fields.meta,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.compute_stats, num_proc=1)
        dataset = dataset.filter(op.process, num_proc=1)
        metas = dataset.select_columns(column_names=[Fields.meta])
        self.assertIn(MetaKeys.video_optical_flow, metas.features[Fields.meta])
        sample_optical_flow = metas[0][Fields.meta][MetaKeys.video_optical_flow]
        self.assertIsInstance(sample_optical_flow, np.ndarray)
        self.assertEqual(len(sample_optical_flow.shape), 4)
        self.assertEqual(sample_optical_flow.shape[-1], 2)

        dataset = dataset.select_columns(column_names=[op.video_key])
        res_list = dataset.to_list()
        self.assertEqual(res_list, tgt_list)


if __name__ == '__main__':
    unittest.main()

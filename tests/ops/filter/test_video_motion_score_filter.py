import os
import unittest

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

    img1_path = os.path.join(data_path, 'img6.jpg')

    def _run_helper(self, op, source_list, target_list, select_field=None):
        dataset = Dataset.from_list(source_list)
        if Fields.stats not in dataset.features:
            dataset = dataset.add_column(name=Fields.stats,
                                         column=[{}] * dataset.num_rows)
        dataset = op.run(dataset)
        if select_field is not None:
            dataset = dataset.select_columns(column_names=select_field)
        else:
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
        op = VideoMotionScoreFilter(min_score=1.5, max_score=3.0, num_proc=2)
        self._run_helper(op, ds_list, tgt_list)

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

        dataset = dataset.select_columns(column_names=[op.video_key])
        res_list = dataset.to_list()
        self.assertEqual(res_list, tgt_list)

    def test_frame_field(self):
        ds_list = [{
            'frames': [[self.img1_path, self.img1_path, self.img1_path]],
        }, {
            'frames': [[self.img1_path, self.img1_path]],
        }, {
            'frames': [[self.img1_path]],
        }]
        tgt_list = [{
            'frames': [[self.img1_path, self.img1_path, self.img1_path]],
        }, {
            'frames': [[self.img1_path, self.img1_path]],
        }]
        op = VideoMotionScoreFilter(min_score=0, max_score=3.0, frame_field='frames', num_proc=2)
        self._run_helper(op, ds_list, tgt_list, select_field=['frames'])

    def test_frame_field_without_original_fps(self):
        """When original_fps is not specified, all frames are processed
        (backward compatible behavior)."""
        ds_list = [{
            'frames': [[self.img1_path, self.img1_path, self.img1_path]],
        }, {
            'frames': [[self.img1_path, self.img1_path]],
        }, {
            'frames': [[self.img1_path]],
        }]
        tgt_list = [{
            'frames': [[self.img1_path, self.img1_path, self.img1_path]],
        }, {
            'frames': [[self.img1_path, self.img1_path]],
        }]
        op = VideoMotionScoreFilter(
            min_score=0, max_score=3.0,
            frame_field='frames', sampling_fps=2,
        )
        self._run_helper(op, ds_list, tgt_list, select_field=['frames'])

    def test_frame_field_with_original_fps(self):
        """When original_fps is specified, frames are sampled at sampling_fps
        rate. With original_fps=30 and sampling_fps=2, sampling_step=15.
        For 3 identical frames (idx 0,1,2), only idx 0 is selected
        (0 % 15 == 0), resulting in no optical flow pairs -> score=-1."""
        ds_list = [{
            'frames': [[self.img1_path, self.img1_path, self.img1_path]],
        }, {
            'frames': [[self.img1_path, self.img1_path]],
        }, {
            'frames': [[self.img1_path]],
        }]
        # With original_fps=30 and sampling_fps=2, step=15,
        # only frame 0 is selected from each video -> only 1 frame ->
        # no optical flow -> score=-1, which is < min_score=0 -> filtered out
        tgt_list = []
        op = VideoMotionScoreFilter(
            min_score=0, max_score=3.0,
            frame_field='frames', sampling_fps=2, original_fps=30,
        )
        self._run_helper(op, ds_list, tgt_list, select_field=['frames'])

    def test_frame_field_with_original_fps_small_step(self):
        """When original_fps is close to sampling_fps, sampling_step is small.
        With original_fps=4 and sampling_fps=2, sampling_step=2.
        For 3 identical frames (idx 0,1,2), frames 0 and 2 are selected
        (0%2==0, 2%2==0), resulting in 1 optical flow pair. Since frames
        are identical, motion score is 0, which is within [0, 3.0]."""
        ds_list = [{
            'frames': [[self.img1_path, self.img1_path, self.img1_path]],
        }, {
            'frames': [[self.img1_path, self.img1_path]],
        }, {
            'frames': [[self.img1_path]],
        }]
        # step=2: for 3 frames, idx 0,2 selected -> 1 flow pair -> score=0
        # for 2 frames, idx 0 selected -> no flow pair -> score=-1
        # for 1 frame, idx 0 selected -> no flow pair -> score=-1
        # score=0 is in [0, 3.0] -> kept; score=-1 is not -> filtered
        tgt_list = [{
            'frames': [[self.img1_path, self.img1_path, self.img1_path]],
        }]
        op = VideoMotionScoreFilter(
            min_score=0, max_score=3.0,
            frame_field='frames', sampling_fps=2, original_fps=4,
        )
        self._run_helper(op, ds_list, tgt_list, select_field=['frames'])

    def test_frame_field_with_original_fps_equal_sampling_fps(self):
        """When original_fps equals sampling_fps, sampling_step=1, all frames
        are processed (same as not specifying original_fps)."""
        ds_list = [{
            'frames': [[self.img1_path, self.img1_path, self.img1_path]],
        }, {
            'frames': [[self.img1_path, self.img1_path]],
        }, {
            'frames': [[self.img1_path]],
        }]
        tgt_list = [{
            'frames': [[self.img1_path, self.img1_path, self.img1_path]],
        }, {
            'frames': [[self.img1_path, self.img1_path]],
        }]
        op = VideoMotionScoreFilter(
            min_score=0, max_score=3.0,
            frame_field='frames', sampling_fps=2, original_fps=2,
        )
        self._run_helper(op, ds_list, tgt_list, select_field=['frames'])


if __name__ == '__main__':
    unittest.main()

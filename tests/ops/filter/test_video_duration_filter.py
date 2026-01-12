import os
import unittest

from data_juicer.core.data import NestedDataset as Dataset

from data_juicer.ops.filter.video_duration_filter import VideoDurationFilter
from data_juicer.utils.constant import Fields
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class VideoDurationFilterTest(DataJuicerTestCaseBase):

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')
    vid1_path = os.path.join(data_path, 'video1.mp4')  # about 12s
    vid2_path = os.path.join(data_path, 'video2.mp4')  # about 23s
    vid3_path = os.path.join(data_path, 'video3.mp4')  # about 50s

    def _run_video_duration_filter(self,
                                   dataset: Dataset,
                                   target_list,
                                   op,
                                   np=1):
        if Fields.stats not in dataset.features:
            dataset = dataset.add_column(name=Fields.stats,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.compute_stats, num_proc=np)
        dataset = dataset.filter(op.process, num_proc=np)
        dataset = dataset.select_columns(column_names=[op.video_key])
        res_list = dataset.to_list()
        self.assertEqual(res_list, target_list)

    def test_default_filter(self):

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
        dataset = Dataset.from_list(ds_list)

        for video_backend in ['av', 'ffmpeg']:
            op = VideoDurationFilter(video_backend=video_backend)
            self._run_video_duration_filter(dataset, tgt_list, op)

    def test_filter_long_videos(self):

        ds_list = [{
            'videos': [self.vid1_path]
        }, {
            'videos': [self.vid2_path]
        }, {
            'videos': [self.vid3_path]
        }]
        tgt_list = [{'videos': [self.vid1_path]}]
        dataset = Dataset.from_list(ds_list)

        for video_backend in ['av', 'ffmpeg']:
            op = VideoDurationFilter(max_duration=15, video_backend=video_backend)
            self._run_video_duration_filter(dataset, tgt_list, op)

    def test_filter_short_videos(self):

        ds_list = [{
            'videos': [self.vid1_path]
        }, {
            'videos': [self.vid2_path]
        }, {
            'videos': [self.vid3_path]
        }]
        tgt_list = [{'videos': [self.vid3_path]}]
        dataset = Dataset.from_list(ds_list)

        for video_backend in ['av', 'ffmpeg']:
            op = VideoDurationFilter(min_duration=30, video_backend=video_backend)
            self._run_video_duration_filter(dataset, tgt_list, op)

    def test_filter_videos_within_range(self):

        ds_list = [{
            'videos': [self.vid1_path]
        }, {
            'videos': [self.vid2_path]
        }, {
            'videos': [self.vid3_path]
        }]
        tgt_list = [{'videos': [self.vid2_path]}]
        dataset = Dataset.from_list(ds_list)

        for video_backend in ['av', 'ffmpeg']:
            op = VideoDurationFilter(min_duration=16, max_duration=42, video_backend=video_backend)
            self._run_video_duration_filter(dataset, tgt_list, op)

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
            'videos': [self.vid2_path, self.vid3_path]
        }]
        dataset = Dataset.from_list(ds_list)

        for video_backend in ['av', 'ffmpeg']:
            op = VideoDurationFilter(min_duration=15,
                                    max_duration=30,
                                    any_or_all='any',
                                    video_backend=video_backend)
            self._run_video_duration_filter(dataset, tgt_list, op)

    def test_all(self):

        ds_list = [{
            'videos': [self.vid1_path, self.vid2_path]
        }, {
            'videos': [self.vid2_path, self.vid3_path]
        }, {
            'videos': [self.vid1_path, self.vid3_path]
        }]
        tgt_list = []
        dataset = Dataset.from_list(ds_list)

        for video_backend in ['av', 'ffmpeg']:
            op = VideoDurationFilter(min_duration=15,
                                    max_duration=30,
                                    any_or_all='all',
                                    video_backend=video_backend)
            self._run_video_duration_filter(dataset, tgt_list, op)

    def test_filter_in_parallel(self):

        ds_list = [{
            'videos': [self.vid1_path]
        }, {
            'videos': [self.vid2_path]
        }, {
            'videos': [self.vid3_path]
        }]
        tgt_list = [{'videos': [self.vid2_path]}]
        dataset = Dataset.from_list(ds_list)

        for video_backend in ['av', 'ffmpeg']:
            op = VideoDurationFilter(min_duration=15, max_duration=30, video_backend=video_backend)
            self._run_video_duration_filter(dataset, tgt_list, op, np=2)


if __name__ == '__main__':
    unittest.main()

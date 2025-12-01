import os
import unittest

from data_juicer.core.data import NestedDataset as Dataset

from data_juicer.ops.filter.video_ocr_area_ratio_filter import \
    VideoOcrAreaRatioFilter
from data_juicer.utils.constant import Fields
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

class VideoOcrAreaRatioFilterTest(DataJuicerTestCaseBase):

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')
    vid1_path = os.path.join(data_path, 'video1.mp4')  # about 0.067
    vid2_path = os.path.join(data_path, 'video2.mp4')  # about 0.288
    vid3_path = os.path.join(data_path, 'video3.mp4')  # about 0.075

    def _run_video_ocr_area_ratio_filter(self,
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
        op = VideoOcrAreaRatioFilter()
        self._run_video_ocr_area_ratio_filter(dataset, tgt_list, op)

    def test_filter_large_ratio_videos(self):

        ds_list = [{
            'videos': [self.vid1_path]
        }, {
            'videos': [self.vid2_path]
        }, {
            'videos': [self.vid3_path]
        }]
        tgt_list = [{'videos': [self.vid1_path]}, {'videos': [self.vid3_path]}]
        dataset = Dataset.from_list(ds_list)
        op = VideoOcrAreaRatioFilter(max_area_ratio=0.1)
        self._run_video_ocr_area_ratio_filter(dataset, tgt_list, op)

    def test_filter_small_ratio_videos(self):

        ds_list = [{
            'videos': [self.vid1_path]
        }, {
            'videos': [self.vid2_path]
        }, {
            'videos': [self.vid3_path]
        }]
        tgt_list = [{'videos': [self.vid2_path]}]
        dataset = Dataset.from_list(ds_list)
        op = VideoOcrAreaRatioFilter(min_area_ratio=0.2)
        self._run_video_ocr_area_ratio_filter(dataset, tgt_list, op)

    def test_filter_videos_within_range(self):

        ds_list = [{
            'videos': [self.vid1_path]
        }, {
            'videos': [self.vid2_path]
        }, {
            'videos': [self.vid3_path]
        }]
        tgt_list = [{'videos': [self.vid3_path]}]
        dataset = Dataset.from_list(ds_list)
        op = VideoOcrAreaRatioFilter(min_area_ratio=0.07, max_area_ratio=0.1)
        self._run_video_ocr_area_ratio_filter(dataset, tgt_list, op)

    def test_any(self):

        ds_list = [{
            'videos': [self.vid1_path, self.vid2_path]
        }, {
            'videos': [self.vid2_path, self.vid3_path]
        }, {
            'videos': [self.vid1_path, self.vid3_path]
        }]
        tgt_list = [{
            'videos': [self.vid2_path, self.vid3_path]
        }, {
            'videos': [self.vid1_path, self.vid3_path]
        }]
        dataset = Dataset.from_list(ds_list)
        op = VideoOcrAreaRatioFilter(min_area_ratio=0.07,
                                     max_area_ratio=0.1,
                                     any_or_all='any')
        self._run_video_ocr_area_ratio_filter(dataset, tgt_list, op)

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
        op = VideoOcrAreaRatioFilter(min_area_ratio=0.07,
                                     max_area_ratio=0.1,
                                     any_or_all='all')
        self._run_video_ocr_area_ratio_filter(dataset, tgt_list, op)

    def test_filter_in_parallel(self):

        # WARNING: current parallel tests only work in spawn method
        import multiprocess
        original_method = multiprocess.get_start_method()
        multiprocess.set_start_method('spawn', force=True)
        # WARNING: current parallel tests only work in spawn method

        ds_list = [{
            'videos': [self.vid1_path]
        }, {
            'videos': [self.vid2_path]
        }, {
            'videos': [self.vid3_path]
        }]
        tgt_list = [{'videos': [self.vid3_path]}]
        dataset = Dataset.from_list(ds_list)
        op = VideoOcrAreaRatioFilter(min_area_ratio=0.07, max_area_ratio=0.1)
        self._run_video_ocr_area_ratio_filter(dataset, tgt_list, op, np=2)

        # WARNING: current parallel tests only work in spawn method
        multiprocess.set_start_method(original_method, force=True)
        # WARNING: current parallel tests only work in spawn method


if __name__ == '__main__':
    unittest.main()

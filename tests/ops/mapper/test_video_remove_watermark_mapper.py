import os
import shutil
import unittest

from data_juicer.core.data import NestedDataset as Dataset

from data_juicer.ops.mapper.video_remove_watermark_mapper import \
    VideoRemoveWatermarkMapper
from data_juicer.utils.constant import Fields
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class VideoRemoveWatermarkMapperTest(DataJuicerTestCaseBase):

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')
    # video1: horizontal resolution 640p, vertical resolution 360p
    vid1_path = os.path.join(data_path, 'video1.mp4')

    def _run_video_remove_watermask_mapper(self,
                                           dataset: Dataset,
                                           op,
                                           test_name,
                                           np=1):
        if Fields.stats not in dataset.features:
            dataset = dataset.add_column(name=Fields.stats,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.process, num_proc=np)
        dataset = dataset.select_columns(column_names=[op.video_key])

        # check each video personally
        output_dir = '../video_remove_watermark_mapper'
        move_to_dir = os.path.join(output_dir, test_name)
        if not os.path.exists(move_to_dir):
            os.makedirs(move_to_dir)
        for sample in dataset.to_list():
            for value in sample['videos']:
                move_to_path = os.path.join(move_to_dir,
                                            os.path.basename(value))
                shutil.copyfile(value, move_to_path)

    def test_roi_pixel_type(self):
        ds_list = [{'videos': [self.vid1_path]}]
        dataset = Dataset.from_list(ds_list)
        op = VideoRemoveWatermarkMapper(roi_strings=['[0, 0, 150, 60]'],
                                        roi_type='pixel')
        self._run_video_remove_watermask_mapper(dataset, op,
                                                'test_roi_pixel_type')

    def test_multi_roi_region(self):
        ds_list = [{'videos': [self.vid1_path]}]
        dataset = Dataset.from_list(ds_list)
        op = VideoRemoveWatermarkMapper(
            roi_strings=['[0, 0, 150, 60]', '[30, 60, 75, 140]'],
            roi_type='pixel')
        self._run_video_remove_watermask_mapper(dataset, op,
                                                'test_multi_roi_region')

    def test_roi_ratio_type(self):
        ds_list = [{'videos': [self.vid1_path]}]
        dataset = Dataset.from_list(ds_list)
        op = VideoRemoveWatermarkMapper(
            roi_strings=['[0, 0, 0.234375, 0.16667]'], roi_type='ratio')
        self._run_video_remove_watermask_mapper(dataset, op,
                                                'test_roi_ratio_type')

    def test_frame_num_and_frame_threshold(self):
        ds_list = [{'videos': [self.vid1_path]}]
        dataset = Dataset.from_list(ds_list)
        op = VideoRemoveWatermarkMapper(roi_strings=['[0, 0, 150, 60]'],
                                        roi_type='pixel',
                                        frame_num=100,
                                        min_frame_threshold=100)
        self._run_video_remove_watermask_mapper(
            dataset, op, 'test_frame_num_and_frame_threshold')

    def test_roi_key(self):
        ds_list = [{
            'videos': [self.vid1_path],
            'roi_strings': ['[30, 60, 75, 300]'],
        }, {
            'videos': [self.vid1_path],
            'roi_strings': ['[30, 60, 75, 140]', '30, 140, 53, 300', 'none'],
        }, {
            'videos': [self.vid1_path],
            'roi_strings':
            ['[30, 60, 75, 140]', '30, 140, 53, 200', '(30, 200, 53, 300)'],
        }]
        dataset = Dataset.from_list(ds_list)
        op = VideoRemoveWatermarkMapper(roi_type='pixel',
                                        roi_key='roi_strings')
        self._run_video_remove_watermask_mapper(dataset, op, 'test_roi_key')

    def test_detection_method(self):
        ds_list = [{'videos': [self.vid1_path]}]
        dataset = Dataset.from_list(ds_list)
        op = VideoRemoveWatermarkMapper(
            roi_strings=['[0, 0, 150, 60]', '[30, 60, 75, 300]'],
            roi_type='pixel',
            detection_method='pixel_diversity')
        self._run_video_remove_watermask_mapper(dataset, op,
                                                'test_detection_method')

    def test_multi_process(self):
        ds_list = [{
            'videos': [self.vid1_path],
            'roi_strings': ['[30, 60, 75, 300]'],
        }, {
            'videos': [self.vid1_path],
            'roi_strings': ['[30, 60, 75, 140]', '30, 140, 53, 300', 'none'],
        }, {
            'videos': [self.vid1_path],
            'roi_strings':
            ['[30, 60, 75, 140]', '30, 140, 53, 200', '(30, 200, 53, 300)'],
        }]
        dataset = Dataset.from_list(ds_list)
        op = VideoRemoveWatermarkMapper(roi_type='pixel',
                                        roi_key='roi_strings')
        self._run_video_remove_watermask_mapper(dataset, op, 'test_multi_process', np=2)


if __name__ == '__main__':
    unittest.main()

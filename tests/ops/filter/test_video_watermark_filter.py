# flake8: noqa: E501

import os
import unittest

from data_juicer.core.data import NestedDataset as Dataset

from data_juicer.utils.resource_utils import cuda_device_count
from data_juicer.ops.filter.video_watermark_filter import VideoWatermarkFilter
from data_juicer.utils.constant import Fields
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

class VideoWatermarkFilterTest(DataJuicerTestCaseBase):

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')

    video1_path = os.path.join(data_path, 'video1.mp4')
    video2_path = os.path.join(data_path, 'video2.mp4')
    video3_path = os.path.join(data_path, 'video3.mp4')
    hf_watermark_model = 'amrul-hzz/watermark_detector'

    @classmethod
    def tearDownClass(cls) -> None:
        super().tearDownClass(cls.hf_watermark_model)

    def _run_filter(self, dataset: Dataset, target_list, op, num_proc=1):

        if Fields.stats not in dataset.features:
            # TODO:
            # this is a temp solution,
            # only add stats when calling filter op
            dataset = dataset.add_column(name=Fields.stats,
                                         column=[{}] * dataset.num_rows)

        dataset = dataset.map(op.compute_stats,
                              num_proc=num_proc,
                              with_rank=True)
        dataset = dataset.filter(op.process, num_proc=num_proc)
        dataset = dataset.select_columns(column_names=['videos'])
        res_list = dataset.to_list()
        self.assertEqual(res_list, target_list)

    def test_all_keyframes(self):

        ds_list = [{
            'videos': [self.video1_path]
        }, {
            'videos': [self.video2_path]
        }, {
            'videos': [self.video3_path]
        }]
        tgt_list = [{
            'videos': [self.video3_path]
        }
        ]

        dataset = Dataset.from_list(ds_list)
        op = VideoWatermarkFilter(hf_watermark_model=self.hf_watermark_model,
                            prob_threshold=0.8,
                            frame_sampling_method='all_keyframes')
        self._run_filter(dataset, tgt_list, op)

    def test_uniform_frames(self):

        ds_list = [{
            'videos': [self.video1_path]
        }, {
            'videos': [self.video2_path]
        }, {
            'videos': [self.video3_path]
        }]
        tgt_list = [{
            'videos': [self.video3_path]
        }
        ]

        dataset = Dataset.from_list(ds_list)
        op = VideoWatermarkFilter(hf_watermark_model=self.hf_watermark_model,
                            prob_threshold=0.8,
                            frame_sampling_method='uniform',
                            frame_num=3)
        self._run_filter(dataset, tgt_list, op)

    def test_reduce_max(self):

        ds_list = [{
            'videos': [self.video1_path]
        }, {
            'videos': [self.video2_path]
        }, {
            'videos': [self.video3_path]
        }]
        tgt_list = []

        dataset = Dataset.from_list(ds_list)
        op = VideoWatermarkFilter(hf_watermark_model=self.hf_watermark_model,
                            prob_threshold=0.9,
                            frame_sampling_method='all_keyframes',
                            reduce_mode='max')
        self._run_filter(dataset, tgt_list, op)

    def test_reduce_min(self):

        ds_list = [{
            'videos': [self.video1_path]
        }, {
            'videos': [self.video2_path]
        }, {
            'videos': [self.video3_path]
        }]
        tgt_list = [{
            'videos': [self.video2_path]
        }, {
            'videos': [self.video3_path]
        }
        ]

        dataset = Dataset.from_list(ds_list)
        op = VideoWatermarkFilter(hf_watermark_model=self.hf_watermark_model,
                            prob_threshold=0.85,
                            frame_sampling_method='all_keyframes',
                            reduce_mode='min')
        self._run_filter(dataset, tgt_list, op)

    def test_any(self):

        ds_list = [{
            'videos': [self.video1_path, self.video2_path]
        }, {
            'videos': [self.video1_path, self.video3_path]
        }]
        tgt_list = [{
            'videos': [self.video1_path, self.video3_path]
        }]

        dataset = Dataset.from_list(ds_list)
        op = VideoWatermarkFilter(hf_watermark_model=self.hf_watermark_model,
                            prob_threshold=0.8,
                            frame_sampling_method='all_keyframes',
                            any_or_all='any')
        self._run_filter(dataset, tgt_list, op)    

    def test_all(self):

        ds_list = [{
            'videos': [self.video1_path, self.video2_path]
        }, {
            'videos': [self.video2_path, self.video3_path]
        }]
        tgt_list = [{
            'videos': [self.video2_path, self.video3_path]
        }]

        dataset = Dataset.from_list(ds_list)
        op = VideoWatermarkFilter(hf_watermark_model=self.hf_watermark_model,
                            prob_threshold=0.9,
                            frame_sampling_method='all_keyframes',
                            any_or_all='all')
        self._run_filter(dataset, tgt_list, op)   

    def test_multi_process(self):

        ds_list = [{
            'videos': [self.video1_path]
        }, {
            'videos': [self.video2_path]
        }, {
            'videos': [self.video3_path]
        }]
        tgt_list = [{
            'videos': [self.video3_path]
        }
        ]

        # set num_proc <= the number of CUDA if it is available
        num_proc = 2
        if cuda_device_count() == 1:
            num_proc = 1

        dataset = Dataset.from_list(ds_list)
        op = VideoWatermarkFilter(hf_watermark_model=self.hf_watermark_model,
                            prob_threshold=0.8,
                            frame_sampling_method='all_keyframes')
        self._run_filter(dataset, tgt_list, op, num_proc=num_proc)

if __name__ == '__main__':
    unittest.main()

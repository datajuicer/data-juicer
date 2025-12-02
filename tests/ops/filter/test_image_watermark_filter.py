# flake8: noqa: E501

import os
import unittest

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.utils.lazy_loader import LazyLoader

from data_juicer.utils.resource_utils import cuda_device_count
from data_juicer.ops.filter.image_watermark_filter import ImageWatermarkFilter
from data_juicer.utils.constant import Fields
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

cv2 = LazyLoader('cv2', 'opencv-python')

class ImageWatermarkFilterTest(DataJuicerTestCaseBase):

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')

    img1_path = os.path.join(data_path, 'img1.png')
    img2_path = os.path.join(data_path, 'img2.jpg')
    img3_path = os.path.join(data_path, 'img3.jpg')
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
        dataset = dataset.select_columns(column_names=['images'])
        res_list = dataset.to_list()
        self.assertEqual(res_list, target_list)

    def test_watermark_filter(self):

        ds_list = [{
            'images': [self.img1_path]
        }, {
            'images': [self.img2_path]
        }, {
            'images': [self.img3_path]
        }]
        tgt_list = [{
            'images': [self.img1_path]
        }, {
            'images': [self.img3_path]
        }
        ]

        dataset = Dataset.from_list(ds_list)
        op = ImageWatermarkFilter(hf_watermark_model=self.hf_watermark_model,
                            prob_threshold=0.8)
        self._run_filter(dataset, tgt_list, op)

    def test_any(self):

        ds_list = [{
            'images': [self.img1_path, self.img2_path]
        }, {
            'images': [self.img1_path, self.img3_path]
        }]
        tgt_list = [{
            'images': [self.img1_path, self.img3_path]
        }]

        dataset = Dataset.from_list(ds_list)
        op = ImageWatermarkFilter(hf_watermark_model=self.hf_watermark_model,
                            prob_threshold=0.4,
                            any_or_all='any')
        self._run_filter(dataset, tgt_list, op)    

    def test_all(self):

        ds_list = [{
            'images': [self.img1_path, self.img2_path]
        }, {
            'images': [self.img1_path, self.img3_path]
        }]
        tgt_list = [{
            'images': [self.img1_path, self.img3_path]
        }]

        dataset = Dataset.from_list(ds_list)
        op = ImageWatermarkFilter(hf_watermark_model=self.hf_watermark_model,
                            prob_threshold=0.8,
                            any_or_all='all')
        self._run_filter(dataset, tgt_list, op)   

    def test_multi_process(self):

        ds_list = [{
            'images': [self.img1_path]
        }, {
            'images': [self.img2_path]
        }, {
            'images': [self.img3_path]
        }]
        tgt_list = [{
            'images': [self.img1_path]
        }, {
            'images': [self.img3_path]
        }
        ]

        # set num_proc <= the number of CUDA if it is available
        num_proc = 2
        if cuda_device_count() == 1:
            num_proc = 1

        dataset = Dataset.from_list(ds_list)
        op = ImageWatermarkFilter(hf_watermark_model=self.hf_watermark_model,
                            prob_threshold=0.8)
        self._run_filter(dataset, tgt_list, op, num_proc=num_proc)

if __name__ == '__main__':
    unittest.main()

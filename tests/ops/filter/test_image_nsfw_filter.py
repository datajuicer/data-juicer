# flake8: noqa: E501

import os
import unittest

from data_juicer.core.data import NestedDataset as Dataset

from data_juicer.utils.resource_utils import cuda_device_count
from data_juicer.ops.filter.image_nsfw_filter import ImageNSFWFilter
from data_juicer.utils.constant import Fields
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

class ImageNSFWFilterTest(DataJuicerTestCaseBase):

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')

    img1_path = os.path.join(data_path, 'img1.png')
    img2_path = os.path.join(data_path, 'img2.jpg')
    img3_path = os.path.join(data_path, 'img3.jpg')
    hf_nsfw_model = 'Falconsai/nsfw_image_detection'

    @classmethod
    def tearDownClass(cls) -> None:
        super().tearDownClass(cls.hf_nsfw_model)

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

    def test_nsfw_filter(self):

        ds_list = [{
            'images': [self.img1_path]
        }, {
            'images': [self.img2_path]
        }, {
            'images': [self.img3_path]
        }]
        tgt_list = [{
            'images': [self.img2_path]
        }, {
            'images': [self.img3_path]
        }
        ]

        dataset = Dataset.from_list(ds_list)
        op = ImageNSFWFilter(hf_nsfw_model=self.hf_nsfw_model,
                            max_score=0.0005)
        self._run_filter(dataset, tgt_list, op)

    def test_any(self):

        ds_list = [{
            'images': [self.img1_path, self.img3_path]
        }, {
            'images': [self.img1_path, self.img2_path]
        }]
        tgt_list = [{
            'images': [self.img1_path, self.img2_path]
        }]

        dataset = Dataset.from_list(ds_list)
        op = ImageNSFWFilter(hf_nsfw_model=self.hf_nsfw_model,
                            max_score=0.00012,
                            any_or_all='any')
        self._run_filter(dataset, tgt_list, op)    

    def test_all(self):

        ds_list = [{
            'images': [self.img1_path, self.img2_path]
        }, {
            'images': [self.img2_path, self.img3_path]
        }]
        tgt_list = [{
            'images': [self.img2_path, self.img3_path]
        }]

        dataset = Dataset.from_list(ds_list)
        op = ImageNSFWFilter(hf_nsfw_model=self.hf_nsfw_model,
                            max_score=0.0005,
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
            'images': [self.img2_path]
        }, {
            'images': [self.img3_path]
        }
        ]

        # set num_proc <= the number of CUDA if it is available
        num_proc = 2
        if cuda_device_count() == 1:
            num_proc = 1

        dataset = Dataset.from_list(ds_list)
        op = ImageNSFWFilter(hf_nsfw_model=self.hf_nsfw_model,
                            max_score=0.0005)
        self._run_filter(dataset, tgt_list, op, num_proc=num_proc)

if __name__ == '__main__':
    unittest.main()

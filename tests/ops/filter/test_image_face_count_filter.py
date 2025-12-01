import os
import unittest

from data_juicer.core.data import NestedDataset as Dataset

from data_juicer.ops.filter.image_face_count_filter import ImageFaceCountFilter
from data_juicer.utils.constant import Fields
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class ImageFaceCountFilterTest(DataJuicerTestCaseBase):

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')
    img1_path = os.path.join(data_path, 'cat.jpg')
    img2_path = os.path.join(data_path, 'lena.jpg')
    img3_path = os.path.join(data_path, 'img8.jpg')

    def _run_face_count_filter(self, dataset: Dataset, target_list, op, num_proc=1):
        if Fields.stats not in dataset.features:
            dataset = dataset.add_column(name=Fields.stats,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.compute_stats, num_proc=num_proc)
        dataset = dataset.filter(op.process, num_proc=num_proc)
        dataset = dataset.remove_columns(Fields.stats)
        res_list = dataset.to_list()
        self.assertEqual(res_list, target_list)

    def test_filter_1(self):
        ds_list = [
            {'images': [self.img1_path]},
            {'images': [self.img2_path]},
            {'images': [self.img3_path]}
        ]
        tgt_list = [{'images': [self.img2_path]}]
        dataset = Dataset.from_list(ds_list)
        op = ImageFaceCountFilter(min_face_count=1, max_face_count=1)
        self._run_face_count_filter(dataset, tgt_list, op)

    def test_filter_2(self):
        ds_list = [
            {'images': [self.img1_path]},
            {'images': [self.img2_path]},
            {'images': [self.img3_path]}
        ]
        tgt_list = [{'images': [self.img2_path]}, {'images': [self.img3_path]}]
        dataset = Dataset.from_list(ds_list)
        op = ImageFaceCountFilter(min_face_count=1, max_face_count=5)
        self._run_face_count_filter(dataset, tgt_list, op)

    def test_filter_multi_proc(self):
        ds_list = [
            {'images': [self.img1_path]},
            {'images': [self.img2_path]},
            {'images': [self.img3_path]}
        ]
        tgt_list = [{'images': [self.img2_path]}, {'images': [self.img3_path]}]
        dataset = Dataset.from_list(ds_list)
        op = ImageFaceCountFilter(min_face_count=1, max_face_count=5)
        self._run_face_count_filter(dataset, tgt_list, op, num_proc=3)

    def test_any(self):
        ds_list = [{
            'images': [self.img1_path, self.img2_path]
        }, {
            'images': [self.img2_path, self.img3_path]
        }, {
            'images': [self.img1_path, self.img3_path]
        }]
        tgt_list = [{
            'images': [self.img1_path, self.img2_path]
        }, {
            'images': [self.img2_path, self.img3_path]
        }]
        dataset = Dataset.from_list(ds_list)
        op = ImageFaceCountFilter(min_face_count=1, max_face_count=1, any_or_all='any')
        self._run_face_count_filter(dataset, tgt_list, op)

    def test_all(self):
        ds_list = [{
            'images': [self.img1_path, self.img2_path]
        }, {
            'images': [self.img2_path, self.img3_path]
        }, {
            'images': [self.img1_path, self.img3_path]
        }]
        tgt_list = [{
            'images': [self.img2_path, self.img3_path]
        }]
        dataset = Dataset.from_list(ds_list)
        op = ImageFaceCountFilter(min_face_count=1, max_face_count=5, any_or_all='all')
        self._run_face_count_filter(dataset, tgt_list, op)


if __name__ == '__main__':
    unittest.main()

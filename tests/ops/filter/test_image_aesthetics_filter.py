import os
import unittest

from data_juicer.core.data import NestedDataset as Dataset

from data_juicer.ops.filter.image_aesthetics_filter import \
    ImageAestheticsFilter
from data_juicer.utils.constant import Fields
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class ImageAestheticsFilterTest(DataJuicerTestCaseBase):

    maxDiff = None

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')
    img1_path = os.path.join(data_path, 'cat.jpg')
    img2_path = os.path.join(data_path, 'blip.jpg')
    img3_path = os.path.join(data_path, 'lena-face.jpg')

    model_id = \
        'shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE'

    # with shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE
    # the img1, img2, img3 gets scores 0.4382, 0.5973, 0.5216 respectively

    def _run_image_aesthetics_filter(self,
                                     dataset: Dataset,
                                     target_list,
                                     op,
                                     num_proc=1):
        if Fields.stats not in dataset.features:
            dataset = dataset.add_column(name=Fields.stats,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.compute_stats, num_proc=num_proc)
        dataset = dataset.filter(op.process, num_proc=num_proc)
        dataset = dataset.remove_columns(Fields.stats)
        res_list = dataset.to_list()
        self.assertEqual(res_list, target_list)

    def test_filter_small(self):

        ds_list = [{
            'images': [self.img1_path]
        }, {
            'images': [self.img2_path]
        }, {
            'images': [self.img3_path]
        }]
        tgt_list = [{'images': [self.img2_path]}]
        dataset = Dataset.from_list(ds_list)
        op = ImageAestheticsFilter(hf_scorer_model=self.model_id,
                                   min_score=0.55,
                                   max_score=1.0)
        self._run_image_aesthetics_filter(dataset, tgt_list, op)

    def test_filter_large(self):

        ds_list = [{
            'images': [self.img1_path]
        }, {
            'images': [self.img2_path]
        }, {
            'images': [self.img3_path]
        }]
        tgt_list = [{'images': [self.img1_path]}, {'images': [self.img3_path]}]
        dataset = Dataset.from_list(ds_list)
        op = ImageAestheticsFilter(hf_scorer_model=self.model_id,
                                   min_score=0.4,
                                   max_score=0.55)
        self._run_image_aesthetics_filter(dataset, tgt_list, op)

    def test_filter_multimodal(self):

        ds_list = [{
            'text': 'a test sentence',
            'images': []
        }, {
            'text': 'a test sentence',
            'images': [self.img1_path]
        }, {
            'text': 'a test sentence',
            'images': [self.img2_path]
        }, {
            'text': 'a test sentence',
            'images': [self.img3_path]
        }]
        tgt_list = [{
            'text': 'a test sentence',
            'images': []
        }, {
            'text': 'a test sentence',
            'images': [self.img2_path]
        }, {
            'text': 'a test sentence',
            'images': [self.img3_path]
        }]
        dataset = Dataset.from_list(ds_list)
        op = ImageAestheticsFilter(hf_scorer_model=self.model_id, )
        self._run_image_aesthetics_filter(dataset, tgt_list, op)

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
        }, {
            'images': [self.img1_path, self.img3_path]
        }]
        dataset = Dataset.from_list(ds_list)
        op = ImageAestheticsFilter(hf_scorer_model=self.model_id,
                                   any_or_all='any')
        self._run_image_aesthetics_filter(dataset, tgt_list, op)

    def test_all(self):

        ds_list = [{
            'images': [self.img1_path, self.img2_path]
        }, {
            'images': [self.img2_path, self.img3_path]
        }, {
            'images': [self.img1_path, self.img3_path]
        }]
        tgt_list = [{'images': [self.img2_path, self.img3_path]}]
        dataset = Dataset.from_list(ds_list)
        op = ImageAestheticsFilter(hf_scorer_model=self.model_id,
                                   any_or_all='all')
        self._run_image_aesthetics_filter(dataset, tgt_list, op)

    def test_filter_multi_process(self):

        ds_list = [{
            'images': [self.img1_path]
        }, {
            'images': [self.img2_path]
        }, {
            'images': [self.img3_path]
        }]
        tgt_list = [{'images': [self.img2_path]}, {'images': [self.img3_path]}]
        dataset = Dataset.from_list(ds_list)
        op = ImageAestheticsFilter(hf_scorer_model=self.model_id, )
        self._run_image_aesthetics_filter(dataset, tgt_list, op, num_proc=3)


if __name__ == '__main__':
    unittest.main()

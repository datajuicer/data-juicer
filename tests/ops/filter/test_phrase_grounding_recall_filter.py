# flake8: noqa: E501

import os
import unittest

from data_juicer.core.data import NestedDataset as Dataset

from data_juicer.ops.filter.phrase_grounding_recall_filter import \
    PhraseGroundingRecallFilter
from data_juicer.utils.constant import Fields
from data_juicer.utils.mm_utils import SpecialTokens
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

class PhraseGroundingRecallFilterTest(DataJuicerTestCaseBase):

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')

    demo_path = os.path.join(data_path, 'blip.jpg')
    cat_path = os.path.join(data_path, 'cat.jpg')
    img1_path = os.path.join(data_path, 'img1.png')
    img2_path = os.path.join(data_path, 'img2.jpg')
    img3_path = os.path.join(data_path, 'img3.jpg')
    face_path = os.path.join(data_path, 'lena-face.jpg')
    hf_owlvit = 'google/owlvit-base-patch32'

    @classmethod
    def tearDownClass(cls) -> None:
        super().tearDownClass(cls.hf_owlvit)

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
        dataset = dataset.select_columns(column_names=['text', 'images'])
        res_list = dataset.to_list()
        self.assertEqual(res_list, target_list)

    def test_general(self):

        ds_list = [{
            'text':
            f'{SpecialTokens.image}a woman sitting on the beach with a dog',
            'images': [self.demo_path]
        }, {
            'text':
            f'Two cats are sleeping on the couch with two remote controls{SpecialTokens.image}',
            'images': [self.cat_path]
        }, {
            'text':
            f'{SpecialTokens.image} select luxury furniture 3 - inch gel memory foam mattress topper {SpecialTokens.eoc}',
            'images': [self.img1_path]
        }, {
            'text':
            f'{SpecialTokens.image} A bus with red advertisements is running on the street. {SpecialTokens.eoc}',
            'images': [self.img2_path]
        }, {
            'text':
            f'{SpecialTokens.image} A woman carrying a bag is walking in a rainy alley holding an umbrella {SpecialTokens.eoc}',
            'images': [self.img3_path]
        }]
        tgt_list = [{
            'text':
            f'{SpecialTokens.image}a woman sitting on the beach with a dog',
            'images': [self.demo_path]
        }, {
            'text':
            f'Two cats are sleeping on the couch with two remote controls{SpecialTokens.image}',
            'images': [self.cat_path]
        }, {
            'text':
            f'{SpecialTokens.image} select luxury furniture 3 - inch gel memory foam mattress topper {SpecialTokens.eoc}',
            'images': [self.img1_path]
        }, {
            'text':
            f'{SpecialTokens.image} A bus with red advertisements is running on the street. {SpecialTokens.eoc}',
            'images': [self.img2_path]
        }, {
            'text':
            f'{SpecialTokens.image} A woman carrying a bag is walking in a rainy alley holding an umbrella {SpecialTokens.eoc}',
            'images': [self.img3_path]
        }]

        dataset = Dataset.from_list(ds_list)
        op = PhraseGroundingRecallFilter(
            hf_owlvit=self.hf_owlvit,
            reduce_mode='avg',
            any_or_all='any',
            min_recall=0.5,
            max_recall=1.0,
        )
        self._run_filter(dataset, tgt_list, op)

    def test_high_recall(self):

        ds_list = [{
            'text':
            f'{SpecialTokens.image}a woman sitting on the beach with a dog',
            'images': [self.demo_path]
        }, {
            'text':
            f'Two cats are sleeping on the couch with two remote controls{SpecialTokens.image}',
            'images': [self.cat_path]
        }, {
            'text':
            f'{SpecialTokens.image} select luxury furniture 3 - inch gel memory foam mattress topper {SpecialTokens.eoc}',
            'images': [self.img1_path]
        }, {
            'text':
            f'{SpecialTokens.image} A bus with red advertisements is running on the street. {SpecialTokens.eoc}',
            'images': [self.img2_path]
        }, {
            'text':
            f'{SpecialTokens.image} A woman carrying a bag is walking in a rainy alley holding an umbrella {SpecialTokens.eoc}',
            'images': [self.img3_path]
        }]
        tgt_list = [{
            'text':
            f'{SpecialTokens.image}a woman sitting on the beach with a dog',
            'images': [self.demo_path]
        }, {
            'text':
            f'Two cats are sleeping on the couch with two remote controls{SpecialTokens.image}',
            'images': [self.cat_path]
        }, {
            'text':
            f'{SpecialTokens.image} A woman carrying a bag is walking in a rainy alley holding an umbrella {SpecialTokens.eoc}',
            'images': [self.img3_path]
        }]

        dataset = Dataset.from_list(ds_list)
        op = PhraseGroundingRecallFilter(
            hf_owlvit=self.hf_owlvit,
            reduce_mode='avg',
            any_or_all='any',
            min_recall=0.7,
            max_recall=1.0,
        )
        self._run_filter(dataset, tgt_list, op)

    def test_high_conf_thr(self):

        ds_list = [{
            'text':
            f'{SpecialTokens.image}a woman sitting on the beach with a dog',
            'images': [self.demo_path]
        }, {
            'text':
            f'{SpecialTokens.image}a man sitting on the grass with a cat',
            'images': [self.demo_path]
        }]
        tgt_list = [{
            'text':
            f'{SpecialTokens.image}a woman sitting on the beach with a dog',
            'images': [self.demo_path]
        }]

        dataset = Dataset.from_list(ds_list)
        op = PhraseGroundingRecallFilter(
            hf_owlvit=self.hf_owlvit,
            reduce_mode='avg',
            any_or_all='any',
            min_recall=0.1,
            max_recall=1.0,
            conf_thr=0.5,
        )
        self._run_filter(dataset, tgt_list, op)

    def test_low_conf_thr(self):
        # some similar but different objects might be detected incorrectly
        ds_list = [{
            'text':
            f'{SpecialTokens.image}a woman sitting on the beach with a dog',
            'images': [self.demo_path]
        }, {
            'text':
            f'{SpecialTokens.image}a man sitting on the grass with a cat',
            'images': [self.demo_path]
        }]
        tgt_list = [{
            'text':
            f'{SpecialTokens.image}a woman sitting on the beach with a dog',
            'images': [self.demo_path]
        }, {
            'text':
            f'{SpecialTokens.image}a man sitting on the grass with a cat',
            'images': [self.demo_path]
        }]

        dataset = Dataset.from_list(ds_list)
        op = PhraseGroundingRecallFilter(
            hf_owlvit=self.hf_owlvit,
            reduce_mode='avg',
            any_or_all='any',
            min_recall=0.1,
            max_recall=1.0,
            conf_thr=0.0,
        )
        self._run_filter(dataset, tgt_list, op)

    def test_low_area_ratio(self):
        # objects with relatively large area will be removed
        ds_list = [{
            'text': f'{SpecialTokens.image} a photo of a woman\'s face',
            'images': [self.face_path]
        }, {
            'text':
            f'{SpecialTokens.image}A bus with red advertisements is running on the street.',
            'images': [self.img2_path]
        }]
        tgt_list = []

        dataset = Dataset.from_list(ds_list)
        op = PhraseGroundingRecallFilter(
            hf_owlvit=self.hf_owlvit,
            reduce_mode='avg',
            any_or_all='any',
            min_recall=0.5,
            max_recall=1.0,
            large_area_ratio_thr=0.50,
        )
        self._run_filter(dataset, tgt_list, op)

    def test_high_area_ratio(self):
        # objects with too large area will be removed
        ds_list = [{
            'text': f'{SpecialTokens.image} a photo of a woman\'s face',
            'images': [self.face_path]
        }, {
            'text':
            f'{SpecialTokens.image}A bus with red advertisements is running on the street.',
            'images': [self.img2_path]
        }]
        tgt_list = [{
            'text':
            f'{SpecialTokens.image}A bus with red advertisements is running on the street.',
            'images': [self.img2_path]
        }]

        dataset = Dataset.from_list(ds_list)
        op = PhraseGroundingRecallFilter(
            hf_owlvit=self.hf_owlvit,
            reduce_mode='avg',
            any_or_all='any',
            min_recall=0.5,
            max_recall=1.0,
            large_area_ratio_thr=0.99,
        )
        self._run_filter(dataset, tgt_list, op)

    def test_reduce_avg(self):

        ds_list = [{
            'text':
            f'{SpecialTokens.image}a woman sitting on the beach with a dog{SpecialTokens.image}',
            'images': [self.demo_path, self.cat_path]
        }, {
            'text':
            f'{SpecialTokens.image} {SpecialTokens.image} select luxury furniture 3 - inch gel memory foam mattress topper {SpecialTokens.eoc}',
            'images': [self.img1_path, self.img2_path]
        }]
        tgt_list = [{
            'text':
            f'{SpecialTokens.image}a woman sitting on the beach with a dog{SpecialTokens.image}',
            'images': [self.demo_path, self.cat_path]
        }, {
            'text':
            f'{SpecialTokens.image} {SpecialTokens.image} select luxury furniture 3 - inch gel memory foam mattress topper {SpecialTokens.eoc}',
            'images': [self.img1_path, self.img2_path]
        }]

        dataset = Dataset.from_list(ds_list)
        op = PhraseGroundingRecallFilter(
            hf_owlvit=self.hf_owlvit,
            reduce_mode='avg',
            any_or_all='any',
            min_recall=0.5,
            max_recall=1.0,
        )
        self._run_filter(dataset, tgt_list, op)

    def test_reduce_max(self):

        ds_list = [{
            'text':
            f'{SpecialTokens.image}a woman sitting on the beach with a dog{SpecialTokens.image}',
            'images': [self.demo_path, self.cat_path]
        }, {
            'text':
            f'{SpecialTokens.image} {SpecialTokens.image} select luxury furniture 3 - inch gel memory foam mattress topper {SpecialTokens.eoc}',
            'images': [self.img1_path, self.img2_path]
        }]
        tgt_list = [{
            'text':
            f'{SpecialTokens.image}a woman sitting on the beach with a dog{SpecialTokens.image}',
            'images': [self.demo_path, self.cat_path]
        }]

        dataset = Dataset.from_list(ds_list)
        op = PhraseGroundingRecallFilter(
            hf_owlvit=self.hf_owlvit,
            reduce_mode='max',
            any_or_all='any',
            min_recall=0.7,
            max_recall=1.0,
        )
        self._run_filter(dataset, tgt_list, op)

    def test_reduce_min(self):

        ds_list = [{
            'text':
            f'{SpecialTokens.image}a woman sitting on the beach with a dog{SpecialTokens.image}',
            'images': [self.demo_path, self.cat_path]
        }, {
            'text':
            f'{SpecialTokens.image} {SpecialTokens.image} select luxury furniture 3 - inch gel memory foam mattress topper {SpecialTokens.eoc}',
            'images': [self.img1_path, self.img2_path]
        }]
        tgt_list = []

        dataset = Dataset.from_list(ds_list)
        op = PhraseGroundingRecallFilter(
            hf_owlvit=self.hf_owlvit,
            reduce_mode='min',
            any_or_all='any',
            min_recall=0.5,
            max_recall=1.0,
        )
        self._run_filter(dataset, tgt_list, op)

    def test_keep_all(self):

        ds_list = [{
            'text':
            f'{SpecialTokens.image} {SpecialTokens.image} select luxury furniture 3 - inch gel memory foam mattress topper {SpecialTokens.eoc}'
            f'{SpecialTokens.image} a woman sitting on the beach with a dog',
            'images': [self.img1_path, self.cat_path, self.demo_path]
        }]
        tgt_list = []

        dataset = Dataset.from_list(ds_list)
        op = PhraseGroundingRecallFilter(
            hf_owlvit=self.hf_owlvit,
            reduce_mode='avg',
            any_or_all='all',
            min_recall=0.7,
            max_recall=1.0,
        )
        self._run_filter(dataset, tgt_list, op)

    def test_keep_any(self):

        ds_list = [{
            'text':
            f'{SpecialTokens.image} {SpecialTokens.image} select luxury furniture 3 - inch gel memory foam mattress topper {SpecialTokens.eoc}'
            f'{SpecialTokens.image} a woman sitting on the beach with a dog',
            'images': [self.img1_path, self.cat_path, self.demo_path]
        }]
        tgt_list = [{
            'text':
            f'{SpecialTokens.image} {SpecialTokens.image} select luxury furniture 3 - inch gel memory foam mattress topper {SpecialTokens.eoc}'
            f'{SpecialTokens.image} a woman sitting on the beach with a dog',
            'images': [self.img1_path, self.cat_path, self.demo_path]
        }]

        dataset = Dataset.from_list(ds_list)
        op = PhraseGroundingRecallFilter(
            hf_owlvit=self.hf_owlvit,
            reduce_mode='avg',
            any_or_all='any',
            min_recall=0.7,
            max_recall=1.0,
        )
        self._run_filter(dataset, tgt_list, op)

    def test_process_in_parallel(self):

        ds_list = [{
            'text':
            f'{SpecialTokens.image}a woman sitting on the beach with a dog',
            'images': [self.demo_path]
        }, {
            'text':
            f'Two cats are sleeping on the couch with two remote controls{SpecialTokens.image}',
            'images': [self.cat_path]
        }, {
            'text':
            f'{SpecialTokens.image} select luxury furniture 3 - inch gel memory foam mattress topper {SpecialTokens.eoc}',
            'images': [self.img1_path]
        }, {
            'text':
            f'{SpecialTokens.image} A bus with red advertisements is running on the street. {SpecialTokens.eoc}',
            'images': [self.img2_path]
        }, {
            'text':
            f'{SpecialTokens.image} A woman carrying a bag is walking in a rainy alley holding an umbrella {SpecialTokens.eoc}',
            'images': [self.img3_path]
        }]
        tgt_list = [{
            'text':
            f'{SpecialTokens.image}a woman sitting on the beach with a dog',
            'images': [self.demo_path]
        }, {
            'text':
            f'Two cats are sleeping on the couch with two remote controls{SpecialTokens.image}',
            'images': [self.cat_path]
        }, {
            'text':
            f'{SpecialTokens.image} A woman carrying a bag is walking in a rainy alley holding an umbrella {SpecialTokens.eoc}',
            'images': [self.img3_path]
        }]

        dataset = Dataset.from_list(ds_list)
        op = PhraseGroundingRecallFilter(
            hf_owlvit=self.hf_owlvit,
            reduce_mode='avg',
            any_or_all='any',
            min_recall=0.7,
            max_recall=1.0,
        )
        self._run_filter(dataset, tgt_list, op, num_proc=2)


if __name__ == '__main__':
    unittest.main()

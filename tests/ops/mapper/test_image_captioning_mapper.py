import os
import unittest

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.image_captioning_mapper import \
    ImageCaptioningMapper
from data_juicer.utils.mm_utils import SpecialTokens
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class ImageCaptioningMapperTest(DataJuicerTestCaseBase):

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')

    cat_path = os.path.join(data_path, 'cat.jpg')
    img3_path = os.path.join(data_path, 'img3.jpg')

    hf_img2seq = 'Salesforce/blip2-opt-2.7b'

    @classmethod
    def tearDownClass(cls) -> None:
        super().tearDownClass(cls.hf_img2seq)

    def _run_mapper(self,
                    dataset: Dataset,
                    op,
                    num_proc=1,
                    caption_num=0):

        dataset = dataset.map(op.process, num_proc=num_proc, with_rank=True)
        dataset_list = dataset.select_columns(column_names=['text']).to_list()
        # assert the caption is generated successfully in terms of not_none
        # as the generated content is not deterministic
        self.assertEqual(len(dataset_list), caption_num)

    def test_no_eoc_special_token(self):

        ds_list = [{
            'text': f'{SpecialTokens.image}a photo of a cat',
            'images': [self.cat_path]
        }, {
            'text': f'{SpecialTokens.image}a photo, a women with an umbrella',
            'images': [self.img3_path]
        }]
        caption_num = 1
        dataset = Dataset.from_list(ds_list)
        op = ImageCaptioningMapper(hf_img2seq=self.hf_img2seq,
                                   caption_num=caption_num,
                                   keep_candidate_mode='random_any')
        self._run_mapper(dataset, op, caption_num=len(dataset) * 2)

    def test_eoc_special_token(self):

        ds_list = [
            {
                'text':
                f'{SpecialTokens.image}a photo of a cat{SpecialTokens.eoc}',
                'images': [self.cat_path]
            },
            {
                'text':
                f'{SpecialTokens.image}a photo, a women with an umbrella{SpecialTokens.eoc}',  # noqa: E501
                'images': [self.img3_path]
            }
        ]
        caption_num = 1
        dataset = Dataset.from_list(ds_list)
        op = ImageCaptioningMapper(hf_img2seq=self.hf_img2seq,
                                   caption_num=caption_num,
                                   keep_candidate_mode='random_any')
        self._run_mapper(dataset, op, caption_num=len(dataset) * 2)

    def test_multi_candidate_keep_random_any(self):

        ds_list = [{
            'text': f'{SpecialTokens.image}a photo of a cat',
            'images': [self.cat_path]
        }, {
            'text': f'{SpecialTokens.image}a photo, a women with an umbrella',
            'images': [self.img3_path]
        }]
        caption_num = 4
        dataset = Dataset.from_list(ds_list)
        op = ImageCaptioningMapper(hf_img2seq=self.hf_img2seq,
                                   caption_num=caption_num,
                                   keep_candidate_mode='random_any')
        self._run_mapper(dataset, op, caption_num=len(dataset) * 2)

    def test_multi_candidate_keep_all(self):

        ds_list = [{
            'text': f'{SpecialTokens.image}a photo of a cat',
            'images': [self.cat_path]
        }, {
            'text': f'{SpecialTokens.image}a photo, a women with an umbrella',
            'images': [self.img3_path]
        }]
        caption_num = 4
        dataset = Dataset.from_list(ds_list)
        op = ImageCaptioningMapper(hf_img2seq=self.hf_img2seq,
                                   caption_num=caption_num,
                                   keep_candidate_mode='all')
        self._run_mapper(dataset,
                         op,
                         caption_num=(1 + caption_num) * len(dataset))

    def test_multi_candidate_keep_similar_one(self):
        ds_list = [{
            'text': f'{SpecialTokens.image}a photo of a cat',
            'images': [self.cat_path]
        }, {
            'text': f'{SpecialTokens.image}a photo, a women with an umbrella',
            'images': [self.img3_path]
        }]
        caption_num = 4
        dataset = Dataset.from_list(ds_list)
        op = ImageCaptioningMapper(hf_img2seq=self.hf_img2seq,
                                   caption_num=caption_num,
                                   keep_candidate_mode='similar_one_simhash')
        self._run_mapper(dataset, op, caption_num=len(dataset) * 2)

    def test_multi_process(self):
        ds_list = [{
            'text': f'{SpecialTokens.image}a photo of a cat',
            'images': [self.cat_path]
        }] * 10
        caption_num = 1
        dataset = Dataset.from_list(ds_list)
        op = ImageCaptioningMapper(hf_img2seq=self.hf_img2seq,
                                   caption_num=caption_num,
                                   keep_candidate_mode='random_any')
        self._run_mapper(dataset, op, num_proc=2, caption_num=len(dataset) * 2)

    def test_no_eoc_special_token_remove_original_sample(self):

        ds_list = [{
            'text': f'{SpecialTokens.image}a photo of a cat',
            'images': [self.cat_path]
        }, {
            'text': f'{SpecialTokens.image}a photo, a women with an umbrella',
            'images': [self.img3_path]
        }]
        caption_num = 1
        dataset = Dataset.from_list(ds_list)
        op = ImageCaptioningMapper(hf_img2seq=self.hf_img2seq,
                                   caption_num=caption_num,
                                   keep_candidate_mode='random_any',
                                   keep_original_sample=False)
        self._run_mapper(dataset, op, caption_num=len(dataset))

    def test_eoc_special_token_remove_original_sample(self):

        ds_list = [
            {
                'text':
                f'{SpecialTokens.image}a photo of a cat{SpecialTokens.eoc}',
                'images': [self.cat_path]
            },
            {
                'text':
                f'{SpecialTokens.image}a photo, a women with an umbrella{SpecialTokens.eoc}',  # noqa: E501
                'images': [self.img3_path]
            }
        ]
        caption_num = 1
        dataset = Dataset.from_list(ds_list)
        op = ImageCaptioningMapper(hf_img2seq=self.hf_img2seq,
                                   caption_num=caption_num,
                                   keep_candidate_mode='random_any',
                                   keep_original_sample=False)
        self._run_mapper(dataset, op, caption_num=len(dataset))

    def test_multi_candidate_keep_random_any_remove_original_sample(self):

        ds_list = [{
            'text': f'{SpecialTokens.image}a photo of a cat',
            'images': [self.cat_path]
        }, {
            'text': f'{SpecialTokens.image}a photo, a women with an umbrella',
            'images': [self.img3_path]
        }]
        caption_num = 4
        dataset = Dataset.from_list(ds_list)
        op = ImageCaptioningMapper(hf_img2seq=self.hf_img2seq,
                                   caption_num=caption_num,
                                   keep_candidate_mode='random_any',
                                   keep_original_sample=False)
        self._run_mapper(dataset, op, caption_num=len(dataset))

    def test_multi_candidate_keep_all_remove_original_sample(self):

        ds_list = [{
            'text': f'{SpecialTokens.image}a photo of a cat',
            'images': [self.cat_path]
        }, {
            'text': f'{SpecialTokens.image}a photo, a women with an umbrella',
            'images': [self.img3_path]
        }]
        caption_num = 4
        dataset = Dataset.from_list(ds_list)
        op = ImageCaptioningMapper(hf_img2seq=self.hf_img2seq,
                                   caption_num=caption_num,
                                   keep_candidate_mode='all',
                                   keep_original_sample=False)
        self._run_mapper(dataset, op, caption_num=caption_num * len(dataset))

    def test_multi_candidate_keep_similar_one_remove_original_sample(self):
        ds_list = [{
            'text': f'{SpecialTokens.image}a photo of a cat',
            'images': [self.cat_path]
        }, {
            'text': f'{SpecialTokens.image}a photo, a women with an umbrella',
            'images': [self.img3_path]
        }]
        caption_num = 4
        dataset = Dataset.from_list(ds_list)
        op = ImageCaptioningMapper(hf_img2seq=self.hf_img2seq,
                                   caption_num=caption_num,
                                   keep_candidate_mode='similar_one_simhash',
                                   keep_original_sample=False)
        self._run_mapper(dataset, op, caption_num=len(dataset))

    def test_multi_process_remove_original_sample(self):
        ds_list = [{
            'text': f'{SpecialTokens.image}a photo of a cat',
            'images': [self.cat_path]
        }] * 10
        caption_num = 1
        dataset = Dataset.from_list(ds_list)
        op = ImageCaptioningMapper(hf_img2seq=self.hf_img2seq,
                                   caption_num=caption_num,
                                   keep_candidate_mode='random_any',
                                   keep_original_sample=False)
        self._run_mapper(dataset, op, num_proc=2, caption_num=len(dataset))


if __name__ == '__main__':
    unittest.main()

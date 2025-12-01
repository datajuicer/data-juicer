import os
import shutil
import unittest

from data_juicer.utils.resource_utils import cuda_device_count
from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.image_diffusion_mapper import ImageDiffusionMapper
from data_juicer.utils.mm_utils import SpecialTokens
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

class ImageDiffusionMapperTest(DataJuicerTestCaseBase):

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')

    cat_path = os.path.join(data_path, 'cat.jpg')
    img3_path = os.path.join(data_path, 'img3.jpg')

    hf_diffusion = 'CompVis/stable-diffusion-v1-4'
    hf_img2seq = 'Salesforce/blip2-opt-2.7b'

    # dir to save the images produced in the tests
    output_dir = '../diffusion_output/'

    @classmethod
    def tearDownClass(cls) -> None:
        super().tearDownClass(cls.hf_diffusion)
        super().tearDownClass(cls.hf_img2seq)

    def _run_mapper(self,
                    dataset: Dataset,
                    op,
                    move_to_dir,
                    num_proc=1,
                    total_num=1):

        dataset = dataset.map(op.process, num_proc=num_proc, with_rank=True)
        dataset_list = dataset.select_columns(
            column_names=['images']).to_list()

        self.assertEqual(len(dataset_list), total_num)
        if not os.path.exists(move_to_dir):
            os.makedirs(move_to_dir)
        for data in dataset_list:
            for image_path in data['images']:
                if str(image_path) != str(self.cat_path) \
                 and str(image_path) != str(self.img3_path):
                    cp_to_path = os.path.join(move_to_dir,
                                              os.path.basename(image_path))
                    shutil.copyfile(image_path, cp_to_path)

    def test_for_strength(self):
        ds_list = [{
            'text': f'{SpecialTokens.image}a photo of a cat',
            'caption': 'a women with an umbrella',
            'images': [self.cat_path]
        }]
        aug_num = 3
        dataset = Dataset.from_list(ds_list)
        op = ImageDiffusionMapper(hf_diffusion=self.hf_diffusion,
                                  strength=1.0,
                                  aug_num=aug_num,
                                  keep_original_sample=True,
                                  caption_key='caption')
        self._run_mapper(dataset,
                         op,
                         os.path.join(self.output_dir, 'test_for_strength'),
                         total_num=(aug_num + 1) * len(ds_list))

    def test_for_given_caption_list(self):

        ds_list = [{
            'text': f'{SpecialTokens.image}, {SpecialTokens.image}',
            'captions': ['A photo of a cat', 'a women with an umbrella'],
            'images': [self.cat_path, self.img3_path]
        }]

        aug_num = 2
        dataset = Dataset.from_list(ds_list)
        op = ImageDiffusionMapper(hf_diffusion=self.hf_diffusion,
                                  aug_num=aug_num,
                                  keep_original_sample=False,
                                  caption_key='captions')
        self._run_mapper(dataset,
                         op,
                         os.path.join(self.output_dir,
                                      'test_for_given_caption_list'),
                         total_num=aug_num * len(ds_list))

    def test_for_given_caption_string(self):

        ds_list = [{
            'text': f'{SpecialTokens.image}a photo of a cat',
            'images': [self.cat_path]
        }, {
            'text': f'{SpecialTokens.image}a photo, a women with an umbrella',
            'images': [self.img3_path]
        }]

        aug_num = 1
        dataset = Dataset.from_list(ds_list)
        op = ImageDiffusionMapper(hf_diffusion=self.hf_diffusion,
                                  aug_num=aug_num,
                                  keep_original_sample=False,
                                  caption_key='text')
        self._run_mapper(dataset,
                         op,
                         os.path.join(self.output_dir,
                                      'test_for_given_caption_string'),
                         total_num=aug_num * len(ds_list))

    def test_for_no_given_caption(self):

        ds_list = [{
            'text': f'{SpecialTokens.image}',
            'images': [self.cat_path]
        }, {
            'text': f'{SpecialTokens.image}',
            'images': [self.img3_path]
        }]

        aug_num = 2
        dataset = Dataset.from_list(ds_list)
        op = ImageDiffusionMapper(hf_diffusion=self.hf_diffusion,
                                  aug_num=aug_num,
                                  keep_original_sample=False,
                                  hf_img2seq=self.hf_img2seq)
        self._run_mapper(dataset,
                         op,
                         os.path.join(self.output_dir,
                                      'test_for_no_given_caption'),
                         total_num=aug_num * len(ds_list))

    def test_for_fp16_given_caption_string(self):

        ds_list = [{
            'text': f'{SpecialTokens.image}a photo of a cat',
            'images': [self.cat_path]
        }, {
            'text': f'{SpecialTokens.image}a photo, a women with an umbrella',
            'images': [self.img3_path]
        }]

        aug_num = 1
        dataset = Dataset.from_list(ds_list)
        op = ImageDiffusionMapper(hf_diffusion=self.hf_diffusion,
                                  torch_dtype='fp16',
                                  revision='fp16',
                                  aug_num=aug_num,
                                  keep_original_sample=False,
                                  caption_key='text')
        self._run_mapper(dataset,
                         op,
                         os.path.join(self.output_dir,
                                      'test_for_fp16_given_caption_string'),
                         total_num=aug_num * len(ds_list))

    def test_for_multi_process_given_caption_string(self):

        ds_list = [{
            'text': f'{SpecialTokens.image}a photo of a cat',
            'images': [self.cat_path]
        }, {
            'text': f'{SpecialTokens.image}a photo, a women with an umbrella',
            'images': [self.img3_path]
        }]

        aug_num = 1
        dataset = Dataset.from_list(ds_list)
        op = ImageDiffusionMapper(hf_diffusion=self.hf_diffusion,
                                  aug_num=aug_num,
                                  keep_original_sample=False,
                                  caption_key='text')

        # set num_proc <= the number of CUDA if it is available
        num_proc = 2
        if cuda_device_count() == 1:
            num_proc = 1

        self._run_mapper(dataset,
                         op,
                         os.path.join(self.output_dir,
                                      'test_for_given_caption_string'),
                         num_proc=num_proc,
                         total_num=aug_num * len(ds_list))


if __name__ == '__main__':
    unittest.main()

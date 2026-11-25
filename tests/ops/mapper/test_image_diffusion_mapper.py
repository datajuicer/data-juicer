import copy
import os
import shutil
import unittest
from unittest.mock import patch

from PIL import Image

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


class ImageDiffusionMapperEmptyCaptionTest(DataJuicerTestCaseBase):
    """Regression tests for missing/empty captions.

    An empty prompt is a valid input for image-to-image generation: it simply
    means "no text guidance, redraw from the image alone". The OP already
    behaved this way for an empty string, so a missing key, None, an empty list
    or None entries inside a list must be normalized to an empty prompt and
    generate as usual, rather than silently dropping the sample.

    ``_real_guidance`` is stubbed so the caption handling runs for real without
    downloading a diffusion model.
    """

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')

    cat_path = os.path.join(data_path, 'cat.jpg')
    img3_path = os.path.join(data_path, 'img3.jpg')

    aug_num = 2

    def _build_op(self, keep_original_sample, caption_key='caption'):
        with patch('data_juicer.ops.mapper.image_diffusion_mapper.'
                   'prepare_model',
                   return_value='fake_model_key'):
            return ImageDiffusionMapper(
                aug_num=self.aug_num,
                keep_original_sample=keep_original_sample,
                caption_key=caption_key)

    def _run(self, op, samples):
        """Run the OP with generation stubbed, returning recorded prompts."""
        prompts = []

        def fake_guidance(inner_self, caption, image, rank=None):
            prompts.append(caption)
            return Image.new('RGB', (8, 8))

        with patch.object(ImageDiffusionMapper, '_real_guidance',
                          fake_guidance):
            res = op.process_batched(copy.deepcopy(samples))
        return res, prompts

    def _assert_generated_with_empty_prompt(self, samples, num_images=1):
        """Empty captions still generate, using an empty prompt."""
        num_samples = len(samples['images'])

        # keep_original_sample=False -> only the generated samples remain
        op = self._build_op(keep_original_sample=False)
        res, prompts = self._run(op, samples)
        self.assertEqual(set(res.keys()), set(samples.keys()))
        self.assertEqual(len(res['images']), self.aug_num * num_samples)
        # every image of every augmentation is generated from an empty prompt
        self.assertEqual(len(prompts),
                         self.aug_num * num_samples * num_images)
        self.assertTrue(all(p == '' for p in prompts),
                        f'expected only empty prompts, got {prompts}')

        # keep_original_sample=True -> originals are kept alongside
        op = self._build_op(keep_original_sample=True)
        res, _ = self._run(op, samples)
        self.assertEqual(len(res['images']),
                         (self.aug_num + 1) * num_samples)

    def test_missing_caption_key(self):
        samples = {
            'text': [f'{SpecialTokens.image}a photo of a cat'],
            'images': [[self.cat_path]],
        }
        self._assert_generated_with_empty_prompt(samples)

    def test_none_caption(self):
        samples = {
            'text': [f'{SpecialTokens.image}a photo of a cat'],
            'caption': [None],
            'images': [[self.cat_path]],
        }
        self._assert_generated_with_empty_prompt(samples)

    def test_blank_string_caption(self):
        samples = {
            'text': [
                f'{SpecialTokens.image}a photo of a cat',
                f'{SpecialTokens.image}a women with an umbrella',
            ],
            'caption': ['', '   \t\n'],
            'images': [[self.cat_path], [self.img3_path]],
        }
        self._assert_generated_with_empty_prompt(samples)

    def test_empty_list_caption(self):
        samples = {
            'text': [f'{SpecialTokens.image}a photo of a cat'],
            'caption': [[]],
            'images': [[self.cat_path]],
        }
        self._assert_generated_with_empty_prompt(samples)

    def test_list_with_none_entries(self):
        samples = {
            'text': [f'{SpecialTokens.image}, {SpecialTokens.image}'],
            'caption': [[None, '  ']],
            'images': [[self.cat_path, self.img3_path]],
        }
        self._assert_generated_with_empty_prompt(samples, num_images=2)

    def test_partial_caption_list_still_raises(self):
        """A partially filled caption list keeps failing loudly.

        Padding a short list would silently pair captions with the wrong
        images, so the length check is kept. Callers must use an explicit
        placeholder instead of omitting entries.
        """
        samples = {
            'text': [f'{SpecialTokens.image}, {SpecialTokens.image}'],
            'caption': [['a photo of a cat']],
            'images': [[self.cat_path, self.img3_path]],
        }
        op = self._build_op(keep_original_sample=False)
        with self.assertRaises(AssertionError):
            self._run(op, samples)

    def test_none_placeholder_keeps_alignment(self):
        """An explicit None placeholder maps to an empty prompt in place."""
        samples = {
            'text': [f'{SpecialTokens.image}, {SpecialTokens.image}'],
            'caption': [['a photo of a cat', None]],
            'images': [[self.cat_path, self.img3_path]],
        }
        op = self._build_op(keep_original_sample=False)
        res, prompts = self._run(op, samples)

        self.assertEqual(len(res['images']), self.aug_num)
        self.assertEqual(prompts, ['a photo of a cat', ''] * self.aug_num)

    def test_leading_none_placeholder_keeps_alignment(self):
        """A placeholder in the first slot must not shift the later caption."""
        samples = {
            'text': [f'{SpecialTokens.image}, {SpecialTokens.image}'],
            'caption': [[None, 'a women with an umbrella']],
            'images': [[self.cat_path, self.img3_path]],
        }
        op = self._build_op(keep_original_sample=False)
        res, prompts = self._run(op, samples)

        self.assertEqual(len(res['images']), self.aug_num)
        self.assertEqual(prompts,
                         ['', 'a women with an umbrella'] * self.aug_num)

    def test_duplicate_images_align_with_key_list(self):
        """Captions align with the image key list, not the deduplicated dict.

        ``load_data_with_context`` returns a dict keyed by image path, so
        repeating the same path collapses it. Alignment must follow the image
        key list, otherwise captions get truncated and indexing overflows.
        """
        samples = {
            'text': [f'{SpecialTokens.image}, {SpecialTokens.image}'],
            'caption': [['first caption', 'second caption']],
            'images': [[self.cat_path, self.cat_path]],
        }
        op = self._build_op(keep_original_sample=False)
        res, prompts = self._run(op, samples)

        self.assertEqual(len(res['images']), self.aug_num)
        self.assertEqual(prompts,
                         ['first caption', 'second caption'] * self.aug_num)

    def test_empty_list_caption_with_duplicate_images(self):
        """An empty caption list pads to the full image key count."""
        samples = {
            'text': [f'{SpecialTokens.image}, {SpecialTokens.image}'],
            'caption': [[]],
            'images': [[self.cat_path, self.cat_path]],
        }
        op = self._build_op(keep_original_sample=False)
        res, prompts = self._run(op, samples)

        self.assertEqual(len(res['images']), self.aug_num)
        self.assertEqual(prompts, ['', ''] * self.aug_num)

    def test_no_images_returns_empty_batch(self):
        """Samples without images are skipped, but the schema is preserved."""
        samples = {
            'text': [f'{SpecialTokens.image}a photo of a cat'],
            'caption': ['a photo of a cat'],
            'images': [[]],
        }
        op = self._build_op(keep_original_sample=False)
        res, prompts = self._run(op, samples)

        self.assertEqual(set(res.keys()), set(samples.keys()))
        for key in samples:
            self.assertEqual(res[key], [])
        self.assertEqual(prompts, [])


if __name__ == '__main__':
    unittest.main()

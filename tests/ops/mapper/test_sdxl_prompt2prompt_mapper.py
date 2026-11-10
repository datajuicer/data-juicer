import io
from PIL import Image
import unittest
from unittest.mock import MagicMock, patch

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.sdxl_prompt2prompt_mapper import SDXLPrompt2PromptMapper
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase
from data_juicer.utils.cache_utils import DATA_JUICER_ASSETS_CACHE


class SDXLPrompt2PromptMapperValidationTest(unittest.TestCase):

    @patch('data_juicer.ops.mapper.sdxl_prompt2prompt_mapper.p2p_pipeline', new_callable=MagicMock)
    @patch('data_juicer.ops.mapper.sdxl_prompt2prompt_mapper.prepare_model')
    def test_missing_text_key_second(self, prepare_model, p2p_pipeline):
        for kwargs in ({}, {'text_key_second': None}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, 'text_key_second is required'):
                    SDXLPrompt2PromptMapper(**kwargs)
        prepare_model.assert_not_called()
        self.assertEqual(p2p_pipeline.mock_calls, [])

    @patch('data_juicer.ops.mapper.sdxl_prompt2prompt_mapper.p2p_pipeline', new_callable=MagicMock)
    @patch('data_juicer.ops.mapper.sdxl_prompt2prompt_mapper.prepare_model')
    def test_valid_text_key_second(self, prepare_model, p2p_pipeline):
        for kwargs, expected_key in (({}, 'text'), ({'text_key': 'caption'}, 'caption')):
            with self.subTest(kwargs=kwargs):
                op = SDXLPrompt2PromptMapper(text_key_second='edited_caption', **kwargs)
                self.assertEqual(op.text_key, expected_key)
                self.assertEqual(op.text_key_second, 'edited_caption')
                self.assertEqual(op.model_key, prepare_model.return_value)
        self.assertEqual(prepare_model.call_count, 2)


class SDXLPrompt2PromptMapperTest(DataJuicerTestCaseBase):

    hf_diffusion = 'stabilityai/stable-diffusion-xl-base-1.0'

    text_key = 'text'

    text_key = "caption1"
    text_key_second = "caption2"

    @classmethod
    def tearDownClass(cls) -> None:
        super().tearDownClass(cls.hf_diffusion)

    def _run_sdxl_prompt2prompt(self):
        op = SDXLPrompt2PromptMapper(
            hf_diffusion=self.hf_diffusion,
            torch_dtype="fp16",
            text_key=self.text_key,
            text_key_second=self.text_key_second,
            output_dir=DATA_JUICER_ASSETS_CACHE
        )

        ds_list = [{self.text_key: "a chocolate cake",
                    self.text_key_second: "a confetti apple bread"},
                   {self.text_key: "a plane in the sky flying alone.",
                    self.text_key_second: "a boat in the ocean sailing alone."}]

        dataset = Dataset.from_list(ds_list)
        dataset = dataset.map(op.process, num_proc=1, with_rank=True)

        for temp_idx, sample in enumerate(dataset):
            self.assertIn("image_path1", sample)
            self.assertIn("image_path2", sample)
            print(sample)

    def test_sdxl_prompt2prompt(self):
        self._run_sdxl_prompt2prompt()


if __name__ == '__main__':
    unittest.main()
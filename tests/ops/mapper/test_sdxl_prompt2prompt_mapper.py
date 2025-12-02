import io
from PIL import Image
import unittest

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.sdxl_prompt2prompt_mapper import SDXLPrompt2PromptMapper
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase
from data_juicer.utils.cache_utils import DATA_JUICER_ASSETS_CACHE


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
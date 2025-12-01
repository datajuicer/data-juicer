import unittest
from data_juicer.ops.mapper.mllm_mapper import MllmMapper
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase
import os

class MllmMapperTest(DataJuicerTestCaseBase):

    hf_model = 'llava-hf/llava-v1.6-vicuna-7b-hf'

    text_key = 'text'
    image_key = "images"

    @classmethod
    def tearDownClass(cls) -> None:
        super().tearDownClass(cls.hf_model)

    def _run_mllm(self):
        op = MllmMapper(
            hf_model=self.hf_model,
            temperature=0.9,
            top_p=0.95,
            max_new_tokens=512
        )

        data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')
        img2_path = os.path.join(data_path, 'img2.jpg')
        img3_path = os.path.join(data_path, 'img3.jpg')
      
        samples = [
            {self.text_key: 'Describe this image.', self.image_key: [img2_path, img3_path]},
        ]

        for sample in samples:
            result = op.process(sample)
            self.assertIsInstance(sample[self.text_key], list)
            self.assertEqual(len(sample[self.text_key]), 2)
            print(f'Output results: {result}')

    def test_mllm(self):
        self._run_mllm()


if __name__ == '__main__':
    unittest.main()

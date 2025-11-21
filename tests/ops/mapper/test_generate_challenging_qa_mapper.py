import unittest
from copy import deepcopy
from data_juicer.ops.mapper.generate_challenging_qa_mapper import GenerateChallengingQAMapper
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class GenerateChallengingQAMapperTest(DataJuicerTestCaseBase):
    hf_model = 'Qwen/Qwen2.5-VL-7B-Instruct'
    category = 'Mathematical Reasoning'
    model_name = "Qwen"

    text_key = 'text'

    @classmethod
    def tearDownClass(cls) -> None:
        super().tearDownClass(cls.hf_model)

    def _run_generate_challenging_qa_mapper(self):
        op = GenerateChallengingQAMapper(
            hf_model = 'Qwen/Qwen2.5-VL-7B-Instruct',
            category = 'Mathematical Reasoning',
            model_name = "Qwen"
        )

        samples = [
            {self.text_key: ''}
        ]

        for sample in samples:
            result = op.process(deepcopy(sample))
            print(f'Output results: {result}')

    def test_generate_challenging_qa_mapper(self):
        self._run_generate_challenging_qa_mapper()


if __name__ == '__main__':
    unittest.main()

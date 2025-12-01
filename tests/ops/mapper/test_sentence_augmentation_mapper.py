import unittest
from copy import deepcopy
from data_juicer.ops.mapper.sentence_augmentation_mapper import SentenceAugmentationMapper
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class SentenceAugmentationMapperTest(DataJuicerTestCaseBase):

    hf_model = 'Qwen/Qwen2-7B-Instruct'

    text_key = "caption1"
    text_key_second = "caption2"

    @classmethod
    def tearDownClass(cls) -> None:
        super().tearDownClass(cls.hf_model)

    def _run_sentence_augmentation_mapper(self):
        op = SentenceAugmentationMapper(
            hf_model=self.hf_model,
            task_sentence="Please replace one entity in this sentence with "
                          "another entity, such as an animal, a vehicle, or a "
                          "piece of furniture. Please only answer with the "
                          "replaced sentence.",
            max_new_tokens=512,
            temperature=0.9,
            top_p=0.95,
            num_beams=1,
            text_key=self.text_key,
            text_key_second=self.text_key_second
        )

        samples = [
            {self.text_key: 'a book is near a cat and a dog'}
        ]

        for sample in samples:
            result = op.process(deepcopy(sample))
            print(f'Output results: {result}')
            self.assertNotEqual(sample, result)

    def test_sentence_augmentation_mapper(self):
        self._run_sentence_augmentation_mapper()


if __name__ == '__main__':
    unittest.main()

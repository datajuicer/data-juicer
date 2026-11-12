import unittest
from copy import deepcopy
from unittest.mock import MagicMock, patch

from data_juicer.ops.mapper.sentence_augmentation_mapper import SentenceAugmentationMapper
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class SentenceAugmentationMapperValidationTest(unittest.TestCase):

    @patch('data_juicer.ops.mapper.sentence_augmentation_mapper.prepare_model')
    def test_missing_task_sentence(self, prepare_model):
        for kwargs in ({}, {'task_sentence': None}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, 'task_sentence is required'):
                    SentenceAugmentationMapper(text_key_second='augmented', **kwargs)
        prepare_model.assert_not_called()

    @patch('data_juicer.ops.mapper.sentence_augmentation_mapper.prepare_model')
    def test_valid_task_sentence(self, prepare_model):
        for kwargs, expected_key in (({}, 'text'), ({'text_key': 'caption'}, 'caption')):
            with self.subTest(kwargs=kwargs):
                op = SentenceAugmentationMapper(
                    task_sentence='Rephrase the sentence.', text_key_second='augmented', **kwargs
                )
                self.assertEqual(op.task_sentence, 'Rephrase the sentence.')
                self.assertEqual(op.text_key, expected_key)
                self.assertEqual(op.text_key_second, 'augmented')
                self.assertEqual(op.model_key, prepare_model.return_value)
        self.assertEqual(prepare_model.call_count, 2)

    @patch('data_juicer.ops.mapper.sentence_augmentation_mapper.get_model')
    @patch('data_juicer.ops.mapper.sentence_augmentation_mapper.prepare_model')
    def test_augmentation_preserves_original_text(self, prepare_model, get_model):
        op = SentenceAugmentationMapper(
            task_sentence='Rephrase the sentence.', text_key_second='augmented'
        )
        model, processor = MagicMock(), MagicMock()
        get_model.return_value = (model, processor)
        inputs = processor.return_value.to.return_value
        inputs.input_ids.shape = (1, 2)
        inputs.input_ids.__ne__.return_value.sum.return_value.item.return_value = 0
        processor.batch_decode.return_value = ['  A cat is sleeping.  ']
        sample = {'text': 'The cat sleeps.', 'augmented': 'Previous content.'}

        result = op.process_single(sample)

        self.assertIs(result, sample)
        self.assertEqual(result['text'], 'The cat sleeps.')
        self.assertEqual(result['augmented'], 'A cat is sleeping.')
        model.generate.assert_called_once()


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

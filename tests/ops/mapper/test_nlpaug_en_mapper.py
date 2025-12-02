# flake8: noqa: E501

import unittest

from data_juicer.core import NestedDataset as Dataset
from data_juicer.ops.mapper.nlpaug_en_mapper import NlpaugEnMapper
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class NlpaugEnMapperTest(DataJuicerTestCaseBase):

    def setUp(self):
        super().setUp()

        self.samples = Dataset.from_dict({
            'text': [
                'I am a deep learning engineer. I love LLM.',
                'A short test with numbers 2023'
            ],
            'meta': ['meta information', 'meta information with numbers'],
        })

    def test_number_of_generated_samples_with_sequential_on(self):
        aug_num = 3
        aug_method_num = 3
        op = NlpaugEnMapper(
            sequential=True,
            aug_num=aug_num,
            delete_random_word=True,
            swap_random_char=True,
            spelling_error_word=True,
        )
        self.assertEqual(len(op.aug), aug_method_num)
        result = self.samples.map(op.process)
        self.assertEqual(len(result['text']),
                         (aug_num + 1) * len(self.samples))
        self.assertEqual(len(result['meta']), len(result['text']))

    def test_number_of_generated_samples_with_sequential_off(self):
        aug_num = 3
        aug_method_num = 3
        op = NlpaugEnMapper(
            sequential=False,
            aug_num=aug_num,
            delete_random_word=True,
            swap_random_char=True,
            spelling_error_word=True,
        )
        self.assertEqual(len(op.aug), aug_method_num)
        result = self.samples.map(op.process)
        self.assertEqual(len(result['text']),
                         (aug_num * aug_method_num + 1) * len(self.samples))
        self.assertEqual(len(result['meta']), len(result['text']))

    def test_zero_aug_methods_with_sequential_on(self):
        aug_num = 3
        aug_method_num = 0
        # sequential on
        op = NlpaugEnMapper(
            sequential=True,
            aug_num=aug_num,
        )
        self.assertEqual(len(op.aug), aug_method_num)
        result = self.samples.map(op.process)
        self.assertEqual(len(result['text']), len(self.samples))
        self.assertEqual(len(result['meta']), len(result['text']))

    def test_zero_aug_methods_with_sequential_off(self):
        aug_num = 3
        aug_method_num = 0
        # sequential off
        op = NlpaugEnMapper(
            sequential=False,
            aug_num=aug_num,
        )
        self.assertEqual(len(op.aug), aug_method_num)
        result = self.samples.map(op.process)
        self.assertEqual(len(result['text']), len(self.samples))
        self.assertEqual(len(result['meta']), len(result['text']))

    def test_all_aug_methods_with_sequential_on(self):
        aug_num = 3
        aug_method_num = 9
        # sequential on
        op = NlpaugEnMapper(
            sequential=True,
            aug_num=aug_num,
            delete_random_word=True,
            swap_random_word=True,
            spelling_error_word=True,
            split_random_word=True,
            keyboard_error_char=True,
            ocr_error_char=True,
            delete_random_char=True,
            swap_random_char=True,
            insert_random_char=True,
        )
        self.assertEqual(len(op.aug), aug_method_num)
        result = self.samples.map(op.process)
        self.assertEqual(len(result['text']),
                         (aug_num + 1) * len(self.samples))
        self.assertEqual(len(result['meta']), len(result['text']))

    def test_all_aug_methods_with_sequential_off(self):
        aug_num = 3
        aug_method_num = 9
        # sequential off
        op = NlpaugEnMapper(
            sequential=False,
            aug_num=aug_num,
            delete_random_word=True,
            swap_random_word=True,
            spelling_error_word=True,
            split_random_word=True,
            keyboard_error_char=True,
            ocr_error_char=True,
            delete_random_char=True,
            swap_random_char=True,
            insert_random_char=True,
        )
        self.assertEqual(len(op.aug), aug_method_num)
        result = self.samples.map(op.process)
        self.assertEqual(len(result['text']),
                         (aug_num * aug_method_num + 1) * len(self.samples))
        self.assertEqual(len(result['meta']), len(result['text']))

    def test_number_of_generated_samples_with_sequential_on_remove_original_sample(
            self):
        aug_num = 3
        aug_method_num = 3
        op = NlpaugEnMapper(
            sequential=True,
            aug_num=aug_num,
            keep_original_sample=False,
            delete_random_word=True,
            swap_random_char=True,
            spelling_error_word=True,
        )
        self.assertEqual(len(op.aug), aug_method_num)
        result = self.samples.map(op.process)
        self.assertEqual(len(result['text']), aug_num * len(self.samples))
        self.assertEqual(len(result['meta']), len(result['text']))

    def test_number_of_generated_samples_with_sequential_off_remove_original_sample(
            self):
        aug_num = 3
        aug_method_num = 3
        op = NlpaugEnMapper(
            sequential=False,
            aug_num=aug_num,
            keep_original_sample=False,
            delete_random_word=True,
            swap_random_char=True,
            spelling_error_word=True,
        )
        self.assertEqual(len(op.aug), aug_method_num)
        result = self.samples.map(op.process)
        self.assertEqual(len(result['text']),
                         aug_num * aug_method_num * len(self.samples))
        self.assertEqual(len(result['meta']), len(result['text']))

    def test_zero_aug_methods_with_sequential_on_remove_original_sample(self):
        aug_num = 3
        aug_method_num = 0
        # sequential on
        op = NlpaugEnMapper(
            sequential=True,
            aug_num=aug_num,
            keep_original_sample=False,
        )
        self.assertEqual(len(op.aug), aug_method_num)
        result = self.samples.map(op.process)
        self.assertEqual(len(result['text']), 0)
        self.assertEqual(len(result['meta']), len(result['text']))

    def test_zero_aug_methods_with_sequential_off_remove_original_sample(self):
        aug_num = 3
        aug_method_num = 0
        # sequential off
        op = NlpaugEnMapper(
            sequential=False,
            aug_num=aug_num,
            keep_original_sample=False,
        )
        self.assertEqual(len(op.aug), aug_method_num)
        result = self.samples.map(op.process)
        self.assertEqual(len(result['text']), 0)
        self.assertEqual(len(result['meta']), len(result['text']))

    def test_all_aug_methods_with_sequential_on_remove_original_sample(self):
        aug_num = 3
        aug_method_num = 9
        # sequential on
        op = NlpaugEnMapper(
            sequential=True,
            aug_num=aug_num,
            keep_original_sample=False,
            delete_random_word=True,
            swap_random_word=True,
            spelling_error_word=True,
            split_random_word=True,
            keyboard_error_char=True,
            ocr_error_char=True,
            delete_random_char=True,
            swap_random_char=True,
            insert_random_char=True,
        )
        self.assertEqual(len(op.aug), aug_method_num)
        result = self.samples.map(op.process)
        self.assertEqual(len(result['text']), aug_num * len(self.samples))
        self.assertEqual(len(result['meta']), len(result['text']))

    def test_all_aug_methods_with_sequential_off_remove_original_sample(self):
        aug_num = 3
        aug_method_num = 9
        # sequential off
        op = NlpaugEnMapper(
            sequential=False,
            aug_num=aug_num,
            keep_original_sample=False,
            delete_random_word=True,
            swap_random_word=True,
            spelling_error_word=True,
            split_random_word=True,
            keyboard_error_char=True,
            ocr_error_char=True,
            delete_random_char=True,
            swap_random_char=True,
            insert_random_char=True,
        )
        self.assertEqual(len(op.aug), aug_method_num)
        result = self.samples.map(op.process)
        self.assertEqual(len(result['text']),
                         aug_num * aug_method_num * len(self.samples))
        self.assertEqual(len(result['meta']), len(result['text']))


if __name__ == '__main__':
    unittest.main()

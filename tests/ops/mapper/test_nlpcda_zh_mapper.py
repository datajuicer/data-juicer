# flake8: noqa: E501

import unittest

from data_juicer.core import NestedDataset as Dataset
from data_juicer.ops.mapper.nlpcda_zh_mapper import NlpcdaZhMapper
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

@unittest.skip('unknown UnicodeEncodeError')
class NlpaugEnMapperTest(DataJuicerTestCaseBase):

    def setUp(self):
        super().setUp()

        self.samples = Dataset.from_dict({
            'text': ['这里一共有5种不同的数据增强方法', '这是不带数字的测试样例'],
            'meta': ['meta information', 'meta information without numbers'],
        })

    def test_create_number_with_sequential_on(self):
        aug_num = 3
        aug_method_num = 3
        op = NlpcdaZhMapper(
            sequential=True,
            aug_num=aug_num,
            replace_similar_word=True,
            replace_homophone_char=True,
            delete_random_char=True,
        )
        create_nums = [method.create_num for method in op.aug_pipeline]
        target_nums = [aug_num + 1] + [2] * (aug_method_num - 1)
        self.assertEqual(create_nums, target_nums)

    def test_create_number_with_sequential_off(self):
        aug_num = 3
        aug_method_num = 3
        op = NlpcdaZhMapper(
            sequential=False,
            aug_num=aug_num,
            replace_similar_word=True,
            replace_homophone_char=True,
            delete_random_char=True,
        )
        create_nums = [method.create_num for method in op.aug_pipeline]
        target_nums = [aug_num + 1] * aug_method_num
        self.assertEqual(create_nums, target_nums)

    def test_number_of_generated_samples_with_sequential_on(self):
        aug_num = 3
        aug_method_num = 3
        op = NlpcdaZhMapper(
            sequential=True,
            aug_num=aug_num,
            replace_similar_word=True,
            replace_homophone_char=True,
            delete_random_char=True,
        )
        self.assertEqual(len(op.aug_pipeline), aug_method_num)
        result = self.samples.map(op.process)
        self.assertLessEqual(len(result['text']),
                             (aug_num + 1) * len(self.samples['text']))
        self.assertGreaterEqual(len(result['text']), len(self.samples['text']))
        self.assertEqual(len(result['meta']), len(result['text']))

    def test_number_of_generated_samples_with_sequential_off(self):
        aug_num = 3
        aug_method_num = 3
        op = NlpcdaZhMapper(
            sequential=False,
            aug_num=aug_num,
            replace_similar_word=True,
            replace_homophone_char=True,
            delete_random_char=True,
        )
        self.assertEqual(len(op.aug_pipeline), aug_method_num)
        result = self.samples.map(op.process)
        self.assertLessEqual(len(result['text']),
                             (aug_num * aug_method_num + 1) *
                             len(self.samples['text']))
        self.assertGreaterEqual(len(result['text']), len(self.samples['text']))
        self.assertEqual(len(result['meta']), len(result['text']))

    def test_zero_aug_methods_with_sequential_on(self):
        aug_num = 3
        aug_method_num = 0
        # sequential on
        op = NlpcdaZhMapper(
            sequential=True,
            aug_num=aug_num,
        )
        self.assertEqual(len(op.aug_pipeline), aug_method_num)
        result = self.samples.map(op.process)
        self.assertEqual(len(result['text']), len(self.samples['text']))
        self.assertEqual(len(result['meta']), len(result['text']))

    def test_zero_aug_methods_with_sequential_off(self):
        aug_num = 3
        aug_method_num = 0
        # sequential off
        op = NlpcdaZhMapper(
            sequential=False,
            aug_num=aug_num,
        )
        self.assertEqual(len(op.aug_pipeline), aug_method_num)
        result = self.samples.map(op.process)
        self.assertEqual(len(result['text']), len(self.samples['text']))
        self.assertEqual(len(result['meta']), len(result['text']))

    def test_all_aug_methods_with_sequential_on(self):
        aug_num = 3
        aug_method_num = 5
        # sequential on
        op = NlpcdaZhMapper(
            sequential=True,
            aug_num=aug_num,
            replace_similar_word=True,
            replace_homophone_char=True,
            delete_random_char=True,
            swap_random_char=True,
            replace_equivalent_num=True,
        )
        self.assertEqual(len(op.aug_pipeline), aug_method_num)
        result = self.samples.map(op.process)
        self.assertLessEqual(len(result['text']),
                             (aug_num + 1) * len(self.samples['text']))
        self.assertGreaterEqual(len(result['text']), len(self.samples['text']))
        self.assertEqual(len(result['meta']), len(result['text']))

    def test_all_aug_methods_with_sequential_off(self):
        aug_num = 3
        aug_method_num = 5
        # sequential off
        op = NlpcdaZhMapper(
            sequential=False,
            aug_num=aug_num,
            replace_similar_word=True,
            replace_homophone_char=True,
            delete_random_char=True,
            swap_random_char=True,
            replace_equivalent_num=True,
        )
        self.assertEqual(len(op.aug_pipeline), aug_method_num)
        result = self.samples.map(op.process)
        self.assertLessEqual(len(result['text']),
                             (aug_num * aug_method_num + 1) *
                             len(self.samples['text']))
        self.assertGreaterEqual(len(result['text']), len(self.samples['text']))
        self.assertEqual(len(result['meta']), len(result['text']))

    def test_number_of_generated_samples_with_sequential_on_remove_original_sample(
            self):
        aug_num = 3
        aug_method_num = 3
        op = NlpcdaZhMapper(
            sequential=True,
            aug_num=aug_num,
            keep_original_sample=False,
            replace_similar_word=True,
            replace_homophone_char=True,
            delete_random_char=True,
        )
        self.assertEqual(len(op.aug_pipeline), aug_method_num)
        result = self.samples.map(op.process)
        self.assertLessEqual(len(result['text']),
                             aug_num * len(self.samples['text']))
        self.assertGreaterEqual(len(result['text']), len(self.samples['text']))
        self.assertEqual(len(result['meta']), len(result['text']))

    def test_number_of_generated_samples_with_sequential_off_remove_original_sample(
            self):
        aug_num = 3
        aug_method_num = 3
        op = NlpcdaZhMapper(
            sequential=False,
            aug_num=aug_num,
            keep_original_sample=False,
            replace_similar_word=True,
            replace_homophone_char=True,
            delete_random_char=True,
        )
        self.assertEqual(len(op.aug_pipeline), aug_method_num)
        result = self.samples.map(op.process)
        self.assertLessEqual(
            len(result['text']),
            aug_num * aug_method_num * len(self.samples['text']))
        self.assertGreaterEqual(len(result['text']), len(self.samples['text']))
        self.assertEqual(len(result['meta']), len(result['text']))

    def test_zero_aug_methods_with_sequential_on_remove_original_sample(self):
        aug_num = 3
        aug_method_num = 0
        # sequential on
        op = NlpcdaZhMapper(
            sequential=True,
            aug_num=aug_num,
            keep_original_sample=False,
        )
        self.assertEqual(len(op.aug_pipeline), aug_method_num)
        result = self.samples.map(op.process)
        self.assertEqual(len(result['text']), 0)
        self.assertEqual(len(result['meta']), len(result['text']))

    def test_zero_aug_methods_with_sequential_off_remove_original_sample(self):
        aug_num = 3
        aug_method_num = 0
        # sequential off
        op = NlpcdaZhMapper(
            sequential=False,
            aug_num=aug_num,
            keep_original_sample=False,
        )
        self.assertEqual(len(op.aug_pipeline), aug_method_num)
        result = self.samples.map(op.process)
        self.assertEqual(len(result['text']), 0)
        self.assertEqual(len(result['meta']), len(result['text']))

    def test_all_aug_methods_with_sequential_on_remove_original_sample(self):
        aug_num = 3
        aug_method_num = 5
        # sequential on
        op = NlpcdaZhMapper(
            sequential=True,
            aug_num=aug_num,
            keep_original_sample=False,
            replace_similar_word=True,
            replace_homophone_char=True,
            delete_random_char=True,
            swap_random_char=True,
            replace_equivalent_num=True,
        )
        self.assertEqual(len(op.aug_pipeline), aug_method_num)
        result = self.samples.map(op.process)
        self.assertLessEqual(len(result['text']),
                             aug_num * len(self.samples['text']))
        self.assertGreaterEqual(len(result['text']), len(self.samples['text']))
        self.assertEqual(len(result['meta']), len(result['text']))

    def test_all_aug_methods_with_sequential_off_remove_original_sample(self):
        aug_num = 3
        aug_method_num = 5
        # sequential off
        op = NlpcdaZhMapper(
            sequential=False,
            aug_num=aug_num,
            keep_original_sample=False,
            replace_similar_word=True,
            replace_homophone_char=True,
            delete_random_char=True,
            swap_random_char=True,
            replace_equivalent_num=True,
        )
        self.assertEqual(len(op.aug_pipeline), aug_method_num)
        result = self.samples.map(op.process)
        self.assertLessEqual(
            len(result['text']),
            aug_num * aug_method_num * len(self.samples['text']))
        self.assertGreaterEqual(len(result['text']), len(self.samples['text']))
        self.assertEqual(len(result['meta']), len(result['text']))


if __name__ == '__main__':
    unittest.main()

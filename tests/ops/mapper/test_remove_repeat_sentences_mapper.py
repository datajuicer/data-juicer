# flake8: noqa: E501

import unittest

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.remove_repeat_sentences_mapper import \
    RemoveRepeatSentencesMapper
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class RemoveRepeatSentencesMapperTest(DataJuicerTestCaseBase):

    def _run_helper(self, samples, op):
        dataset = Dataset.from_list(samples)
        dataset = dataset.map(op.process, batch_size=2)
                
        for data in dataset:
            self.assertEqual(data['text'], data['target'])

    def test_text(self):

        samples = [{
            'text':
            '今天天气真不错，阳光明媚，适合出去散步。小明说：“今天天气真不错，我们去海边吧。” 小红回答说：“好主意！” 但是，小李觉得：“今天天气真不错，我们去爬山吧。” 今天天气真不错，阳光明媚，适合出去散步。昨天下了一整天的雨，今天终于放晴了。昨天下了一整天的雨，今天终于放晴了。',
            'target':
            '今天天气真不错，阳光明媚，适合出去散步。小明说：“今天天气真不错，我们去海边吧。” 小红回答说：“好主意！” 但是，小李觉得：“今天天气真不错，我们去爬山吧。”昨天下了一整天的雨，今天终于放晴了。',
        }, {
            'text':
            'The quick brown fox jumps over the lazy dog. Isn\'t it amazing how a simple sentence can contain every letter of the alphabet? The quick brown fox jumps over the lazy dog. Speaking of weather, yesterday was quite dreary; however, today is absolutely delightful. Isn\'t it amazing how a simple sentence can contain every letter of the alphabet? "Let\'s seize the day," Tom exclaimed, full of enthusiasm. "Let\'s seize the day," Tom exclaimed, full of enthusiasm.',
            'target':
            'The quick brown fox jumps over the lazy dog. Isn\'t it amazing how a simple sentence can contain every letter of the alphabet? Speaking of weather, yesterday was quite dreary; however, today is absolutely delightful. "Let\'s seize the day," Tom exclaimed, full of enthusiasm.'
        }, {
            'text': '''我很开心 。但是你不开心  。我很开心 。\n你好呀！我很开心 。我好的。你好呀！''',
            'target': '''我很开心 。但是你不开心  。\n你好呀！我好的。'''
        }, {
            'text':
            '默认配置下，长度低于2的句子不会被去重。去重？去重。去重！重。重...... 重! 1234？3215. 1234. 3. 3. 3',
            'target':
            '默认配置下，长度低于2的句子不会被去重。去重？重。重...... 重! 1234？3215. 3. 3. 3'
        }]

        op = RemoveRepeatSentencesMapper()
        self._run_helper(samples, op)

    def test_text2(self):

        samples = [{
            'text':
            'Life is what happens when you\'re busy making other plans. John Lennon once said. Life is what happens when you\'re busy making other plans. This phrase has resonated with many people over the years. 人生就是当你忙于制定其他计划时发生的事情。对很多人来说，这句话引起了共鸣。',
            'target':
            'Life is what happens when you\'re busy making other plans. John Lennon once said. This phrase has resonated with many people over the years. 人生就是当你忙于制定其他计划时发生的事情。对很多人来说，这句话引起了共鸣。',
        }, {
            'text':
            'The quick brown fox jumps over the lazy dog. Isn\'t it amazing how a simple sentence can contain every letter of the alphabet? The quick brown fox jumps over the lazy dog. Speaking of weather, yesterday was quite dreary; however, today is absolutely delightful. Isn\'t it amazing how a simple sentence can contain every letter of the alphabet? "Let\'s seize the day," Tom exclaimed, full of enthusiasm. "Let\'s seize the day," Tom exclaimed, full of enthusiasm.',
            'target':
            'The quick brown fox jumps over the lazy dog. Isn\'t it amazing how a simple sentence can contain every letter of the alphabet? Speaking of weather, yesterday was quite dreary; however, today is absolutely delightful. "Let\'s seize the day," Tom exclaimed, full of enthusiasm.'
        }, {
            'text': '''我很开心 。但是你不开心  。我很开心 。\n你好呀！我很开心 。我好的。你好呀！''',
            'target': '''我很开心 。但是你不开心  。\n你好呀！我好的。你好呀！'''
        }, {
            'text': '去重？去重。去重！重。重...... 重! 1234？3215. 1234. 3. 3. 3',
            'target': '去重？去重。去重！重。重...... 重! 1234？3215. 1234. 3. 3. 3'
        }]

        op = RemoveRepeatSentencesMapper(lowercase=True,
                                         ignore_special_character=False,
                                         min_repeat_sentence_length=5)
        self._run_helper(samples, op)


if __name__ == '__main__':
    unittest.main()

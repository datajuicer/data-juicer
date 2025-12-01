import unittest

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.replace_content_mapper import ReplaceContentMapper
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class ReplaceContentMapperTest(DataJuicerTestCaseBase):

    def _run_helper(self, op, samples):
        dataset = Dataset.from_list(samples)
        dataset = dataset.map(op.process, batch_size=2)
                
        for data in dataset:
            self.assertEqual(data['text'], data['target'])

    def test_special_char_pattern_text(self):

        samples = [
            {
                'text': '这是一个干净的文本。Including Chinese and English.',
                'target': '这是一个干净的文本。Including Chinese and English.',
            },
            {
                'text': '◆●■►▼▲▴∆▻▷❖♡□',
                'target': '◆<SPEC>►▼▲▴∆▻▷❖♡□',
            },
            {
                'text': '多个●■►▼这样的特殊字符可以►▼▲▴∆吗？',
                'target': '多个<SPEC>►▼这样的特殊字符可以►▼▲▴∆吗？',
            },
            {
                'text': '未指定的●■☛₨➩►▼▲特殊字符会☻▷❖被删掉吗？？',
                'target': '未指定的<SPEC>☛₨➩►▼▲特殊字符会☻▷❖被删掉吗？？',
            },
        ]
        op = ReplaceContentMapper(pattern='●■', repl='<SPEC>')
        self._run_helper(op, samples)

    def test_raw_digit_pattern_text(self):

        samples = [
            {
                'text': '这是一个123。Including 456 and English.',
                'target': '这是一个<DIGIT>。Including <DIGIT> and English.',
            },
        ]
        op = ReplaceContentMapper(pattern=r'\d+(?:,\d+)*', repl='<DIGIT>')
        self._run_helper(op, samples)

    def test_regular_digit_pattern_text(self):

        samples = [
            {
                'text': '这是一个123。Including 456 and English.',
                'target': '这是一个<DIGIT>。Including <DIGIT> and English.',
            },
        ]
        op = ReplaceContentMapper(pattern='\\d+(?:,\\d+)*', repl='<DIGIT>')
        self._run_helper(op, samples)


if __name__ == '__main__':
    unittest.main()

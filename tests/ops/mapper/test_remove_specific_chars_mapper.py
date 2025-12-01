import unittest

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.remove_specific_chars_mapper import \
    RemoveSpecificCharsMapper
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class RemoveSpecificCharsMapperTest(DataJuicerTestCaseBase):

    def setUp(self):
        super().setUp()
        self.op = RemoveSpecificCharsMapper()

    def _run_helper(self, samples):
        dataset = Dataset.from_list(samples)
        dataset = dataset.map(self.op.process, batch_size=2)
                
        for data in dataset:
            self.assertEqual(data['text'], data['target'])

    def test_complete_html_text(self):

        samples = [
            {
                'text': '这是一个干净的文本。Including Chinese and English.',
                'target': '这是一个干净的文本。Including Chinese and English.',
            },
            {
                'text': '◆●■►▼▲▴∆▻▷❖♡□',
                'target': '',
            },
            {
                'text': '►This is a dirty text ▻ 包括中文和英文',
                'target': 'This is a dirty text  包括中文和英文',
            },
            {
                'text': '多个●■►▼这样的特殊字符可以►▼▲▴∆吗？',
                'target': '多个这样的特殊字符可以吗？',
            },
            {
                'text': '未指定的●■☛₨➩►▼▲特殊字符会☻▷❖被删掉吗？？',
                'target': '未指定的☛₨➩特殊字符会☻被删掉吗？？',
            },
        ]
        self._run_helper(samples)


if __name__ == '__main__':
    unittest.main()

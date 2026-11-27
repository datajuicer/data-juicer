import unittest

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.clean_copyright_mapper import CleanCopyrightMapper
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class CleanCopyrightMapperTest(DataJuicerTestCaseBase):

    def setUp(self):
        super().setUp()

        self.op = CleanCopyrightMapper()

    def _run_clean_copyright(self, samples):
        dataset = Dataset.from_list(samples)
        dataset = dataset.map(self.op.process, batch_size=2)
                
        for data in dataset:
            self.assertEqual(data['text'], data['target'])

    def test_preserves_exact_line_boundaries(self):
        cases = [
            ("\n// header\r\n\n# header\nbody\r\n# keep\n\n", "body\r\n# keep\n\n"),
            ("\r\n# keep\nbody", "\r\n# keep\nbody"),
            (" # keep\nbody", " # keep\nbody"),
            ("\n-- no final newline", ""),
            ("\n\n", ""),
            ("# keep\nbody /* non-license comment */\n", "# keep\nbody /* non-license comment */\n"),
            ("# keep\nbody /* Copyright 2025 */\n", "# keep\nbody \n"),
        ]
        self._run_clean_copyright([{"text": text, "target": target} for text, target in cases])

    def test_clean_copyright(self):

        samples = [{
            'text': '这是一段 /* 多行注释\n注释内容copyright\n*/ 的文本。另外还有一些 // 单行注释。',
            'target': '这是一段  的文本。另外还有一些 // 单行注释。'
        }, {
            'text': '如果多行/*注释中没有\n关键词,那么\n这部分注释也不会\n被清除*/\n会保留下来',
            'target': '如果多行/*注释中没有\n关键词,那么\n这部分注释也不会\n被清除*/\n会保留下来'
        }, {
            'text': '//if start with\n//that will be cleaned \n evenly',
            'target': ' evenly'
        }, {
            'text': 'http://www.nasosnsncc.com',
            'target': 'http://www.nasosnsncc.com'
        }, {
            'text': '#if start with\nthat will be cleaned \n#evenly',
            'target': 'that will be cleaned \n#evenly'
        }, {
            'text': '--if start with\n--that will be cleaned \n#evenly',
            'target': ''
        }]
        self._run_clean_copyright(samples)


if __name__ == '__main__':
    unittest.main()

import unittest

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.whitespace_normalization_mapper import \
    WhitespaceNormalizationMapper
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class WhitespaceNormalizationMapperTest(DataJuicerTestCaseBase):

    def setUp(self):
        super().setUp()
        self.op = WhitespaceNormalizationMapper()

    def _run_whitespace_normalization(self, samples):
        dataset = Dataset.from_list(samples)
        dataset = dataset.map(self.op.process, batch_size=2)
                
        for data in dataset:
            self.assertEqual(data['text'], data['target'])

    def test_case(self):

        samples = [{
            'text': 'x \t              　\u200B\u200C\u200D\u2060￼\u0084y',
            'target': 'x                       y'
        }]

        self._run_whitespace_normalization(samples)


if __name__ == '__main__':
    unittest.main()

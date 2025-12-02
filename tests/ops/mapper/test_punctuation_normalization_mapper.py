import unittest

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.punctuation_normalization_mapper import \
    PunctuationNormalizationMapper
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class PunctuationNormalizationMapperTest(DataJuicerTestCaseBase):

    def setUp(self):
        super().setUp()
        self.op = PunctuationNormalizationMapper()

    def _run_punctuation_normalization(self, samples):
        dataset = Dataset.from_list(samples)
        dataset = dataset.map(self.op.process, batch_size=2)
                
        for data in dataset:
            self.assertEqual(data['text'], data['target'])

    def test_case(self):

        samples = [{
            'text':
            '，。、„”“«»１」「《》´∶：？！（）；–—．～’…━〈〉【】％►',
            'target':
            ",.,\"\"\"\"\"\"\"\"\"\"'::?!();- - . ~'...-<>[]%-"
        }]

        self._run_punctuation_normalization(samples)


if __name__ == '__main__':
    unittest.main()

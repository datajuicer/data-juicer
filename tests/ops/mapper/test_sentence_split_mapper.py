import unittest

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.sentence_split_mapper import SentenceSplitMapper
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class SentenceSplitMapperTest(DataJuicerTestCaseBase):

    def _run_helper(self, op, samples):
        dataset = Dataset.from_list(samples)
        dataset = dataset.map(op.process, batch_size=2)
                
        for data in dataset:
            self.assertEqual(data['text'], data['target'])

    def test_en_text(self):

        samples = [
            {
                'text':
                'Smithfield employs 3,700 people at its plant in Sioux Falls, '
                'South Dakota. The plant slaughters 19,500 pigs a day — 5 '
                'percent of U.S. pork.',
                'target':
                'Smithfield employs 3,700 people at its plant in Sioux Falls, '
                'South Dakota.\nThe plant slaughters 19,500 pigs a day — 5 '
                'percent of U.S. pork.'
            },
        ]
        op = SentenceSplitMapper('en')
        self._run_helper(op, samples)

    def test_fr_text(self):

        samples = [
            {
                'text':
                'Smithfield emploie 3,700 personnes dans son usine de'
                ' Sioux Falls, dans le Dakota du Sud. L\'usine '
                'abat 19 500 porcs par jour, soit 5 % du porc américain.',
                'target':
                'Smithfield emploie 3,700 personnes dans son usine de'
                ' Sioux Falls, dans le Dakota du Sud.\nL\'usine '
                'abat 19 500 porcs par jour, soit 5 % du porc américain.'
            },
        ]
        op = SentenceSplitMapper('fr')
        self._run_helper(op, samples)

    def test_pt_text(self):

        samples = [
            {
                'text':
                'A Smithfield emprega 3.700 pessoas em sua fábrica em '
                'Sioux Falls, Dakota do Sul. A fábrica '
                'abate 19.500 porcos por dia – 5% da carne suína dos EUA.',
                'target':
                'A Smithfield emprega 3.700 pessoas em sua fábrica em '
                'Sioux Falls, Dakota do Sul.\nA fábrica abate 19.500 '
                'porcos por dia – 5% da carne suína dos EUA.'
            },
        ]
        op = SentenceSplitMapper('pt')
        self._run_helper(op, samples)

    def test_es_text(self):

        samples = [
            {
                'text':
                'Smithfield emplea a 3.700 personas en su planta de '
                'Sioux Falls, Dakota del Sur. La planta sacrifica 19.500 '
                'cerdos al día, el 5 por ciento de la carne de cerdo de EE.',
                'target':
                'Smithfield emplea a 3.700 personas en su planta de Sioux '
                'Falls, Dakota del Sur.\nLa planta sacrifica 19.500 cerdos '
                'al día, el 5 por ciento de la carne de cerdo de EE.'
            },
        ]
        op = SentenceSplitMapper('es')
        self._run_helper(op, samples)


if __name__ == '__main__':
    unittest.main()

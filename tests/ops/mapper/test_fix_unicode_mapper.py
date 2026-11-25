import unittest

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.fix_unicode_mapper import FixUnicodeMapper
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class FixUnicodeMapperTest(DataJuicerTestCaseBase):

    def setUp(self):
        super().setUp()

        self.op = FixUnicodeMapper()

    def _run_fix_unicode(self, samples):
        dataset = Dataset.from_list(samples)
        dataset = dataset.map(self.op.process, batch_size=2)
                                  
        for data in dataset:
            self.assertEqual(data['text'], data['target'])

    def test_bad_unicode_text(self):

        samples = [
            {
                'text': 'âœ” No problems',
                'target': '✔ No problems'
            },
            {
                'text':
                'The Mona Lisa doesnÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢t have eyebrows.',
                'target': 'The Mona Lisa doesn\'t have eyebrows.'
            },
        ]

        self._run_fix_unicode(samples)

    def test_good_unicode_text(self):
        samples = [
            {
                'text': 'No problems',
                'target': 'No problems'
            },
            {
                'text': '阿里巴巴',
                'target': '阿里巴巴'
            },
        ]
        self._run_fix_unicode(samples)


    def test_custom_normalization_nfkc(self):
        """Custom normalization mode NFKC should work."""
        op = FixUnicodeMapper(normalization='nfkc')
        samples = [{'text': 'ﬁ', 'target': 'fi'}]  # ﬁ ligature → fi in NFKC
        dataset = Dataset.from_list(samples)
        dataset = dataset.map(op.process, batch_size=2)
        for data in dataset:
            self.assertEqual(data['text'], data['target'])

    def test_invalid_normalization_raises(self):
        with self.assertRaises(ValueError):
            FixUnicodeMapper(normalization='INVALID')

    def test_empty_normalization_defaults_nfc(self):
        """Empty string normalization should default to NFC."""
        op = FixUnicodeMapper(normalization='')
        self.assertEqual(op.normalization, 'NFC')


if __name__ == '__main__':
    unittest.main()

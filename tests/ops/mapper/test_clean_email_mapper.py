import unittest

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.clean_email_mapper import CleanEmailMapper
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class CleanEmailMapperTest(DataJuicerTestCaseBase):

    def _run_clean_email(self, op, samples):
        dataset = Dataset.from_list(samples)
        dataset = dataset.map(op.process, batch_size=2)
                
        for data in dataset:
            self.assertEqual(data['text'], data['target'])

    def test_clean_email(self):

        samples = [{
            'text': 'happy day euqdh@cjqi.com',
            'target': 'happy day '
        }, {
            'text': '请问你是谁dasoidhao@1264fg.45om',
            'target': '请问你是谁dasoidhao@1264fg.45om'
        }, {
            'text': 'ftp://examplema-nièrdash@hqbchd.ckdhnfes.cds',
            'target': 'ftp://examplema-niè'
        }, {
            'text': '👊23da44sh12@46hqb12chd.ckdhnfes.comd.dasd.asd.dc',
            'target': '👊'
        }]
        op = CleanEmailMapper()
        self._run_clean_email(op, samples)

    def test_replace_email(self):

        samples = [{
            'text': 'happy day euqdh@cjqi.com',
            'target': 'happy day <EMAIL>'
        }, {
            'text': '请问你是谁dasoidhao@1264fg.45om',
            'target': '请问你是谁dasoidhao@1264fg.45om'
        }, {
            'text': 'ftp://examplema-nièrdash@hqbchd.ckdhnfes.cds',
            'target': 'ftp://examplema-niè<EMAIL>'
        }, {
            'text': '👊23da44sh12@46hqb12chd.ckdhnfes.comd.dasd.asd.dc',
            'target': '👊<EMAIL>'
        }]
        op = CleanEmailMapper(repl='<EMAIL>')
        self._run_clean_email(op, samples)


    def test_custom_pattern(self):
        """Custom pattern must produce different results from default pattern.

        Default pattern: [A-Za-z0-9.\\-+_]+@[a-z0-9.\\-+_]+\\.[a-z]+
        Custom pattern below also matches {user}@host forms (curly braces).
        The input '{admin}@srv.co' does NOT match the default pattern (because
        '{' and '}' are not in the default character class), but DOES match
        the custom one — proving the custom pattern is actually used.
        """
        input_text = 'Contact: {admin}@srv.co for info'
        # Verify default pattern does NOT match this input
        default_op = CleanEmailMapper(repl='<MAIL>')
        ds = Dataset.from_list([{'text': input_text}])
        default_result = ds.map(default_op.process, batch_size=2)
        self.assertEqual(default_result[0]['text'], input_text,
                         "Precondition failed: default pattern should NOT "
                         "match '{admin}@srv.co'")

        # Now verify custom pattern DOES match
        samples = [{
            'text': input_text,
            'target': 'Contact: <MAIL> for info',
        }]
        custom_op = CleanEmailMapper(
            pattern=r"r'[A-Za-z0-9.+_{}\-]+@[a-z0-9.+_\-]+\.[a-z]+'",
            repl='<MAIL>',
        )
        self._run_clean_email(custom_op, samples)

    def test_no_email_unchanged(self):
        samples = [{
            'text': 'No emails here!',
            'target': 'No emails here!',
        }]
        op = CleanEmailMapper()
        self._run_clean_email(op, samples)

    def test_batched_process(self):
        """Ensure process_batched is exercised via generate_dataset + run_single_op."""
        ds_list = [
            {'text': 'hello user@test.com world'},
            {'text': 'clean text here'},
        ]
        tgt_list = [
            {'text': 'hello  world'},
            {'text': 'clean text here'},
        ]
        dataset = self.generate_dataset(ds_list)
        op = CleanEmailMapper(batch_size=2)
        result = self.run_single_op(dataset, op, ['text'])
        self.assertDatasetEqual(result, tgt_list)


if __name__ == '__main__':
    unittest.main()

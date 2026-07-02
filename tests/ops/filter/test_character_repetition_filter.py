import unittest

from data_juicer.core.data import NestedDataset as Dataset

from data_juicer.ops.filter.character_repetition_filter import \
    CharacterRepetitionFilter
from data_juicer.utils.constant import Fields
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class CharacterRepetitionFilterTest(DataJuicerTestCaseBase):

    def _run_character_repetition_filter(self, dataset: Dataset, target_list,
                                         op):
        if Fields.stats not in dataset.features:
            # TODO:
            # this is a temp solution,
            # only add stats when calling filter op
            dataset = dataset.add_column(name=Fields.stats,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.compute_stats, batch_size=op.batch_size, num_proc=1)
        dataset = dataset.filter(op.process, batch_size=op.batch_size, num_proc=2)
        dataset = dataset.select_columns(column_names=['text'])
        res_list = dataset.to_list()
        self.assertEqual(res_list, target_list)

    def test_case(self):

        ds_list = [{
            'text':
            "Today is Sund Sund Sund Sund Sund Sunda and it's a happy day!"
        }, {
            'text': 'a v s e c s f e f g a a a a a a a a a a'
        }, {
            'text': '，。、„”“«»１」「《》´∶：？！（）；–—．～’…━〈〉【】％►'
        }, {
            'text': '中文也是一个字算一个长度'
        }]
        tgt_list = [{
            'text': '，。、„”“«»１」「《》´∶：？！（）；–—．～’…━〈〉【】％►'
        }, {
            'text': '中文也是一个字算一个长度'
        }]
        dataset = Dataset.from_list(ds_list)
        op = CharacterRepetitionFilter(
            rep_len=5, 
            min_ratio=0.0, 
            max_ratio=0.4,
            batch_size=2)
        self._run_character_repetition_filter(dataset, tgt_list, op)

    def test_existing_stats(self):
        ds_list = [{
            'text':
            "Today is Sund Sund Sund Sund Sund Sunda and it's a happy day!",
            Fields.stats: {
                'char_rep_ratio': 0.5
            }
        }, {
            'text': 'a v s e c s f e f g a a a a a a a a a a',
            Fields.stats: {
                'char_rep_ratio': 0.5
            }
        }]
        dataset = Dataset.from_list(ds_list)
        op = CharacterRepetitionFilter(
            rep_len=5,
            min_ratio=0.0,
            max_ratio=0.4,
            batch_size=2)
        dataset_after_compute_stats = op.compute_stats(dataset)
        self.assertEqual(dataset_after_compute_stats.to_list(), ds_list)


    def test_compute_stats_batched_directly(self):
        """Call compute_stats_batched directly to verify stat computation."""
        op = CharacterRepetitionFilter(rep_len=5, min_ratio=0.0, max_ratio=0.4, batch_size=2)
        samples = {
            'text': [
                'aaaaaaaaaaaaaaaaaaaaaaaaa',  # all same char -> high ratio
                'abcdefghijklmnopqrstuvwxyz',  # all unique ngrams -> ratio 0
            ],
            Fields.stats: [{}, {}],
        }
        result = op.compute_stats_batched(samples)
        # Highly repetitive text should have ratio close to 1.0
        self.assertGreater(result[Fields.stats][0]['char_rep_ratio'], 0.9)
        # Unique text should have ratio 0.0
        self.assertAlmostEqual(result[Fields.stats][1]['char_rep_ratio'], 0.0)

    def test_process_batched_directly(self):
        """Call process_batched directly to verify filtering logic."""
        op = CharacterRepetitionFilter(rep_len=5, min_ratio=0.0, max_ratio=0.4, batch_size=2)
        samples = {
            Fields.stats: [
                {'char_rep_ratio': 0.9},  # exceeds max, filtered out
                {'char_rep_ratio': 0.2},  # within range, kept
                {'char_rep_ratio': 0.0},  # within range, kept
            ],
        }
        keep_flags = list(op.process_batched(samples))
        self.assertEqual(keep_flags, [False, True, True])

    def test_compute_stats_batched_short_text(self):
        """Text shorter than rep_len produces ratio 0.0."""
        op = CharacterRepetitionFilter(rep_len=10, min_ratio=0.0, max_ratio=0.5, batch_size=2)
        samples = {
            'text': ['hi', 'abcde', ''],
            Fields.stats: [{}, {}, {}],
        }
        result = op.compute_stats_batched(samples)
        for stat in result[Fields.stats]:
            self.assertAlmostEqual(stat['char_rep_ratio'], 0.0)

    def test_compute_stats_batched_skips_existing(self):
        """Already computed stats are not overwritten."""
        op = CharacterRepetitionFilter(rep_len=5, min_ratio=0.0, max_ratio=0.5, batch_size=2)
        samples = {
            'text': ['aaaaaaaaaaaaaaaaaaaaaaaaa'],
            Fields.stats: [{'char_rep_ratio': 0.42}],  # pre-computed
        }
        result = op.compute_stats_batched(samples)
        # Should preserve existing value
        self.assertAlmostEqual(result[Fields.stats][0]['char_rep_ratio'], 0.42)

    def test_compute_stats_batched_mixed_repetition(self):
        """Verify correct ratio for text with moderate repetition."""
        op = CharacterRepetitionFilter(rep_len=5, min_ratio=0.0, max_ratio=1.0, batch_size=2)
        # "Today is Sund Sund Sund..." has some repeated 5-grams
        samples = {
            'text': [
                "Today is Sund Sund Sund Sund Sund Sunda and it's a happy day!",
                'abcdefghijklmnopqrstuvwxyz0123456789',
            ],
            Fields.stats: [{}, {}],
        }
        result = op.compute_stats_batched(samples)
        # First text has repeated ngrams so ratio > 0
        self.assertGreater(result[Fields.stats][0]['char_rep_ratio'], 0.0)
        # Second text has no repetition
        self.assertAlmostEqual(result[Fields.stats][1]['char_rep_ratio'], 0.0)


if __name__ == '__main__':
    unittest.main()

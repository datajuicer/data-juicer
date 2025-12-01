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


if __name__ == '__main__':
    unittest.main()

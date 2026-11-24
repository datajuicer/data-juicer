import unittest

from data_juicer.core.data import NestedDataset as Dataset

from data_juicer.ops.filter.word_repetition_filter import WordRepetitionFilter
from data_juicer.utils.constant import Fields
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class WordRepetitionFilterTest(DataJuicerTestCaseBase):

    def _run_word_repetition_filter(self, dataset: Dataset, target_list, op):
        if Fields.stats not in dataset.features:
            # TODO:
            # this is a temp solution,
            # only add stats when calling filter op
            dataset = dataset.add_column(name=Fields.stats,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.compute_stats, batch_size=op.batch_size)
        dataset = dataset.filter(op.process, batch_size=op.batch_size)
        dataset = dataset.select_columns(column_names=['text'])
        res_list = dataset.to_list()
        self.assertEqual(res_list, target_list)

    def test_en_case(self):

        ds_list = [{
            'text':
            "Today is Sunday Sunday Sunday Sunday Sunday and it's a happy day!"
        }, {
            'text':
            "Today is Sunday Sunday Sunday and it's a happy day!"
        }, {
            'text':
            "Today is Sund Sund Sund Sund Sund Sunda and it's a happy day!"
        }, {
            'text':
            "plusieurs èrdash@hqbchd.ckd d'accéder à ces wwwasdasd fonc"
        }, {
            'text':
            'This proposed a novel proposed pretraining proposed pretraining.'
        }]
        tgt_list = [{
            'text':
            "Today is Sunday Sunday Sunday and it's a happy day!"
        }, {
            'text':
            "plusieurs èrdash@hqbchd.ckd d'accéder à ces wwwasdasd fonc"
        }, {
            'text':
            'This proposed a novel proposed pretraining proposed pretraining.'
        }]
        dataset = Dataset.from_list(ds_list)
        op = WordRepetitionFilter(
            rep_len=3, 
            min_ratio=0.0, 
            max_ratio=0.2,
            batch_size=2)
        self._run_word_repetition_filter(dataset, tgt_list, op)

    def test_zh_case(self):

        ds_list = [{
            'text': '去除字母、数字、下划线占比过低或过高的代码'
        }, {
            'text': '欢迎来到阿里巴巴巴巴巴巴巴巴'
        }, {
            'text': '使用片段分词器对每个页面进行分词，使用语言模型计算每个段落的困惑度得分'
        }, {
            'text': '根据算子使用使用使用使用安装方案确定'
        }, {
            'text': '基于前一步结果，在同一个聚类中找出那些过长文档为假正例，暂不进行滤除'
        }]
        tgt_list = [{
            'text': '去除字母、数字、下划线占比过低或过高的代码'
        }, {
            'text': '使用片段分词器对每个页面进行分词，使用语言模型计算每个段落的困惑度得分'
        }, {
            'text': '基于前一步结果，在同一个聚类中找出那些过长文档为假正例，暂不进行滤除'
        }]
        dataset = Dataset.from_list(ds_list)
        op = WordRepetitionFilter(lang='zh',
                                  tokenization=True,
                                  rep_len=3,
                                  min_ratio=0.0,
                                  max_ratio=0.2,
                                  batch_size=1)
        self._run_word_repetition_filter(dataset, tgt_list, op)


    def test_compute_stats_batched_directly(self):
        """Directly call compute_stats_batched for non-tokenization path."""
        from data_juicer.utils.constant import Fields, StatsKeys

        op = WordRepetitionFilter(rep_len=2, min_ratio=0.0, max_ratio=0.5, tokenization=False)
        # 'hello world' repeated 5 times -> high word 2-gram repetition
        text_repetitive = ' '.join(['hello world'] * 5 + ['unique text here'])
        text_unique = 'all different words in this sentence no repeats at all'
        samples = {
            'text': [text_repetitive, text_unique],
            Fields.stats: [{}, {}],
        }
        result = op.compute_stats_batched(samples)
        # Repetitive text should have high ratio
        self.assertGreater(result[Fields.stats][0][StatsKeys.word_rep_ratio], 0.5)
        # Unique text should have ratio 0.0
        self.assertAlmostEqual(result[Fields.stats][1][StatsKeys.word_rep_ratio], 0.0)

    def test_process_batched_directly(self):
        """Directly call process_batched to verify filtering."""
        from data_juicer.utils.constant import Fields, StatsKeys

        op = WordRepetitionFilter(rep_len=2, min_ratio=0.0, max_ratio=0.5, tokenization=False)
        samples = {
            Fields.stats: [
                {StatsKeys.word_rep_ratio: 0.8},   # exceeds max -> filtered
                {StatsKeys.word_rep_ratio: 0.3},   # within range -> kept
                {StatsKeys.word_rep_ratio: 0.0},   # within range -> kept
            ],
        }
        keep_flags = list(op.process_batched(samples))
        self.assertEqual(keep_flags, [False, True, True])

    def test_compute_stats_batched_empty_words(self):
        """Text that produces no word n-grams gets ratio 0.0."""
        from data_juicer.utils.constant import Fields, StatsKeys

        op = WordRepetitionFilter(rep_len=5, min_ratio=0.0, max_ratio=1.0, tokenization=False)
        # Very short text with fewer words than rep_len=5
        samples = {
            'text': ['hi there', ''],
            Fields.stats: [{}, {}],
        }
        result = op.compute_stats_batched(samples)
        self.assertAlmostEqual(result[Fields.stats][0][StatsKeys.word_rep_ratio], 0.0)
        self.assertAlmostEqual(result[Fields.stats][1][StatsKeys.word_rep_ratio], 0.0)

    def test_compute_stats_batched_skips_existing(self):
        """Already computed stats are preserved."""
        from data_juicer.utils.constant import Fields, StatsKeys

        op = WordRepetitionFilter(rep_len=2, min_ratio=0.0, max_ratio=1.0, tokenization=False)
        samples = {
            'text': ['hello world hello world hello world'],
            Fields.stats: [{StatsKeys.word_rep_ratio: 0.42}],
        }
        result = op.compute_stats_batched(samples)
        self.assertAlmostEqual(result[Fields.stats][0][StatsKeys.word_rep_ratio], 0.42)


if __name__ == '__main__':
    unittest.main()

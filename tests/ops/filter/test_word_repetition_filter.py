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


if __name__ == '__main__':
    unittest.main()

import unittest

from data_juicer.core.data import NestedDataset as Dataset

from data_juicer.ops.filter.stopwords_filter import StopWordsFilter
from data_juicer.utils.constant import Fields
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class StopWordsFilterTest(DataJuicerTestCaseBase):

    def _run_stopwords_filter(self, dataset: Dataset, target_list, op):
        if Fields.stats not in dataset.features:
            # TODO:
            # this is a temp solution,
            # only add stats when calling filter op
            dataset = dataset.add_column(name=Fields.stats,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.compute_stats)
        dataset = dataset.filter(op.process)
        dataset = dataset.select_columns(column_names=['text'])
        res_list = dataset.to_list()
        self.assertEqual(res_list, target_list)

    def test_en_case(self):

        ds_list = [{
            'text': "Today is Sunday and it's a happy day!"
        }, {
            'text':
            "Today is Sund Sund Sund Sund Sunda and it's a happy day!"
        }, {
            'text': 'a v s e c s f e f g a qkc'
        }, {
            'text': '，。、„”“«»１」「《》´∶：？！（）；–—．～’…━〈〉【】％►'
        }, {
            'text': 'Do you need a cup of coffee?'
        }]
        tgt_list = [{
            'text': "Today is Sunday and it's a happy day!"
        }, {
            'text':
            "Today is Sund Sund Sund Sund Sunda and it's a happy day!"
        }, {
            'text': 'Do you need a cup of coffee?'
        }]
        dataset = Dataset.from_list(ds_list)
        op = StopWordsFilter(lang='en', min_ratio=0.3)
        self._run_stopwords_filter(dataset, tgt_list, op)

    def test_zh_case(self):

        ds_list = [{
            'text': '你好，请问你是谁'
        }, {
            'text': '字母、数字、下划线、占比、代码'
        }, {
            'text': '基于前一步结果，在同一个聚类中找出那些过长文档为假正例，暂不进行滤除'
        }, {
            'text': '使用片段分词器对每个页面进行分词，使用语言模型计算每个段落的困惑度得分，由此过滤低质量文本'
        }]
        tgt_list = [{
            'text': '你好，请问你是谁'
        }, {
            'text': '基于前一步结果，在同一个聚类中找出那些过长文档为假正例，暂不进行滤除'
        }, {
            'text': '使用片段分词器对每个页面进行分词，使用语言模型计算每个段落的困惑度得分，由此过滤低质量文本'
        }]
        dataset = Dataset.from_list(ds_list)
        op = StopWordsFilter(lang='zh',
                             tokenization=True,
                             min_ratio=0.2,
                             use_words_aug=True)
        self._run_stopwords_filter(dataset, tgt_list, op)


if __name__ == '__main__':
    unittest.main()

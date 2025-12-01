import unittest

from data_juicer.core.data import NestedDataset as Dataset

from data_juicer.ops.filter.words_num_filter import WordsNumFilter
from data_juicer.utils.constant import Fields
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class WordsNumFilterTest(DataJuicerTestCaseBase):

    def _run_words_num_filter(self, dataset: Dataset, target_list, op):
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

    def test_case(self):

        ds_list = [{
            'text': 'Today is Sun'
        }, {
            'text':
            "Today is Sund Sund Sund Sund Sund Sunda and it's a happy day!"
        }, {
            'text': 'a v s e c s f e f g a a a  '
        }, {
            'text': '，。、„”“«»１」「《》´∶：？！（）；–—．～’…━〈〉【】％►'
        }]
        tgt_list = [{
            'text':
            "Today is Sund Sund Sund Sund Sund Sunda and it's a happy day!"
        }, {
            'text': 'a v s e c s f e f g a a a  '
        }]
        dataset = Dataset.from_list(ds_list)
        op = WordsNumFilter(min_num=5, max_num=15, batch_size=2)
        self._run_words_num_filter(dataset, tgt_list, op)

    def test_zh_case(self):

        ds_list = [{
            'text': '你好，请问你是谁'
        }, {
            'text': '欢迎来到阿里巴巴'
        }, {
            'text': '根据算子使用情况增量安装方案确定'
        }, {
            'text': '去除字母、数字、下划线占比过低或过高的代码'
        }, {
            'text': '使用片段分词器对每个页面进行分词，使用语言模型计算每个段落的困惑度得分，由此过滤低质量文本'
        }, {
            'text': '基于前一步结果，在同一个聚类中找出那些过长文档为假正例，暂不进行滤除'
        }]
        tgt_list = [{
            'text': '去除字母、数字、下划线占比过低或过高的代码'
        }, {
            'text': '基于前一步结果，在同一个聚类中找出那些过长文档为假正例，暂不进行滤除'
        }]
        dataset = Dataset.from_list(ds_list)
        op = WordsNumFilter(lang='zh',
                           tokenization=True,
                           min_num=10,
                           max_num=25,
                           batch_size=1)
        self._run_words_num_filter(dataset, tgt_list, op)


if __name__ == '__main__':
    unittest.main()

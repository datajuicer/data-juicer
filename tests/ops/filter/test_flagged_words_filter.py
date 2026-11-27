import unittest

from data_juicer.core.data import NestedDataset as Dataset

from data_juicer.ops.filter.flagged_words_filter import FlaggedWordFilter
from data_juicer.utils.constant import Fields
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class FlaggedWordFilterTest(DataJuicerTestCaseBase):

    def _run_flagged_words_filter(self, dataset: Dataset, target_list, op):
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
            'text': 'Today is anal cumshot day'
        }, {
            'text': 'Fuck you doggystyle!'
        }, {
            'text': '，。、„”“«»１」「《》´∶：？！（）；–—．～’…━〈〉【】％►'
        }, {
            'text': 'Do you need a cup of coffee?'
        }, {
            'text': 'emoji表情测试下😊，😸31231\n'
        }]
        tgt_list = [{
            'text': '，。、„”“«»１」「《》´∶：？！（）；–—．～’…━〈〉【】％►'
        }, {
            'text': 'Do you need a cup of coffee?'
        }, {
            'text': 'emoji表情测试下😊，😸31231\n'
        }]
        dataset = Dataset.from_list(ds_list)
        op = FlaggedWordFilter(lang='en', max_ratio=0.045)
        self._run_flagged_words_filter(dataset, tgt_list, op)

    def test_zh_case(self):

        ds_list = [{
            'text': '你是个卖淫女'
        }, {
            'text': '根据算子使用情况增量安装方案确定'
        }, {
            'text': '去除字母、数字、下划线占比过低或过高的代码'
        }, {
            'text': '基于前一步结果，除掉打飞机、三级片等敏感词'
        }, {
            'text': '使用片段分词器对每个页面进行分词，使用语言模型计算每个段落的困惑度得分，由此过滤低质量文本'
        }]
        tgt_list = [{
            'text': '根据算子使用情况增量安装方案确定'
        }, {
            'text': '去除字母、数字、下划线占比过低或过高的代码'
        }, {
            'text': '使用片段分词器对每个页面进行分词，使用语言模型计算每个段落的困惑度得分，由此过滤低质量文本'
        }]
        dataset = Dataset.from_list(ds_list)
        op = FlaggedWordFilter(lang='zh',
                               tokenization=True,
                               max_ratio=0.045,
                               use_words_aug=True)
        self._run_flagged_words_filter(dataset, tgt_list, op)

    def test_capitalized_entries_case(self):
        # Some entries in the flagged words list are capitalized
        # (e.g. 'Cabrón' in es), while words are lower-cased before
        # matching. They should still be counted as flagged words.
        ds_list = [{
            'text': 'Esto es un Cabrón muy fuerte'
        }]
        tgt_list = []
        dataset = Dataset.from_list(ds_list)
        op = FlaggedWordFilter(lang='es', max_ratio=0.045)
        self._run_flagged_words_filter(dataset, tgt_list, op)


if __name__ == '__main__':
    unittest.main()

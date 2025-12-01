import unittest

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.remove_words_with_incorrect_substrings_mapper import \
    RemoveWordsWithIncorrectSubstringsMapper  # noqa: E501
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class RemoveWordsWithIncorrectSubstringsMapperTest(DataJuicerTestCaseBase):

    def _run_remove_words_with_incorrect_sbstrings(self, samples, op):
        dataset = Dataset.from_list(samples)
        dataset = dataset.map(op.process, batch_size=2)
                
        for data in dataset:
            self.assertEqual(data['text'], data['target'])

    def test_en_case(self):

        samples = [
            {
                'text':
                'This paper proposed a novel https://whiugc.com method on LLM',
                'target': 'This paper proposed a novel method on LLM'
            },
            {
                'text':
                "plusieurs èrdash@hqbchd.ckd d'accéder à ces wwwasdasd fonc",
                'target': "plusieurs èrdash@hqbchd.ckd d'accéder à ces fonc"
            },
        ]

        op = RemoveWordsWithIncorrectSubstringsMapper(
            substrings=['http', 'www', '.com', 'href', '//'])
        self._run_remove_words_with_incorrect_sbstrings(samples, op)

    def test_zh_case(self):

        samples = [{
            'text': '你好，请问你是谁',
            'target': '你好，请问你是谁'
        }, {
            'text': '欢迎来到阿里巴巴！',
            'target': '欢迎来到阿里巴巴！'
        }, {
            'text': '根据算子使用情况增量安装方案确定',
            'target': '根据使用情况增量安装方案确定'
        }, {
            'text': '请用百度www.baidu.com进行搜索',
            'target': '请用百度www.baidu.进行搜索'
        }]

        op = RemoveWordsWithIncorrectSubstringsMapper(lang='zh',
                                                      tokenization=True,
                                                      substrings=['com', '算子'])
        self._run_remove_words_with_incorrect_sbstrings(samples, op)


if __name__ == '__main__':
    unittest.main()

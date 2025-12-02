import unittest

from data_juicer.core.data import NestedDataset as Dataset

from data_juicer.ops.filter.token_num_filter import TokenNumFilter
from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class TokenNumFilterTest(DataJuicerTestCaseBase):

    def test_token_num(self):
        ds_list = [
            {
                'text': "Today is Sunday and it's a happy day!"
            },
            {
                'text': 'Do you need a cup of coffee?'
            },
            {
                'text': '你好，请问你是谁'
            },
            {
                'text':
                "Sur la plateforme MT4, plusieurs manières d'accéder à "
                'ces fonctionnalités sont conçues simultanément.'
            },
            {
                'text': '欢迎来到阿里巴巴！'
            },
            {
                'text':
                'This paper proposed a novel method on LLM pretraining.'
            },
        ]
        tgt_list = [
            10,
            8,
            9,
            31,
            14,
            12,
        ]
        ds = Dataset.from_list(ds_list)
        op = TokenNumFilter()
        ds = ds.add_column(Fields.stats, [{}] * len(ds))
        ds = ds.map(op.compute_stats)
        stats = ds[Fields.stats]
        self.assertEqual([item[StatsKeys.num_token] for item in stats], tgt_list)

    def test_token_num_filter(self):
        ds_list = [
            {
                'text': "Today is Sunday and it's a happy day!"
            },
            {
                'text': 'Do you need a cup of coffee?'
            },
            {
                'text': '你好，请问你是谁'
            },
            {
                'text':
                "Sur la plateforme MT4, plusieurs manières d'accéder à "
                'ces fonctionnalités sont conçues simultanément.'
            },
            {
                'text': '欢迎来到阿里巴巴！'
            },
            {
                'text':
                'This paper proposed a novel method on LLM pretraining.'
            },
        ]
        tgt_list = [
            {
                'text': "Today is Sunday and it's a happy day!"
            },
            {
                'text': '欢迎来到阿里巴巴！'
            },
            {
                'text':
                'This paper proposed a novel method on LLM pretraining.'
            },
        ]
        ds = Dataset.from_list(ds_list)
        tgt_list = Dataset.from_list(tgt_list)
        op = TokenNumFilter(min_num=10, max_num=20)
        ds = ds.add_column(Fields.stats, [{}] * len(ds))
        ds = ds.map(op.compute_stats)
        ds = ds.filter(op.process)
        self.assertEqual(ds['text'], tgt_list['text'])


if __name__ == '__main__':
    unittest.main()

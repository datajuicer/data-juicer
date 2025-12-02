import unittest

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.grouper.naive_grouper import NaiveGrouper
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class NaiveGrouperTest(DataJuicerTestCaseBase):

    def _run_helper(self, op, samples, target):
        dataset = Dataset.from_list(samples)
        new_dataset = op.run(dataset)

        for d, t in zip(new_dataset, target):
            self.assertEqual(d['text'], t['text'])

    def test_naive_group(self):

        ds_list = [
            {
                'text': "Today is Sunday and it's a happy day!"
            },
            {
                'text':
                "Sur la plateforme MT4, plusieurs manières d'accéder à \n"
                'ces fonctionnalités sont conçues simultanément.'
            },
            {
                'text': '欢迎来到阿里巴巴！'
            },
        ]
        tgt_list = [
            {
                'text':[
                    "Today is Sunday and it's a happy day!",
                    "Sur la plateforme MT4, plusieurs manières d'accéder à \n"
                    'ces fonctionnalités sont conçues simultanément.',
                    '欢迎来到阿里巴巴！'
                ]
            }
        ]

        op = NaiveGrouper()
        self._run_helper(op, ds_list, tgt_list)

if __name__ == '__main__':
    unittest.main()
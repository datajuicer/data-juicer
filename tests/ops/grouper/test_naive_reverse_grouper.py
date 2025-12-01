import unittest
import json
import os

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.grouper.naive_reverse_grouper import NaiveReverseGrouper
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase
from data_juicer.utils.constant import Fields


class NaiveReverseGrouperTest(DataJuicerTestCaseBase):

    def _run_helper(self, op, samples, target, meta_target=None, meta_path=None):
        dataset = Dataset.from_list(samples)
        new_dataset = op.run(dataset)

        for d, t in zip(new_dataset, target):
            self.assertEqual(d['text'], t['text'])
        
        if meta_target is not None:
            batch_meta = []
            with open(meta_path) as f:
                for line in f.readlines():
                    batch_meta.append(json.loads(line))
            self.assertEqual(batch_meta, meta_target)
            os.remove(meta_path)

    def test_one_batched_sample(self):

        ds_list = [
            {
                'text':[
                    "Today is Sunday and it's a happy day!",
                    "Sur la plateforme MT4, plusieurs manières d'accéder à \n"
                    'ces fonctionnalités sont conçues simultanément.',
                    '欢迎来到阿里巴巴！'
                ]
            }
        ]

        tgt_list = [
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

        op = NaiveReverseGrouper()
        self._run_helper(op, ds_list, tgt_list)


    def test_two_batch_sample(self):

        ds_list = [
            {
                'text':[
                    "Today is Sunday and it's a happy day!",
                    "Sur la plateforme MT4, plusieurs manières d'accéder à \n"
                    'ces fonctionnalités sont conçues simultanément.'
                ]
            },
            {
                'text':[
                    '欢迎来到阿里巴巴！'
                ]
            }
        ]

        tgt_list = [
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

        op = NaiveReverseGrouper()
        self._run_helper(op, ds_list, tgt_list)
    
    def test_rm_unbatched_keys1(self):
        ds_list = [
            {
                'text':[
                    "Today is Sunday and it's a happy day!",
                    "Sur la plateforme MT4, plusieurs manières d'accéder à \n"
                    'ces fonctionnalités sont conçues simultanément.'
                ],
                Fields.batch_meta: {'batch_size': 2},
            }
        ]

        tgt_list = [
            {
                'text': "Today is Sunday and it's a happy day!"
            },
            {
                'text':
                "Sur la plateforme MT4, plusieurs manières d'accéder à \n"
                'ces fonctionnalités sont conçues simultanément.'
            }
        ]

        op = NaiveReverseGrouper()
        self._run_helper(op, ds_list, tgt_list)

    def test_rm_unbatched_keys2(self):
        ds_list = [
            {
                'text':[
                    '欢迎来到阿里巴巴！'
                ],
                'query':[
                    'Can I help you?'
                ],
                Fields.batch_meta: {
                    'response':[
                        'No',
                        'Yes'
                    ],
                    'batch_size': 1,
                }
            },
            {
                'text':[
                    'Can I help you?'
                ],
                'query':[
                    '欢迎来到阿里巴巴！'
                ],
                Fields.batch_meta: {
                    'response':[
                        'No',
                        'Yes'
                    ],
                    'batch_size': 1,
                }
            }
        ]

        target = [
            {
                'text': '欢迎来到阿里巴巴！',
                'query': 'Can I help you?',
            },
            {
                'text': 'Can I help you?',
                'query': '欢迎来到阿里巴巴！',
            },
        ]

        target_meta = [
            {
                'response':[
                    'No',
                    'Yes'
                ],
                'batch_size': 1,
            },
            {
                'response':[
                    'No',
                    'Yes'
                ],
                'batch_size': 1,
            }
        ]

        export_path = '__dj__naive_reverse_grouper_test_file.jsonl'
        op = NaiveReverseGrouper(export_path)
        self._run_helper(op, ds_list, target,
            meta_target=target_meta,
            meta_path=export_path)

if __name__ == '__main__':
    unittest.main()
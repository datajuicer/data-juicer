import unittest

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.grouper.key_value_grouper import KeyValueGrouper
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class KeyValueGrouperTest(DataJuicerTestCaseBase):

    def _run_helper(self, op, samples, target):
        dataset = Dataset.from_list(samples)
        new_dataset = op.run(dataset)

        for batched_sample in new_dataset:
            lang = batched_sample['meta'][0]['language']
            self.assertEqual(batched_sample['text'], target[lang])

    def test_key_value_grouper(self):

        ds_list = [
            {
                'text': "Today is Sunday and it's a happy day!",
                'meta': {
                    'language': 'en'
                }
            },
            {
                'text': "Welcome to Alibaba.",
                'meta': {
                    'language': 'en'
                }
            },
            {
                'text': '欢迎来到阿里巴巴！',
                'meta': {
                    'language': 'zh'
                }
            },
        ]
        tgt_list = {
            'en':[
                "Today is Sunday and it's a happy day!",
                "Welcome to Alibaba."
            ],
            'zh':[
                '欢迎来到阿里巴巴！'
            ]
        }

        op = KeyValueGrouper(['meta.language'])
        self._run_helper(op, ds_list, tgt_list)

if __name__ == '__main__':
    unittest.main()
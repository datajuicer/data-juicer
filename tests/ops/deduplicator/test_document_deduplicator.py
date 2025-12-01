import unittest

from data_juicer.core.data import NestedDataset as Dataset

from data_juicer.ops.deduplicator.document_deduplicator import \
    DocumentDeduplicator
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class DocumentDeduplicatorTest(DataJuicerTestCaseBase):

    def _run_doc_dedup(self, dataset: Dataset, target_list, op, show_num=0):
        dataset = dataset.map(op.compute_hash)
        dataset, dup_pairs = op.process(dataset, show_num=show_num)
        dataset = dataset.select_columns(column_names=['text'])
        res_list = dataset.to_list()
        self.assertEqual(res_list, target_list)
        return dup_pairs

    def test_english_deduplication(self):
        ds_list = [
            {
                'text': 'Today is Sunday and it\'s a happy day!'
            },
            {
                'text': 'Do you need a cup of coffee?'
            },
            {
                'text': 'Today is sunday and it\'s a happy day!'
            },
            {
                'text':
                'This paper proposed a novel method on LLM pretraining.'
            },
            {
                'text':
                'This paper proposed a novel method on LLM pretraining.'
            },
        ]
        tgt_list = [{
            'text': 'Today is Sunday and it\'s a happy day!'
        }, {
            'text': 'Do you need a cup of coffee?'
        }, {
            'text': 'Today is sunday and it\'s a happy day!'
        }, {
            'text':
            'This paper proposed a novel method on LLM pretraining.'
        }]
        dataset = Dataset.from_list(ds_list)
        op = DocumentDeduplicator(lowercase=False, ignore_non_character=False)
        dup_pairs = self._run_doc_dedup(dataset, tgt_list, op)
        self.assertEqual(len(dup_pairs), 0)
        
    def test_english_deduplication_with_params(self):
        ds_list = [
            {
                'text': 'Today is Sunday and it\'s a happy day!'
            },
            {
                'text': 'Do you need a cup of coffee?'
            },
            {
                'text': 'Today is sunday and it\'s a happy day!'
            },
            {
                'text': 'Today is sunday and it\'s a happy day?'
            },
            {
                'text':
                'This paper proposed a novel method on LLM pretraining.'
            },
            {
                'text':
                'This paper proposed a novel method on LLM pretraining.'
            },
        ]
        tgt_list = [{
            'text': 'Today is Sunday and it\'s a happy day!'
        }, {
            'text': 'Do you need a cup of coffee?'
        }, {
            'text':
            'This paper proposed a novel method on LLM pretraining.'
        }]
        dataset = Dataset.from_list(ds_list)
        op = DocumentDeduplicator(lowercase=True, ignore_non_character=True)
        dup_pairs = self._run_doc_dedup(dataset, tgt_list, op)
        self.assertEqual(len(dup_pairs), 0)

    def test_chinese_deduplication(self):
        ds_list = [
            {
                'text': '你好，请问你是谁'
            },
            {
                'text': '欢迎来到阿里巴巴！'
            },
            {
                'text':
                '第九届会议\n2003年7月28日至8月8日\n牙买加金斯敦\n为来自发展中国家的法'
                '律和技术委员会以及财务委员会成员\n参加委员会会议支付费用的方式\n1.'
            },
            {
                'text':
                '第九届会议\n2003年7月28日至8月8日\n牙买加金斯敦\n为来自发展中国家的法'
                '律和技术委员会以及财务委员会成员\n参加委员会会议支付费用的方式\n1.'
            },
            {
                'text':
                '第九届会议\n时间：2003年7月28日至8月8日\n牙买加金斯敦\n为来自发展中国家的法'
                '律和技术委员会以及财务委员会成员\n参加委员会会议支付费用的方式\n1.'
            },
        ]
        tgt_list = [
            {
                'text': '你好，请问你是谁'
            },
            {
                'text': '欢迎来到阿里巴巴！'
            },
            {
                'text':
                '第九届会议\n2003年7月28日至8月8日\n牙买加金斯敦\n为来自发展中国家的法'
                '律和技术委员会以及财务委员会成员\n参加委员会会议支付费用的方式\n1.'
            },
            {
                'text':
                '第九届会议\n时间：2003年7月28日至8月8日\n牙买加金斯敦\n为来自发展中国家的法'
                '律和技术委员会以及财务委员会成员\n参加委员会会议支付费用的方式\n1.'
            },
        ]
        dataset = Dataset.from_list(ds_list)
        op = DocumentDeduplicator(lowercase=False, ignore_non_character=False)
        dup_pairs = self._run_doc_dedup(dataset, tgt_list, op)
        self.assertEqual(len(dup_pairs), 0)
    
    def test_show_num(self):
        ds_list = [
            {
                'text': 'Today is Sunday and it\'s a happy day!'
            },
            {
                'text': 'Do you need a cup of coffee?'
            },
            {
                'text': 'Today is sunday and it\'s a happy day!'
            },
            {
                'text':
                'This paper proposed a novel method on LLM pretraining.'
            },
            {
                'text':
                'This paper proposed a novel method on LLM pretraining.'
            },
        ]
        tgt_list = [{
            'text': 'Today is Sunday and it\'s a happy day!'
        }, {
            'text': 'Do you need a cup of coffee?'
        }, {
            'text': 'Today is sunday and it\'s a happy day!'
        }, {
            'text':
            'This paper proposed a novel method on LLM pretraining.'
        }]
        dataset = Dataset.from_list(ds_list)
        op = DocumentDeduplicator(lowercase=False, ignore_non_character=False)
        dup_pairs = self._run_doc_dedup(dataset, tgt_list, op, show_num=1)
        self.assertEqual(len(dup_pairs), 1)


if __name__ == '__main__':
    unittest.main()

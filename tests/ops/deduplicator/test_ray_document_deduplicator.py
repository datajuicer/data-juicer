import unittest

from data_juicer.core.data import NestedDataset as Dataset

from data_juicer.ops.deduplicator.ray_document_deduplicator import \
    RayDocumentDeduplicator
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase, TEST_TAG


class RayDocumentDeduplicatorTest(DataJuicerTestCaseBase):

    def _run_doc_dedup(self, dataset: Dataset, target_list, op):
        res_list = self.run_single_op(dataset, op, [op.text_key])
        res_list.sort(key=lambda x: x['text'])
        target_list.sort(key=lambda x: x['text'])
        self.assertEqual(res_list, target_list)

    @TEST_TAG("ray")
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
        dataset = self.generate_dataset(ds_list)
        op = RayDocumentDeduplicator(lowercase=False, ignore_non_character=False)
        self._run_doc_dedup(dataset, tgt_list, op)

    @TEST_TAG("ray")
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
        dataset = self.generate_dataset(ds_list)
        op = RayDocumentDeduplicator(lowercase=False, ignore_non_character=False)
        self._run_doc_dedup(dataset, tgt_list, op)


if __name__ == '__main__':
    unittest.main()

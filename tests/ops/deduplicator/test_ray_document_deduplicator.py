import unittest
from unittest.mock import patch

from data_juicer.core.data import NestedDataset as Dataset

from data_juicer.ops.deduplicator.ray_document_deduplicator import \
    RayDocumentDeduplicator
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase, TEST_TAG


class RayDocumentDeduplicatorTest(DataJuicerTestCaseBase):

    def _run_ray_cross_block_dedup(self, samples, op):
        dataset = self._build_ray_cross_block_dataset(samples)
        dataset.process([op])
        return dataset.data.take_all()

    def _build_ray_cross_block_dataset(self, samples):
        import ray

        from data_juicer.core.data.ray_dataset import RayDataset
        from data_juicer.utils.constant import Fields

        ds_list = [{Fields.stats: {}, **sample} for sample in samples]
        return RayDataset(
            ray.data.from_items(ds_list, override_num_blocks=len(ds_list)),
            cfg={'auto_op_parallelism': False},
            auto_op_parallelism=False,
        )

    def _run_doc_dedup(self, dataset: Dataset, target_list, op):
        import ray

        from data_juicer.core.data.ray_dataset import RayDataset

        dataset = RayDataset(
            ray.data.from_items(dataset.to_list()),
            cfg={'auto_op_parallelism': False},
            auto_op_parallelism=False,
        )
        dataset.process([op])
        res_list = [
            {op.text_key: sample[op.text_key]}
            for sample in dataset.data.take_all()
        ]
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
        op = RayDocumentDeduplicator(
            lowercase=False,
            ignore_non_character=False,
            dedup_set_num=1,
            batch_size=1,
            num_proc=2,
            auto_op_parallelism=False,
        )
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
        op = RayDocumentDeduplicator(
            lowercase=False,
            ignore_non_character=False,
            dedup_set_num=1,
            batch_size=1,
            num_proc=2,
            auto_op_parallelism=False,
        )
        self._run_doc_dedup(dataset, tgt_list, op)

    @TEST_TAG("ray")
    def test_ray_actor_backend_deduplicates_across_blocks(self):
        op = RayDocumentDeduplicator(
            lowercase=False,
            ignore_non_character=False,
            dedup_set_num=1,
            batch_size=1,
            num_proc=4,
            auto_op_parallelism=False,
        )

        res_list = self._run_ray_cross_block_dedup(
            [{'text': 'duplicate across ray blocks'} for _ in range(8)],
            op,
        )

        self.assertEqual(len(res_list), 1)
        self.assertEqual(res_list[0]['text'], 'duplicate across ray blocks')

    @TEST_TAG("ray")
    def test_ray_actor_execution_mode_still_shares_dedup_sets(self):
        op = RayDocumentDeduplicator(
            lowercase=False,
            ignore_non_character=False,
            dedup_set_num=1,
            batch_size=1,
            num_proc=4,
            auto_op_parallelism=False,
            ray_execution_mode='actor',
        )

        res_list = self._run_ray_cross_block_dedup(
            [{'text': 'duplicate with actor execution mode'} for _ in range(8)],
            op,
        )

        self.assertEqual(len(res_list), 1)
        self.assertEqual(res_list[0]['text'], 'duplicate with actor execution mode')

    @TEST_TAG("ray")
    def test_ray_basic_deduplicator_subclasses_share_dedup_sets(self):
        from data_juicer.ops.deduplicator.ray_image_deduplicator import RayImageDeduplicator
        from data_juicer.ops.deduplicator.ray_video_deduplicator import RayVideoDeduplicator

        cases = [
            (RayImageDeduplicator, {'images': []}),
            (RayVideoDeduplicator, {'videos': []}),
        ]
        for op_cls, sample in cases:
            with self.subTest(op_cls=op_cls.__name__):
                op = op_cls(
                    dedup_set_num=1,
                    batch_size=1,
                    num_proc=4,
                    auto_op_parallelism=False,
                )

                res_list = self._run_ray_cross_block_dedup([sample for _ in range(8)], op)

                self.assertEqual(len(res_list), 1)

    @TEST_TAG("ray")
    def test_repeated_execution_keeps_materialized_dedup_result(self):
        op = RayDocumentDeduplicator(
            lowercase=False,
            ignore_non_character=False,
            dedup_set_num=1,
            batch_size=1,
            num_proc=4,
            auto_op_parallelism=False,
        )
        dataset = self._build_ray_cross_block_dataset([{
            'text': 'duplicate across repeated executions',
        } for _ in range(8)])

        dataset.process([op])
        self.assertEqual(dataset.data.count(), 1)
        res_list = dataset.data.take_all()

        self.assertEqual(len(res_list), 1)
        self.assertEqual(res_list[0]['text'], 'duplicate across repeated executions')

    @TEST_TAG("ray")
    def test_stats_export_does_not_consume_dedup_state_before_filter(self):
        def materializing_write_json(dataset, *args, **kwargs):
            return dataset.count()

        with patch('ray.data.Dataset.write_json', materializing_write_json):
            op = RayDocumentDeduplicator(
                lowercase=False,
                ignore_non_character=False,
                dedup_set_num=1,
                batch_size=1,
                num_proc=4,
                auto_op_parallelism=False,
                stats_export_path='mock_stats_export_path',
            )
            dataset = self._build_ray_cross_block_dataset([{
                'text': 'duplicate with stats export',
            } for _ in range(8)])

            dataset.process([op])
            res_list = dataset.data.take_all()

        self.assertEqual(len(res_list), 1)
        self.assertEqual(res_list[0]['text'], 'duplicate with stats export')

if __name__ == '__main__':
    unittest.main()

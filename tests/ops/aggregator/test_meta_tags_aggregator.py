import unittest

from loguru import logger

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.aggregator import MetaTagsAggregator
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase, FROM_FORK

@unittest.skipIf(FROM_FORK, "Skipping API-based test because running from a fork repo")
class MetaTagsAggregatorTest(DataJuicerTestCaseBase):

    def _run_helper(self, op, samples):

        # before running this test, set below environment variables:
        # export OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1/
        # export OPENAI_API_KEY=your_dashscope_key

        dataset = Dataset.from_list(samples)
        new_dataset = op.run(dataset)

        for data in new_dataset:
            for k in data:
                logger.info(f"{k}: {data[k]}")

        self.assertEqual(len(new_dataset), len(samples))

    def test_default_aggregator(self):
        samples = [
            {
                Fields.meta: [
                    {
                        MetaKeys.query_sentiment_label: '开心'
                    },
                    {
                        MetaKeys.query_sentiment_label: '快乐'
                    },
                    {
                        MetaKeys.query_sentiment_label: '难过'
                    },
                    {
                        MetaKeys.query_sentiment_label: '不开心'
                    },
                    {
                        MetaKeys.query_sentiment_label: '愤怒'
                    }
                ]
            },
        ]
        op = MetaTagsAggregator(
            api_model='qwen2.5-72b-instruct',
            meta_tag_key=MetaKeys.query_sentiment_label,
        )
        self._run_helper(op, samples)
    

    def test_target_tags(self):
        samples = [
            {
                Fields.meta: [
                    {
                        MetaKeys.query_sentiment_label: '开心'
                    },
                    {
                        MetaKeys.query_sentiment_label: '快乐'
                    },
                    {
                        MetaKeys.query_sentiment_label: '难过'
                    },
                    {
                        MetaKeys.query_sentiment_label: '不开心'
                    },
                    {
                        MetaKeys.query_sentiment_label: '愤怒'
                    }
                ]
            },
        ]
        op = MetaTagsAggregator(
            api_model='qwen2.5-72b-instruct',
            meta_tag_key=MetaKeys.query_sentiment_label,
            target_tags=['开心', '难过', '其他']
        )
        self._run_helper(op, samples)

    def test_tag_list(self):
        samples = [
            {
                Fields.meta: [
                    {
                        MetaKeys.dialog_sentiment_labels: ['开心', '平静']
                    },
                    {
                        MetaKeys.dialog_sentiment_labels: ['快乐', '开心', '幸福']
                    },
                    {
                        MetaKeys.dialog_sentiment_labels: ['难过']
                    },
                    {
                        MetaKeys.dialog_sentiment_labels: ['不开心', '没头脑', '不高兴']
                    },
                    {
                        MetaKeys.dialog_sentiment_labels: ['愤怒', '愤慨']
                    }
                ]
            },
        ]
        op = MetaTagsAggregator(
            api_model='qwen2.5-72b-instruct',
            meta_tag_key=MetaKeys.dialog_sentiment_labels,
            target_tags=['开心', '难过', '其他']
        )
        self._run_helper(op, samples)

if __name__ == '__main__':
    unittest.main()
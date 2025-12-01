import unittest

from loguru import logger

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.query_sentiment_detection_mapper import QuerySentimentDetectionMapper
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase
from data_juicer.utils.constant import Fields, MetaKeys

class TestQuerySentimentDetectionMapper(DataJuicerTestCaseBase):

    hf_model = 'mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis'
    zh_to_en_hf_model = 'Helsinki-NLP/opus-mt-zh-en'

    def _run_op(self, op, samples, targets, label_key=None, score_key=None):
        dataset = Dataset.from_list(samples)
        dataset = op.run(dataset)

        label_key = label_key or MetaKeys.query_sentiment_label
        score_key = score_key or MetaKeys.query_sentiment_score

        for sample, target in zip(dataset, targets):
            label = sample[Fields.meta][label_key]
            score = sample[Fields.meta][score_key]
            logger.info(f'{label}: {score}')
            self.assertEqual(label, target)
        
    def test_default(self):
        
        samples = [{
            'query': '太棒了！'
        },{
            'query': '嗯嗯'
        },{
            'query': '没有希望。'
        },
        ]
        targets = ['positive', 'neutral', 'negative']

        op = QuerySentimentDetectionMapper(
            hf_model = self.hf_model,
            zh_to_en_hf_model = self.zh_to_en_hf_model,
        )
        self._run_op(op, samples, targets)
    
    def test_no_zh_to_en(self):
        
        samples = [{
            'query': '太棒了！'
        },{
            'query': 'That is great!'
        }
        ]
        targets = ['neutral', 'positive']

        op = QuerySentimentDetectionMapper(
            hf_model = self.hf_model,
            zh_to_en_hf_model = None,
        )
        self._run_op(op, samples, targets)

    def test_rename_keys(self):
        
        samples = [{
            'query': '太棒了！'
        },{
            'query': '嗯嗯'
        },{
            'query': '没有希望。'
        },
        ]
        targets = ['positive', 'neutral', 'negative']

        label_key = 'my_label'
        score_key = 'my_score'
        op = QuerySentimentDetectionMapper(
            hf_model = self.hf_model,
            zh_to_en_hf_model = self.zh_to_en_hf_model,
            label_key = label_key,
            score_key = score_key,
        )
        self._run_op(op, samples, targets, label_key, score_key)


if __name__ == '__main__':
    unittest.main()

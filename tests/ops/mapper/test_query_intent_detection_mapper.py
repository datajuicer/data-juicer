import unittest
import json

from loguru import logger

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.query_intent_detection_mapper import QueryIntentDetectionMapper
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase
from data_juicer.utils.constant import Fields, MetaKeys

class TestQueryIntentDetectionMapper(DataJuicerTestCaseBase):

    hf_model = 'bespin-global/klue-roberta-small-3i4k-intent-classification'
    zh_to_en_hf_model = 'Helsinki-NLP/opus-mt-zh-en'

    def _run_op(self, op, samples, targets, label_key=None, score_key=None):
        dataset = Dataset.from_list(samples)
        dataset = op.run(dataset)

        label_key = label_key or MetaKeys.query_intent_label
        score_key = score_key or MetaKeys.query_intent_score

        for sample, target in zip(dataset, targets):
            label = sample[Fields.meta][label_key]
            score = sample[Fields.meta][score_key]
            logger.info(f'{label}: {score}')
            self.assertEqual(label, target)
        
    def test_default(self):
        
        samples = [{
            'query': '这样好吗？'
        },{
            'query': '站住！'
        },{
            'query': '今天阳光灿烂。'
        }
        ]
        targets = ['question', 'command', 'statement']

        op = QueryIntentDetectionMapper(
            hf_model = self.hf_model,
            zh_to_en_hf_model = self.zh_to_en_hf_model,
        )
        self._run_op(op, samples, targets)
    
    def test_no_zh_to_en(self):
        
        samples = [{
            'query': '这样好吗？'
        },{
            'query': 'Is this okay?'
        }
        ]
        targets = ['question', 'rhetorical question']

        op = QueryIntentDetectionMapper(
            hf_model = self.hf_model,
            zh_to_en_hf_model = None,
        )
        self._run_op(op, samples, targets)

    def test_rename_keys(self):
        
        samples = [{
            'query': '这样好吗？'
        },{
            'query': '站住！'
        },{
            'query': '今天阳光灿烂。'
        }
        ]
        targets = ['question', 'command', 'statement']

        label_key = 'my_label'
        score_key = 'my_score'
        op = QueryIntentDetectionMapper(
            hf_model = self.hf_model,
            zh_to_en_hf_model = self.zh_to_en_hf_model,
            label_key = label_key,
            score_key = score_key,
        )
        self._run_op(op, samples, targets, label_key, score_key)

if __name__ == '__main__':
    unittest.main()

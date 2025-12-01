import os
import unittest

import torch.distributions

from data_juicer.analysis.collector import TextTokenDistCollector

from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

class TextTokenDistCollectorTest(DataJuicerTestCaseBase):

    test_data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                  '..',
                                  '..',
                                  'demos',
                                  'data',
                                  'demo-dataset.jsonl')

    tokenizer_model = 'EleutherAI/pythia-6.9b-deduped'

    @classmethod
    def tearDownClass(cls) -> None:
        super().tearDownClass(cls.tokenizer_model)

    def test_basic_func(self):
        collector = TextTokenDistCollector(self.tokenizer_model)
        dist = collector.collect(self.test_data_path, 'text')
        self.assertIsInstance(dist, torch.distributions.Categorical)


if __name__ == '__main__':
    unittest.main()

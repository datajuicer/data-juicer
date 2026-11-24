import unittest
from unittest.mock import patch

import torch

from data_juicer.core.data import NestedDataset as Dataset

from data_juicer.ops.filter.text_pair_similarity_filter import TextPairSimilarityFilter
from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class TextPairSimilarityFilterTest(DataJuicerTestCaseBase):

    hf_clip = "openai/clip-vit-base-patch32"

    text_key = "text"
    text_key_second = "target_text"

    
    @classmethod
    def tearDownClass(cls) -> None:
        super().tearDownClass(cls.hf_clip)

    def _run_filter(self, dataset: Dataset, op, tgt_list, num_proc=1):

        if Fields.stats not in dataset.features:
            # TODO:
            # this is a temp solution,
            # only add stats when calling filter op
            dataset = dataset.add_column(name=Fields.stats,
                                         column=[{}] * dataset.num_rows)

        dataset = dataset.map(op.compute_stats,
                              num_proc=num_proc,
                              with_rank=True)
        dataset = dataset.filter(op.process, num_proc=num_proc)
        dataset = dataset.select_columns(column_names=[self.text_key,
                                                       self.text_key_second])
        res_list = dataset.to_list()
        self.assertEqual(res_list, tgt_list)

    def test_no_eoc_special_token(self):

        ds_list = [{
            self.text_key_second: 'a lovely cat',
            self.text_key: 'a lovely cat',
        }, {
            self.text_key_second: 'a lovely cat',
            self.text_key: 'a cute cat',
        }, {
            self.text_key_second: 'a lovely cat',
            self.text_key: 'a black dog',
        }]
        tgt_list = [{
            self.text_key_second: 'a lovely cat',
            self.text_key: 'a cute cat',
        }]


        dataset = Dataset.from_list(ds_list)
        op = TextPairSimilarityFilter(hf_clip=self.hf_clip,
                                      any_or_all='any',
                                      min_score=0.85,
                                      max_score=0.99,
                                      text_key_second=self.text_key_second)
        self._run_filter(dataset, op, tgt_list)

    def test_compute_stats_stores_plain_python_float(self):
        class FakeBatch(dict):
            def to(self, _device):
                return self

        class FakeProcessor:
            def __call__(self, **_kwargs):
                return FakeBatch()

        class FakeModel:
            device = "cpu"

            def get_text_features(self, **_kwargs):
                return torch.tensor([[1.0, 0.0], [1.0, 0.0]])

        op = object.__new__(TextPairSimilarityFilter)
        op.model_key = "fake-model"
        op.text_key = self.text_key
        op.text_key_second = self.text_key_second
        op.use_cuda = lambda: False
        sample = {
            self.text_key: "first",
            self.text_key_second: "second",
            Fields.stats: {},
        }

        with patch(
            "data_juicer.ops.filter.text_pair_similarity_filter.get_model",
            return_value=(FakeModel(), FakeProcessor()),
        ):
            result = op.compute_stats_single(sample)

        similarity = result[Fields.stats][StatsKeys.text_pair_similarity]
        self.assertEqual(similarity, [1.0])
        self.assertIs(type(similarity[0]), float)


if __name__ == '__main__':
    unittest.main()

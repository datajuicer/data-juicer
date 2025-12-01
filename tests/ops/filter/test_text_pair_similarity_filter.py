import unittest

from data_juicer.core.data import NestedDataset as Dataset

from data_juicer.ops.filter.text_pair_similarity_filter import TextPairSimilarityFilter
from data_juicer.utils.constant import Fields
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


if __name__ == '__main__':
    unittest.main()

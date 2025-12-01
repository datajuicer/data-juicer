import os
import unittest

from data_juicer.core.data import NestedDataset as Dataset

from data_juicer.ops.filter.image_pair_similarity_filter import ImagePairSimilarityFilter
from data_juicer.utils.constant import Fields
from data_juicer.utils.mm_utils import SpecialTokens
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class ImagePairSimilarityFilterTest(DataJuicerTestCaseBase):

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')
    cat_path = os.path.join(data_path, 'cat.jpg')
    img2_path = os.path.join(data_path, 'img2.jpg')
    img3_path = os.path.join(data_path, 'img3.jpg')
    img5_path = os.path.join(data_path, 'img5.jpg')
    img7_path = os.path.join(data_path, 'img7.jpg')
    hf_clip = 'openai/clip-vit-base-patch32'
    
    @classmethod
    def tearDownClass(cls) -> None:
        super().tearDownClass(cls.hf_clip)

    def _run_filter(self, dataset: Dataset, op, num_proc=1):

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
        dataset = dataset.select_columns(column_names=['text', 'images'])
        res_list = dataset.to_list()
        print(res_list)

    def test_no_eoc_special_token(self):

        ds_list = [{
            'text': 'image pair 1',
            'images': [self.cat_path, self.img3_path]
        }, {
            'text': 'image pair 2',
            'images': [self.img3_path, self.img7_path]
        }, {
            'text': 'image pair 3',
            'images': [self.img2_path, self.img5_path]
        }]


        dataset = Dataset.from_list(ds_list)
        op = ImagePairSimilarityFilter(hf_clip=self.hf_clip,
                                       any_or_all='any',
                                       min_score=0.85,
                                       max_score=1)
        self._run_filter(dataset, op)


if __name__ == '__main__':
    unittest.main()

import os
import unittest

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.deduplicator.image_deduplicator import ImageDeduplicator
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class ImageDeduplicatorTest(DataJuicerTestCaseBase):

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')
    img1_path = os.path.join(data_path, 'img1.png')
    img2_path = os.path.join(data_path, 'img2.jpg')
    img3_path = os.path.join(data_path, 'img3.jpg')
    # img4.png is a duplicate sample of img1.png
    img4_path = os.path.join(data_path, 'img4.png')
    if not os.path.exists(img4_path):
        os.symlink(img1_path, img4_path)
    # img5.jpg is a duplicate sample of img2.jpg
    img5_path = os.path.join(data_path, 'img5.jpg')
    if not os.path.exists(img5_path):
        os.symlink(img2_path, img5_path)
    # img6.jpg is a duplicate sample of img3.jpg
    img6_path = os.path.join(data_path, 'img6.jpg')
    if not os.path.exists(img6_path):
        os.symlink(img3_path, img6_path)
    # img7.jpg is a duplicate sample of img6.jpg
    img7_path = os.path.join(data_path, 'img7.jpg')
    if not os.path.exists(img7_path):
        os.symlink(img6_path, img7_path)

    def _run_image_deduplicator(self, dataset: Dataset, target_list, op, show_num: int = 0):
        expected_keys = [op.image_key, op.text_key]
        key_list = [key for key in expected_keys
                    if len(target_list) > 0 and key in target_list[0]]

        dataset = dataset.map(op.compute_hash)
        dataset, dup_pairs = op.process(dataset, show_num=show_num)
        dataset = dataset.select_columns(column_names=key_list)
        res_list = dataset.to_list()
        self.assertEqual(res_list, target_list)
        return dup_pairs

    def test_0(self):

        ds_list = [{
            'images': [self.img1_path]
        }]
        tgt_list = [{'images': [self.img1_path]}]
        dataset = Dataset.from_list(ds_list)
        op = ImageDeduplicator()
        dup_pairs = self._run_image_deduplicator(dataset, tgt_list, op)
        self.assertEqual(len(dup_pairs), 0)

    def test_1(self):

        ds_list = [{
            'images': [self.img1_path]
        }, {
            'images': [self.img2_path]
        }, {
            'images': [self.img3_path]
        }]
        tgt_list = [{
            'images': [self.img1_path]
        }, {
            'images': [self.img2_path]
        }, {
            'images': [self.img3_path]
        }]
        dataset = Dataset.from_list(ds_list)
        op = ImageDeduplicator()
        self._run_image_deduplicator(dataset, tgt_list, op)
    
    def test_1_show_num(self):

        ds_list = [{
            'images': [self.img1_path]
        }, {
            'images': [self.img2_path]
        }, {
            'images': [self.img3_path]
        }]
        tgt_list = [{
            'images': [self.img1_path]
        }, {
            'images': [self.img2_path]
        }, {
            'images': [self.img3_path]
        }]
        dataset = Dataset.from_list(ds_list)
        op = ImageDeduplicator()
        dup_pairs = self._run_image_deduplicator(dataset, tgt_list, op, show_num=1)
        self.assertEqual(len(dup_pairs), 0)

    def test_2(self):

        ds_list = [{
            'images': [self.img1_path]
        }, {
            'images': [self.img2_path]
        }, {
            'images': [self.img2_path]
        }]
        tgt_list = [{'images': [self.img1_path]}, {'images': [self.img2_path]}]
        dataset = Dataset.from_list(ds_list)
        op = ImageDeduplicator()
        self._run_image_deduplicator(dataset, tgt_list, op)
    
    def test_2_show_num(self):
        ds_list = [{
            'images': [self.img1_path]
        }, {
            'images': [self.img2_path]
        }, {
            'images': [self.img2_path]
        }]
        tgt_list = [{'images': [self.img1_path]}, {'images': [self.img2_path]}]
        dataset = Dataset.from_list(ds_list)
        op = ImageDeduplicator()
        dup_pairs = self._run_image_deduplicator(dataset, tgt_list, op, show_num=1)
        self.assertEqual(len(dup_pairs), 1)

    def test_3(self):

        ds_list = [{
            'images': [self.img1_path]
        }, {
            'images': [self.img2_path]
        }, {
            'images': [self.img3_path]
        }, {
            'images': [self.img4_path]
        }, {
            'images': [self.img5_path]
        }, {
            'images': [self.img6_path]
        }, {
            'images': [self.img7_path]
        }]
        tgt_list = [{
            'images': [self.img1_path]
        }, {
            'images': [self.img2_path]
        }, {
            'images': [self.img3_path]
        }]
        dataset = Dataset.from_list(ds_list)
        op = ImageDeduplicator()
        self._run_image_deduplicator(dataset, tgt_list, op)

    def test_3_consider_text(self):

        ds_list = [{
            'images': [self.img1_path],
            'text': '<video> text1'
        }, {
            'images': [self.img2_path],
            'text': '<video> text2'
        }, {
            'images': [self.img3_path],
            'text': '<video> text3'
        }, {
            'images': [self.img4_path],
            'text': '<video> text1'
        }, {
            'images': [self.img5_path],
            'text': '<video> text5'
        }, {
            'images': [self.img6_path],
            'text': '<video> text3'
        }, {
            'images': [self.img7_path],
            'text': '<video> text7'
        }]
        tgt_list = [{
            'images': [self.img1_path],
            'text': '<video> text1'
        }, {
            'images': [self.img2_path],
            'text': '<video> text2'
        }, {
            'images': [self.img3_path],
            'text': '<video> text3'
        }, {
            'images': [self.img5_path],
            'text': '<video> text5'
        }, {
            'images': [self.img7_path],
            'text': '<video> text7'
        }]
        dataset = Dataset.from_list(ds_list)
        op = ImageDeduplicator(consider_text=True)
        self._run_image_deduplicator(dataset, tgt_list, op)
    
    def test_3_consider_text_show_num(self):

        ds_list = [{
            'images': [self.img1_path],
            'text': '<video> text1'
        }, {
            'images': [self.img2_path],
            'text': '<video> text2'
        }, {
            'images': [self.img3_path],
            'text': '<video> text3'
        }, {
            'images': [self.img4_path],
            'text': '<video> text1'
        }, {
            'images': [self.img5_path],
            'text': '<video> text5'
        }, {
            'images': [self.img6_path],
            'text': '<video> text3'
        }, {
            'images': [self.img7_path],
            'text': '<video> text7'
        }]
        tgt_list = [{
            'images': [self.img1_path],
            'text': '<video> text1'
        }, {
            'images': [self.img2_path],
            'text': '<video> text2'
        }, {
            'images': [self.img3_path],
            'text': '<video> text3'
        }, {
            'images': [self.img5_path],
            'text': '<video> text5'
        }, {
            'images': [self.img7_path],
            'text': '<video> text7'
        }]
        dataset = Dataset.from_list(ds_list)
        op = ImageDeduplicator(consider_text=True)
        dup_pairs = self._run_image_deduplicator(dataset, tgt_list, op, show_num=2)
        self.assertEqual(len(dup_pairs), 2)

    def test_4(self):

        ds_list = [{
            'images': [self.img1_path, self.img2_path, self.img3_path]
        }, {
            'images': [self.img4_path, self.img5_path, self.img6_path]
        }, {
            'images': [self.img7_path]
        }, {
            'images': [self.img6_path]
        }]
        tgt_list = [{
            'images': [self.img1_path, self.img2_path, self.img3_path]
        }, {
            'images': [self.img7_path]
        }]
        dataset = Dataset.from_list(ds_list)
        op = ImageDeduplicator()
        self._run_image_deduplicator(dataset, tgt_list, op)

    def test_4_consider_text(self):

        ds_list = [{
            'images': [self.img1_path, self.img2_path, self.img3_path],
            'text': '<image> text1 <image> text2 <image> text3',
        }, {
            'images': [self.img4_path, self.img5_path, self.img6_path],
            'text': '<image> text1 <image> text5 <image> text3',
        }, {
            'images': [self.img7_path],
            'text': '<image> text6',
        }, {
            'images': [self.img6_path],
            'text': '<image> text6',
        }]
        tgt_list = [{
            'images': [self.img1_path, self.img2_path, self.img3_path],
            'text': '<image> text1 <image> text2 <image> text3',
        }, {
            'images': [self.img4_path, self.img5_path, self.img6_path],
            'text': '<image> text1 <image> text5 <image> text3',
        }, {
            'images': [self.img7_path],
            'text': '<image> text6',
        }]
        dataset = Dataset.from_list(ds_list)
        op = ImageDeduplicator(consider_text=True)
        self._run_image_deduplicator(dataset, tgt_list, op)

    def test_5(self):

        ds_list = [{
            'images': [self.img1_path, self.img2_path]
        }, {
            'images': [self.img2_path, self.img1_path]
        }, {
            'images': [self.img4_path, self.img5_path]
        }, {
            'images': [self.img7_path, self.img7_path]
        }, {
            'images': [self.img6_path, self.img6_path]
        }]
        tgt_list = [{
            'images': [self.img1_path, self.img2_path]
        }, {
            'images': [self.img2_path, self.img1_path]
        }, {
            'images': [self.img7_path, self.img7_path]
        }]
        dataset = Dataset.from_list(ds_list)
        op = ImageDeduplicator()
        self._run_image_deduplicator(dataset, tgt_list, op)

    def test_6(self):

        ds_list = [{
            'images': [self.img1_path, self.img2_path]
        }, {
            'images': [self.img2_path, self.img1_path]
        }, {
            'images': [self.img4_path, self.img5_path]
        }, {
            'images': [self.img7_path, self.img7_path]
        }, {
            'images': [self.img6_path, self.img6_path]
        }]
        tgt_list = [{
            'images': [self.img1_path, self.img2_path]
        }, {
            'images': [self.img2_path, self.img1_path]
        }, {
            'images': [self.img7_path, self.img7_path]
        }]
        dataset = Dataset.from_list(ds_list)
        op = ImageDeduplicator(method='dhash')
        self._run_image_deduplicator(dataset, tgt_list, op)

    def test_7(self):

        ds_list = [{
            'images': [self.img1_path, self.img2_path]
        }, {
            'images': [self.img2_path, self.img1_path]
        }, {
            'images': [self.img4_path, self.img5_path]
        }, {
            'images': [self.img7_path, self.img7_path]
        }, {
            'images': [self.img6_path, self.img6_path]
        }]
        tgt_list = [{
            'images': [self.img1_path, self.img2_path]
        }, {
            'images': [self.img2_path, self.img1_path]
        }, {
            'images': [self.img7_path, self.img7_path]
        }]
        dataset = Dataset.from_list(ds_list)
        op = ImageDeduplicator(method='whash')
        self._run_image_deduplicator(dataset, tgt_list, op)

    def test_8(self):

        ds_list = [{
            'images': [self.img1_path, self.img2_path]
        }, {
            'images': [self.img2_path, self.img1_path]
        }, {
            'images': [self.img4_path, self.img5_path]
        }, {
            'images': [self.img7_path, self.img7_path]
        }, {
            'images': [self.img6_path, self.img6_path]
        }]
        tgt_list = [{
            'images': [self.img1_path, self.img2_path]
        }, {
            'images': [self.img2_path, self.img1_path]
        }, {
            'images': [self.img7_path, self.img7_path]
        }]
        dataset = Dataset.from_list(ds_list)
        op = ImageDeduplicator(method='ahash')
        self._run_image_deduplicator(dataset, tgt_list, op)

    def test_no_image(self):

        ds_list = [{
            'images': [],
            'text': 'text1',
        }, {
            'images': [],
            'text': 'text2',
        }, {
            'images': [self.img7_path],
            'text': '<image> text6',
        }, {
            'images': [self.img6_path],
            'text': '<image> text6',
        }]
        tgt_list = [{
            'images': [],
            'text': 'text1',
        }, {
            'images': [],
            'text': 'text2',
        }, {
            'images': [self.img7_path],
            'text': '<image> text6',
        }]
        dataset = Dataset.from_list(ds_list)
        op = ImageDeduplicator()
        self._run_image_deduplicator(dataset, tgt_list, op)

    def test_no_image_consider_text(self):

        ds_list = [{
            'images': [],
            'text': 'text1',
        }, {
            'images': [],
            'text': 'text2',
        }, {
            'images': [],
            'text': 'text1',
        }, {
            'images': [],
            'text': 'text3',
        }, {
            'images': [self.img7_path],
            'text': '<image> text6',
        }, {
            'images': [self.img6_path],
            'text': '<image> text6',
        }]
        tgt_list = [{
            'images': [],
            'text': 'text1',
        }, {
            'images': [],
            'text': 'text2',
        }, {
            'images': [],
            'text': 'text3',
        }, {
            'images': [self.img7_path],
            'text': '<image> text6',
        }]
        dataset = Dataset.from_list(ds_list)
        op = ImageDeduplicator(consider_text=True)
        self._run_image_deduplicator(dataset, tgt_list, op)
    
    def test_error_hash_method(self):
        with self.assertRaises(ValueError):
            op = ImageDeduplicator(method='error')
            


if __name__ == '__main__':
    unittest.main()

import os
import unittest

import numpy as np

from data_juicer.core.data import NestedDataset as Dataset

from data_juicer.ops.deduplicator.ray_image_deduplicator import RayImageDeduplicator
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase, TEST_TAG


class RayImageDeduplicatorTest(DataJuicerTestCaseBase):

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

    def _run_image_deduplicator(self, dataset: Dataset, target_list, op):
        expected_keys = [op.image_key, op.text_key]
        key_list = [key for key in expected_keys
                    if len(target_list) > 0 and key in target_list[0]]

        res_list = self.run_single_op(dataset, op, key_list)
        self.assertEqual(len(res_list), len(target_list))

    @TEST_TAG("ray")
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
        dataset = self.generate_dataset(ds_list)
        op = RayImageDeduplicator()
        self._run_image_deduplicator(dataset, tgt_list, op)

    @TEST_TAG("ray")
    def test_2(self):

        ds_list = [{
            'images': [self.img1_path]
        }, {
            'images': [self.img2_path]
        }, {
            'images': [self.img2_path]
        }]
        tgt_list = [{'images': [self.img1_path]}, {'images': [self.img2_path]}]
        dataset = self.generate_dataset(ds_list)
        op = RayImageDeduplicator()
        self._run_image_deduplicator(dataset, tgt_list, op)

    @TEST_TAG("ray")
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
        dataset = self.generate_dataset(ds_list)
        op = RayImageDeduplicator()
        self._run_image_deduplicator(dataset, tgt_list, op)

    # @TEST_TAG("ray")
    # def test_3_consider_text(self):

    #     ds_list = [{
    #         'images': [self.img1_path],
    #         'text': '<video> text1'
    #     }, {
    #         'images': [self.img2_path],
    #         'text': '<video> text2'
    #     }, {
    #         'images': [self.img3_path],
    #         'text': '<video> text3'
    #     }, {
    #         'images': [self.img4_path],
    #         'text': '<video> text1'
    #     }, {
    #         'images': [self.img5_path],
    #         'text': '<video> text5'
    #     }, {
    #         'images': [self.img6_path],
    #         'text': '<video> text3'
    #     }, {
    #         'images': [self.img7_path],
    #         'text': '<video> text7'
    #     }]
    #     tgt_list = [{
    #         'images': [self.img1_path],
    #         'text': '<video> text1'
    #     }, {
    #         'images': [self.img2_path],
    #         'text': '<video> text2'
    #     }, {
    #         'images': [self.img3_path],
    #         'text': '<video> text3'
    #     }, {
    #         'images': [self.img5_path],
    #         'text': '<video> text5'
    #     }, {
    #         'images': [self.img7_path],
    #         'text': '<video> text7'
    #     }]
    #     dataset = self.generate_dataset(ds_list)
    #     op = RayImageDeduplicator(consider_text=True)
    #     self._run_image_deduplicator(dataset, tgt_list, op)

    @TEST_TAG("ray")
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
        dataset = self.generate_dataset(ds_list)
        op = RayImageDeduplicator()
        self._run_image_deduplicator(dataset, tgt_list, op)

    # @TEST_TAG("ray")
    # def test_4_consider_text(self):

    #     ds_list = [{
    #         'images': [self.img1_path, self.img2_path, self.img3_path],
    #         'text': '<image> text1 <image> text2 <image> text3',
    #     }, {
    #         'images': [self.img4_path, self.img5_path, self.img6_path],
    #         'text': '<image> text1 <image> text5 <image> text3',
    #     }, {
    #         'images': [self.img7_path],
    #         'text': '<image> text6',
    #     }, {
    #         'images': [self.img6_path],
    #         'text': '<image> text6',
    #     }]
    #     tgt_list = [{
    #         'images': [self.img1_path, self.img2_path, self.img3_path],
    #         'text': '<image> text1 <image> text2 <image> text3',
    #     }, {
    #         'images': [self.img4_path, self.img5_path, self.img6_path],
    #         'text': '<image> text1 <image> text5 <image> text3',
    #     }, {
    #         'images': [self.img7_path],
    #         'text': '<image> text6',
    #     }]
    #     dataset = self.generate_dataset(ds_list)
    #     op = RayImageDeduplicator(consider_text=True)
    #     self._run_image_deduplicator(dataset, tgt_list, op)

    @TEST_TAG("ray")
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
        dataset = self.generate_dataset(ds_list)
        op = RayImageDeduplicator()
        self._run_image_deduplicator(dataset, tgt_list, op)

    @TEST_TAG("ray")
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
        dataset = self.generate_dataset(ds_list)
        op = RayImageDeduplicator(method='dhash')
        self._run_image_deduplicator(dataset, tgt_list, op)

    @TEST_TAG("ray")
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
        dataset = self.generate_dataset(ds_list)
        op = RayImageDeduplicator(method='whash')
        self._run_image_deduplicator(dataset, tgt_list, op)

    @TEST_TAG("ray")
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
        dataset = self.generate_dataset(ds_list)
        op = RayImageDeduplicator(method='ahash')
        self._run_image_deduplicator(dataset, tgt_list, op)

    # @TEST_TAG("ray")
    # def test_no_image(self):

    #     ds_list = [{
    #         'images': [],
    #         'text': 'text1',
    #     }, {
    #         'images': [],
    #         'text': 'text2',
    #     }, {
    #         'images': [self.img7_path],
    #         'text': '<image> text6',
    #     }, {
    #         'images': [self.img6_path],
    #         'text': '<image> text6',
    #     }]
    #     tgt_list = [{
    #         'images': [],
    #         'text': 'text1',
    #     }, {
    #         'images': [],
    #         'text': 'text2',
    #     }, {
    #         'images': [self.img7_path],
    #         'text': '<image> text6',
    #     }]
    #     dataset = self.generate_dataset(ds_list)
    #     op = RayImageDeduplicator()
    #     self._run_image_deduplicator(dataset, tgt_list, op)

    # @TEST_TAG("ray")
    # def test_no_image_consider_text(self):

    #     ds_list = [{
    #         'images': [],
    #         'text': 'text1',
    #     }, {
    #         'images': [],
    #         'text': 'text2',
    #     }, {
    #         'images': [],
    #         'text': 'text1',
    #     }, {
    #         'images': [],
    #         'text': 'text3',
    #     }, {
    #         'images': [self.img7_path],
    #         'text': '<image> text6',
    #     }, {
    #         'images': [self.img6_path],
    #         'text': '<image> text6',
    #     }]
    #     tgt_list = [{
    #         'images': [],
    #         'text': 'text1',
    #     }, {
    #         'images': [],
    #         'text': 'text2',
    #     }, {
    #         'images': [],
    #         'text': 'text3',
    #     }, {
    #         'images': [self.img7_path],
    #         'text': '<image> text6',
    #     }]
    #     dataset = self.generate_dataset(ds_list)
    #     op = RayImageDeduplicator(consider_text=True)
    #     self._run_image_deduplicator(dataset, tgt_list, op)


if __name__ == '__main__':
    unittest.main()

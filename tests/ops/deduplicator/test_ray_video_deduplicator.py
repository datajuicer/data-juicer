import os
import unittest

import numpy as np

from data_juicer.core.data import NestedDataset as Dataset

from data_juicer.ops.deduplicator.ray_video_deduplicator import RayVideoDeduplicator
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase, TEST_TAG


class RayVideoDeduplicatorTest(DataJuicerTestCaseBase):

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')
    video1_path = os.path.join(data_path, 'video1.mp4')
    video2_path = os.path.join(data_path, 'video2.mp4')
    video3_path = os.path.join(data_path, 'video3.mp4')
    # video6.mp4 is a duplicate sample of video1.mp4
    video6_path = os.path.join(data_path, 'video6.mp4')
    if not os.path.exists(video6_path):
        os.symlink(video1_path, video6_path)
    # video7.mp4 is a duplicate sample of video2.mp4
    video7_path = os.path.join(data_path, 'video7.mp4')
    if not os.path.exists(video7_path):
        os.symlink(video2_path, video7_path)
    # video8.mp4 is a duplicate sample of video3.mp4
    video8_path = os.path.join(data_path, 'video8.mp4')
    if not os.path.exists(video8_path):
        os.symlink(video3_path, video8_path)
    # video9.mp4 is a duplicate sample of video8.mp4
    video9_path = os.path.join(data_path, 'video9.mp4')
    if not os.path.exists(video9_path):
        os.symlink(video8_path, video9_path)

    def _run_video_deduplicator(self, dataset: Dataset, target_list, op):
        expected_keys = [op.video_key, op.text_key]
        key_list = [key for key in expected_keys
                    if len(target_list) > 0 and key in target_list[0]]

        res_list = self.run_single_op(dataset, op, key_list)
        self.assertEqual(len(res_list), len(target_list))

    @TEST_TAG("ray")
    def test_1(self):

        ds_list = [{
            'videos': [self.video1_path]
        }, {
            'videos': [self.video2_path]
        }, {
            'videos': [self.video3_path]
        }]
        tgt_list = [{
            'videos': [self.video1_path]
        }, {
            'videos': [self.video2_path]
        }, {
            'videos': [self.video3_path]
        }]
        dataset = self.generate_dataset(ds_list)
        op = RayVideoDeduplicator()
        self._run_video_deduplicator(dataset, tgt_list, op)

    @TEST_TAG("ray")
    def test_2(self):

        ds_list = [{
            'videos': [self.video1_path]
        }, {
            'videos': [self.video2_path]
        }, {
            'videos': [self.video2_path]
        }]
        tgt_list = [{
            'videos': [self.video1_path]
        }, {
            'videos': [self.video2_path]
        }]
        dataset = self.generate_dataset(ds_list)
        op = RayVideoDeduplicator()
        self._run_video_deduplicator(dataset, tgt_list, op)

    @TEST_TAG("ray")
    def test_3(self):

        ds_list = [{
            'videos': [self.video1_path]
        }, {
            'videos': [self.video2_path]
        }, {
            'videos': [self.video3_path]
        }, {
            'videos': [self.video6_path]
        }, {
            'videos': [self.video7_path]
        }, {
            'videos': [self.video8_path]
        }, {
            'videos': [self.video9_path]
        }]
        tgt_list = [{
            'videos': [self.video1_path]
        }, {
            'videos': [self.video2_path]
        }, {
            'videos': [self.video3_path]
        }]
        dataset = self.generate_dataset(ds_list)
        op = RayVideoDeduplicator()
        self._run_video_deduplicator(dataset, tgt_list, op)

    # @TEST_TAG("ray")
    # def test_3_consider_text(self):

    #     ds_list = [{
    #         'videos': [self.video1_path],
    #         'text': '<video> text1'
    #     }, {
    #         'videos': [self.video2_path],
    #         'text': '<video> text2'
    #     }, {
    #         'videos': [self.video3_path],
    #         'text': '<video> text3'
    #     }, {
    #         'videos': [self.video6_path],
    #         'text': '<video> text1'
    #     }, {
    #         'videos': [self.video7_path],
    #         'text': '<video> text5'
    #     }, {
    #         'videos': [self.video8_path],
    #         'text': '<video> text3'
    #     }, {
    #         'videos': [self.video9_path],
    #         'text': '<video> text7'
    #     }]
    #     tgt_list = [{
    #         'videos': [self.video1_path],
    #         'text': '<video> text1'
    #     }, {
    #         'videos': [self.video2_path],
    #         'text': '<video> text2'
    #     }, {
    #         'videos': [self.video3_path],
    #         'text': '<video> text3'
    #     }, {
    #         'videos': [self.video7_path],
    #         'text': '<video> text5'
    #     }, {
    #         'videos': [self.video9_path],
    #         'text': '<video> text7'
    #     }]
    #     dataset = self.generate_dataset(ds_list)
    #     op = RayVideoDeduplicator(consider_text=True)
    #     self._run_video_deduplicator(dataset, tgt_list, op)

    @TEST_TAG("ray")
    def test_4(self):

        ds_list = [{
            'videos': [self.video1_path, self.video2_path, self.video3_path]
        }, {
            'videos': [self.video6_path, self.video7_path, self.video8_path]
        }, {
            'videos': [self.video9_path, self.video7_path]
        }, {
            'videos': [self.video8_path, self.video7_path]
        }]
        tgt_list = [{
            'videos': [self.video1_path, self.video2_path, self.video3_path]
        }, {
            'videos': [self.video9_path, self.video7_path]
        }]
        dataset = self.generate_dataset(ds_list)
        op = RayVideoDeduplicator()
        self._run_video_deduplicator(dataset, tgt_list, op)

    # @TEST_TAG("ray")
    # def test_4_consider_text(self):

    #     ds_list = [{
    #         'videos': [self.video1_path, self.video2_path, self.video3_path],
    #         'text': '<video> text1 <video> text2 <video> text3',
    #     }, {
    #         'videos': [self.video6_path, self.video7_path, self.video8_path],
    #         'text': '<video> text1 <video> text2 <video> text3',
    #     }, {
    #         'videos': [self.video9_path, self.video7_path],
    #         'text': '<video> text3 <video> text2',
    #     }, {
    #         'videos': [self.video8_path, self.video7_path],
    #         'text': '<video> text6 <video> text2',
    #     }]
    #     tgt_list = [{
    #         'videos': [self.video1_path, self.video2_path, self.video3_path],
    #         'text': '<video> text1 <video> text2 <video> text3',
    #     }, {
    #         'videos': [self.video9_path, self.video7_path],
    #         'text': '<video> text3 <video> text2',
    #     }, {
    #         'videos': [self.video8_path, self.video7_path],
    #         'text': '<video> text6 <video> text2',
    #     }]
    #     dataset = self.generate_dataset(ds_list)
    #     op = RayVideoDeduplicator(consider_text=True)
    #     self._run_video_deduplicator(dataset, tgt_list, op)

    @TEST_TAG("ray")
    def test_5(self):

        ds_list = [{
            'videos': [self.video1_path, self.video2_path]
        }, {
            'videos': [self.video2_path, self.video1_path]
        }, {
            'videos': [self.video6_path, self.video7_path]
        }, {
            'videos': [self.video9_path, self.video9_path]
        }, {
            'videos': [self.video8_path, self.video8_path]
        }]
        tgt_list = [{
            'videos': [self.video1_path, self.video2_path]
        }, {
            'videos': [self.video2_path, self.video1_path]
        }, {
            'videos': [self.video9_path, self.video9_path]
        }]
        dataset = self.generate_dataset(ds_list)
        op = RayVideoDeduplicator()
        self._run_video_deduplicator(dataset, tgt_list, op)

    # @TEST_TAG("ray")
    # def test_no_video(self):

    #     ds_list = [{
    #         'videos': [],
    #         'text': '<video> text1'
    #     }, {
    #         'videos': [self.video2_path],
    #         'text': '<video> text2'
    #     }, {
    #         'videos': [self.video3_path],
    #         'text': '<video> text3'
    #     }, {
    #         'videos': [],
    #         'text': '<video> text1'
    #     }, {
    #         'videos': [self.video7_path],
    #         'text': '<video> text5'
    #     }, {
    #         'videos': [],
    #         'text': '<video> text3'
    #     }, {
    #         'videos': [self.video9_path],
    #         'text': '<video> text7'
    #     }]
    #     tgt_list = [{
    #         'videos': [],
    #         'text': '<video> text1'
    #     }, {
    #         'videos': [self.video2_path],
    #         'text': '<video> text2'
    #     }, {
    #         'videos': [self.video3_path],
    #         'text': '<video> text3'
    #     }, {
    #         'videos': [],
    #         'text': '<video> text1'
    #     }, {
    #         'videos': [],
    #         'text': '<video> text3'
    #     }]
    #     dataset = self.generate_dataset(ds_list)
    #     op = RayVideoDeduplicator()
    #     self._run_video_deduplicator(dataset, tgt_list, op)

    # @TEST_TAG("ray")
    # def test_no_video_consider_text(self):

    #     ds_list = [{
    #         'videos': [],
    #         'text': '<video> text1'
    #     }, {
    #         'videos': [self.video2_path],
    #         'text': '<video> text2'
    #     }, {
    #         'videos': [self.video3_path],
    #         'text': '<video> text3'
    #     }, {
    #         'videos': [],
    #         'text': '<video> text1'
    #     }, {
    #         'videos': [self.video7_path],
    #         'text': '<video> text5'
    #     }, {
    #         'videos': [],
    #         'text': '<video> text3'
    #     }, {
    #         'videos': [self.video9_path],
    #         'text': '<video> text3'
    #     }]
    #     tgt_list = [{
    #         'videos': [],
    #         'text': '<video> text1'
    #     }, {
    #         'videos': [self.video2_path],
    #         'text': '<video> text2'
    #     }, {
    #         'videos': [self.video3_path],
    #         'text': '<video> text3'
    #     }, {
    #         'videos': [self.video7_path],
    #         'text': '<video> text5'
    #     }, {
    #         'videos': [],
    #         'text': '<video> text3'
    #     }]
    #     dataset = self.generate_dataset(ds_list)
    #     op = RayVideoDeduplicator(consider_text=True)
    #     self._run_video_deduplicator(dataset, tgt_list, op)


if __name__ == '__main__':
    unittest.main()

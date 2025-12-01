import os
import unittest

from data_juicer.core.data import NestedDataset as Dataset

from data_juicer.ops.deduplicator.video_deduplicator import VideoDeduplicator
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class VideoDeduplicatorTest(DataJuicerTestCaseBase):

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

    def _run_video_deduplicator(self, dataset: Dataset, target_list, op, show_num: int = 0):
        expected_keys = [op.video_key, op.text_key]
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
            'videos': [self.video1_path]
        }]
        tag_list = [{
            'videos': [self.video1_path]
        }]
        dataset = Dataset.from_list(ds_list)
        op = VideoDeduplicator()
        dup_pairs = self._run_video_deduplicator(dataset, tag_list, op)
        self.assertEqual(len(dup_pairs), 0)

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
        dataset = Dataset.from_list(ds_list)
        op = VideoDeduplicator()
        self._run_video_deduplicator(dataset, tgt_list, op)
    
    def test_1_show_num(self):
        
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
        dataset = Dataset.from_list(ds_list)
        op = VideoDeduplicator()
        dup_pairs = self._run_video_deduplicator(dataset, tgt_list, op, show_num=1)
        self.assertEqual(len(dup_pairs), 0)

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
        dataset = Dataset.from_list(ds_list)
        op = VideoDeduplicator()
        self._run_video_deduplicator(dataset, tgt_list, op)
    
    def test_2_show_num(self):

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
        dataset = Dataset.from_list(ds_list)
        op = VideoDeduplicator()
        dup_pairs = self._run_video_deduplicator(dataset, tgt_list, op, show_num=1)
        self.assertEqual(len(dup_pairs), 1)

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
        dataset = Dataset.from_list(ds_list)
        op = VideoDeduplicator()
        self._run_video_deduplicator(dataset, tgt_list, op)

    def test_3_consider_text(self):

        ds_list = [{
            'videos': [self.video1_path],
            'text': '<video> text1'
        }, {
            'videos': [self.video2_path],
            'text': '<video> text2'
        }, {
            'videos': [self.video3_path],
            'text': '<video> text3'
        }, {
            'videos': [self.video6_path],
            'text': '<video> text1'
        }, {
            'videos': [self.video7_path],
            'text': '<video> text5'
        }, {
            'videos': [self.video8_path],
            'text': '<video> text3'
        }, {
            'videos': [self.video9_path],
            'text': '<video> text7'
        }]
        tgt_list = [{
            'videos': [self.video1_path],
            'text': '<video> text1'
        }, {
            'videos': [self.video2_path],
            'text': '<video> text2'
        }, {
            'videos': [self.video3_path],
            'text': '<video> text3'
        }, {
            'videos': [self.video7_path],
            'text': '<video> text5'
        }, {
            'videos': [self.video9_path],
            'text': '<video> text7'
        }]
        dataset = Dataset.from_list(ds_list)
        op = VideoDeduplicator(consider_text=True)
        self._run_video_deduplicator(dataset, tgt_list, op)
    
    def test_3_consider_text_show_num(self):

        ds_list = [{
            'videos': [self.video1_path],
            'text': '<video> text1'
        }, {
            'videos': [self.video2_path],
            'text': '<video> text2'
        }, {
            'videos': [self.video3_path],
            'text': '<video> text3'
        }, {
            'videos': [self.video6_path],
            'text': '<video> text1'
        }, {
            'videos': [self.video7_path],
            'text': '<video> text5'
        }, {
            'videos': [self.video8_path],
            'text': '<video> text3'
        }, {
            'videos': [self.video9_path],
            'text': '<video> text7'
        }]
        tgt_list = [{
            'videos': [self.video1_path],
            'text': '<video> text1'
        }, {
            'videos': [self.video2_path],
            'text': '<video> text2'
        }, {
            'videos': [self.video3_path],
            'text': '<video> text3'
        }, {
            'videos': [self.video7_path],
            'text': '<video> text5'
        }, {
            'videos': [self.video9_path],
            'text': '<video> text7'
        }]
        dataset = Dataset.from_list(ds_list)
        op = VideoDeduplicator(consider_text=True)
        dup_pairs = self._run_video_deduplicator(dataset, tgt_list, op, show_num=1)
        self.assertEqual(len(dup_pairs), 1)

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
        dataset = Dataset.from_list(ds_list)
        op = VideoDeduplicator()
        self._run_video_deduplicator(dataset, tgt_list, op)

    def test_4_consider_text(self):

        ds_list = [{
            'videos': [self.video1_path, self.video2_path, self.video3_path],
            'text': '<video> text1 <video> text2 <video> text3',
        }, {
            'videos': [self.video6_path, self.video7_path, self.video8_path],
            'text': '<video> text1 <video> text2 <video> text3',
        }, {
            'videos': [self.video9_path, self.video7_path],
            'text': '<video> text3 <video> text2',
        }, {
            'videos': [self.video8_path, self.video7_path],
            'text': '<video> text6 <video> text2',
        }]
        tgt_list = [{
            'videos': [self.video1_path, self.video2_path, self.video3_path],
            'text': '<video> text1 <video> text2 <video> text3',
        }, {
            'videos': [self.video9_path, self.video7_path],
            'text': '<video> text3 <video> text2',
        }, {
            'videos': [self.video8_path, self.video7_path],
            'text': '<video> text6 <video> text2',
        }]
        dataset = Dataset.from_list(ds_list)
        op = VideoDeduplicator(consider_text=True)
        self._run_video_deduplicator(dataset, tgt_list, op)

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
        dataset = Dataset.from_list(ds_list)
        op = VideoDeduplicator()
        self._run_video_deduplicator(dataset, tgt_list, op)

    def test_no_video(self):

        ds_list = [{
            'videos': [],
            'text': '<video> text1'
        }, {
            'videos': [self.video2_path],
            'text': '<video> text2'
        }, {
            'videos': [self.video3_path],
            'text': '<video> text3'
        }, {
            'videos': [],
            'text': '<video> text1'
        }, {
            'videos': [self.video7_path],
            'text': '<video> text5'
        }, {
            'videos': [],
            'text': '<video> text3'
        }, {
            'videos': [self.video9_path],
            'text': '<video> text7'
        }]
        tgt_list = [{
            'videos': [],
            'text': '<video> text1'
        }, {
            'videos': [self.video2_path],
            'text': '<video> text2'
        }, {
            'videos': [self.video3_path],
            'text': '<video> text3'
        }, {
            'videos': [],
            'text': '<video> text1'
        }, {
            'videos': [],
            'text': '<video> text3'
        }]
        dataset = Dataset.from_list(ds_list)
        op = VideoDeduplicator()
        self._run_video_deduplicator(dataset, tgt_list, op)

    def test_no_video_consider_text(self):

        ds_list = [{
            'videos': [],
            'text': '<video> text1'
        }, {
            'videos': [self.video2_path],
            'text': '<video> text2'
        }, {
            'videos': [self.video3_path],
            'text': '<video> text3'
        }, {
            'videos': [],
            'text': '<video> text1'
        }, {
            'videos': [self.video7_path],
            'text': '<video> text5'
        }, {
            'videos': [],
            'text': '<video> text3'
        }, {
            'videos': [self.video9_path],
            'text': '<video> text3'
        }]
        tgt_list = [{
            'videos': [],
            'text': '<video> text1'
        }, {
            'videos': [self.video2_path],
            'text': '<video> text2'
        }, {
            'videos': [self.video3_path],
            'text': '<video> text3'
        }, {
            'videos': [self.video7_path],
            'text': '<video> text5'
        }, {
            'videos': [],
            'text': '<video> text3'
        }]
        dataset = Dataset.from_list(ds_list)
        op = VideoDeduplicator(consider_text=True)
        self._run_video_deduplicator(dataset, tgt_list, op)


if __name__ == '__main__':
    unittest.main()

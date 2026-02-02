import os
import unittest
import cv2
import math

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.video_split_by_frame_mapper import VideoSplitByFrameMapper
from data_juicer.utils.mm_utils import SpecialTokens
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

class VideoSplitByFrameMapperTest(DataJuicerTestCaseBase):

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data')
    vid1_path = os.path.join(data_path, 'video1.mp4')
    vid2_path = os.path.join(data_path, 'video2.mp4')
    vid3_path = os.path.join(data_path, 'video3.mp4')

    def _get_res_list(self, dataset, source_list):
        res_list = []
        origin_paths = [self.vid1_path, self.vid2_path, self.vid3_path]
        idx = 0
        for sample in dataset.to_list():
            output_paths = sample['videos']

            # for keep_original_sample=True
            if set(output_paths) <= set(origin_paths):
                res_list.append({
                    'text': sample['text'],
                    'videos': sample['videos']
                })
                continue

            source = source_list[idx]
            idx += 1

            output_file_names = [
                os.path.splitext(os.path.basename(p))[0] for p in output_paths
            ]
            split_frames_nums = []
            for origin_path in source['videos']:
                origin_file_name = os.path.splitext(
                    os.path.basename(origin_path))[0]
                cnt = 0
                for output_file_name in output_file_names:
                    if origin_file_name in output_file_name:
                        cnt += 1
                split_frames_nums.append(cnt)

            res_list.append({
                'text': sample['text'],
                'split_frames_num': split_frames_nums
            })

        return res_list

    def _run_video_split_by_frame_mapper(self,
                                            op,
                                            source_list,
                                            target_list,
                                            num_proc=1):
        dataset = Dataset.from_list(source_list)
        dataset = dataset.map(op.process, num_proc=num_proc)
        res_list = self._get_res_list(dataset, source_list)
        self.assertEqual(res_list, target_list)

    def _get_frame_count(self, video_path):
        cap = cv2.VideoCapture(video_path)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return count

    def test_split_by_frame(self):
        # Calculate expected splits
        # video1
        frames1 = self._get_frame_count(self.vid1_path)
        split_len = 100
        overlap_len = 0
        expected_splits1 = math.ceil(frames1 / split_len)
        
        # video2
        frames2 = self._get_frame_count(self.vid2_path)
        expected_splits2 = math.ceil(frames2 / split_len)
        
        # video3
        frames3 = self._get_frame_count(self.vid3_path)
        expected_splits3 = math.ceil(frames3 / split_len)

        ds_list = [{
            'text': f'{SpecialTokens.video} video1',
            'videos': [self.vid1_path]
        }, {
            'text': f'{SpecialTokens.video} video2 {SpecialTokens.eoc}',
            'videos': [self.vid2_path]
        }, {
            'text': f'{SpecialTokens.video} video3 {SpecialTokens.eoc}',
            'videos': [self.vid3_path]
        }]
        
        tgt_list = [{
            'text': f'{SpecialTokens.video * expected_splits1} video1{SpecialTokens.eoc}',
            'split_frames_num': [expected_splits1]
        }, {
            'text': f'{SpecialTokens.video * expected_splits2} video2 {SpecialTokens.eoc}',
            'split_frames_num': [expected_splits2]
        }, {
            'text': f'{SpecialTokens.video * expected_splits3} video3 {SpecialTokens.eoc}',
            'split_frames_num': [expected_splits3]
        }]
        
        op = VideoSplitByFrameMapper(split_len=split_len, overlap_len=overlap_len, unit='frame', keep_original_sample=False)
        self._run_video_split_by_frame_mapper(op, ds_list, tgt_list)

    def test_split_by_frame_with_overlap(self):
        # video1
        frames1 = self._get_frame_count(self.vid1_path)
        split_len = 50
        overlap_len = 10
        # stride = 50
        # num_parts = ceil(frames1 / 50)
        expected_splits1 = math.ceil(frames1 / split_len)
        
        ds_list = [{
            'text': f'{SpecialTokens.video} video1',
            'videos': [self.vid1_path]
        }]
        
        tgt_list = [{
            'text': f'{SpecialTokens.video * expected_splits1} video1{SpecialTokens.eoc}',
            'split_frames_num': [expected_splits1]
        }]
        
        op = VideoSplitByFrameMapper(split_len=split_len, overlap_len=overlap_len, unit='frame', keep_original_sample=False)
        self._run_video_split_by_frame_mapper(op, ds_list, tgt_list)

if __name__ == '__main__':
    unittest.main()

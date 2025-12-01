import os
import unittest

from data_juicer.core.data import NestedDataset as Dataset

from data_juicer.ops.mapper.video_split_by_scene_mapper import \
    VideoSplitBySceneMapper
from data_juicer.utils.mm_utils import SpecialTokens
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

class VideoSplitBySceneMapperTest(DataJuicerTestCaseBase):

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')
    vid1_path = os.path.join(data_path, 'video1.mp4')  # about 12s
    vid2_path = os.path.join(data_path, 'video2.mp4')  # about 23s
    vid3_path = os.path.join(data_path, 'video3.mp4')  # about 50s

    vid1_base, vid1_ext = os.path.splitext(os.path.basename(vid1_path))
    vid2_base, vid2_ext = os.path.splitext(os.path.basename(vid2_path))
    vid3_base, vid3_ext = os.path.splitext(os.path.basename(vid3_path))

    op_name = 'video_split_by_scene_mapper'

    def get_res_list(self, dataset: Dataset):
        res_list = []
        for sample in dataset.to_list():
            scene_num = len(sample['videos'])
            if 'text' in sample:
                res_list.append({
                    'scene_num': scene_num,
                    'text': sample['text']
                })
            else:
                res_list.append({'scene_num': scene_num})
        return res_list

    def _run_helper(self, op, source_list, target_list):
        dataset = Dataset.from_list(source_list)
        dataset = dataset.map(op.process)
        res_list = self.get_res_list(dataset)
        self.assertEqual(res_list, target_list)

    def test_ContentDetector(self):
        ds_list = [
            {
                'videos': [self.vid1_path]  # 3 scenes
            },
            {
                'videos': [self.vid2_path]  # 1 scene
            },
            {
                'videos': [self.vid3_path]  # 2 scenes
            }
        ]
        tgt_list = [{'scene_num': 3}, {'scene_num': 1}, {'scene_num': 2}]
        op = VideoSplitBySceneMapper(detector='ContentDetector',
                                     threshold=27.0,
                                     min_scene_len=15)
        self._run_helper(op, ds_list, tgt_list)

    def test_AdaptiveDetector(self):
        ds_list = [
            {
                'videos': [self.vid1_path]  # 3 scenes
            },
            {
                'videos': [self.vid2_path]  # 1 scene
            },
            {
                'videos': [self.vid3_path]  # 8 scenes
            }
        ]
        tgt_list = [{'scene_num': 3}, {'scene_num': 1}, {'scene_num': 8}]
        op = VideoSplitBySceneMapper(detector='AdaptiveDetector',
                                     threshold=3.0,
                                     min_scene_len=15)
        self._run_helper(op, ds_list, tgt_list)

    def test_ThresholdDetector(self):
        ds_list = [
            {
                'videos': [self.vid1_path]  # 1 scene
            },
            {
                'videos': [self.vid2_path]  # 1 scene
            },
            {
                'videos': [self.vid3_path]  # 1 scene
            }
        ]
        tgt_list = [{'scene_num': 1}, {'scene_num': 1}, {'scene_num': 1}]
        op = VideoSplitBySceneMapper(detector='ThresholdDetector',
                                     threshold=12.0,
                                     min_scene_len=15)
        self._run_helper(op, ds_list, tgt_list)

    def test_default_progress(self):
        ds_list = [
            {
                'videos': [self.vid1_path]  # 3 scenes
            },
            {
                'videos': [self.vid2_path]  # 1 scene
            },
            {
                'videos': [self.vid3_path]  # 2 scenes
            }
        ]
        tgt_list = [{'scene_num': 3}, {'scene_num': 1}, {'scene_num': 2}]
        op = VideoSplitBySceneMapper(show_progress=True)
        self._run_helper(op, ds_list, tgt_list)

    def test_default_kwargs(self):
        ds_list = [
            {
                'videos': [self.vid1_path]  # 2 scenes
            },
            {
                'videos': [self.vid2_path]  # 1 scene
            },
            {
                'videos': [self.vid3_path]  # 2 scenes
            }
        ]
        tgt_list = [{'scene_num': 2}, {'scene_num': 1}, {'scene_num': 2}]
        op = VideoSplitBySceneMapper(luma_only=True, kernel_size=5)
        self._run_helper(op, ds_list, tgt_list)

    def test_default_with_text(self):
        ds_list = [
            {
                'text':
                f'{SpecialTokens.video} this is video1 {SpecialTokens.eoc}',
                'videos': [self.vid1_path]  # 3 scenes
            },
            {
                'text':
                f'{SpecialTokens.video} this is video2 {SpecialTokens.eoc}',
                'videos': [self.vid2_path]  # 1 scene
            },
            {
                'text':
                f'{SpecialTokens.video} this is video3 {SpecialTokens.eoc}',
                'videos': [self.vid3_path]  # 2 scenes
            }
        ]
        tgt_list = [
            {
                'text':
                f'{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video} this is video1 {SpecialTokens.eoc}',  # noqa: E501
                'scene_num': 3
            },
            {
                'text':
                f'{SpecialTokens.video} this is video2 {SpecialTokens.eoc}',
                'scene_num': 1
            },
            {
                'text':
                f'{SpecialTokens.video}{SpecialTokens.video} this is video3 {SpecialTokens.eoc}',  # noqa: E501
                'scene_num': 2
            }
        ]
        op = VideoSplitBySceneMapper()
        self._run_helper(op, ds_list, tgt_list)


if __name__ == '__main__':
    unittest.main()

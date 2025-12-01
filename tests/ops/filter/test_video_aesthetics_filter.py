import os
import unittest

from data_juicer.core.data import NestedDataset as Dataset

from data_juicer.ops.filter.video_aesthetics_filter import \
    VideoAestheticsFilter
from data_juicer.utils.constant import Fields
from data_juicer.utils.mm_utils import SpecialTokens
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

class VideoAestheticsFilterTest(DataJuicerTestCaseBase):

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')
    # vid-low:  keyframes -- 0.410, uniform-3 -- 0.410, uniform-5 -- 0.406
    # vid-mid:  keyframes -- 0.448, uniform-3 -- 0.419, uniform-5 -- 0.449
    # vid-high: keyframes -- 0.468, uniform-3 -- 0.474, uniform-5 -- 0.480
    vid_low_path = os.path.join(data_path, 'video4.mp4')
    vid_mid_path = os.path.join(data_path, 'video1.mp4')
    vid_high_path = os.path.join(data_path, 'video3.mp4')
    vid_low_text = (
        f'{SpecialTokens.video} [[q]]: Can you summarize what the girls '
        f'are doing in the video?\n", "[[a]]: Sure. The video shows a girl'
        f' brushing the hair of another girl who keeps moving her face '
        f'around while the first girl keeps brushing the hair.'
        f'{SpecialTokens.eoc}')
    vid_mid_text = (f'{SpecialTokens.video} 白色的小羊站在一旁讲话。'
                    f'旁边还有两只灰色猫咪和一只拉着灰狼的猫咪'
                    f'{SpecialTokens.eoc}')
    vid_high_text = (f'两个长头发的女子正坐在一张圆桌前讲话互动。 '
                     f'{SpecialTokens.video} {SpecialTokens.eoc}')

    hf_aesthetics_scorer = \
        'shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE'

    @classmethod
    def tearDownClass(cls) -> None:
        super().tearDownClass(cls.hf_aesthetics_scorer)

    def _run_video_aesthetics_filter(self,
                                     dataset: Dataset,
                                     target_list,
                                     op,
                                     np=1):
        if Fields.stats not in dataset.features:
            dataset = dataset.add_column(name=Fields.stats,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.compute_stats, num_proc=np)
        dataset = dataset.filter(op.process, num_proc=np)
        dataset = dataset.select_columns(column_names=[op.video_key])
        res_list = dataset.to_list()
        self.assertEqual(res_list, target_list)

    def test_default_filter(self):
        ds_list = [{
            'videos': [self.vid_low_path],
            'text': self.vid_low_text,
        }, {
            'videos': [self.vid_mid_path],
            'text': self.vid_mid_text,
        }, {
            'videos': [self.vid_high_path],
            'text': self.vid_high_text,
        }]
        tgt_list = [{
            'videos': [self.vid_low_path]
        }, {
            'videos': [self.vid_mid_path]
        }, {
            'videos': [self.vid_high_path]
        }]
        dataset = Dataset.from_list(ds_list)
        op = VideoAestheticsFilter(self.hf_aesthetics_scorer)
        self._run_video_aesthetics_filter(dataset, tgt_list, op)

    def test_filter_large_score_videos(self):
        ds_list = [{
            'videos': [self.vid_low_path],
            'text': self.vid_low_text,
        }, {
            'videos': [self.vid_mid_path],
            'text': self.vid_mid_text,
        }, {
            'videos': [self.vid_high_path],
            'text': self.vid_high_text,
        }]
        tgt_list = [{
            'videos': [self.vid_low_path]
        }, {
            'videos': [self.vid_mid_path]
        }]
        dataset = Dataset.from_list(ds_list)
        op = VideoAestheticsFilter(self.hf_aesthetics_scorer, max_score=0.45)
        self._run_video_aesthetics_filter(dataset, tgt_list, op)

    def test_filter_small_score_videos(self):
        ds_list = [{
            'videos': [self.vid_low_path],
            'text': self.vid_low_text,
        }, {
            'videos': [self.vid_mid_path],
            'text': self.vid_mid_text,
        }, {
            'videos': [self.vid_high_path],
            'text': self.vid_high_text,
        }]
        tgt_list = [{
            'videos': [self.vid_mid_path]
        }, {
            'videos': [self.vid_high_path]
        }]
        dataset = Dataset.from_list(ds_list)
        op = VideoAestheticsFilter(self.hf_aesthetics_scorer, min_score=0.415)
        self._run_video_aesthetics_filter(dataset, tgt_list, op)

    def test_filter_videos_within_range_keyframes(self):
        ds_list = [{
            'videos': [self.vid_low_path],
            'text': self.vid_low_text,
        }, {
            'videos': [self.vid_mid_path],
            'text': self.vid_mid_text,
        }, {
            'videos': [self.vid_high_path],
            'text': self.vid_high_text,
        }]
        tgt_list = [{'videos': [self.vid_mid_path]}]
        dataset = Dataset.from_list(ds_list)
        op = VideoAestheticsFilter(self.hf_aesthetics_scorer,
                                   min_score=0.415,
                                   max_score=0.47)
        self._run_video_aesthetics_filter(dataset, tgt_list, op)

    def test_filter_keyframes(self):
        ds_list = [{
            'videos': [self.vid_low_path],
            'text': self.vid_low_text,
        }, {
            'videos': [self.vid_mid_path],
            'text': self.vid_mid_text,
        }, {
            'videos': [self.vid_high_path],
            'text': self.vid_high_text,
        }]
        tgt_list = [
            {
                'videos': [self.vid_mid_path]
            },
        ]
        dataset = Dataset.from_list(ds_list)
        op = VideoAestheticsFilter(self.hf_aesthetics_scorer,
                                   min_score=0.411,
                                   max_score=0.45,
                                   frame_sampling_method='all_keyframes')
        self._run_video_aesthetics_filter(dataset, tgt_list, op)

    def test_filter_uniform_frames_with_different_frame_num(self):
        ds_list = [{
            'videos': [self.vid_low_path],
            'text': self.vid_low_text,
        }, {
            'videos': [self.vid_mid_path],
            'text': self.vid_mid_text,
        }, {
            'videos': [self.vid_high_path],
            'text': self.vid_high_text,
        }]
        tgt_list = [{'videos': [self.vid_mid_path]}]
        dataset = Dataset.from_list(ds_list)
        op = VideoAestheticsFilter(self.hf_aesthetics_scorer,
                                   min_score=0.41,
                                   max_score=0.48,
                                   frame_sampling_method='uniform',
                                   frame_num=5)
        self._run_video_aesthetics_filter(dataset, tgt_list, op)

    def test_any(self):
        ds_list = [{
            'videos': [self.vid_low_path, self.vid_mid_path],
            'text': self.vid_low_text + self.vid_mid_text,
        }, {
            'videos': [self.vid_mid_path, self.vid_high_path],
            'text': self.vid_mid_text + self.vid_high_text,
        }, {
            'videos': [self.vid_low_path, self.vid_high_path],
            'text': self.vid_low_text + self.vid_high_text,
        }]
        tgt_list = [{
            'videos': [self.vid_low_path, self.vid_mid_path]
        }, {
            'videos': [self.vid_mid_path, self.vid_high_path]
        }]
        dataset = Dataset.from_list(ds_list)
        op = VideoAestheticsFilter(self.hf_aesthetics_scorer,
                                   min_score=0.415,
                                   max_score=0.45,
                                   any_or_all='any')
        self._run_video_aesthetics_filter(dataset, tgt_list, op)

    def test_all(self):
        ds_list = [{
            'videos': [self.vid_low_path, self.vid_mid_path],
            'text': self.vid_low_text + self.vid_mid_text,
        }, {
            'videos': [self.vid_mid_path, self.vid_high_path],
            'text': self.vid_mid_text + self.vid_high_text,
        }, {
            'videos': [self.vid_low_path, self.vid_high_path],
            'text': self.vid_low_text + self.vid_high_text,
        }]
        tgt_list = []
        dataset = Dataset.from_list(ds_list)
        op = VideoAestheticsFilter(self.hf_aesthetics_scorer,
                                   min_score=0.415,
                                   max_score=0.45,
                                   any_or_all='all')
        self._run_video_aesthetics_filter(dataset, tgt_list, op)

    def test_filter_in_parallel(self):

        ds_list = [{
            'videos': [self.vid_low_path],
            'text': self.vid_low_text,
        }, {
            'videos': [self.vid_mid_path],
            'text': self.vid_mid_text,
        }, {
            'videos': [self.vid_high_path],
            'text': self.vid_high_text,
        }]
        tgt_list = [{'videos': [self.vid_mid_path]}]
        dataset = Dataset.from_list(ds_list)
        op = VideoAestheticsFilter(
            self.hf_aesthetics_scorer,
            min_score=0.415,
            max_score=0.45,
        )
        self._run_video_aesthetics_filter(dataset, tgt_list, op, np=2)


if __name__ == '__main__':
    unittest.main()

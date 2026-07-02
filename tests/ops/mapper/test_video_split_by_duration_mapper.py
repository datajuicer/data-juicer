# flake8: noqa: E501

import os
import shutil
import tempfile
import unittest

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.base_op import Fields
from data_juicer.ops.mapper.video_split_by_duration_mapper import \
    VideoSplitByDurationMapper
from data_juicer.utils.file_utils import add_suffix_to_filename
from data_juicer.utils.mm_utils import SpecialTokens, load_file_byte
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class VideoSplitByDurationMapperTest(DataJuicerTestCaseBase):

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')
    vid1_path = os.path.join(data_path, 'video1.mp4')
    vid2_path = os.path.join(data_path, 'video2.mp4')
    vid3_path = os.path.join(data_path, 'video3.mp4')
    tmp_dir = tempfile.TemporaryDirectory().name

    def tearDown(self):
        super().tearDown()
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

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

    def _run_video_split_by_duration_mapper(self,
                                            op,
                                            source_list,
                                            target_list,
                                            num_proc=1):
        dataset = Dataset.from_list(source_list)
        dataset = dataset.map(op.process, num_proc=num_proc)
        res_list = self._get_res_list(dataset, source_list)
        self.assertEqual(res_list, target_list)

    def test(self):
        ds_list = [{
            'text': f'{SpecialTokens.video} 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。',
            'videos': [self.vid1_path]
        }, {
            'text':
            f'{SpecialTokens.video} 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。{SpecialTokens.eoc}',
            'videos': [self.vid2_path]
        }, {
            'text':
            f'{SpecialTokens.video} 两个长头发的女子正坐在一张圆桌前讲话互动。 {SpecialTokens.eoc}',
            'videos': [self.vid3_path]
        }]
        tgt_list = [{
            'text':
            f'{SpecialTokens.video}{SpecialTokens.video} 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。{SpecialTokens.eoc}',
            'split_frames_num': [2]
        }, {
            'text':
            f'{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video} 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。{SpecialTokens.eoc}',
            'split_frames_num': [3]
        }, {
            'text':
            f'{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video} 两个长头发的女子正坐在一张圆桌前讲话互动。 {SpecialTokens.eoc}',
            'split_frames_num': [5]
        }]
        op = VideoSplitByDurationMapper(split_duration=10,
                                        keep_original_sample=False)
        self._run_video_split_by_duration_mapper(op, ds_list, tgt_list)

    def test_keep_ori_sample(self):
        ds_list = [{
            'text': f'{SpecialTokens.video} 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。',
            'videos': [self.vid1_path]
        }, {
            'text':
            f'{SpecialTokens.video} 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。{SpecialTokens.eoc}',
            'videos': [self.vid2_path]
        }, {
            'text':
            f'{SpecialTokens.video} 两个长头发的女子正坐在一张圆桌前讲话互动。 {SpecialTokens.eoc}',
            'videos': [self.vid3_path]
        }]
        tgt_list = [{
            'text': f'{SpecialTokens.video} 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。',
            'videos': [self.vid1_path]
        }, {
            'text':
            f'{SpecialTokens.video}{SpecialTokens.video} 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。{SpecialTokens.eoc}',
            'split_frames_num': [2]
        }, {
            'text':
            f'{SpecialTokens.video} 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。{SpecialTokens.eoc}',
            'videos': [self.vid2_path]
        }, {
            'text':
            f'{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video} 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。{SpecialTokens.eoc}',
            'split_frames_num': [3]
        }, {
            'text':
            f'{SpecialTokens.video} 两个长头发的女子正坐在一张圆桌前讲话互动。 {SpecialTokens.eoc}',
            'videos': [self.vid3_path]
        }, {
            'text':
            f'{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video} 两个长头发的女子正坐在一张圆桌前讲话互动。 {SpecialTokens.eoc}',
            'split_frames_num': [5]
        }]
        op = VideoSplitByDurationMapper()
        self._run_video_split_by_duration_mapper(op, ds_list, tgt_list)

    def test_multi_process(self):
        ds_list = [{
            'text': f'{SpecialTokens.video} 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。',
            'videos': [self.vid1_path]
        }, {
            'text':
            f'{SpecialTokens.video} 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。{SpecialTokens.eoc}',
            'videos': [self.vid2_path]
        }, {
            'text':
            f'{SpecialTokens.video} 两个长头发的女子正坐在一张圆桌前讲话互动。 {SpecialTokens.eoc}',
            'videos': [self.vid3_path]
        }]
        tgt_list = [{
            'text':
            f'{SpecialTokens.video}{SpecialTokens.video} 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。{SpecialTokens.eoc}',
            'split_frames_num': [2]
        }, {
            'text':
            f'{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video} 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。{SpecialTokens.eoc}',
            'split_frames_num': [3]
        }, {
            'text':
            f'{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video} 两个长头发的女子正坐在一张圆桌前讲话互动。 {SpecialTokens.eoc}',
            'split_frames_num': [5]
        }]
        op = VideoSplitByDurationMapper(keep_original_sample=False)
        self._run_video_split_by_duration_mapper(op,
                                                 ds_list,
                                                 tgt_list,
                                                 num_proc=2)

    def test_multi_chunk(self):
        ds_list = [{
            'text':
            f'{SpecialTokens.video} 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。{SpecialTokens.eoc}{SpecialTokens.video} 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。',
            'videos': [self.vid1_path, self.vid2_path]
        }, {
            'text':
            f'{SpecialTokens.video} 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。{SpecialTokens.eoc}{SpecialTokens.video} 两个长头发的女子正坐在一张圆桌前讲话互动。 {SpecialTokens.eoc}',
            'videos': [self.vid2_path, self.vid3_path]
        }, {
            'text':
            f'{SpecialTokens.video} 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。{SpecialTokens.eoc}{SpecialTokens.video} 两个长头发的女子正坐在一张圆桌前讲话互动。 {SpecialTokens.eoc}',
            'videos': [self.vid1_path, self.vid3_path]
        }]
        tgt_list = [{
            'text':
            f'{SpecialTokens.video}{SpecialTokens.video} 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。{SpecialTokens.eoc}{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video} 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。{SpecialTokens.eoc}',
            'split_frames_num': [2, 3]
        }, {
            'text':
            f'{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video} 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。{SpecialTokens.eoc}{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video} 两个长头发的女子正坐在一张圆桌前讲话互动。 {SpecialTokens.eoc}',
            'split_frames_num': [3, 5]
        }, {
            'text':
            f'{SpecialTokens.video}{SpecialTokens.video} 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。{SpecialTokens.eoc}{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video} 两个长头发的女子正坐在一张圆桌前讲话互动。 {SpecialTokens.eoc}',
            'split_frames_num': [2, 5]
        }]
        op = VideoSplitByDurationMapper(keep_original_sample=False, ffmpeg_extra_args='-movflags frag_keyframe+empty_moov')
        self._run_video_split_by_duration_mapper(op, ds_list, tgt_list)

    def test_min_last_split_duration(self):
        ds_list = [{
            'text': f'{SpecialTokens.video} 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。',
            'videos': [self.vid1_path]
        }, {
            'text':
            f'{SpecialTokens.video} 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。{SpecialTokens.eoc}',
            'videos': [self.vid2_path]
        }, {
            'text':
            f'{SpecialTokens.video} 两个长头发的女子正坐在一张圆桌前讲话互动。 {SpecialTokens.eoc}',
            'videos': [self.vid3_path]
        }]
        tgt_list = [{
            'text':
            f'{SpecialTokens.video} 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。{SpecialTokens.eoc}',
            'split_frames_num': [1]
        }, {
            'text':
            f'{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video} 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。{SpecialTokens.eoc}',
            'split_frames_num': [3]
        }, {
            'text':
            f'{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video} 两个长头发的女子正坐在一张圆桌前讲话互动。 {SpecialTokens.eoc}',
            'split_frames_num': [5]
        }]
        op = VideoSplitByDurationMapper(split_duration=10,
                                        min_last_split_duration=3,
                                        keep_original_sample=False)
        self._run_video_split_by_duration_mapper(op, ds_list, tgt_list)

    def test_output_format_bytes(self, save_field=None):
        ds_list = [{
            'text': f'{SpecialTokens.video} 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。',
            'videos': [self.vid1_path]
        }, {
            'text':
            f'{SpecialTokens.video} 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。{SpecialTokens.eoc}',
            'videos': [self.vid2_path]
        }, {
            'text':
            f'{SpecialTokens.video} 两个长头发的女子正坐在一张圆桌前讲话互动。 {SpecialTokens.eoc}',
            'videos': [self.vid3_path]
        }]
        tgt_list = [{
            'text':
            f'{SpecialTokens.video}{SpecialTokens.video} 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。{SpecialTokens.eoc}',
            'split_frames_num': 2
        }, {
            'text':
            f'{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video} 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。{SpecialTokens.eoc}',
            'split_frames_num': 3
        }, {
            'text':
            f'{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video} 两个长头发的女子正坐在一张圆桌前讲话互动。 {SpecialTokens.eoc}',
            'split_frames_num': 5
        }]
        op = VideoSplitByDurationMapper(
            split_duration=10,
            keep_original_sample=False,
            output_format="bytes",
            save_field=save_field,
            save_dir=self.tmp_dir,
            legacy_split_by_text_token=True)

        dataset = Dataset.from_list(ds_list)
        dataset = dataset.map(op.process, num_proc=1)
        res_list = dataset.to_list()

        save_field = save_field or "videos"
        for i in range(len(ds_list)):
            res = res_list[i]
            tgt = tgt_list[i]
            self.assertEqual(res['text'], tgt['text'])
            self.assertEqual(len(res[Fields.source_file]), tgt['split_frames_num'])
            for clip_path in res[Fields.source_file]:
                self.assertTrue(os.path.exists(clip_path))
            self.assertEqual(len(res[save_field]), tgt['split_frames_num'])
            self.assertTrue(all(isinstance(v, bytes) for v in res[save_field]))

    def test_output_format_bytes_save_field(self):
        self.test_output_format_bytes(save_field="clips")

    def test_input_video_bytes(self):
        ds_list = [{
            'text': f'{SpecialTokens.video} 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。',
            'videos': [load_file_byte(self.vid1_path)]
        }, {
            'text':
            f'{SpecialTokens.video} 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。{SpecialTokens.eoc}',
            'videos': [load_file_byte(self.vid2_path)]
        }, {
            'text':
            f'{SpecialTokens.video} 两个长头发的女子正坐在一张圆桌前讲话互动。 {SpecialTokens.eoc}',
            'videos': [load_file_byte(self.vid3_path)]
        }]
        tgt_list = [{
            'text':
            f'{SpecialTokens.video}{SpecialTokens.video} 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。{SpecialTokens.eoc}',
            'split_frames_num': 2
        }, {
            'text':
            f'{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video} 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。{SpecialTokens.eoc}',
            'split_frames_num': 3
        }, {
            'text':
            f'{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video} 两个长头发的女子正坐在一张圆桌前讲话互动。 {SpecialTokens.eoc}',
            'split_frames_num': 5
        }]

        save_field = "clips"
        op = VideoSplitByDurationMapper(
            split_duration=10,
            keep_original_sample=False,
            output_format="bytes",
            save_field=save_field,
            save_dir=self.tmp_dir,
            legacy_split_by_text_token=True,
            video_backend="ffmpeg")

        dataset = Dataset.from_list(ds_list)
        dataset = dataset.map(op.process, num_proc=1)
        res_list = dataset.to_list()

        for i in range(len(ds_list)):
            res = res_list[i]
            tgt = tgt_list[i]
            self.assertEqual(res['text'], tgt['text'])
            self.assertEqual(len(res[Fields.source_file]), tgt['split_frames_num'])
            for clip_path in res[Fields.source_file]:
                self.assertTrue(os.path.exists(clip_path))
            self.assertEqual(len(res[save_field]), tgt['split_frames_num'])
            self.assertTrue(all(isinstance(v, bytes) for v in res[save_field]))


    # ─── overlap_duration tests ───────────────────────────────────

    def test_overlap_basic(self):
        """split=10, overlap=5 → step=5.
        video1(11.76s): [0-10],[5-11.76] → 2 clips
        video2(23.17s): [0-10],[5-15],[10-20],[15-23.17] → 4 clips
        video3(49.58s): 9 clips
        """
        ds_list = [{
            'text': f'{SpecialTokens.video} vid1.',
            'videos': [self.vid1_path]
        }, {
            'text': f'{SpecialTokens.video} vid2.{SpecialTokens.eoc}',
            'videos': [self.vid2_path]
        }, {
            'text': f'{SpecialTokens.video} vid3.{SpecialTokens.eoc}',
            'videos': [self.vid3_path]
        }]
        tgt_list = [{
            'text': f'{SpecialTokens.video}{SpecialTokens.video} vid1.{SpecialTokens.eoc}',
            'split_frames_num': [2]
        }, {
            'text': f'{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video} vid2.{SpecialTokens.eoc}',
            'split_frames_num': [4]
        }, {
            'text': f'{"".join([SpecialTokens.video] * 9)} vid3.{SpecialTokens.eoc}',
            'split_frames_num': [9]
        }]
        op = VideoSplitByDurationMapper(
            split_duration=10,
            overlap_duration=5,
            keep_original_sample=False)
        self._run_video_split_by_duration_mapper(op, ds_list, tgt_list)

    def test_overlap_more_clips_than_no_overlap(self):
        """Overlap produces more clips than no-overlap for the same video.
        video2(23.17s):
          no-overlap split=10: [0-10],[10-20],[20-23.17] → 3 clips
          overlap=5  split=10: [0-10],[5-15],[10-20],[15-23.17] → 4 clips
        """
        ds_list = [{
            'text': f'{SpecialTokens.video} vid2.{SpecialTokens.eoc}',
            'videos': [self.vid2_path]
        }]
        op_no_overlap = VideoSplitByDurationMapper(
            split_duration=10,
            overlap_duration=0,
            keep_original_sample=False)
        op_overlap = VideoSplitByDurationMapper(
            split_duration=10,
            overlap_duration=5,
            keep_original_sample=False)

        dataset_no = Dataset.from_list(ds_list)
        dataset_no = dataset_no.map(op_no_overlap.process, num_proc=1)
        no_clips = len(dataset_no.to_list()[0]['videos'])

        dataset_ov = Dataset.from_list(ds_list)
        dataset_ov = dataset_ov.map(op_overlap.process, num_proc=1)
        ov_clips = len(dataset_ov.to_list()[0]['videos'])

        self.assertEqual(no_clips, 3)
        self.assertEqual(ov_clips, 4)
        self.assertGreater(ov_clips, no_clips)

    def test_overlap_short_video_no_split(self):
        """Video shorter than split_duration → return original, overlap irrelevant.
        video1(11.76s) with split=20, overlap=5: no split.
        """
        ds_list = [{
            'text': f'{SpecialTokens.video} vid1.',
            'videos': [self.vid1_path]
        }]
        op = VideoSplitByDurationMapper(
            split_duration=20,
            overlap_duration=5,
            keep_original_sample=False)
        dataset = Dataset.from_list(ds_list)
        dataset = dataset.map(op.process, num_proc=1)
        res = dataset.to_list()[0]
        # Original path returned as-is
        self.assertEqual(res['videos'], [self.vid1_path])

    def test_overlap_with_min_last_split_duration(self):
        """split=10, overlap=5, min_last=10 → drop short last segments.
        video1(11.76s): [0-10], then start=5 remaining=6.76<10 → 1 clip
        video2(23.17s): [0-10],[5-15],[10-20], then start=15 remaining=8.17<10 → 3 clips
        video3(49.58s): 8 clips, then start=40 remaining=9.58<10 → 8 clips
        """
        ds_list = [{
            'text': f'{SpecialTokens.video} vid1.',
            'videos': [self.vid1_path]
        }, {
            'text': f'{SpecialTokens.video} vid2.{SpecialTokens.eoc}',
            'videos': [self.vid2_path]
        }, {
            'text': f'{SpecialTokens.video} vid3.{SpecialTokens.eoc}',
            'videos': [self.vid3_path]
        }]
        tgt_list = [{
            'text': f'{SpecialTokens.video} vid1.{SpecialTokens.eoc}',
            'split_frames_num': [1]
        }, {
            'text': f'{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video} vid2.{SpecialTokens.eoc}',
            'split_frames_num': [3]
        }, {
            'text': f'{"".join([SpecialTokens.video] * 8)} vid3.{SpecialTokens.eoc}',
            'split_frames_num': [8]
        }]
        op = VideoSplitByDurationMapper(
            split_duration=10,
            overlap_duration=5,
            min_last_split_duration=10,
            keep_original_sample=False)
        self._run_video_split_by_duration_mapper(op, ds_list, tgt_list)

    def test_overlap_bytes_output(self):
        """Overlap with bytes output format."""
        ds_list = [{
            'text': f'{SpecialTokens.video} vid2.{SpecialTokens.eoc}',
            'videos': [self.vid2_path]
        }]
        save_field = "clips"
        op = VideoSplitByDurationMapper(
            split_duration=10,
            overlap_duration=5,
            keep_original_sample=False,
            output_format="bytes",
            save_field=save_field,
            save_dir=self.tmp_dir,
            legacy_split_by_text_token=True)

        dataset = Dataset.from_list(ds_list)
        dataset = dataset.map(op.process, num_proc=1)
        res = dataset.to_list()[0]

        # video2(23.17s), split=10, overlap=5 → 4 clips
        self.assertEqual(len(res[save_field]), 4)
        self.assertTrue(all(isinstance(v, bytes) for v in res[save_field]))
        self.assertEqual(len(res[Fields.source_file]), 4)

    def test_overlap_non_legacy_save_field(self):
        """Overlap with non-legacy mode (save_field, no text token update)."""
        ds_list = [{
            'text': '',
            'videos': [self.vid3_path]
        }]
        save_field = "clips"
        op = VideoSplitByDurationMapper(
            split_duration=10,
            overlap_duration=5,
            keep_original_sample=False,
            save_field=save_field,
            save_dir=self.tmp_dir,
            legacy_split_by_text_token=False)

        dataset = Dataset.from_list(ds_list)
        dataset = dataset.map(op.process, num_proc=1)
        res = dataset.to_list()[0]

        # video3(49.58s), split=10, overlap=5 → 9 clips
        self.assertEqual(len(res[save_field]), 9)
        # Original videos field untouched
        self.assertEqual(res['videos'], [self.vid3_path])

    def test_overlap_zero_same_as_default(self):
        """overlap_duration=0 produces identical results to the default."""
        ds_list = [{
            'text': f'{SpecialTokens.video} vid2.{SpecialTokens.eoc}',
            'videos': [self.vid2_path]
        }]
        op_default = VideoSplitByDurationMapper(
            split_duration=10,
            keep_original_sample=False)
        op_zero = VideoSplitByDurationMapper(
            split_duration=10,
            overlap_duration=0,
            keep_original_sample=False)

        dataset_d = Dataset.from_list(ds_list)
        dataset_d = dataset_d.map(op_default.process, num_proc=1)
        res_d = dataset_d.to_list()[0]

        dataset_z = Dataset.from_list(ds_list)
        dataset_z = dataset_z.map(op_zero.process, num_proc=1)
        res_z = dataset_z.to_list()[0]

        self.assertEqual(len(res_d['videos']), len(res_z['videos']))
        self.assertEqual(res_d['text'], res_z['text'])

    def test_overlap_validation_negative(self):
        """overlap_duration < 0 should raise AssertionError."""
        with self.assertRaises(AssertionError):
            VideoSplitByDurationMapper(split_duration=10, overlap_duration=-1)

    def test_overlap_validation_exceeds_split(self):
        """overlap_duration >= split_duration should raise AssertionError."""
        with self.assertRaises(AssertionError):
            VideoSplitByDurationMapper(split_duration=10, overlap_duration=10)
        with self.assertRaises(AssertionError):
            VideoSplitByDurationMapper(split_duration=10, overlap_duration=15)

    def test_overlap_multi_chunk(self):
        """Overlap with multi-video sample.
        video1(11.76s) split=10,overlap=5: 2 clips
        video3(49.58s) split=10,overlap=5: 9 clips
        """
        ds_list = [{
            'text': f'{SpecialTokens.video} v1.{SpecialTokens.eoc}{SpecialTokens.video} v3.{SpecialTokens.eoc}',
            'videos': [self.vid1_path, self.vid3_path]
        }]
        tgt_list = [{
            'text': f'{SpecialTokens.video}{SpecialTokens.video} v1.{SpecialTokens.eoc}{"".join([SpecialTokens.video] * 9)} v3.{SpecialTokens.eoc}',
            'split_frames_num': [2, 9]
        }]
        op = VideoSplitByDurationMapper(
            split_duration=10,
            overlap_duration=5,
            keep_original_sample=False)
        self._run_video_split_by_duration_mapper(op, ds_list, tgt_list)


if __name__ == '__main__':
    unittest.main()

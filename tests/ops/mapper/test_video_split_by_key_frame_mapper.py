# flake8: noqa: E501

import os
import unittest
import shutil
import tempfile

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.video_split_by_key_frame_mapper import \
    VideoSplitByKeyFrameMapper
from data_juicer.utils.mm_utils import SpecialTokens
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase, TEST_TAG


class VideoSplitByKeyFrameMapperTest(DataJuicerTestCaseBase):

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
        dataset = sorted(dataset, key=lambda x: x["id"])
        source_list = sorted(source_list, key=lambda x: x["id"])
        res_list = []
        origin_paths = [self.vid1_path, self.vid2_path, self.vid3_path]
        idx = 0
        for sample in dataset:
            output_paths = sample['videos']
            # for keep_original_sample=True
            if set(output_paths) <= set(origin_paths):
                res_list.append({
                    'id': sample['id'],
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
                'id': sample['id'],
                'text': sample['text'],
                'split_frames_num': split_frames_nums
            })

        return res_list

    def _run_video_split_by_key_frame_mapper(self,
                                             op,
                                             source_list,
                                             target_list,
                                             num_proc=1):
        dataset = self.generate_dataset(source_list)
        # TODO: use num_proc
        dataset = self.run_single_op(dataset, op, ["id", "text", "videos"])
        res_list = self._get_res_list(dataset, source_list)
        self.assertDatasetEqual(res_list, target_list)

    @TEST_TAG("standalone", "ray")
    def test(self):
        ds_list = [{
            'id': 0,
            'text': f'{SpecialTokens.video} 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。',
            'videos': [self.vid1_path]
        }, {
            'id': 1,
            'text':
            f'{SpecialTokens.video} 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。{SpecialTokens.eoc}',
            'videos': [self.vid2_path]
        }, {
            'id': 2,
            'text':
            f'{SpecialTokens.video} 两个长头发的女子正坐在一张圆桌前讲话互动。 {SpecialTokens.eoc}',
            'videos': [self.vid3_path]
        }]
        tgt_list = [{
            'id': 0,
            'text':
            f'{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video} 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。{SpecialTokens.eoc}',
            'split_frames_num': [3]
        }, {
            'id': 1,
            'text':
            f'{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video} 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。{SpecialTokens.eoc}',
            'split_frames_num': [3]
        }, {
            'id': 2,
            'text':
            f'{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video} 两个长头发的女子正坐在一张圆桌前讲话互动。 {SpecialTokens.eoc}',
            'split_frames_num': [6]
        }]
        op = VideoSplitByKeyFrameMapper(keep_original_sample=False)
        self._run_video_split_by_key_frame_mapper(op, ds_list, tgt_list)

    @TEST_TAG("standalone", "ray")
    def test_keep_ori_sample(self):
        ds_list = [{
            'id': 0,
            'text': f'{SpecialTokens.video} 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。',
            'videos': [self.vid1_path]
        }, {
            'id': 1,
            'text':
            f'{SpecialTokens.video} 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。{SpecialTokens.eoc}',
            'videos': [self.vid2_path]
        }, {
            'id': 2,
            'text':
            f'{SpecialTokens.video} 两个长头发的女子正坐在一张圆桌前讲话互动。 {SpecialTokens.eoc}',
            'videos': [self.vid3_path]
        }]
        tgt_list = [{
            'id': 0,
            'text': f'{SpecialTokens.video} 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。',
            'videos': [self.vid1_path]
        }, {
            'id': 0,
            'text':
            f'{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video} 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。{SpecialTokens.eoc}',
            'split_frames_num': [3]
        }, {
            'id': 1,
            'text':
            f'{SpecialTokens.video} 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。{SpecialTokens.eoc}',
            'videos': [self.vid2_path]
        }, {
            'id': 1,
            'text':
            f'{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video} 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。{SpecialTokens.eoc}',
            'split_frames_num': [3]
        }, {
            'id': 2,
            'text':
            f'{SpecialTokens.video} 两个长头发的女子正坐在一张圆桌前讲话互动。 {SpecialTokens.eoc}',
            'videos': [self.vid3_path]
        }, {
            'id': 2,
            'text':
            f'{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video} 两个长头发的女子正坐在一张圆桌前讲话互动。 {SpecialTokens.eoc}',
            'split_frames_num': [6]
        }]
        op = VideoSplitByKeyFrameMapper(ffmpeg_extra_args='-movflags frag_keyframe+empty_moov')
        self._run_video_split_by_key_frame_mapper(op, ds_list, tgt_list)

    @TEST_TAG("standalone", "ray")
    def test_multi_process(self):
        ds_list = [{
            'id': 0,
            'text': f'{SpecialTokens.video} 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。',
            'videos': [self.vid1_path]
        }, {
            'id': 1,
            'text':
            f'{SpecialTokens.video} 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。{SpecialTokens.eoc}',
            'videos': [self.vid2_path]
        }, {
            'id': 2,
            'text':
            f'{SpecialTokens.video} 两个长头发的女子正坐在一张圆桌前讲话互动。 {SpecialTokens.eoc}',
            'videos': [self.vid3_path]
        }]
        tgt_list = [{
            'id': 0,
            'text':
            f'{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video} 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。{SpecialTokens.eoc}',
            'split_frames_num': [3]
        }, {
            'id': 1,
            'text':
            f'{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video} 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。{SpecialTokens.eoc}',
            'split_frames_num': [3]
        }, {
            'id': 2,
            'text':
            f'{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video} 两个长头发的女子正坐在一张圆桌前讲话互动。 {SpecialTokens.eoc}',
            'split_frames_num': [6]
        }]
        op = VideoSplitByKeyFrameMapper(keep_original_sample=False)
        self._run_video_split_by_key_frame_mapper(op,
                                                  ds_list,
                                                  tgt_list,
                                                  num_proc=2)

    @TEST_TAG("standalone", "ray")
    def test_multi_chunk(self):
        # wrong because different order
        ds_list = [{
            'id': 0,
            'text':
            f'{SpecialTokens.video} 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。{SpecialTokens.eoc}{SpecialTokens.video} 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。',
            'videos': [self.vid1_path, self.vid2_path]
        }, {
            'id': 1,
            'text':
            f'{SpecialTokens.video} 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。{SpecialTokens.eoc}{SpecialTokens.video} 两个长头发的女子正坐在一张圆桌前讲话互动。 {SpecialTokens.eoc}',
            'videos': [self.vid2_path, self.vid3_path]
        }, {
            'id': 2,
            'text':
            f'{SpecialTokens.video} 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。{SpecialTokens.eoc}{SpecialTokens.video} 两个长头发的女子正坐在一张圆桌前讲话互动。 {SpecialTokens.eoc}',
            'videos': [self.vid1_path, self.vid3_path]
        }]
        tgt_list = [{
            'id': 0,
            'text':
            f'{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video} 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。{SpecialTokens.eoc}{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video} 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。{SpecialTokens.eoc}',
            'split_frames_num': [3, 3]
        }, {
            'id': 1,
            'text':
            f'{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video} 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。{SpecialTokens.eoc}{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video} 两个长头发的女子正坐在一张圆桌前讲话互动。 {SpecialTokens.eoc}',
            'split_frames_num': [3, 6]
        }, {
            'id': 2,
            'text':
            f'{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video} 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。{SpecialTokens.eoc}{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video} 两个长头发的女子正坐在一张圆桌前讲话互动。 {SpecialTokens.eoc}',
            'split_frames_num': [3, 6]
        }]
        op = VideoSplitByKeyFrameMapper(keep_original_sample=False)
        self._run_video_split_by_key_frame_mapper(op, ds_list, tgt_list)

    @TEST_TAG("standalone", "ray")
    def test_output_format(self, save_field=None):
        ds_list = [{
            'id': 0,
            'text': f'{SpecialTokens.video} 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。',
            'videos': [self.vid1_path]
        }, {
            'id': 1,
            'text':
            f'{SpecialTokens.video} 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。{SpecialTokens.eoc}',
            'videos': [self.vid2_path]
        }, {
            'id': 2,
            'text':
            f'{SpecialTokens.video} 两个长头发的女子正坐在一张圆桌前讲话互动。 {SpecialTokens.eoc}',
            'videos': [self.vid3_path]
        }]
        tgt_list = [{
            'id': 0,
            'text':
            f'{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video} 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。{SpecialTokens.eoc}',
            'split_frames_num': 3
        }, {
            'id': 1,
            'text':
            f'{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video} 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。{SpecialTokens.eoc}',
            'split_frames_num': 3
        }, {
            'id': 2,
            'text':
            f'{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video}{SpecialTokens.video} 两个长头发的女子正坐在一张圆桌前讲话互动。 {SpecialTokens.eoc}',
            'split_frames_num': 6
        }]
        op = VideoSplitByKeyFrameMapper(
            keep_original_sample=False,
            output_format="bytes",
            save_field=save_field,
            legacy_split_by_text_token=True)

        dataset = Dataset.from_list(ds_list)
        dataset = dataset.map(op.process, num_proc=2)
        res_list = dataset.to_list()
        res_list = sorted(res_list, key=lambda x: x["id"])

        save_field = save_field or "videos"
        for i in range(len(ds_list)):
            res = res_list[i] 
            tgt = tgt_list[i]
            self.assertEqual(res['id'], tgt['id'])
            self.assertEqual(res['text'], tgt['text'])
            self.assertEqual(len(res[save_field]), tgt['split_frames_num'])
            self.assertTrue(all(isinstance(v, bytes) for v in res[save_field]))

    @TEST_TAG("standalone", "ray")
    def test_save_field(self):
        self.test_output_format(save_field="clips")
    
    @TEST_TAG("standalone", "ray")
    def test_legacy_split_by_text_token_false(self):
        ds_list = [{
            'id': 0,
            'text': '',
            'videos': [self.vid1_path]
        }, {
            'id': 1,
            'text': '',
            'videos': [self.vid2_path]
        }, {
            'id': 2,
            'text': '',
            'videos': [self.vid3_path]
        }]
        tgt_list = [{
            'id': 0,
            'text': '',
            'split_frames_num': 3
        }, {
            'id': 1,
            'text': '',
            'split_frames_num': 3
        }, {
            'id': 2,
            'text': '',
            'split_frames_num': 6
        }]

        save_field = 'clips_path'
        op = VideoSplitByKeyFrameMapper(
            keep_original_sample=False,
            output_format="path",
            save_dir=self.tmp_dir,
            save_field=save_field,
            legacy_split_by_text_token=False)

        dataset = Dataset.from_list(ds_list)
        dataset = dataset.map(op.process, num_proc=2)
        res_list = dataset.to_list()
        res_list = sorted(res_list, key=lambda x: x["id"])

        all_clips = []
        for i in range(len(ds_list)):
            res = res_list[i] 
            tgt = tgt_list[i]
            self.assertEqual(res['id'], tgt['id'])
            self.assertEqual(res['text'], tgt['text'])
            self.assertEqual(len(res[save_field]), tgt['split_frames_num'])
            for clip_path in res[save_field]:
                self.assertTrue(os.path.exists(clip_path))
            all_clips.extend(res[save_field])

        self.assertListEqual(
            sorted([os.path.join(self.tmp_dir, f) for f in os.listdir(self.tmp_dir)]),
            sorted(all_clips))


if __name__ == '__main__':
    unittest.main()

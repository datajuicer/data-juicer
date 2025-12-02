import os
import os.path as osp
import re
import unittest
import tempfile
import shutil

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.video_extract_frames_mapper import \
    VideoExtractFramesMapper
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.mm_utils import SpecialTokens
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class VideoExtractFramesMapperTest(DataJuicerTestCaseBase):

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data')
    vid1_path = os.path.join(data_path, 'video1.mp4')
    vid2_path = os.path.join(data_path, 'video2.mp4')
    vid3_path = os.path.join(data_path, 'video3.mp4')
    tmp_dir = tempfile.TemporaryDirectory().name

    def tearDown(self):
        super().tearDown()
        if osp.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

        default_frame_dir_prefix = self._get_default_frame_dir_prefix()
        if osp.exists(default_frame_dir_prefix):
            shutil.rmtree(osp.dirname(default_frame_dir_prefix))

    def _get_default_frame_dir_prefix(self):
        from data_juicer.ops.mapper.video_extract_frames_mapper import OP_NAME
        default_frame_dir_prefix = osp.abspath(osp.join(self.data_path, 
            f'{Fields.multimodal_data_output_dir}/{OP_NAME}/'))
        return default_frame_dir_prefix

    def _get_frames_list(self, filepath, frame_dir, frame_num):
        frames_dir = osp.join(frame_dir, osp.splitext(osp.basename(filepath))[0])
        frames_list = [osp.join(frames_dir, f'frame_{i}.jpg') for i in range(frame_num)]
        return frames_list

    def _get_frames_dir(self, filepath, frame_dir):
        frames_dir = osp.join(frame_dir, osp.splitext(osp.basename(filepath))[0])
        return frames_dir

    def _sort_files(self, file_list):
        return sorted(file_list, key=lambda x: int(re.search(r'(\d+)', x).group()))

    def test_duration(self):
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

        frame_num = 2
        frame_dir=os.path.join(self.tmp_dir, 'test1')
        vid1_frame_dir =  self._get_frames_dir(self.vid1_path, frame_dir)
        vid2_frame_dir =  self._get_frames_dir(self.vid2_path, frame_dir)
        vid3_frame_dir =  self._get_frames_dir(self.vid3_path, frame_dir)

        op = VideoExtractFramesMapper(
            frame_sampling_method='uniform',
            frame_num=frame_num,
            output_format="path",
            duration=0,
            frame_dir=frame_dir,
            batch_size=2,
            num_proc=1)

        dataset = Dataset.from_list(ds_list)
        dataset = op.run(dataset)
        res_list = dataset.to_list()

        tgt_frames_num = [[frame_num], [frame_num], [frame_num]]
        tgt_frames_dir = [[vid1_frame_dir], [vid2_frame_dir], [vid3_frame_dir]]
        for sample_i in range(len(res_list)):
            num_videos = len(ds_list[sample_i]['videos'])
            for video_idx in range(num_videos):
                self.assertEqual(
                    res_list[sample_i][MetaKeys.video_frames][video_idx],
                    [osp.join(
                        tgt_frames_dir[sample_i][video_idx],
                        f'frame_{f_i}.jpg') for f_i in range(tgt_frames_num[sample_i][video_idx])
                    ])

    def test_uniform_sampling(self):
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
        frame_num = 3
        frame_dir = os.path.join(self.tmp_dir, 'test1')
        vid1_frame_dir =  self._get_frames_dir(self.vid1_path, frame_dir)
        vid2_frame_dir =  self._get_frames_dir(self.vid2_path, frame_dir)
        vid3_frame_dir =  self._get_frames_dir(self.vid3_path, frame_dir)

        op = VideoExtractFramesMapper(
            frame_sampling_method='uniform',
            frame_num=frame_num,
            duration=10,
            frame_dir=frame_dir,
            batch_size=2,
            num_proc=1)

        dataset = Dataset.from_list(ds_list)
        dataset = op.run(dataset)
        res_list = dataset.to_list()

        tgt_frames_num = [[3], [6], [12]]
        tgt_frames_dir = [[vid1_frame_dir], [vid2_frame_dir], [vid3_frame_dir]]
        for sample_i in range(len(res_list)):
            num_videos = len(ds_list[sample_i]['videos'])
            for video_idx in range(num_videos):
                self.assertEqual(
                    res_list[sample_i][MetaKeys.video_frames][video_idx],
                    [osp.join(
                        tgt_frames_dir[sample_i][video_idx],
                        f'frame_{f_i}.jpg') for f_i in range(tgt_frames_num[sample_i][video_idx])
                    ])

    def test_all_keyframes_sampling(self):
        ds_list = [{
            'text': f'{SpecialTokens.video} 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。',
            'videos': [self.vid1_path]
        }, {
            'text':
            f'{SpecialTokens.video} 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。{SpecialTokens.eoc}' + \
            f'{SpecialTokens.video} 两个长头发的女子正坐在一张圆桌前讲话互动。 {SpecialTokens.eoc}',
            'videos': [self.vid2_path, self.vid3_path]
        }, {
            'text':
            f'{SpecialTokens.video} 两个长头发的女子正坐在一张圆桌前讲话互动。 {SpecialTokens.eoc}',
            'videos': [self.vid3_path]
        }]
        frame_dir=os.path.join(self.tmp_dir, 'test2')
        vid1_frame_dir =  self._get_frames_dir(self.vid1_path, frame_dir)
        vid2_frame_dir =  self._get_frames_dir(self.vid2_path, frame_dir)
        vid3_frame_dir =  self._get_frames_dir(self.vid3_path, frame_dir)

        op = VideoExtractFramesMapper(
            frame_sampling_method='all_keyframes',
            frame_dir=frame_dir,
            duration=5,
            batch_size=2,
            num_proc=2)

        dataset = Dataset.from_list(ds_list)
        dataset = op.run(dataset)
        res_list = dataset.to_list()

        tgt_frames_num = [[4], [5, 13], [13]]
        tgt_frames_dir = [[vid1_frame_dir], [vid2_frame_dir, vid3_frame_dir], [vid3_frame_dir]]
        for sample_i in range(len(res_list)):
            num_videos = len(ds_list[sample_i]['videos'])
            for video_idx in range(num_videos):
                self.assertEqual(
                    res_list[sample_i][MetaKeys.video_frames][video_idx],
                    [osp.join(
                        tgt_frames_dir[sample_i][video_idx],
                        f'frame_{f_i}.jpg') for f_i in range(tgt_frames_num[sample_i][video_idx])
                    ])

    def test_bytes_format(self):
        ds_list = [{
            'text': f'{SpecialTokens.video} 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。',
            'videos': [self.vid1_path]
        }, {
            'text':
            f'{SpecialTokens.video} 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。{SpecialTokens.eoc}' + \
            f'{SpecialTokens.video} 两个长头发的女子正坐在一张圆桌前讲话互动。 {SpecialTokens.eoc}',
            'videos': [self.vid2_path, self.vid3_path]
        }, {
            'text':
            f'{SpecialTokens.video} 两个长头发的女子正坐在一张圆桌前讲话互动。 {SpecialTokens.eoc}',
            'videos': [self.vid3_path]
        }]

        frame_key = 'frames_bytes'
        op = VideoExtractFramesMapper(
            frame_sampling_method='all_keyframes',
            frame_key=frame_key,
            output_format='bytes',
            duration=5,
            batch_size=2,
            num_proc=2)

        dataset = Dataset.from_list(ds_list)
        dataset = op.run(dataset)
        res_list = dataset.to_list()

        tgt_frames_num = [[4], [5, 13], [13]]
        for sample_i in range(len(res_list)):
            num_videos = len(ds_list[sample_i]['videos'])
            for video_idx in range(num_videos):
                self.assertEqual(
                    len(res_list[sample_i][frame_key][video_idx]),
                    tgt_frames_num[sample_i][video_idx]
                    )
                self.assertTrue(isinstance(res_list[sample_i][frame_key][video_idx][0], bytes))
    
    @unittest.skip('The default frame dir is not supported as it may contaminate the data. The frame dir must be specified')
    def test_default_frame_dir(self):
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

        frame_num = 2
        op = VideoExtractFramesMapper(
            frame_sampling_method='uniform',
            frame_num=frame_num,
            duration=5,
            batch_size=2,
            num_proc=1
            )

        vid1_frame_dir =  op._get_default_frame_dir(self.vid1_path)
        vid2_frame_dir =  op._get_default_frame_dir(self.vid2_path)
        vid3_frame_dir =  op._get_default_frame_dir(self.vid3_path)

        dataset = Dataset.from_list(ds_list)
        dataset = op.run(dataset)
        res_list = dataset.to_list()

        frame_dir_prefix = self._get_default_frame_dir_prefix()
        self.assertIn(frame_dir_prefix, osp.abspath(vid1_frame_dir))
        self.assertIn(frame_dir_prefix, osp.abspath(vid2_frame_dir))
        self.assertIn(frame_dir_prefix, osp.abspath(vid3_frame_dir))

        tgt_frames_num = [[3], [6], [12]]
        tgt_frames_dir = [[vid1_frame_dir], [vid2_frame_dir], [vid3_frame_dir]]
        for sample_i in range(len(res_list)):
            num_videos = len(ds_list[sample_i]['videos'])
            for video_idx in range(num_videos):
                self.assertEqual(
                    res_list[sample_i][MetaKeys.video_frames][video_idx],
                    [osp.join(
                        tgt_frames_dir[sample_i][video_idx],
                        f'frame_{f_i}.jpg') for f_i in range(tgt_frames_num[sample_i][video_idx])
                    ])


if __name__ == '__main__':
    unittest.main()

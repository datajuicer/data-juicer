import os
import unittest

from data_juicer.core.data import NestedDataset as Dataset

from data_juicer.ops.mapper.video_resize_aspect_ratio_mapper import \
    VideoResizeAspectRatioMapper
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.mm_utils import close_video, load_video
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

ffmpeg = LazyLoader('ffmpeg', 'ffmpeg-python')


class VideoResizeAspectRatioMapperTest(DataJuicerTestCaseBase):

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')
    vid1_path = os.path.join(data_path, 'video1.mp4')  # 640x360, 16:9
    vid2_path = os.path.join(data_path, 'video2.mp4')  # 480x640, 3:4
    vid3_path = os.path.join(data_path, 'video3.mp4')  # 362x640, 181:320
    vid3_no_audio_path = os.path.join(data_path, 'video3-no-audio.mp4')

    def _run_op(self, dataset: Dataset, target_list, op, np=1):
        dataset = dataset.map(op.process, num_proc=np)

        def get_size(dataset):
            sizes = []
            res_list = dataset.to_list()
            for sample in res_list:
                sample_list = []
                for value in sample['videos']:
                    video = load_video(value)
                    width = video.streams.video[0].codec_context.width
                    height = video.streams.video[0].codec_context.height
                    sample_list.append((width, height))
                    close_video(video)
                sizes.append(sample_list)
            return sizes

        sizes = get_size(dataset)
        self.assertEqual(sizes, target_list)

    def test_default_params(self):

        ds_list = [{
            'videos': [self.vid1_path]
        }, {
            'videos': [self.vid2_path]
        }, {
            'videos': [self.vid3_path]
        }]
        tgt_list = [
            [(640, 360)],  # no change
            [(480, 640)],  # no change
            [(362, 640)]  # no change
        ]
        dataset = Dataset.from_list(ds_list)
        op = VideoResizeAspectRatioMapper()
        self._run_op(dataset, tgt_list, op)

    def test_min_ratio_increase(self):

        ds_list = [{
            'videos': [self.vid1_path]
        }, {
            'videos': [self.vid2_path]
        }, {
            'videos': [self.vid3_path]
        }]
        tgt_list = [
            [(640, 360)],  # no change
            [(480, 640)],  # no change
            [(480, 640)]  # 181:320 to 3:4
        ]
        dataset = Dataset.from_list(ds_list)
        op = VideoResizeAspectRatioMapper(min_ratio='3/4', strategy='increase')
        self._run_op(dataset, tgt_list, op)

    def test_min_ratio_decrease(self):

        ds_list = [{
            'videos': [self.vid1_path]
        }, {
            'videos': [self.vid2_path]
        }, {
            'videos': [self.vid3_path]
        }]
        tgt_list = [
            [(640, 360)],  # no change
            [(480, 640)],  # no change
            [(362, 482)]  # ratio 181:320 to 3:4
        ]
        dataset = Dataset.from_list(ds_list)
        op = VideoResizeAspectRatioMapper(min_ratio='3/4', strategy='decrease')
        self._run_op(dataset, tgt_list, op)

    def test_max_ratio_increase(self):

        ds_list = [{
            'videos': [self.vid1_path]
        }, {
            'videos': [self.vid2_path]
        }, {
            'videos': [self.vid3_path]
        }]
        tgt_list = [
            [(640, 480)],  # 16:9 to  4:3
            [(480, 640)],  # no change
            [(362, 640)]  # no change
        ]
        dataset = Dataset.from_list(ds_list)
        op = VideoResizeAspectRatioMapper(max_ratio='4/3', strategy='increase')
        self._run_op(dataset, tgt_list, op)

    def test_max_ratio_decrease(self):

        ds_list = [{
            'videos': [self.vid1_path]
        }, {
            'videos': [self.vid2_path]
        }, {
            'videos': [self.vid3_path]
        }]
        tgt_list = [
            [(480, 360)],  # 16:9 to  4:3
            [(480, 640)],  # no change
            [(362, 640)]  # no change
        ]
        dataset = Dataset.from_list(ds_list)
        op = VideoResizeAspectRatioMapper(max_ratio='4/3', strategy='decrease')
        self._run_op(dataset, tgt_list, op)

    def test_parallel(self):

        ds_list = [{
            'videos': [self.vid1_path]
        }, {
            'videos': [self.vid2_path]
        }, {
            'videos': [self.vid3_path]
        }]
        tgt_list = [
            [(480, 360)],  # 16:9 to  4:3
            [(480, 640)],  # no change
            [(362, 640)]  # no change
        ]
        dataset = Dataset.from_list(ds_list)
        op = VideoResizeAspectRatioMapper(max_ratio='4/3', strategy='decrease')
        self._run_op(dataset, tgt_list, op, np=2)

    def _count_audio_streams(self, video_path):
        probe = ffmpeg.probe(video_path)
        return sum(1 for stream in probe['streams']
                   if stream['codec_type'] == 'audio')

    def test_preserves_audio_when_resizing(self):
        """Resizing must keep audio streams that were present on the input."""
        input_audio = self._count_audio_streams(self.vid1_path)
        self.assertGreater(
            input_audio, 0,
            'fixture video1.mp4 should include an audio stream')

        dataset = Dataset.from_list([{'videos': [self.vid1_path]}])
        op = VideoResizeAspectRatioMapper(max_ratio='4/3', strategy='decrease')
        dataset = dataset.map(op.process)
        out_path = dataset.to_list()[0]['videos'][0]
        self.assertNotEqual(out_path, self.vid1_path)
        self.assertEqual(self._count_audio_streams(out_path), input_audio)

    def test_silent_video_still_resizes(self):
        """Videos without audio should still resize successfully."""
        self.assertEqual(self._count_audio_streams(self.vid3_no_audio_path), 0)
        dataset = Dataset.from_list([{'videos': [self.vid3_no_audio_path]}])
        # 362x640 → force a wider ratio so a resize happens
        op = VideoResizeAspectRatioMapper(min_ratio='1/1', strategy='increase')
        dataset = dataset.map(op.process)
        out_path = dataset.to_list()[0]['videos'][0]
        self.assertEqual(self._count_audio_streams(out_path), 0)


if __name__ == '__main__':
    unittest.main()

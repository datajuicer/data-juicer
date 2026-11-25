import os
import unittest

from data_juicer.core.data import NestedDataset as Dataset

from data_juicer.ops.mapper.video_resize_resolution_mapper import \
    VideoResizeResolutionMapper
from data_juicer.utils.constant import Fields
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase
from data_juicer.utils.lazy_loader import LazyLoader

ffmpeg = LazyLoader('ffmpeg', 'ffmpeg-python')


class VideoResizeResolutionMapperTest(DataJuicerTestCaseBase):

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')
    # video1: horizontal resolution 640p, vertical resolution 360p
    # video2: horizontal resolution 480p, vertical resolution 640p
    # video3: horizontal resolution 362p, vertical resolution 640p
    vid1_path = os.path.join(data_path, 'video1.mp4')
    vid2_path = os.path.join(data_path, 'video2.mp4')
    vid3_path = os.path.join(data_path, 'video3.mp4')
    vid3_no_audio_path = os.path.join(data_path, 'video3-no-audio.mp4')

    def _get_size_list(self, dataset: Dataset):
        res_list = []
        for sample in dataset.to_list():
            cur_list = []
            for value in sample['videos']:
                probe = ffmpeg.probe(value)
                video_stream = next((stream for stream in probe['streams']
                                     if stream['codec_type'] == 'video'), None)
                width = int(video_stream['width'])
                height = int(video_stream['height'])
                cur_list.append((width, height))
            res_list.append(cur_list)
        return res_list

    def _count_audio_streams(self, video_path):
        probe = ffmpeg.probe(video_path)
        return sum(1 for stream in probe['streams']
                   if stream['codec_type'] == 'audio')

    def _run_video_resize_resolution_mapper(self,
                                            dataset: Dataset,
                                            target_list,
                                            op,
                                            test_name,
                                            np=1):
        if Fields.stats not in dataset.features:
            dataset = dataset.add_column(name=Fields.stats,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.process, num_proc=np)
        dataset = dataset.select_columns(column_names=[op.video_key])

        # check each video personally
        # output_dir = '../video_resize_resolution_mapper'
        # move_to_dir = os.path.join(output_dir, test_name)
        # if not os.path.exists(move_to_dir):
        #     os.makedirs(move_to_dir)
        # for sample in dataset.to_list():
        #     for value in sample['videos']:
        #         move_to_path = os.path.join(move_to_dir,
        #           os.path.basename(value))
        #         shutil.copyfile(value, move_to_path)

        res_list = self._get_size_list(dataset)
        self.assertEqual(res_list, target_list)

    def test_default_mapper(self):

        ds_list = [{
            'videos': [self.vid1_path]
        }, {
            'videos': [self.vid2_path]
        }, {
            'videos': [self.vid3_path]
        }]
        tgt_list = [[(640, 360)], [(480, 640)], [(362, 640)]]
        dataset = Dataset.from_list(ds_list)
        op = VideoResizeResolutionMapper()
        self._run_video_resize_resolution_mapper(dataset, tgt_list, op,
                                                 'test_default_mapper')

    def test_width_mapper(self):

        ds_list = [{
            'videos': [self.vid1_path]
        }, {
            'videos': [self.vid2_path]
        }, {
            'videos': [self.vid3_path]
        }]
        tgt_list = [[(480, 270)], [(480, 640)], [(400, 708)]]
        dataset = Dataset.from_list(ds_list)
        op = VideoResizeResolutionMapper(min_width=400, max_width=480)
        self._run_video_resize_resolution_mapper(dataset, tgt_list, op,
                                                 'test_width_mapper')

    def test_height_mapper(self):

        ds_list = [{
            'videos': [self.vid1_path]
        }, {
            'videos': [self.vid2_path]
        }, {
            'videos': [self.vid3_path]
        }]
        tgt_list = [[(854, 480)], [(360, 480)], [(272, 480)]]
        dataset = Dataset.from_list(ds_list)
        op = VideoResizeResolutionMapper(min_height=480, max_height=480)
        self._run_video_resize_resolution_mapper(dataset, tgt_list, op,
                                                 'test_width_mapper')

    def test_width_and_height_mapper(self):

        ds_list = [{
            'videos': [self.vid1_path, self.vid2_path, self.vid3_path]
        }]
        tgt_list = [[(480, 480), (400, 480), (400, 480)]]
        dataset = Dataset.from_list(ds_list)
        op = VideoResizeResolutionMapper(min_width=400,
                                         max_width=480,
                                         min_height=480,
                                         max_height=480)
        self._run_video_resize_resolution_mapper(
            dataset, tgt_list, op, 'test_width_and_height_mapper')

    def test_keep_aspect_ratio_decrease_mapper(self):

        ds_list = [{'videos': [self.vid1_path]}]
        tgt_list = [[(480, 270)]]
        dataset = Dataset.from_list(ds_list)
        op = VideoResizeResolutionMapper(
            min_width=400,
            max_width=480,
            min_height=480,
            max_height=480,
            force_original_aspect_ratio='decrease')
        self._run_video_resize_resolution_mapper(
            dataset, tgt_list, op, 'test_keep_aspect_ratio_decrease_mapper')

    def test_keep_aspect_ratio_increase_mapper(self):

        ds_list = [{'videos': [self.vid1_path]}]
        tgt_list = [[(854, 480)]]
        dataset = Dataset.from_list(ds_list)
        op = VideoResizeResolutionMapper(
            min_width=400,
            max_width=480,
            min_height=480,
            max_height=480,
            force_original_aspect_ratio='increase')
        self._run_video_resize_resolution_mapper(
            dataset, tgt_list, op, 'test_keep_aspect_ratio_increase_mapper')

    def test_force_divisible_by(self):

        ds_list = [{'videos': [self.vid1_path]}]
        tgt_list = [[(480, 272)]]
        dataset = Dataset.from_list(ds_list)
        op = VideoResizeResolutionMapper(
            min_width=400,
            max_width=480,
            min_height=480,
            max_height=480,
            force_original_aspect_ratio='decrease',
            force_divisible_by=4)
        self._run_video_resize_resolution_mapper(dataset, tgt_list, op,
                                                 'test_force_divisible_by')

    def test_filter_in_parallel(self):

        ds_list = [{
            'videos': [self.vid1_path]
        }, {
            'videos': [self.vid2_path]
        }, {
            'videos': [self.vid3_path]
        }]
        tgt_list = [[(480, 270)], [(480, 640)], [(400, 708)]]
        dataset = Dataset.from_list(ds_list)
        op = VideoResizeResolutionMapper(min_width=400, max_width=480)
        self._run_video_resize_resolution_mapper(dataset,
                                                 tgt_list,
                                                 op,
                                                 'test_filter_in_parallel',
                                                 np=2)

    def test_preserves_audio_when_resizing(self):
        """Resizing must keep audio streams that were present on the input."""
        input_audio = self._count_audio_streams(self.vid1_path)
        self.assertGreater(
            input_audio, 0,
            'fixture video1.mp4 should include an audio stream')

        dataset = Dataset.from_list([{'videos': [self.vid1_path]}])
        if Fields.stats not in dataset.features:
            dataset = dataset.add_column(name=Fields.stats,
                                         column=[{}] * dataset.num_rows)
        op = VideoResizeResolutionMapper(max_width=320)
        dataset = dataset.map(op.process)
        out_path = dataset.to_list()[0]['videos'][0]
        self.assertNotEqual(out_path, self.vid1_path)
        self.assertEqual(self._count_audio_streams(out_path), input_audio)

    def test_silent_video_still_resizes(self):
        """Videos without audio should still resize successfully."""
        self.assertEqual(self._count_audio_streams(self.vid3_no_audio_path), 0)
        dataset = Dataset.from_list([{'videos': [self.vid3_no_audio_path]}])
        if Fields.stats not in dataset.features:
            dataset = dataset.add_column(name=Fields.stats,
                                         column=[{}] * dataset.num_rows)
        op = VideoResizeResolutionMapper(max_width=180)
        dataset = dataset.map(op.process)
        out_path = dataset.to_list()[0]['videos'][0]
        self.assertEqual(self._count_audio_streams(out_path), 0)
        probe = ffmpeg.probe(out_path)
        video_stream = next(stream for stream in probe['streams']
                            if stream['codec_type'] == 'video')
        self.assertLessEqual(int(video_stream['width']), 180)


if __name__ == '__main__':
    unittest.main()

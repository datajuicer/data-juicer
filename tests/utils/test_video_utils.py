import os
import shutil
import unittest
import numpy as np
import subprocess

from data_juicer.utils.video_utils import Clip, AVReader, FFmpegReader, DecordReader
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


def is_valid_mp4_ffprobe(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
           '-show_entries', 'stream=codec_name', '-of', 'csv=p=0', file_path]
    output = subprocess.run(cmd, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, timeout=10)
    
    if output.returncode != 0:
        return False
    if not output.stdout.decode().strip():
        return False
    return True


class TestVideoReader(DataJuicerTestCaseBase):

    def setUp(self) -> None:
        super().setUp()
        data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                 '..',
                                 'ops',
                                 'data')
        self.vid_path1 = os.path.join(data_path, 'video1.mp4')
        self.temp_output_path = 'tmp/test_video_utils/'
        os.makedirs(self.temp_output_path, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.temp_output_path):
            shutil.rmtree(self.temp_output_path)

        super().tearDown()

    def get_backends(self):
        """Get backends"""
        return {
            'av': AVReader,
            'ffmpeg': FFmpegReader,
            'decord': DecordReader
        }
    
    def test_metadata_consistency(self):
        """Test metadata consistency"""
        test_video_path = self.vid_path1
        backends = self.get_backends()
        
        for name, cls in backends.items():
            self.assertTrue(cls.is_available())

        metadata_results = {}
        readers = {}
        try:
            for name, cls in backends.items():
                reader = cls(test_video_path)
                readers[name] = reader
                metadata = reader.get_metadata()
                metadata_results[name] = metadata
        
            self.assertEqual(len(metadata_results), len(backends))            
            metadata_list = list(metadata_results.values())

            for metadata in metadata_list:
                self.assertEqual(metadata.height, 360)
                self.assertEqual(metadata.width, 640)
                self.assertEqual(metadata.fps, 24.0)
                self.assertEqual(metadata.num_frames, 282)
                self.assertEqual(metadata.duration, 11.75)
        finally:
            for reader in readers.values():
                reader.close()

    def _test_extract_frames(self, start_time, end_time, tgt_frames_num, 
                             extract_keyframes=False, check_np=True):
        test_video_path = self.vid_path1
        backends = self.get_backends()
        frame_results = {}
        indices_results = {}
        pts_time_results = {}
        readers = {}
        
        try:
            for name, cls in backends.items():
                reader = cls(test_video_path)
                readers[name] = reader
                if extract_keyframes:
                    frame_obj = reader.extract_keyframes(start_time, end_time)
                    frames = frame_obj.frames
                    indices = frame_obj.indices
                    pts_time = frame_obj.pts_time
                    indices_results[name] = indices
                    pts_time_results[name] = pts_time
                else:
                    frames = list(reader.extract_frames(start_time, end_time))
                frame_results[name] = frames

            backends_frames = list(frame_results.values())
            first_backend_frames = backends_frames[0]
            for cur_backend_frames in backends_frames:
                self.assertEqual(len(cur_backend_frames), tgt_frames_num)
                for i, frame in enumerate(cur_backend_frames):
                    self.assertEqual(frame.shape, (360, 640, 3))
                    self.assertTrue(isinstance(frame, np.ndarray))
                    if check_np:
                        np.testing.assert_array_equal(frame, first_backend_frames[i])
        finally:
            for reader in readers.values():
                reader.close()

        return indices_results, pts_time_results

    def test_extract_frames_full_video(self):
        self._test_extract_frames(0, None, 282, extract_keyframes=False, check_np=True)

    def test_extract_frames_time_range(self):
        self._test_extract_frames(1.0, 3.0, 48, extract_keyframes=False, check_np=True)

    def test_extract_keyframes(self):
        indices_results, pts_time_results = self._test_extract_frames(
            0, None, 3, extract_keyframes=True, check_np=True)
        
        backends = self.get_backends()
        for name, _ in backends.items():
            self.assertEqual(indices_results[name], [0, 144, 237])
            self.assertEqual(pts_time_results[name], [0.0, 6.0, 9.875])

    def test_extract_keyframes_time_range(self):
        indices_results, pts_time_results = self._test_extract_frames(
            0, 2.0, 1, extract_keyframes=True, check_np=False)

        backends = self.get_backends()
        for name, _ in backends.items():
            self.assertEqual(indices_results[name], [0])
            self.assertEqual(pts_time_results[name], [0.0])

    def test_extract_clip_numpy(self):
        test_video_path = self.vid_path1
        backends = self.get_backends()
        
        start_time, end_time = 1.0, 3.0
        clip_results = {}
        readers = {}
        
        try:
            for name, cls in backends.items():
                reader = cls(test_video_path)
                readers[name] = reader
                clip = reader.extract_clip(start_time, end_time)
                clip_results[name] = clip
                
                self.assertTrue(isinstance(clip, Clip))
                self.assertEqual(clip.encoded_data, None)
                self.assertEqual(clip.source_video, test_video_path)
                self.assertEqual(clip.span, (start_time, end_time))
                self.assertEqual(len(clip.frames), 48)
                np.testing.assert_array_equal(clip.frames, clip_results['av'].frames) 
        
        finally:
            for reader in readers.values():
                reader.close()

    def test_extract_clip_bytes(self):
        test_video_path = self.vid_path1
        backends = self.get_backends()
        
        start_time, end_time = 1.0, 3.0
        clip_results = {}
        readers = {}
        
        try:
            backends.pop('decord')
            for name, cls in backends.items():
                reader = cls(test_video_path)
                readers[name] = reader
                clip = reader.extract_clip(start_time, end_time, to_numpy=False)
                clip_results[name] = clip
                
                self.assertTrue(isinstance(clip, Clip))
                self.assertTrue(isinstance(clip.encoded_data, bytes))
                self.assertEqual(clip.source_video, test_video_path)
                self.assertEqual(clip.span, (start_time, end_time))
                self.assertEqual(clip.frames, None)
        finally:
            for reader in readers.values():
                reader.close()

    def test_extract_clip_output_path(self):
        test_video_path = self.vid_path1
        backends = self.get_backends()
        backends.pop('decord')
        
        start_time, end_time = 0, None
        clip_results = {}
        readers = {}

        output_clips_path = []
        try:
            for name, cls in backends.items():
                output_path = os.path.join(self.temp_output_path, f'{name}_out.mp4')
                output_clips_path.append(output_path)
                reader = cls(test_video_path)
                readers[name] = reader
                clip = reader.extract_clip(start_time, end_time, output_path=output_path)
                clip_results[name] = clip
                
                self.assertTrue(isinstance(clip, Clip))
                self.assertEqual(clip.encoded_data, None)
                self.assertEqual(clip.source_video, test_video_path)
                self.assertEqual(clip.span, (start_time, end_time))
                self.assertEqual(clip.frames, None)
                self.assertEqual(clip.path, output_path)
                self.assertTrue(os.path.exists(output_path))
                self.assertTrue(is_valid_mp4_ffprobe(output_path))
        finally:
            for reader in readers.values():
                reader.close()
        
        self.assertEqual(len(output_clips_path), 2)

        # check clip content
        check_backends = self.get_backends()
        check_readers = {}
        for clip_path in output_clips_path:
            try:
                for check_name, check_cls in check_backends.items():
                    check_reader = check_cls(clip_path)
                    check_readers[check_name] = check_reader
                    self.assertListEqual(check_reader.extract_keyframes().indices, [0, 144, 237])
                    self.assertEqual(len(list(check_reader.extract_frames())), 282)
            finally:
                for check_reader in check_readers.values():
                    check_reader.close()

    def test_context_manager(self):
        test_video_path = self.vid_path1
        start_time, end_time = 1.0, 3.0

        backends = self.get_backends()

        for name, cls in backends.items():
            with cls(test_video_path) as reader:
                metadata = reader.metadata
                extracted_frames = list(reader.extract_frames(start_time, end_time))
                all_frames = list(reader.extract_frames())
                clip_frames = reader.extract_clip(start_time, end_time).frames
                all_keyframes = reader.extract_keyframes()

                self.assertEqual(metadata.height, 360)
                self.assertEqual(metadata.width, 640)
                self.assertEqual(metadata.fps, 24.0)
                self.assertEqual(metadata.num_frames, 282)
                self.assertEqual(metadata.duration, 11.75)

                self.assertEqual(len(extracted_frames), 48)
                self.assertEqual(len(clip_frames), 48)
                np.testing.assert_array_equal(clip_frames, extracted_frames) 
                self.assertEqual(len(all_frames), 282)
                self.assertEqual(len(all_keyframes.frames), 3)

    def test_edge_cases(self):
        test_video_path = self.vid_path1
        backends = self.get_backends()

        for name, cls in backends.items():
            reader = cls(test_video_path)
            try:
                with self.assertRaises(ValueError) as cm:
                    list(reader.extract_frames(-1, 1))
                self.assertEqual(str(cm.exception), "start_time cannot be negative")
                
                with self.assertRaises(ValueError) as cm:
                    list(reader.extract_frames(3, 1))
                self.assertEqual(str(cm.exception), "end_time must be greater than start_time")
                
                metadata = reader.metadata
                video_duration = metadata.duration
                frames = list(reader.extract_frames(video_duration - 1, video_duration + 10))
                self.assertEqual(len(frames), 24)
                self.assertEqual(frames[0].shape, (360, 640, 3))
                
            finally:
                reader.close()

    def test_extract_keyframes_meta_only(self):
        """Test extract_keyframes with return_meta_only=True"""
        test_video_path = self.vid_path1
        backends = self.get_backends()

        for name, cls in backends.items():
            with cls(test_video_path) as reader:
                frames = reader.extract_keyframes(return_meta_only=True)
                self.assertEqual(frames.frames, [])
                self.assertEqual(frames.indices, [0, 144, 237])
                self.assertEqual(frames.pts_time, [0.0, 6.0, 9.875])


if __name__ == '__main__':
    unittest.main()

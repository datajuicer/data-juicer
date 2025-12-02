import os
import unittest
import numpy as np
import av
import io
from typing import Iterator

from PIL import ImageFilter
from PIL.Image import Image
from data_juicer.utils.lazy_loader import LazyLoader
cv2 = LazyLoader('cv2', 'opencv-python')

from data_juicer.utils.mm_utils import (
    remove_special_tokens, remove_non_special_tokens, load_data_with_context,
    load_image, load_video, load_audio, Fields, load_images, load_audios,
    load_videos, load_image_byte, load_images_byte, image_path_to_base64,
    image_byte_to_base64, pil_to_opencv, get_file_size, iou,
    calculate_resized_dimensions, get_video_duration, process_each_frame,
    get_decoded_frames_from_video, cut_video_by_seconds,
    extract_key_frames_by_seconds, extract_key_frames, get_key_frame_seconds,
    extract_video_frames_uniformly_by_seconds, extract_video_frames_uniformly,
    extract_audio_from_video, size_to_bytes, insert_texts_after_placeholders,
    timecode_string_to_seconds, parse_string_to_roi,
)
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

class MMTest(DataJuicerTestCaseBase):

    def setUp(self) -> None:
        super().setUp()
        data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                 '..',
                                 'ops',
                                 'data')
        self.img_path = os.path.join(data_path, 'img1.png')
        self.vid_path = os.path.join(data_path, 'video1.mp4')
        self.aud_path = os.path.join(data_path, 'audio1.wav')
        self.temp_output_path = 'tmp/test_mm_utils/'
        os.makedirs(self.temp_output_path, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.temp_output_path):
            os.system(f'rm -rf {self.temp_output_path}')
        super().tearDown()

    def test_special_tokens(self):
        self.assertEqual(
            remove_special_tokens(
                '<__dj__image> <|__dj__eoc|> <|image|> <|eoc|> normal text'),
            '<|image|> <|eoc|> normal text')
        self.assertEqual(
            remove_non_special_tokens(
                '<__dj__image> <|__dj__eoc|> <|image|> <|eoc|> normal text'),
            '<__dj__image><|__dj__eoc|>')

    def test_load_data_with_context(self):
        # image, context off
        img_smple = {'images': [self.img_path]}
        res_img_sample, data = load_data_with_context(img_smple, False, [self.img_path], load_image)
        self.assertEqual(res_img_sample, img_smple)
        self.assertIsInstance(data[self.img_path], Image)

        # audio, context on, no context
        aud_smple = {'audios': [self.aud_path], Fields.context: {}}
        res_aud_smple, data = load_data_with_context(aud_smple, True, [self.aud_path], load_audio)
        self.assertIn(Fields.context, res_aud_smple)
        self.assertIn(self.aud_path, res_aud_smple[Fields.context])
        self.assertIsInstance(res_aud_smple[Fields.context][self.aud_path][0], np.ndarray)
        self.assertIsInstance(res_aud_smple[Fields.context][self.aud_path][1], int)
        self.assertIsInstance(data[self.aud_path], tuple)

        # video, context on, already in context
        vid_smple = {'videos': [self.vid_path], Fields.context: {self.vid_path: load_video(self.vid_path)}}
        res_vid_smple, data = load_data_with_context(vid_smple, True, [self.vid_path], load_video)
        self.assertEqual(res_vid_smple, vid_smple)
        self.assertIsInstance(data[self.vid_path], av.container.Container)

    def test_load_multiple_data(self):
        # images
        imgs = load_images([self.img_path, self.img_path])
        self.assertEqual(len(imgs), 2)
        for img in imgs:
            self.assertIsInstance(img, Image)

        # image bytes
        imgs = load_images_byte([self.img_path, self.img_path])
        self.assertEqual(len(imgs), 2)
        for img in imgs:
            self.assertIsInstance(img, bytes)

        # audios
        auds = load_audios([self.aud_path, self.aud_path])
        self.assertEqual(len(auds), 2)
        for aud in auds:
            self.assertIsInstance(aud[0], np.ndarray)
            self.assertIsInstance(aud[1], int)

        # videos
        vids = load_videos([self.vid_path, self.vid_path])
        self.assertEqual(len(vids), 2)
        for vid in vids:
            self.assertIsInstance(vid, av.container.Container)

        with self.assertRaises(FileNotFoundError):
            vid = load_video('invalid_path')

    def test_img_to_base64(self):
        self.assertIsInstance(image_path_to_base64(self.img_path), str)
        self.assertIsInstance(image_byte_to_base64(load_image_byte(self.img_path)), str)

    def test_pil_to_opencv(self):
        pil_img = load_image(self.img_path)
        opencv_img = pil_to_opencv(pil_img)
        self.assertIsInstance(opencv_img, np.ndarray)
        self.assertTrue((opencv_img == cv2.imread(self.img_path)).all())

        # test other mode
        pil_img = load_image(self.img_path)
        pil_img = pil_img.convert('LAB')
        opencv_img = pil_to_opencv(pil_img)
        self.assertIsInstance(opencv_img, np.ndarray)

    def test_get_file_size(self):
        self.assertIsInstance(get_file_size(self.img_path), int)

    def test_iou(self):
        test_data = [
            ((0, 0, 1, 1), (2, 2, 3, 3), 0),  # no overlap
            ((0, 0, 1, 1), (1, 1, 2, 2), 0),  # overlap on a point
            ((0, 0, 1, 1), (0, 1, 1, 2), 0),  # overlap on a border
            ((0, 0, 2, 2), (1, 1, 3, 3), 1.0/7.0),  # normal overlap
            ((0, 0, 3, 3), (1, 1, 2, 2), 1.0/9.0),  # contained
            ((0, 0, 1, 1), (0, 0, 1, 1), 1.0),  # same

            ((0, 0, 0, 0), (0, 0, 0, 0), 0),  # union is 0
            ((1, 1, 1, 1), (0, 0, 2, 2), 0),  # one of the box is a point
        ]
        for box1, box2, iou_gt in test_data:
            self.assertEqual(iou_gt, iou(box1, box2))

    def test_calculate_resized_dimensions(self):
        test_data = [
            ((224, 224), (336, 336), None, 1, (336, 336)),  # x1.5
            ((224, 224), None, None, 1, (224, 224)),  # no target size
            ((224, 336), 560, None, 1, (560, 840)),  # target short length
            ((224, 336), 560, 720, 1, (480, 720)),  # max_length
            ((224, 336), 560, None, 30, (540, 840)),  # divisible
        ]
        for ori_size, tgt_size, max_len, divisible, res_size in test_data:
            self.assertEqual(res_size, calculate_resized_dimensions(
                ori_size, tgt_size, max_len, divisible))

    def test_get_video_duration(self):
        # video path
        duration = get_video_duration(self.vid_path)
        self.assertIsInstance(duration, float)
        # video container
        vid = load_video(self.vid_path)
        container_duration = get_video_duration(vid)
        self.assertIsInstance(container_duration, float)
        self.assertEqual(duration, container_duration)

        # invalid input video
        with self.assertRaises(ValueError):
            get_video_duration(42)

    def test_get_decoded_frames_from_video(self):
        # video path
        iter = get_decoded_frames_from_video(self.vid_path)
        self.assertIsInstance(iter, Iterator)
        # video container
        vid = load_video(self.vid_path)
        container_iter = get_decoded_frames_from_video(vid)
        self.assertIsInstance(container_iter, Iterator)

    def test_cut_video_by_seconds(self):
        input_path = self.vid_path  # about 11.75 seconds
        output_path = os.path.join(self.temp_output_path, 'cut_res.mp4')
        # cut the video from 3 seconds to 5 seconds
        self.assertTrue(cut_video_by_seconds(input_path, output_path, 3, 5))
        self.assertAlmostEqual(get_video_duration(output_path), 2, delta=0.1)
        # cut the video from 3 seconds to 5 seconds with container
        container = load_video(input_path)
        self.assertTrue(cut_video_by_seconds(container, output_path, 3, 5))
        self.assertAlmostEqual(get_video_duration(output_path), 2, delta=0.1)
        # cut the video from 3 seconds to 5 seconds and return a container
        ret = cut_video_by_seconds(input_path, None, start_seconds=3, end_seconds=5)
        self.assertIsInstance(ret, io.BytesIO)
        self.assertAlmostEqual(get_video_duration(av.open(ret)), 2, delta=0.1)
        # different starts and ends
        self.assertTrue(cut_video_by_seconds(input_path, output_path, 0, 12))
        self.assertAlmostEqual(get_video_duration(output_path), 11.75, delta=0.1)
        self.assertTrue(cut_video_by_seconds(input_path, output_path, -1, 3))
        self.assertAlmostEqual(get_video_duration(output_path), 3, delta=0.1)
        self.assertTrue(cut_video_by_seconds(input_path, output_path, 10, 15))
        self.assertAlmostEqual(get_video_duration(output_path), 1.75, delta=0.1)
        self.assertTrue(cut_video_by_seconds(input_path, output_path, 10))
        self.assertAlmostEqual(get_video_duration(output_path), 1.75, delta=0.1)
        self.assertTrue(cut_video_by_seconds(input_path, output_path, 0))
        self.assertAlmostEqual(get_video_duration(output_path), get_video_duration(input_path))

    def test_process_each_frame(self):
        input_path = self.vid_path  # about 11.75 seconds
        output_path = os.path.join(self.temp_output_path, 'process_res.mp4')
        self.assertEqual(process_each_frame(input_path, output_path, lambda x: x), input_path)
        container = load_video(input_path)
        self.assertEqual(process_each_frame(container, output_path, lambda x: x), input_path)
        # tiny change
        def change_frame(frame: av.VideoFrame):
            img = frame.to_image()
            img = img.filter(ImageFilter.BLUR)
            return av.VideoFrame.from_image(img)
        self.assertEqual(process_each_frame(input_path, output_path, change_frame), output_path)

    def test_extract_key_frames(self):
        input_path = self.vid_path  # 11.75s
        key_frames_1 = extract_key_frames_by_seconds(input_path)
        self.assertIsInstance(key_frames_1, list)
        container = load_video(input_path)
        key_frames_2 = extract_key_frames_by_seconds(container, 3)
        self.assertIsInstance(key_frames_2, list)

        key_frames_3 = extract_key_frames(input_path)
        self.assertIsInstance(key_frames_3, list)

        # invalid input video
        with self.assertRaises(ValueError):
            extract_key_frames_by_seconds(42)
        with self.assertRaises(ValueError):
            extract_key_frames(42)

    def test_get_key_frame_seconds(self):
        input_path = self.vid_path  # 11.75s
        self.assertIsInstance(get_key_frame_seconds(input_path), list)

    def test_extract_frames_uniformly(self):
        input_path = self.vid_path  # 11.75s, 282 frames
        frames_1 = extract_video_frames_uniformly_by_seconds(input_path, 3, 1)
        self.assertEqual(len(frames_1), 33)  # floor(11.75 / 1) * 3

        container = load_video(input_path)
        frames_2 = extract_video_frames_uniformly_by_seconds(container, 5, 4)
        self.assertEqual(len(frames_2), 10)  # ceil(11.75 / 4) * 5

        frames_3 = extract_video_frames_uniformly(input_path, 1)
        self.assertEqual(len(frames_3), 1)
        frames_3 = extract_video_frames_uniformly(input_path, 7)
        self.assertEqual(len(frames_3), 7)
        frames_4 = extract_video_frames_uniformly(input_path, 300)
        self.assertEqual(len(frames_4), 282)

        with self.assertRaises(ValueError):
            extract_video_frames_uniformly_by_seconds(42, 3, 1)
        with self.assertRaises(ValueError):
            extract_video_frames_uniformly(42, 7)

    def test_extract_audio_from_video(self):
        input_path = self.vid_path
        output_path = os.path.join(self.temp_output_path, 'extracted_audio.mp3')
        data, sampling_rates, valid_stream_indexes = extract_audio_from_video(input_path, output_path)
        self.assertIsInstance(data, list)
        self.assertIsInstance(sampling_rates, list)
        self.assertIsInstance(valid_stream_indexes, list)
        self.assertEqual(len(data), len(sampling_rates))
        self.assertEqual(len(data), len(valid_stream_indexes))

        # set stream_indexes
        container = load_video(input_path)
        data, sampling_rates, valid_stream_indexes = extract_audio_from_video(container, output_path, stream_indexes=0)
        self.assertIsInstance(data, list)
        self.assertIsInstance(sampling_rates, list)
        self.assertIsInstance(valid_stream_indexes, list)
        self.assertEqual(len(data), len(sampling_rates))
        self.assertEqual(len(data), len(valid_stream_indexes))

        data, sampling_rates, valid_stream_indexes = extract_audio_from_video(input_path, output_path, stream_indexes=[0])
        self.assertIsInstance(data, list)
        self.assertIsInstance(sampling_rates, list)
        self.assertIsInstance(valid_stream_indexes, list)
        self.assertEqual(len(data), len(sampling_rates))
        self.assertEqual(len(data), len(valid_stream_indexes))

        # unsupported audio type
        with self.assertRaises(ValueError):
            extract_audio_from_video(42, output_path)
        with self.assertRaises(ValueError):
            extract_audio_from_video(input_path, output_path.replace('.mp3', '.wav'))

    def test_size_to_bytes(self):
        test_data = [
            ('1kb', 1024),
            ('5mib', 5 * 1024 * 1024),
            ('10GB', 10 * 1024 * 1024 * 1024),
            ('100TiB', 100 * 1024 * 1024 * 1024 * 1024),
            ('1024 PB', 1024 * 1024 * 1024 * 1024 * 1024 * 1024),
            ('3 eib', 3 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024),
            ('2 zb', 2 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024),
            ('9 YiB', 9 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024),
            ('15', 15),
        ]
        for size, expected_bytes in test_data:
            self.assertEqual(size_to_bytes(size), expected_bytes)

        with self.assertRaises(ValueError):
            size_to_bytes('mb')

        with self.assertRaises(ValueError):
            size_to_bytes('2 k')

    def test_insert_texts_after_placeholders(self):
        test_data = [
            ('ab<ph>cde', ['<ph>'], ['f'], ' ', 'ab<ph> f cde'),
            ('ab<ph1>cd<ph2>e', ['<ph1>', '<ph2>'], ['f', 'g'], '', 'ab<ph1>fcd<ph2>ge'),
        ]
        for string, phs, new_txts, delimiter, tgt_res in test_data:
            self.assertEqual(insert_texts_after_placeholders(string, phs, new_txts, delimiter), tgt_res)

        with self.assertRaises(ValueError):
            insert_texts_after_placeholders('test', ['<ph1>', '<ph2>'], ['only 1 text'])
        with self.assertRaises(ValueError):
            insert_texts_after_placeholders('no place holders', ['<ph1>', '<ph2>'], ['f', 'g'])

    def test_timecode_string_to_seconds(self):
        test_data = [
            ('23:59:59.123', 86399.123),
            ('23:59:59.123456', 86399.123456),
        ]
        for timecode, tgt_res in test_data:
            self.assertEqual(timecode_string_to_seconds(timecode), tgt_res)

        with self.assertRaises(ValueError):
            # missing milliseconds
            timecode_string_to_seconds('23:59:59')
        with self.assertRaises(ValueError):
            timecode_string_to_seconds('invalid timecode')

    def test_parse_string_to_roi(self):
        test_data = [
            ('3,30,5,50', 'pixel', (3, 30, 5, 50)),
            ('(0.1, 0, 1.3, 1)', 'ratio', (0.1, 0, 1, 1)),
            ('[22,    0,13,   1]', 'pixel', (22, 0, 13, 1)),
            ('[0.1, 0, 1.3, 1]', 'invalid_type', None),
            ('invalid roi string [0.1, 0, 1.3, 1]', 'pixel', None),
            ('', 'ratio', None),
        ]
        for roi_string, roi_type, tgt_res in test_data:
            self.assertEqual(parse_string_to_roi(roi_string, roi_type), tgt_res)

        # float pixels is invalid
        with self.assertRaises(ValueError):
            parse_string_to_roi('[0.1, 0, 1.3, 1]', 'pixel')


if __name__ == '__main__':
    unittest.main()

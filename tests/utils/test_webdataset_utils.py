import io
import os
import pickle
import shutil
import tempfile
import unittest

import numpy as np
from PIL import Image

from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase
from data_juicer.utils.webdataset_utils import (
    _custom_default_decoder,
    _custom_default_encoder,
    _encode_audio,
    _encode_image,
    _load_image,
    read_file_as_bytes,
    reconstruct_custom_webdataset_format,
)


class WebDatasetUtilsTest(DataJuicerTestCaseBase):

    def setUp(self):
        super().setUp()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        super().tearDown()

    def _image_array(self, value=64):
        arr = np.zeros((4, 5, 3), dtype=np.uint8)
        arr[:, :, 0] = value
        arr[:, :, 1] = 128
        arr[:, :, 2] = 255 - value
        return arr

    def _image_bytes(self, value=64, fmt="PNG"):
        stream = io.BytesIO()
        Image.fromarray(self._image_array(value)).save(stream, format=fmt)
        return stream.getvalue()

    def _write_bytes(self, name, content):
        path = os.path.join(self.temp_dir, name)
        with open(path, "wb") as f:
            f.write(content)
        return path

    def test_read_file_and_encode_audio_accept_bytes_or_paths(self):
        path = self._write_bytes("audio.wav", b"abc")

        self.assertEqual(read_file_as_bytes(path), b"abc")
        self.assertEqual(_encode_audio(path), b"abc")
        self.assertEqual(_encode_audio(b"raw"), b"raw")
        with self.assertRaisesRegex(AssertionError, "value should be a bytes"):
            _encode_audio(123)

    def test_encode_image_accepts_ndarray_bytes_pil_and_path(self):
        png_bytes = self._image_bytes(7, fmt="PNG")
        path = self._write_bytes("image.png", png_bytes)

        self.assertEqual(_encode_image(png_bytes, "png"), png_bytes)
        self.assertEqual(_encode_image(path, "png"), png_bytes)
        self.assertIsInstance(_encode_image(Image.fromarray(self._image_array(8)), "png"), bytes)
        self.assertIsInstance(_encode_image(self._image_array(9), "png"), bytes)

        pil = _load_image(png_bytes, "PIL")
        arr = _load_image(png_bytes, "numpy")
        self.assertIsInstance(pil, Image.Image)
        self.assertEqual(arr.shape, (4, 5, 3))

    def test_custom_encoder_handles_scalar_and_collection_extensions(self):
        img_a = self._image_bytes(10)
        img_b = self._image_bytes(20)
        sample = {
            "__key__": "kept",
            "caption.txt": "hello",
            "tokens.txt": ["a", "b"],
            "label.cls": 3,
            "meta.json": {"x": 1},
            "array.npy": np.array([[1, 2], [3, 4]], dtype=np.int64),
            "blob.pickle": {"nested": True},
            "image.png": img_a,
            "images.pngs": [img_a, img_b],
            "video.mp4": [img_a, img_b],
            "videos.mp4s": [[img_a], [img_b]],
            "unknown.bin": b"unchanged",
        }

        encoded = _custom_default_encoder(sample)

        self.assertEqual(encoded["__key__"], "kept")
        self.assertEqual(encoded["caption.txt"], b"hello")
        self.assertEqual(encoded["tokens.txt"], [b"a", b"b"])
        self.assertEqual(encoded["label.cls"], b"3")
        self.assertEqual(encoded["meta.json"], b'{"x": 1}')
        self.assertTrue(encoded["array.npy"].startswith(b"\x93NUMPY"))
        self.assertEqual(pickle.loads(encoded["blob.pickle"]), {"nested": True})
        self.assertEqual(encoded["image.png"], img_a)
        self.assertEqual(len(pickle.loads(encoded["images.pngs"])), 2)
        self.assertEqual(len(pickle.loads(encoded["video.mp4"])), 2)
        self.assertEqual([len(v) for v in pickle.loads(encoded["videos.mp4s"])], [1, 1])
        self.assertEqual(encoded["unknown.bin"], b"unchanged")

    def test_custom_decoder_handles_text_json_numpy_pickle_images_and_videos(self):
        img_a = self._image_bytes(11)
        img_b = self._image_bytes(22)
        np_stream = io.BytesIO()
        np.save(np_stream, np.array([1, 2, 3], dtype=np.int64))
        sample = {
            "__key__": "kept",
            "caption.txt": b"hello",
            "label.cls2": b"5",
            "meta.json": b'{"x": 1}',
            "array.npy": np_stream.getvalue(),
            "blob.pkl": pickle.dumps({"nested": True}),
            "image.png": img_a,
            "images.pngs": pickle.dumps([img_a, img_b]),
            "video.mp4": pickle.dumps([img_a, img_b]),
            "videos.mp4s": pickle.dumps([[img_a], [img_b]]),
            "unknown.bin": b"unchanged",
        }

        decoded = _custom_default_decoder(sample, format="PIL")

        self.assertEqual(decoded["__key__"], "kept")
        self.assertEqual(decoded["caption.txt"], "hello")
        self.assertEqual(decoded["label.cls2"], 5)
        self.assertEqual(decoded["meta.json"], {"x": 1})
        np.testing.assert_array_equal(decoded["array.npy"], np.array([1, 2, 3], dtype=np.int64))
        self.assertEqual(decoded["blob.pkl"], {"nested": True})
        self.assertIsInstance(decoded["image.png"], Image.Image)
        self.assertEqual(len(decoded["images.pngs"]), 2)
        self.assertEqual(len(decoded["video.mp4"]), 2)
        self.assertEqual([len(v) for v in decoded["videos.mp4s"]], [1, 1])
        self.assertEqual(decoded["unknown.bin"], b"unchanged")

    def test_custom_encoder_decoder_round_trips_core_extensions(self):
        image = self._image_array(77)
        array = np.array([[1, 2], [3, 4]], dtype=np.int64)
        sample = {
            "caption.txt": "round trip",
            "label.cls": 9,
            "meta.json": {"source": "unit", "count": 2},
            "array.npy": array,
            "image.png": image,
        }

        encoded = _custom_default_encoder(sample)
        decoded = _custom_default_decoder(encoded, format="numpy")

        self.assertEqual(decoded["caption.txt"], "round trip")
        self.assertEqual(decoded["label.cls"], 9)
        self.assertEqual(decoded["meta.json"], {"source": "unit", "count": 2})
        np.testing.assert_array_equal(decoded["array.npy"], array)
        np.testing.assert_array_equal(decoded["image.png"], image)

    def test_custom_encoder_decoder_handle_msgpack_and_audio_lists(self):
        import msgpack

        sample = {
            "payload.mp": {"items": [1, "two"]},
            "clips.wavs": [b"first", b"second"],
        }

        encoded = _custom_default_encoder(sample)

        self.assertEqual(msgpack.unpackb(encoded["payload.mp"], raw=False), {"items": [1, "two"]})
        self.assertEqual(pickle.loads(encoded["clips.wavs"]), [b"first", b"second"])

        decoded = _custom_default_decoder(
            {
                "payload.mp": encoded["payload.mp"],
            }
        )
        self.assertEqual(decoded["payload.mp"], {"items": [1, "two"]})

    def test_custom_encoder_encodes_video_frames_from_paths(self):
        frame_a = self._write_bytes("frame_a.png", self._image_bytes(33))
        frame_b = self._write_bytes("frame_b.png", self._image_bytes(44))

        encoded = _custom_default_encoder(
            {
                "clip.mp4": [frame_a, self._image_bytes(55)],
                "clips.mp4s": [[frame_a], [frame_b]],
            }
        )

        self.assertEqual(len(pickle.loads(encoded["clip.mp4"])), 2)
        decoded_videos = pickle.loads(encoded["clips.mp4s"])
        self.assertEqual([len(frames) for frames in decoded_videos], [1, 1])

    def test_reconstruct_custom_webdataset_format(self):
        samples = {
            "text": ["a", "b"],
            "image": ["i1", "i2"],
            "audio": ["a1", "a2"],
        }

        self.assertIs(reconstruct_custom_webdataset_format(samples), samples)
        self.assertEqual(
            reconstruct_custom_webdataset_format(
                samples,
                field_mapping={
                    "caption.txt": "text",
                    "media.json": ["image", "audio"],
                },
            ),
            {
                "caption.txt": ["a", "b"],
                "media.json": {"image": ["i1", "i2"], "audio": ["a1", "a2"]},
            },
        )
        with self.assertRaises(AssertionError):
            reconstruct_custom_webdataset_format(samples, field_mapping="bad")
        with self.assertRaises(AssertionError):
            reconstruct_custom_webdataset_format(samples, field_mapping={"bad": 1})


if __name__ == "__main__":
    unittest.main()

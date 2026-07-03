import os
import tempfile
import unittest
from types import SimpleNamespace

from cryptography.fernet import Fernet

from data_juicer.core.data import NestedDataset as Dataset

from data_juicer.format.formatter import load_dataset, unify_format
from data_juicer.utils.constant import Fields
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class UnifyFormatTest(DataJuicerTestCaseBase):

    def run_test(self, sample, args=None):
        if args is None:
            args = {}
        ds = Dataset.from_list(sample['source'])
        ds = unify_format(ds, **args)
        self.assertEqual(ds.to_list(), sample['target'])

    def test_text_key(self):
        samples = [
            {
                'source': [{
                    'text': 'This is a test text',
                    'outer_key': 1,
                }],
                'target': [{
                    'text': 'This is a test text',
                    'outer_key': 1,
                }]
            },
            {
                'source': [{
                    'content': 'This is a test text',
                    'outer_key': 1,
                }],
                'target': [{
                    'content': 'This is a test text',
                    'outer_key': 1,
                }]
            },
            {
                'source': [{
                    'input': 'This is a test text, input part',
                    'instruction': 'This is a test text, instruction part',
                    'outer_key': 1,
                }],
                'target': [{
                    'input': 'This is a test text, input part',
                    'instruction': 'This is a test text, instruction part',
                    'outer_key': 1,
                }]
            },
        ]
        self.run_test(samples[0])
        self.run_test(samples[1], args={'text_keys': ['content']})
        self.run_test(samples[2], args={'text_keys': ['input', 'instruction']})

    def test_empty_text(self):
        # filter out samples containing None field, but '' is OK
        samples = [
            {
                'source': [{
                    'text': '',
                    'outer_key': 1,
                }],
                'target': [{
                    'text': '',
                    'outer_key': 1,
                }],
            },
            {
                'source': [{
                    'text': None,
                    'outer_key': 1,
                }],
                'target': [],
            },
        ]
        for sample in samples:
            self.run_test(sample)

    def test_no_extra_fields(self):
        samples = [{
            'source': [{
                'text': 'This is a test text.',
                Fields.stats: {
                    'lang': 'en'
                },
            }],
            'target': [{
                'text': 'This is a test text.',
                Fields.stats: {
                    'lang': 'en'
                },
            }],
        }, {
            'source': [{
                'text': 'This is a test text.',
            }],
            'target': [{
                'text': 'This is a test text.',
            }],
        }]
        for sample in samples:
            self.run_test(sample)

    def test_no_extra_fields_except_meta(self):
        samples = [{
            'source': [{
                'text': 'This is a test text.',
                'meta': {
                    'version': 1
                },
                Fields.stats: {
                    'lang': 'en'
                },
            }],
            'target': [{
                'text': 'This is a test text.',
                'meta': {
                    'version': 1
                },
                Fields.stats: {
                    'lang': 'en'
                },
            }],
        }, {
            'source': [{
                'text': 'This is a test text.',
                'meta': {
                    'version': 1
                },
            }],
            'target': [{
                'text': 'This is a test text.',
                'meta': {
                    'version': 1
                },
            }],
        }]
        for sample in samples:
            self.run_test(sample)

    def test_invalid_stats(self):
        # non-dict stats will be unified into stats
        samples = [{
            'source': [{
                'text': 'This is a test text.',
                'stats': 'nice',
            }],
            'target': [{
                'text': 'This is a test text.',
                'stats': 'nice'
            }],
        }, {
            'source': [{
                'text': 'This is a test text.',
                Fields.stats: {
                    'version': 1
                },
            }],
            'target': [{
                'text': 'This is a test text.',
                Fields.stats: {
                    'version': 1
                },
            }],
        }]
        for sample in samples:
            self.run_test(sample)

    def test_outer_fields(self):
        samples = [
            {
                'source': [{
                    'text': 'This is a test text.',
                    'meta': {
                        'meta_inner': 'nice'
                    },
                    'outer_field': 'value'
                }],
                'target': [{
                    'text': 'This is a test text.',
                    'meta': {
                        'meta_inner': 'nice',
                    },
                    'outer_field': 'value',
                }],
            },
            {
                'source': [{
                    'text': 'This is a test text.',
                    'outer_key': 'nice',
                    'outer_field': 'value'
                }],
                'target': [{
                    'text': 'This is a test text.',
                    'outer_key': 'nice',
                    'outer_field': 'value',
                }],
            },
            {
                'source': [{
                    'text': 'This is a test text.',
                    'meta': 'nice',
                    'outer_field': 'value'
                }],
                'target': [{
                    'text': 'This is a test text.',
                    'meta': 'nice',
                    'outer_field': 'value',
                }],
            },
            {
                'source': [{
                    'text': 'This is a test text.',
                    'meta': {
                        'meta_inner': 'nice'
                    },
                    'outer_field': 'value',
                    Fields.stats: {
                        'lang': 'en'
                    },
                }],
                'target': [{
                    'text': 'This is a test text.',
                    'meta': {
                        'meta_inner': 'nice'
                    },
                    'outer_field': 'value',
                    Fields.stats: {
                        'lang': 'en'
                    },
                }],
            },
            {
                'source': [{
                    'text': 'This is a test text.',
                    'outer_key': 'nice',
                    'outer_field': 'value',
                    Fields.stats: {
                        'lang': 'en'
                    },
                }],
                'target': [{
                    'text': 'This is a test text.',
                    'outer_key': 'nice',
                    'outer_field': 'value',
                    Fields.stats: {
                        'lang': 'en'
                    },
                }],
            },
            {
                'source': [{
                    'text': 'This is a test text.',
                    'meta': 'nice',
                    'outer_field': 'value',
                    Fields.stats: {
                        'lang': 'en'
                    },
                }],
                'target': [{
                    'text': 'This is a test text.',
                    'meta': 'nice',
                    'outer_field': 'value',
                    Fields.stats: {
                        'lang': 'en'
                    },
                }],
            },
            {
                'source': [{
                    'text': 'This is a test text.',
                    'meta': {
                        'meta_inner': 'nice'
                    },
                    'outer_field': 'value',
                    'stats': 'en'
                }],
                'target': [{
                    'text': 'This is a test text.',
                    'meta': {
                        'meta_inner': 'nice'
                    },
                    'outer_field': 'value',
                    'stats': 'en'
                }],
            },
            {
                'source': [{
                    'text': 'This is a test text.',
                    'outer_key': 'nice',
                    'outer_field': 'value',
                    'stats': 'en'
                }],
                'target': [{
                    'text': 'This is a test text.',
                    'outer_key': 'nice',
                    'outer_field': 'value',
                    'stats': 'en'
                }],
            },
            {
                'source': [{
                    'text': 'This is a test text.',
                    'meta': 'nice',
                    'outer_field': 'value',
                    'stats': 'en',
                }],
                'target': [{
                    'text': 'This is a test text.',
                    'meta': 'nice',
                    'outer_field': 'value',
                    'stats': 'en'
                }],
            },
        ]
        for sample in samples:
            self.run_test(sample)

    def test_recursive_meta(self):
        samples = [{
            'source': [{
                'text': 'This is a test text.',
                'outer_field': {
                    'rec1': {
                        'rec2': 'value'
                    }
                },
            }],
            'target': [{
                'text': 'This is a test text.',
                'outer_field': {
                    'rec1': {
                        'rec2': 'value'
                    }
                },
            }],
        }]
        for sample in samples:
            self.run_test(sample)

    def test_hetero_meta(self):
        cur_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                               'data', 'structured')
        file_path = os.path.join(cur_dir, 'demo-dataset.jsonl')
        ds = load_dataset('json', data_files=file_path, split='train')
        ds = unify_format(ds)
        # test nested and missing field for the following cases:
        # Fields present in a row are always accessible; fields absent in the raw
        # data may be filled with None (datasets <=4.4 struct merge) OR simply
        # missing (datasets >=4.8 Json type).  Use .get() for absent fields so
        # the test is compatible with both behaviours.
        # 1. first row, then nested key
        unified_sample_first = ds[0]
        unified_sample_second = ds[1]
        self.assertEqual(unified_sample_first['meta']['src'], 'Arxiv')
        self.assertIsNone(unified_sample_first['meta'].get('author'))  # absent or None
        self.assertIsNone(unified_sample_second['meta'].get('date'))   # absent or None
        # 2. meta column (struct/json), then index
        meta_col = ds['meta']
        self.assertEqual(meta_col[0]['src'], 'Arxiv')
        self.assertEqual(meta_col[1]['src'], 'code')
        self.assertIsNone(meta_col[0].get('author'))  # absent or None
        self.assertIsNone(meta_col[1].get('date'))    # absent or None
        # 3. first partial rows, then column, final row
        unified_ds_first = ds.select([0])
        unified_ds_second = ds.select([1])
        self.assertEqual(unified_ds_first['meta'][0]['src'], 'Arxiv')
        self.assertIsNone(unified_ds_first['meta'][0].get('author'))   # absent or None
        self.assertIsNone(unified_ds_second['meta'][0].get('date'))    # absent or None

    def test_empty_meta(self):
        samples = [{
            'source': [{
                'text': 'This is a test text.',
                'meta': {},
            }],
            'target': [{
                'text': 'This is a test text.',
                'meta': {},
            }],
        }]
        for sample in samples:
            self.run_test(sample)

    def test_empty_stats(self):
        samples = [{
            'source': [{
                'text': 'This is a test text.',
                'meta': {},
                Fields.stats: {},
            }],
            'target': [{
                'text': 'This is a test text.',
                'meta': {},
                Fields.stats: {},
            }],
        }]
        for sample in samples:
            self.run_test(sample)

    def test_empty_outer_fields(self):
        samples = [{
            'source': [{
                'text': 'This is a test text.',
                'meta': {},
                'out_field': {},
            }],
            'target': [{
                'text': 'This is a test text.',
                'meta': {},
                'out_field': {},
            }],
        }, {
            'source': [{
                'text': 'This is a test text.',
                'out_field': {},
            }],
            'target': [{
                'text': 'This is a test text.',
                'out_field': {},
            }],
        }]
        for sample in samples:
            self.run_test(sample)


if __name__ == '__main__':
    unittest.main()


# ---------------------------------------------------------------------------
# Tests for LocalFormatter.load_dataset with decrypt_after_reading
# ---------------------------------------------------------------------------

_STRUCTURED_DATA_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "data", "structured"
)


def _make_global_cfg(key_path, decrypt=True):
    return SimpleNamespace(
        decrypt_after_reading=decrypt,
        encryption_key_path=key_path,
    )


class LocalFormatterDecryptTest(DataJuicerTestCaseBase):
    """Tests for the decrypt_after_reading path in LocalFormatter.load_dataset.

    We test using the JsonFormatter (backed by LocalFormatter) and the
    demo-dataset.jsonl fixture that already exists in the test data directory.
    """

    def setUp(self):
        super().setUp()
        self.key = Fernet.generate_key()
        self.fernet = Fernet(self.key)
        self.src_jsonl = os.path.join(_STRUCTURED_DATA_DIR, "demo-dataset.jsonl")

    def _write_key_file(self, tmp_dir):
        key_path = os.path.join(tmp_dir, "test.key")
        with open(key_path, "wb") as f:
            f.write(self.key)
        return key_path

    def _encrypt_file(self, src_path, dst_path):
        with open(src_path, "rb") as f:
            plaintext = f.read()
        with open(dst_path, "wb") as f:
            f.write(self.fernet.encrypt(plaintext))

    # ------------------------------------------------------------------

    def test_decrypt_jsonl_sample_count(self):
        """load_dataset with decrypt_after_reading returns correct row count."""
        from data_juicer.format.json_formatter import JsonFormatter

        with tempfile.TemporaryDirectory() as tmp:
            key_path = self._write_key_file(tmp)
            enc_path = os.path.join(tmp, "demo-dataset.jsonl")
            self._encrypt_file(self.src_jsonl, enc_path)

            formatter = JsonFormatter(enc_path)
            global_cfg = _make_global_cfg(key_path)
            ds = formatter.load_dataset(num_proc=1, global_cfg=global_cfg)

            # demo-dataset.jsonl has 6 rows
            self.assertEqual(len(ds), 6)
            self.assertIn("text", ds.features)

    def test_decrypt_jsonl_content_matches_plaintext(self):
        """Decrypted content is identical to loading the plaintext directly."""
        from data_juicer.format.json_formatter import JsonFormatter

        with tempfile.TemporaryDirectory() as tmp:
            key_path = self._write_key_file(tmp)
            enc_path = os.path.join(tmp, "demo-dataset.jsonl")
            self._encrypt_file(self.src_jsonl, enc_path)

            formatter_enc = JsonFormatter(enc_path)
            global_cfg = _make_global_cfg(key_path)
            ds_enc = formatter_enc.load_dataset(num_proc=1, global_cfg=global_cfg)

            formatter_plain = JsonFormatter(self.src_jsonl)
            ds_plain = formatter_plain.load_dataset(num_proc=1)

            self.assertEqual(
                sorted(r["text"] for r in ds_enc.to_list()),
                sorted(r["text"] for r in ds_plain.to_list()),
            )

    def test_no_decrypt_when_flag_false(self):
        """With decrypt_after_reading=False the plaintext file loads normally."""
        from data_juicer.format.json_formatter import JsonFormatter

        formatter = JsonFormatter(self.src_jsonl)
        global_cfg = _make_global_cfg(key_path=None, decrypt=False)
        ds = formatter.load_dataset(num_proc=1, global_cfg=global_cfg)
        self.assertEqual(len(ds), 6)

    def test_tmp_files_cleaned_up_after_load(self):
        """Temporary decrypt files must be removed after load_dataset returns."""
        import tempfile as _tempfile
        from data_juicer.format.json_formatter import JsonFormatter

        created_tmp_files = []
        original_ntf = _tempfile.NamedTemporaryFile

        def tracking_ntf(*args, **kwargs):
            ntf = original_ntf(*args, **kwargs)
            created_tmp_files.append(ntf.name)
            return ntf

        with tempfile.TemporaryDirectory() as tmp:
            key_path = self._write_key_file(tmp)
            enc_path = os.path.join(tmp, "demo-dataset.jsonl")
            self._encrypt_file(self.src_jsonl, enc_path)

            formatter = JsonFormatter(enc_path)
            global_cfg = _make_global_cfg(key_path)

            from unittest.mock import patch
            with patch("tempfile.NamedTemporaryFile", side_effect=tracking_ntf):
                formatter.load_dataset(num_proc=1, global_cfg=global_cfg)

        # All tracked temp files must have been deleted
        for p in created_tmp_files:
            self.assertFalse(
                os.path.exists(p),
                f"Temporary file {p} was not cleaned up after load_dataset",
            )


# ---------------------------------------------------------------------------
# Additional coverage tests for unify_format edge cases and add_suffixes
# ---------------------------------------------------------------------------

from datasets import Dataset as HFDataset, DatasetDict
from jsonargparse import Namespace as JANamespace
from data_juicer.format.formatter import add_suffixes


class UnifyFormatNoneFilterTest(DataJuicerTestCaseBase):
    """Test that unify_format filters out samples with None text."""

    def test_filters_none_text(self):
        ds = HFDataset.from_dict({
            "text": ["hello", None, "world", None],
        })
        result = unify_format(ds, text_keys=["text"])
        self.assertEqual(len(result), 2)
        self.assertEqual(list(result["text"]), ["hello", "world"])

    def test_keeps_all_non_none(self):
        ds = HFDataset.from_dict({"text": ["a", "b", "c"]})
        result = unify_format(ds, text_keys=["text"])
        self.assertEqual(len(result), 3)

    def test_filters_all_none(self):
        ds = HFDataset.from_dict({"text": [None, None]})
        result = unify_format(ds, text_keys=["text"])
        self.assertEqual(len(result), 0)

    def test_empty_dataset(self):
        ds = HFDataset.from_dict({"text": []})
        result = unify_format(ds, text_keys=["text"])
        self.assertEqual(len(result), 0)


class UnifyFormatTextKeysTest(DataJuicerTestCaseBase):

    def test_missing_text_key_raises(self):
        ds = HFDataset.from_dict({"content": ["hello"]})
        with self.assertRaises(ValueError) as ctx:
            unify_format(ds, text_keys=["text"])
        self.assertIn("no key [text]", str(ctx.exception).lower())

    def test_string_text_key_converted_to_list(self):
        ds = HFDataset.from_dict({"text": ["hello"]})
        result = unify_format(ds, text_keys="text")
        self.assertEqual(len(result), 1)

    def test_none_text_keys_skips_filtering(self):
        ds = HFDataset.from_dict({"text": ["hello", None]})
        result = unify_format(ds, text_keys=None)
        self.assertEqual(len(result), 2)

    def test_empty_text_keys_skips_filtering(self):
        ds = HFDataset.from_dict({"text": ["hello", None]})
        result = unify_format(ds, text_keys=[])
        self.assertEqual(len(result), 2)


class UnifyFormatDatasetDictTest(DataJuicerTestCaseBase):

    def test_unwraps_single_split_datasetdict(self):
        ds = HFDataset.from_dict({"text": ["hello", "world"]})
        dd = DatasetDict({"train": ds})
        result = unify_format(dd, text_keys=["text"])
        self.assertEqual(len(result), 2)

    def test_multiple_splits_raises(self):
        ds1 = HFDataset.from_dict({"text": ["a"]})
        ds2 = HFDataset.from_dict({"text": ["b"]})
        dd = DatasetDict({"train": ds1, "test": ds2})
        with self.assertRaises(AssertionError):
            unify_format(dd, text_keys=["text"])


class UnifyFormatPathConversionTest(DataJuicerTestCaseBase):
    """Test relative-to-absolute path conversion in unify_format."""

    def setUp(self):
        super().setUp()
        self.tmp_dir = tempfile.mkdtemp()
        self.ds_dir = os.path.join(self.tmp_dir, "dataset")
        os.makedirs(self.ds_dir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        super().tearDown()

    def test_converts_relative_image_paths(self):
        ds = HFDataset.from_dict({
            "text": ["sample1", "sample2"],
            "images": [["img1.jpg"], ["img2.jpg"]],
        })
        cfg = JANamespace(
            dataset_path=self.ds_dir,
            image_key="images",
        )
        result = unify_format(ds, text_keys=["text"], global_cfg=cfg)
        for row in result:
            for path in row["images"]:
                self.assertTrue(os.path.isabs(path))
                self.assertTrue(path.startswith(self.ds_dir))

    def test_preserves_absolute_paths(self):
        abs_path = "/absolute/path/img.jpg"
        ds = HFDataset.from_dict({
            "text": ["sample"],
            "images": [[abs_path]],
        })
        cfg = JANamespace(
            dataset_path=self.ds_dir,
            image_key="images",
        )
        result = unify_format(ds, text_keys=["text"], global_cfg=cfg)
        self.assertEqual(result[0]["images"][0], abs_path)

    def test_no_media_keys_returns_unchanged(self):
        ds = HFDataset.from_dict({"text": ["sample"]})
        cfg = JANamespace(
            dataset_path=self.ds_dir,
            image_key="images",
        )
        result = unify_format(ds, text_keys=["text"], global_cfg=cfg)
        self.assertEqual(len(result), 1)

    def test_no_global_cfg_warns(self):
        ds = HFDataset.from_dict({"text": ["sample"]})
        result = unify_format(ds, text_keys=["text"], global_cfg=None)
        self.assertEqual(len(result), 1)

    def test_empty_ds_dir_skips_conversion(self):
        ds = HFDataset.from_dict({
            "text": ["sample"],
            "images": [["relative/img.jpg"]],
        })
        cfg = JANamespace(
            dataset_path="/nonexistent/path",
            image_key="images",
        )
        result = unify_format(ds, text_keys=["text"], global_cfg=cfg)
        self.assertEqual(result[0]["images"][0], "relative/img.jpg")

    def test_dict_global_cfg(self):
        ds = HFDataset.from_dict({
            "text": ["sample"],
            "images": [["img.jpg"]],
        })
        cfg = {
            "dataset_path": self.ds_dir,
            "image_key": "images",
        }
        result = unify_format(ds, text_keys=["text"], global_cfg=cfg)
        self.assertTrue(os.path.isabs(result[0]["images"][0]))

    def test_dataset_path_is_file(self):
        ds_file = os.path.join(self.ds_dir, "data.jsonl")
        with open(ds_file, "w") as f:
            f.write('{"text": "hello"}\n')

        ds = HFDataset.from_dict({
            "text": ["sample"],
            "images": [["img.jpg"]],
        })
        cfg = JANamespace(
            dataset_path=ds_file,
            image_key="images",
        )
        result = unify_format(ds, text_keys=["text"], global_cfg=cfg)
        self.assertTrue(result[0]["images"][0].startswith(self.ds_dir))


class AddSuffixesTest(DataJuicerTestCaseBase):

    def test_adds_suffix_column(self):
        ds1 = HFDataset.from_dict({"text": ["a", "b"]})
        ds2 = HFDataset.from_dict({"text": ["c"]})
        dd = DatasetDict({"json": ds1, "csv": ds2})

        result = add_suffixes(dd)
        self.assertIn("__dj__suffix__", result.column_names)
        suffixes = result["__dj__suffix__"]
        self.assertIn(".json", suffixes)
        self.assertIn(".csv", suffixes)
        self.assertEqual(len(result), 3)


class UnifyFormatAudioVideoPathTest(DataJuicerTestCaseBase):
    """Cover audio_key / video_key relative-to-absolute path conversion
    (lines 226-232, 244, 253, 261, 272-302 in formatter.py)."""

    def setUp(self):
        super().setUp()
        self.tmp_dir = tempfile.mkdtemp()
        self.ds_dir = os.path.join(self.tmp_dir, "dataset")
        os.makedirs(self.ds_dir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        super().tearDown()

    def test_converts_relative_audio_paths(self):
        ds = HFDataset.from_dict({
            "text": ["s1"],
            "audios": [["clip.wav"]],
        })
        cfg = JANamespace(
            dataset_path=self.ds_dir,
            audio_key="audios",
        )
        result = unify_format(ds, text_keys=["text"], global_cfg=cfg)
        audio_path = result[0]["audios"][0]
        self.assertTrue(os.path.isabs(audio_path))
        self.assertTrue(audio_path.startswith(self.ds_dir))

    def test_converts_relative_video_paths(self):
        ds = HFDataset.from_dict({
            "text": ["s1"],
            "videos": [["clip.mp4"]],
        })
        cfg = JANamespace(
            dataset_path=self.ds_dir,
            video_key="videos",
        )
        result = unify_format(ds, text_keys=["text"], global_cfg=cfg)
        video_path = result[0]["videos"][0]
        self.assertTrue(os.path.isabs(video_path))
        self.assertTrue(video_path.startswith(self.ds_dir))

    def test_mixed_media_keys(self):
        """All three media keys (image/audio/video) converted in one call."""
        ds = HFDataset.from_dict({
            "text": ["s1"],
            "images": [["img.jpg"]],
            "audios": [["clip.wav"]],
            "videos": [["clip.mp4"]],
        })
        cfg = JANamespace(
            dataset_path=self.ds_dir,
            image_key="images",
            audio_key="audios",
            video_key="videos",
        )
        result = unify_format(ds, text_keys=["text"], global_cfg=cfg)
        row = result[0]
        for key in ("images", "audios", "videos"):
            self.assertTrue(os.path.isabs(row[key][0]),
                            f"{key} path not absolute: {row[key][0]}")

    def test_custom_audio_key_name(self):
        ds = HFDataset.from_dict({
            "text": ["s1"],
            "my_audio": [["clip.wav"]],
        })
        cfg = JANamespace(
            dataset_path=self.ds_dir,
            audio_key="my_audio",
        )
        result = unify_format(ds, text_keys=["text"], global_cfg=cfg)
        self.assertTrue(os.path.isabs(result[0]["my_audio"][0]))

    def test_custom_video_key_name(self):
        ds = HFDataset.from_dict({
            "text": ["s1"],
            "my_video": [["clip.mp4"]],
        })
        cfg = JANamespace(
            dataset_path=self.ds_dir,
            video_key="my_video",
        )
        result = unify_format(ds, text_keys=["text"], global_cfg=cfg)
        self.assertTrue(os.path.isabs(result[0]["my_video"][0]))

    def test_preserves_absolute_audio_path(self):
        abs_path = "/abs/audio.wav"
        ds = HFDataset.from_dict({
            "text": ["s1"],
            "audios": [[abs_path]],
        })
        cfg = JANamespace(
            dataset_path=self.ds_dir,
            audio_key="audios",
        )
        result = unify_format(ds, text_keys=["text"], global_cfg=cfg)
        self.assertEqual(result[0]["audios"][0], abs_path)

    def test_empty_media_list_unchanged(self):
        ds = HFDataset.from_dict({
            "text": ["s1"],
            "audios": [[]],
        })
        cfg = JANamespace(
            dataset_path=self.ds_dir,
            audio_key="audios",
        )
        result = unify_format(ds, text_keys=["text"], global_cfg=cfg)
        self.assertEqual(result[0]["audios"], [])


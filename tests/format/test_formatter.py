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
        # import datetime

        # the 'None' fields are missing fields after merging
        # sample = [{
        #     'text': "Today is Sunday and it's a happy day!",
        #     'meta': {
        #         'src': 'Arxiv',
        #         'date': datetime.datetime(2023, 4, 27, 0, 0),
        #         'version': '1.0',
        #         'author': None
        #     }
        # }, {
        #     'text': 'Do you need a cup of coffee?',
        #     'meta': {
        #         'src': 'code',
        #         'date': None,
        #         'version': None,
        #         'author': 'xxx'
        #     }
        # }]
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

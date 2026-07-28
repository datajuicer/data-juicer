import json
import os
import shutil
import tempfile
import unittest
import jsonlines as jl
from unittest.mock import patch

import numpy as np
from datasets import Dataset
from cryptography.fernet import Fernet

from data_juicer.core import Exporter
from data_juicer.core import exporter as exporter_module
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase
from data_juicer.utils.constant import Fields, HashKeys
from data_juicer.utils.file_utils import add_suffix_to_filename

class ExporterTest(DataJuicerTestCaseBase):

    def setUp(self) -> None:
        super().setUp()
        self.work_dir = 'tmp/test_exporter/'
        os.makedirs(self.work_dir, exist_ok=True)

        self.test_data = Dataset.from_list([
            {
                'text': 'text 1',
                Fields.stats: {
                    'a': 1,
                    'b': 2
                },
                Fields.meta: {
                    'c': 'tag1'
                },
                HashKeys.hash: 'hash1'
            },
            {
                'text': 'text 2',
                Fields.stats: {
                    'a': 3,
                    'b': 4
                },
                Fields.meta: {
                    'c': 'tag2'
                },
                HashKeys.hash: 'hash2'
            },
            {
                'text': 'text 3',
                Fields.stats: {
                    'a': 5,
                    'b': 6
                },
                Fields.meta: {
                    'c': 'tag3'
                },
                HashKeys.hash: 'hash3'
            },
        ])

    def tearDown(self):
        super().tearDown()
        if os.path.exists(self.work_dir):
            os.system(f'rm -rf {self.work_dir}')

    def test_normal_function(self):
        export_path = os.path.join(self.work_dir, 'normal', 'test.jsonl')
        exporter = Exporter(
            export_path=export_path,
            export_shard_size=0,
            export_in_parallel=True,
            num_proc=1,
            export_ds=True,
            keep_stats_in_res_ds=False,
            keep_hashes_in_res_ds=False,
            export_stats=True,
        )
        exporter.export(self.test_data)

        # check exported files
        self.assertTrue(os.path.exists(export_path))
        self.assertTrue(os.path.exists(add_suffix_to_filename(export_path, '_stats')))

    def test_different_shard_size(self):
        export_path = os.path.join(self.work_dir, 'shard_size', 'test.json')
        # bytes
        exporter = Exporter(
            export_path=export_path,
            export_shard_size=0,
        )
        self.assertIn('Bytes', exporter.max_shard_size_str)

        # KiB
        exporter = Exporter(
            export_path=export_path,
            export_shard_size=2 * 2 ** 10,
        )
        self.assertIn('KiB', exporter.max_shard_size_str)

        # MiB
        exporter = Exporter(
            export_path=export_path,
            export_shard_size=2 * 2 ** 20,
        )
        self.assertIn('MiB', exporter.max_shard_size_str)

        # GiB
        exporter = Exporter(
            export_path=export_path,
            export_shard_size=2 * 2 ** 30,
        )
        self.assertIn('GiB', exporter.max_shard_size_str)

        # TiB
        exporter = Exporter(
            export_path=export_path,
            export_shard_size=2 * 2 ** 40,
        )
        self.assertIn('TiB', exporter.max_shard_size_str)

        # more --> TiB
        exporter = Exporter(
            export_path=export_path,
            export_shard_size=2 * 2 ** 50,
        )
        self.assertIn('TiB', exporter.max_shard_size_str)

    def test_supported_suffix(self):
        exporter = Exporter(
            export_path=os.path.join(self.work_dir, 'json', 'test.json'),
        )
        self.assertEqual('json', exporter.suffix)
        exporter.export(self.test_data)
        self.assertTrue(os.path.exists(os.path.join(self.work_dir, 'json', 'test.json')))
        self.assertTrue(os.path.exists(os.path.join(self.work_dir, 'json', 'test_stats.jsonl')))

        exporter = Exporter(
            export_path=os.path.join(self.work_dir, 'jsonl', 'test.jsonl'),
        )
        self.assertEqual('jsonl', exporter.suffix)
        exporter.export(self.test_data)
        self.assertTrue(os.path.exists(os.path.join(self.work_dir, 'jsonl', 'test.jsonl')))
        self.assertTrue(os.path.exists(os.path.join(self.work_dir, 'jsonl', 'test_stats.jsonl')))

        exporter = Exporter(
            export_path=os.path.join(self.work_dir, 'parquet', 'test.parquet'),
        )
        self.assertEqual('parquet', exporter.suffix)
        exporter.export(self.test_data)
        self.assertTrue(os.path.exists(os.path.join(self.work_dir, 'parquet', 'test.parquet')))
        self.assertTrue(os.path.exists(os.path.join(self.work_dir, 'parquet', 'test_stats.jsonl')))

        with self.assertRaises(NotImplementedError):
            Exporter(
                export_path=os.path.join(self.work_dir, 'txt', 'test.txt'),
            )

    def test_export_multiple_shards(self):
        export_path = os.path.join(self.work_dir, 'shards', 'test.jsonl')
        exporter = Exporter(
            export_path=export_path,
            export_shard_size=1024,
            export_in_parallel=True,
            num_proc=1,
            export_ds=True,
            keep_stats_in_res_ds=False,
            keep_hashes_in_res_ds=False,
            export_stats=True,
        )
        exporter.export(self.test_data)

        # check exported files
        self.assertTrue(os.path.exists(add_suffix_to_filename(export_path, '-00-of-01')))
        self.assertTrue(os.path.exists(add_suffix_to_filename(export_path, '_stats')))

    def test_export_compute_stats(self):
        export_path = os.path.join(self.work_dir, 'stats', 'res.jsonl')
        exporter = Exporter(
            export_path=export_path,
        )
        exporter.export_compute_stats(self.test_data, export_path)

        self.assertTrue(os.path.exists(export_path))
        self.assertFalse(os.path.exists(add_suffix_to_filename(export_path, '_stats')))


class ExporterEncryptTest(DataJuicerTestCaseBase):
    """Tests for the encrypt_before_export feature of Exporter."""

    def setUp(self):
        super().setUp()
        self.work_dir = 'tmp/test_exporter_encrypt/'
        os.makedirs(self.work_dir, exist_ok=True)
        self.key = Fernet.generate_key()
        self.fernet = Fernet(self.key)
        self.test_data = Dataset.from_list([
            {'text': 'hello', Fields.stats: {'score': 1}},
            {'text': 'world', Fields.stats: {'score': 2}},
        ])

    def tearDown(self):
        super().tearDown()
        if os.path.exists(self.work_dir):
            os.system(f'rm -rf {self.work_dir}')

    def _write_key_file(self):
        key_file = os.path.join(self.work_dir, 'test.key')
        with open(key_file, 'wb') as f:
            f.write(self.key)
        return key_file

    # ------------------------------------------------------------------
    # __init__ parameter handling
    # ------------------------------------------------------------------

    def test_encrypt_flag_disabled_by_default(self):
        os.makedirs(os.path.join(self.work_dir, 'default'), exist_ok=True)
        exporter = Exporter(
            export_path=os.path.join(self.work_dir, 'default', 'out.jsonl'),
        )
        self.assertFalse(exporter.encrypt_before_export)
        self.assertIsNone(exporter._fernet)

    def test_encrypt_flag_enabled_with_key_file(self):
        key_file = self._write_key_file()
        os.makedirs(os.path.join(self.work_dir, 'enabled'), exist_ok=True)
        exporter = Exporter(
            export_path=os.path.join(self.work_dir, 'enabled', 'out.jsonl'),
            encrypt_before_export=True,
            encryption_key_path=key_file,
        )
        self.assertTrue(exporter.encrypt_before_export)
        self.assertIsNotNone(exporter._fernet)

    def test_s3_path_disables_encryption_with_warning(self):
        """S3 export_path should disable local-file encryption with a warning."""
        from loguru import logger

        key_file = self._write_key_file()
        warning_messages = []
        handler_id = logger.add(
            lambda msg: warning_messages.append(str(msg)),
            level='WARNING',
            format='{message}',
        )
        try:
            exporter = Exporter(
                export_path='s3://bucket/prefix/out.jsonl',
                encrypt_before_export=True,
                encryption_key_path=key_file,
            )
        finally:
            logger.remove(handler_id)

        self.assertFalse(exporter.encrypt_before_export)
        self.assertTrue(
            len(warning_messages) > 0,
            'Expected a loguru WARNING about S3 path skipping encryption',
        )

    @patch("fsspec.implementations.arrow.ArrowFSWrapper")
    @patch("data_juicer.utils.hdfs_utils.create_pyarrow_hdfs_filesystem")
    def test_hdfs_path_disables_encryption_with_warning(self, mock_create_hdfs_fs, mock_arrow_wrapper):
        """HDFS export_path should disable local-file encryption with a warning."""
        from loguru import logger

        key_file = self._write_key_file()
        warning_messages = []
        handler_id = logger.add(
            lambda msg: warning_messages.append(str(msg)),
            level='WARNING',
            format='{message}',
        )
        try:
            exporter = Exporter(
                export_path='hdfs://namenode:8020/user/data/out.jsonl',
                encrypt_before_export=True,
                encryption_key_path=key_file,
            )
        finally:
            logger.remove(handler_id)

        self.assertFalse(exporter.encrypt_before_export)
        self.assertTrue(
            len(warning_messages) > 0,
            'Expected a loguru WARNING about HDFS path skipping encryption',
        )


    # ------------------------------------------------------------------
    # _encrypt_local_file helper
    # ------------------------------------------------------------------

    def test_encrypt_local_file_encrypts_in_place(self):
        key_file = self._write_key_file()
        os.makedirs(os.path.join(self.work_dir, 'inplace'), exist_ok=True)
        exporter = Exporter(
            export_path=os.path.join(self.work_dir, 'inplace', 'out.jsonl'),
            encrypt_before_export=True,
            encryption_key_path=key_file,
        )
        plain_path = os.path.join(self.work_dir, 'plain.txt')
        plaintext = b'plaintext content'
        with open(plain_path, 'wb') as f:
            f.write(plaintext)

        exporter._encrypt_local_file(plain_path)

        with open(plain_path, 'rb') as f:
            content = f.read()
        # File must have been overwritten with ciphertext
        self.assertNotEqual(content, plaintext)
        self.assertEqual(self.fernet.decrypt(content), plaintext)

    def test_encrypt_local_file_noop_when_disabled(self):
        os.makedirs(os.path.join(self.work_dir, 'noop'), exist_ok=True)
        exporter = Exporter(
            export_path=os.path.join(self.work_dir, 'noop', 'out.jsonl'),
            encrypt_before_export=False,
        )
        plain_path = os.path.join(self.work_dir, 'plain.txt')
        plaintext = b'untouched'
        with open(plain_path, 'wb') as f:
            f.write(plaintext)

        exporter._encrypt_local_file(plain_path)

        with open(plain_path, 'rb') as f:
            self.assertEqual(f.read(), plaintext)

    # ------------------------------------------------------------------
    # Full export round-trip
    # ------------------------------------------------------------------

    def test_export_single_file_is_encrypted(self):
        key_file = self._write_key_file()
        export_path = os.path.join(self.work_dir, 'enc', 'out.jsonl')
        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        exporter = Exporter(
            export_path=export_path,
            export_shard_size=0,
            export_in_parallel=False,
            num_proc=1,
            export_ds=True,
            keep_stats_in_res_ds=False,
            keep_hashes_in_res_ds=False,
            export_stats=False,
            encrypt_before_export=True,
            encryption_key_path=key_file,
        )
        exporter.export(self.test_data)

        self.assertTrue(os.path.exists(export_path))
        with open(export_path, 'rb') as f:
            raw = f.read()
        # Must not be plaintext JSON
        self.assertFalse(raw.lstrip().startswith(b'{'))
        # Must be decryptable
        decrypted = self.fernet.decrypt(raw)
        self.assertIn(b'hello', decrypted)

    def test_export_stats_file_is_encrypted(self):
        key_file = self._write_key_file()
        export_path = os.path.join(self.work_dir, 'enc_stats', 'out.jsonl')
        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        exporter = Exporter(
            export_path=export_path,
            export_shard_size=0,
            export_in_parallel=False,
            num_proc=1,
            export_ds=True,
            keep_stats_in_res_ds=False,
            keep_hashes_in_res_ds=False,
            export_stats=True,
            encrypt_before_export=True,
            encryption_key_path=key_file,
        )
        exporter.export(self.test_data)

        # stats file naming rule: replace ".jsonl" with "_stats.jsonl"
        stats_path = export_path.replace('.jsonl', '_stats.jsonl')
        self.assertTrue(os.path.exists(stats_path))
        with open(stats_path, 'rb') as f:
            raw = f.read()
        # Stats file must be encrypted
        self.assertFalse(raw.lstrip().startswith(b'{'))
        self.fernet.decrypt(raw)  # must not raise


class CoreExporterFileTest(DataJuicerTestCaseBase):
    def setUp(self):
        super().setUp()
        self.tmp_dir = tempfile.mkdtemp(prefix="dj_exporter_file_")
        self.Exporter = exporter_module.Exporter
        self.Fields = exporter_module.Fields
        self.HashKeys = exporter_module.HashKeys
        self.dataset = Dataset.from_list([
            {
                "text": "text 1",
                self.Fields.stats: {"score": 1},
                self.Fields.meta: {"source": "a"},
                self.HashKeys.hash: "h1",
            },
            {
                "text": "text 2",
                self.Fields.stats: {"score": 2},
                self.Fields.meta: {"source": "b"},
                self.HashKeys.hash: "h2",
            },
            {
                "text": "text 3",
                self.Fields.stats: {"score": 3},
                self.Fields.meta: {"source": "c"},
                self.HashKeys.hash: "h3",
            },
        ])

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        super().tearDown()

    def test_meta_stats_json_strings_are_restored_before_export(self):
        ds = Dataset.from_dict({
            self.Fields.meta: ['{"source": "zh"}', '{"source": "en"}'],
            self.Fields.stats: ['{"score": 1}', '{"score": 2}'],
        })

        fixed = self.Exporter._ensure_meta_stats_dicts_for_export(ds)

        self.assertEqual(fixed[0][self.Fields.meta], {"source": "zh"})
        self.assertEqual(fixed[0][self.Fields.stats], {"score": 1})
        self.assertEqual(fixed[1][self.Fields.meta], {"source": "en"})
        self.assertEqual(fixed[1][self.Fields.stats], {"score": 2})

    def test_invalid_meta_stats_json_strings_are_left_unchanged(self):
        ds = Dataset.from_dict({
            self.Fields.meta: ["not-json"],
            self.Fields.stats: ["also-not-json"],
        })

        fixed = self.Exporter._ensure_meta_stats_dicts_for_export(ds)

        self.assertEqual(fixed[0][self.Fields.meta], "not-json")
        self.assertEqual(fixed[0][self.Fields.stats], "also-not-json")

    def test_meta_stats_restore_is_noop_without_columns(self):
        ds = Dataset.from_list([{"text": "plain"}])

        self.assertIs(self.Exporter._ensure_meta_stats_dicts_for_export(ds), ds)

    def test_row_to_json_serializable_handles_scalars_lists_and_arrow_values(self):
        class ArrowLike:
            def as_py(self):
                return {"nested": np.int64(3)}

        class ListLike:
            def tolist(self):
                return [1, 2]

        row = {
            "scalar": np.int64(7),
            "array": ListLike(),
            "nested": [ArrowLike()],
        }

        self.assertEqual(
            self.Exporter._row_to_json_serializable(row),
            {"scalar": 7, "array": [1, 2], "nested": [{"nested": 3}]},
        )

    def test_json_jsonl_parquet_exports_and_filtered_shards(self):
        jsonl_path = os.path.join(self.tmp_dir, "out.jsonl")
        json_path = os.path.join(self.tmp_dir, "out.json")
        parquet_path = os.path.join(self.tmp_dir, "out.parquet")
        shard_path = os.path.join(self.tmp_dir, "shards", "out.jsonl")

        self.Exporter.to_jsonl(self.dataset, jsonl_path)
        self.Exporter.to_json(self.dataset, json_path, num_proc=1)
        self.Exporter.to_parquet(self.dataset, parquet_path)

        with open(jsonl_path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        self.assertEqual(rows[0]["text"], "text 1")
        self.assertTrue(os.path.exists(json_path))
        self.assertTrue(os.path.exists(parquet_path))

        filtered = self.dataset.filter(lambda row: row["text"] != "text 2")
        exporter = self.Exporter(
            export_path=shard_path,
            export_shard_size=1,
            export_in_parallel=False,
            num_proc=1,
            export_ds=True,
            keep_stats_in_res_ds=False,
            keep_hashes_in_res_ds=False,
            export_stats=False,
        )
        exporter.export(filtered)

        shard_files = os.listdir(os.path.dirname(shard_path))
        self.assertTrue(any(name.endswith(".jsonl") for name in shard_files))


if __name__ == '__main__':
    unittest.main()

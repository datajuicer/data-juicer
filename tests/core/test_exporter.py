from datetime import date, datetime, timezone

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
            "date": date(2024, 1, 2),
            "datetime": datetime(2024, 1, 2, 3, 4, 5),
            "datetime_tz": datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
        }

        self.assertEqual(
            self.Exporter._row_to_json_serializable(row),
            {
                "scalar": 7,
                "array": [1, 2],
                "nested": [{"nested": 3}],
                "date": "2024-01-02",
                "datetime": "2024-01-02T03:04:05",
                "datetime_tz": "2024-01-02T03:04:05+00:00",
            },
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


class ExporterNoSuffixTest(DataJuicerTestCaseBase):
    """Regression test: export_path without extension should give clear error."""

    def test_no_suffix_raises_valueerror(self):
        with self.assertRaises(ValueError) as ctx:
            Exporter(export_path='s3://bucket/data/result', num_proc=1)
        self.assertIn('no file extension', str(ctx.exception))

    def test_no_suffix_local_path_raises_valueerror(self):
        with self.assertRaises(ValueError) as ctx:
            Exporter(export_path='/tmp/output/dataset', num_proc=1)
        self.assertIn('no file extension', str(ctx.exception))

    def test_valid_suffix_still_works(self):
        exp = Exporter(export_path='/tmp/output/data.jsonl', num_proc=1)
        self.assertEqual(exp.suffix, 'jsonl')

    def test_explicit_export_type_bypasses_suffix(self):
        exp = Exporter(
            export_path='s3://bucket/data/result',
            export_type='jsonl',
            num_proc=1,
        )
        self.assertEqual(exp.suffix, 'jsonl')


class IcebergWriterFixture:
    """Existing tables in a SQLite catalog and a strictly local warehouse."""

    def setUp(self):
        super().setUp()
        import importlib.util
        from pathlib import Path

        if importlib.util.find_spec("pyiceberg") is None:
            self.skipTest("Install py-data-juicer[iceberg] for Iceberg writer tests")
        import pyarrow as pa
        from pyiceberg.catalog import load_catalog

        temp = tempfile.TemporaryDirectory(prefix="dj_iceberg_write_")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.catalog_kwargs = {
            "name": "local_writer_test",
            "type": "sql",
            "uri": f"sqlite:///{self.root / 'catalog.db'}",
            "warehouse": self.root.as_uri(),
            "py-io-impl": "pyiceberg.io.pyarrow.PyArrowFileIO",
        }
        self.catalog = load_catalog(**self.catalog_kwargs)
        self.catalog.create_namespace("training")
        self.identifier = "training.documents"
        self.schema = pa.schema([("id", pa.int64()), ("text", pa.string())])
        self.table = self.catalog.create_table(self.identifier, schema=self.schema)
        self.initial = [{"id": 0, "text": "original"}]
        self.rows = [{"id": 1, "text": "first"}, {"id": 2, "text": "second"}]
        self.table.append(pa.Table.from_pylist(self.initial, schema=self.schema))
        self.anchor = str(self.root / "artifacts" / "output.jsonl")

    def options(self, **kwargs):
        return dict(export_path=self.anchor, export_type="iceberg", table_identifier=self.identifier,
                    catalog_kwargs=self.catalog_kwargs, **kwargs)

    def read_rows(self):
        self.table.refresh()
        return self.table.scan().to_arrow().to_pylist()

    def snapshot(self):
        self.table.refresh()
        return self.table.current_snapshot().snapshot_id

    def rich_arrow(self):
        from decimal import Decimal
        import pyarrow as pa

        schema = pa.schema([
            ("id", pa.int64()), ("price", pa.decimal128(10, 2)), ("blob", pa.binary()),
            ("created", pa.timestamp("us")), ("tags", pa.list_(pa.string())),
            ("details", pa.struct([("source", pa.string()), ("page", pa.int32())])),
            ("note", pa.string()),
        ])
        rows = [
            {"id": 1, "price": Decimal("12.34"), "blob": b"\x00\xffbin",
             "created": datetime(2026, 1, 2, 3, 4, 5, 678901), "tags": ["a", None],
             "details": {"source": "web", "page": 7}, "note": None},
            {"id": 2, "price": None, "blob": None, "created": None, "tags": [],
             "details": None, "note": "present"},
        ]
        self.catalog.create_table("training.typed", schema=schema)
        return pa.Table.from_pylist(rows, schema=schema)


class IcebergExporterTest(IcebergWriterFixture, unittest.TestCase):
    def test_append_overwrite_repeated_and_empty(self):
        import pyarrow as pa

        ds = Dataset(pa.Table.from_pylist(self.rows, schema=self.schema))
        empty = ds.select([])
        for mode in ("append", "overwrite"):
            with self.subTest(mode=mode):
                self.table.overwrite(pa.Table.from_pylist(self.initial, schema=self.schema))
                writer = Exporter(**self.options(mode=mode))
                writer.export(ds)
                writer.export(ds)
                expected = self.initial + self.rows * 2 if mode == "append" else self.rows
                self.assertCountEqual(self.read_rows(), expected)
                snapshot = self.snapshot()
                writer.export(empty)
                if mode == "overwrite":
                    self.assertEqual(self.read_rows(), [])
                    self.assertNotEqual(self.snapshot(), snapshot)
                    self.assertEqual(self.table.current_snapshot().parent_snapshot_id, snapshot)
                if mode == "append":
                    self.assertCountEqual(self.read_rows(), expected)
                    self.assertEqual(self.snapshot(), snapshot)
        self.assertFalse(os.path.exists(self.anchor))

    def test_default_overwrite_and_logical_indices(self):
        from data_juicer.core.data import NestedDataset

        rows = [{"id": i, "text": str(i)} for i in range(6)]
        selected = Dataset.from_list(rows).select([5, 1, 4, 0, 3])
        filtered = selected.filter(lambda row: row["id"] != 4)
        reordered = filtered.select([2, 0, 3, 1])
        writer = Exporter(**self.options())
        self.assertEqual(writer.iceberg_config["mode"], "overwrite")
        for ds in (selected, filtered, reordered, NestedDataset(reordered)):
            with self.subTest(dataset=type(ds).__name__, rows=len(ds)):
                self.assertIsNotNone(ds._indices)
                writer.export(ds)
                ids = Dataset.__getitem__(ds, slice(None))["id"]
                self.assertEqual(self.read_rows(), [{"id": i, "text": str(i)} for i in ids])

    def test_custom_transform_is_not_exported_and_format_is_preserved(self):
        from copy import deepcopy
        from data_juicer.core.data import NestedDataset

        rows = [{"id": i, "text": str(i)} for i in range(5)]
        calls = []

        def transform(batch):
            calls.append(True)
            return {"id": [i + 100 for i in batch["id"]], "text": ["changed"] * len(batch["id"])}

        base = Dataset.from_list(rows)
        base.set_transform(transform)
        for ds, indices in ((base, list(range(5))), (base.select([4, 1, 3]).select([2, 0, 1]), [3, 4, 1]),
                            (NestedDataset(base.select([2, 0])), [2, 0])):
            with self.subTest(dataset=type(ds).__name__, indices=indices):
                original_format = deepcopy(ds.format)
                calls.clear()
                Exporter(**self.options(export_stats=False)).export(ds)
                self.assertEqual(self.read_rows(), [rows[i] for i in indices])
                self.assertEqual(calls, [])
                self.assertEqual(ds.format, original_format)
                self.assertEqual(Dataset.__getitem__(ds, 0)["id"], rows[indices[0]]["id"] + 100)

    def test_stats_hash_columns_and_independent_artifacts(self):
        ds = Dataset.from_list([
            dict(row, **{Fields.stats: {"score": i}, Fields.meta: {"source": "local"},
                         HashKeys.hash: f"h{i}"}) for i, row in enumerate(self.rows)
        ])
        for keep_stats in (False, True):
            for keep_hashes in (False, True):
                with self.subTest(stats=keep_stats, hashes=keep_hashes):
                    columns = ["id", "text"]
                    if keep_stats:
                        columns += [Fields.stats, Fields.meta]
                    if keep_hashes:
                        columns += [HashKeys.hash]
                    identifier = f"training.kept_{int(keep_stats)}_{int(keep_hashes)}"
                    table = self.catalog.create_table(identifier, schema=ds.select_columns(columns).data.table.schema)
                    writer = Exporter(**dict(self.options(), table_identifier=identifier),
                                      keep_stats_in_res_ds=keep_stats, keep_hashes_in_res_ds=keep_hashes)
                    writer.export(ds)
                    self.assertEqual(table.refresh().scan().to_arrow().to_pylist(),
                                     ds.select_columns(columns).data.table.to_pylist())
                    stats_path = str(self.root / "artifacts" / "output_stats.jsonl")
                    with open(stats_path, encoding="utf-8") as f:
                        self.assertEqual([json.loads(line) for line in f],
                                         ds.select_columns([Fields.stats, Fields.meta]).data.table.to_pylist())
                    snapshot = table.current_snapshot().snapshot_id
                    checkpoint = str(self.root / "artifacts" / "compute.jsonl")
                    writer.export_compute_stats(ds, checkpoint)
                    checkpoint_ds = ds if keep_hashes else ds.remove_columns(HashKeys.hash)
                    with open(checkpoint, encoding="utf-8") as f:
                        self.assertEqual([json.loads(line) for line in f], checkpoint_ds.data.table.to_pylist())
                    self.assertEqual(writer.keep_stats_in_res_ds, keep_stats)
                    self.assertEqual(table.refresh().current_snapshot().snapshot_id, snapshot)
        self.assertFalse(os.path.exists(self.anchor))

    def test_rich_types_arrow_roundtrip(self):
        arrow = self.rich_arrow()
        writer = Exporter(**dict(self.options(), table_identifier="training.typed"))
        writer.export(Dataset(arrow))
        result = self.catalog.load_table("training.typed").scan().to_arrow()
        self.assertEqual(result.to_pylist(), arrow.to_pylist())

    def test_read_then_write_existing_table(self):
        from jsonargparse import Namespace
        from data_juicer.core.data.load_strategy import DefaultIcebergDataLoadStrategy

        config = {"table_identifier": self.identifier, "catalog_kwargs": self.catalog_kwargs}
        reader = DefaultIcebergDataLoadStrategy(config, Namespace(text_keys=["text"], text_key="text", np=1))
        writer = Exporter(**self.options())
        for _ in range(2):
            writer.export(reader.load_data(num_proc=1))
            self.assertEqual(self.read_rows(), self.initial)

    def test_artifact_anchor_without_extension_or_with_iceberg_suffix(self):
        ds = Dataset.from_list([dict(self.rows[0], **{Fields.stats: {"score": 1}})])
        before = self.snapshot()
        for name in ("result", "result.iceberg"):
            anchor = str(self.root / "artifacts" / name)
            options = dict(self.options(), export_path=anchor, export_ds=False)
            if name.endswith(".iceberg"):
                options["export_type"] = None
            writer = Exporter(**options)
            writer.export(ds)
            self.assertFalse(os.path.exists(anchor))
            stats = os.path.splitext(anchor)[0] + "_stats.jsonl"
            with open(stats, encoding="utf-8") as f:
                self.assertEqual([json.loads(line) for line in f], [{Fields.stats: {"score": 1}}])
            self.assertEqual(self.snapshot(), before)

    def test_required_columns_accept_nullable_arrow_without_nulls(self):
        import pyarrow as pa

        schema = pa.schema([pa.field("id", pa.int64(), nullable=False), ("text", pa.string())])
        table = self.catalog.create_table("training.required", schema=schema)
        writer = Exporter(**dict(self.options(), table_identifier="training.required"))
        ds = Dataset.from_list(self.rows)
        self.assertTrue(ds.data.table.schema.field("id").nullable)
        writer.export(ds.filter(lambda row: row["id"] > 0))
        self.assertEqual(table.refresh().scan().to_arrow().to_pylist(), self.rows)
        snapshot = table.current_snapshot().snapshot_id
        with self.assertRaisesRegex(ValueError, "required column id contains null"):
            writer.export(Dataset.from_list([{"id": None, "text": "missing"}]))
        self.assertEqual(table.refresh().current_snapshot().snapshot_id, snapshot)

    def test_schema_error_does_not_change_snapshot_or_metadata(self):
        before = self.snapshot()
        location = self.table.metadata_location
        writer = Exporter(**self.options())
        for ds in (Dataset.from_list([{"id": "wrong", "text": "x"}]),
                   Dataset.from_list([{"id": 1, "text": "x", "extra": 2}])):
            with self.subTest(schema=ds.features), self.assertRaises(ValueError):
                writer.export(ds)
            self.assertEqual(self.snapshot(), before)
            self.assertEqual(self.table.metadata_location, location)
            self.assertEqual(self.read_rows(), self.initial)

    def test_missing_table_even_empty_no_fallback(self):
        import pyarrow as pa

        for mode in ("append", "overwrite"):
            writer = Exporter(**dict(self.options(mode=mode), table_identifier="training.missing"))
            for ds in (Dataset.from_list(self.rows), Dataset(pa.Table.from_batches([], schema=self.schema))):
                with self.assertRaisesRegex(ValueError, "target table does not exist"):
                    writer.export(ds)
        self.assertFalse(os.path.exists(self.anchor))

    def test_configuration_rejected_before_files_or_credentials(self):
        invalid = [
            {"mode": "upsert"}, {"mode": None}, {"table_identifier": " "}, {"catalog_kwargs": None},
            {"catalog_kwargs": {"name": None}}, {"catalog_kwargs": {}},
            {"catalog_kwargs": {k: v for k, v in self.catalog_kwargs.items() if k != "name"}},
            {"export_path": None}, {"export_path": ""}, {"export_path": "file:///tmp/output"},
            {"export_path": "s3://bucket/output"}, {"export_path": "hdfs://host/output"},
            {"export_path": "https://host/output"}, {"export_shard_size": 1},
            {"encrypt_before_export": True, "encryption_key_path": "/must/not/read"},
            {"overwrite_filter": "id > 0"}, {"overwrite_kwargs": {"branch": "other"}},
        ]
        with patch("data_juicer.utils.encryption_utils.load_fernet_key") as key_loader, \
                patch("data_juicer.utils.s3_utils.get_aws_credentials") as credentials:
            for changes in invalid:
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    Exporter(**dict(self.options(), **changes))
            key_loader.assert_not_called()
            credentials.assert_not_called()
        options = self.options()
        writer = Exporter(**options)
        self.catalog_kwargs["warehouse"] = "changed-after-construction"
        self.assertNotEqual(writer.iceberg_config["catalog_kwargs"]["warehouse"], self.catalog_kwargs["warehouse"])


class IcebergArtifactTest(unittest.TestCase):
    def test_stats_only_needs_no_table_or_catalog(self):
        with tempfile.TemporaryDirectory() as root:
            anchor = os.path.join(root, "output.jsonl")
            ds = Dataset.from_list([{"text": "original", Fields.stats: {"score": 1}}])
            with patch("data_juicer.utils.iceberg_utils.iceberg_export_config") as config, \
                    patch("data_juicer.utils.iceberg_utils.load_iceberg_table") as load_table:
                writer = Exporter(anchor, export_type="iceberg", export_ds=False)
                writer.export(ds)
                writer.export_compute_stats(ds, os.path.join(root, "compute.jsonl"))
                self.assertIsNone(writer.iceberg_config)
                config.assert_not_called()
                load_table.assert_not_called()
            with open(os.path.join(root, "output_stats.jsonl"), encoding="utf-8") as f:
                self.assertEqual([json.loads(line) for line in f], [{Fields.stats: {"score": 1}}])
            self.assertFalse(os.path.exists(anchor))

    def test_analyzer_initializes_without_write_options(self):
        from jsonargparse import Namespace
        from data_juicer.core.analyzer import Analyzer

        cfg = Namespace(work_dir="tmp", use_cache=False, export_path="tmp/output.jsonl",
                        export_type="iceberg", export_shard_size=0, export_in_parallel=False,
                        np=1, export_original_dataset=False)
        with patch("data_juicer.core.analyzer.DatasetBuilder"):
            analyzer = Analyzer(cfg)
        self.assertFalse(analyzer.exporter.export_ds)
        self.assertIsNone(analyzer.exporter.iceberg_config)

    def test_local_windows_artifact_and_unsupported_stat_only_options(self):
        writer = Exporter("C:/tmp/result.jsonl", export_type="iceberg", export_ds=False)
        self.assertEqual(writer.export_path, "C:/tmp/result.jsonl")
        for changes in ({"export_path": "file:///tmp/result.jsonl"}, {"export_path": "s3://bucket/result.jsonl"},
                        {"export_shard_size": 1}, {"encrypt_before_export": True}):
            options = dict(export_path="tmp/result.jsonl", export_type="iceberg", export_ds=False)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                Exporter(**dict(options, **changes))

    def test_load_table_requires_explicit_catalog_name(self):
        from data_juicer.utils.iceberg_utils import load_iceberg_table

        for properties in ({}, {"type": "sql"}, {"name": " "}, {"name": None}):
            with self.subTest(properties=properties), self.assertRaisesRegex(ValueError, "catalog name"):
                load_iceberg_table("training.documents", properties)


if __name__ == '__main__':
    unittest.main()

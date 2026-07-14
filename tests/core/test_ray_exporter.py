import copy
import os
import os.path as osp
import shutil
import sys
import types
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from data_juicer.utils.unittest_utils import TEST_TAG, DataJuicerTestCaseBase
from data_juicer.core.ray_exporter import RayExporter
from data_juicer.utils.constant import Fields, HashKeys
from data_juicer.utils.mm_utils import load_images_byte


class TestRayExporter(DataJuicerTestCaseBase):

    def setUp(self):
        """Set up test data"""
        super().setUp()

        import ray
        from data_juicer.core.data.ray_dataset import RayDataset

        cur_dir = osp.dirname(osp.abspath(__file__))
        self.tmp_dir = f"{cur_dir}/tmp/{self.__class__.__name__}/{self._testMethodName}"
        os.makedirs(self.tmp_dir, exist_ok=True)

        self.data = [
            {"text": "hello", Fields.stats: {"score": 1}, HashKeys.hash: "a1"},
            {"text": "world", Fields.stats: {"score": 2}, HashKeys.hash: "b2"},
            {"text": "test", Fields.stats: {"score": 3}, HashKeys.hash: "c3"},
        ]
        self.dataset = RayDataset(ray.data.from_items(self.data))

    def tearDown(self):
        """Clean up temporary outputs"""

        self.dataset = None
        if osp.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

        super().tearDown()

    def _pop_raw_data_keys(self, keys):
        res = copy.deepcopy(self.data)
        for d_i in res:
            for k in keys:
                d_i.pop(k, None)

        return res

    @TEST_TAG("ray")
    def test_json_not_keep_stats_and_hashes(self):
        import ray

        out_path = osp.join(self.tmp_dir, "outdata.json")
        ray_exporter = RayExporter(out_path, keep_stats_in_res_ds=False, keep_hashes_in_res_ds=False)
        ray_exporter.export(self.dataset.data)

        ds = ray.data.read_json(out_path)
        data_list = ds.take_all()

        self.assertListOfDictEqual(data_list, self._pop_raw_data_keys([Fields.stats, HashKeys.hash]))

    @TEST_TAG("ray")
    def test_jsonl_keep_stats_and_hashes(self):
        import ray

        out_path = osp.join(self.tmp_dir, "outdata.jsonl")
        ray_exporter = RayExporter(out_path, keep_stats_in_res_ds=True, keep_hashes_in_res_ds=True)
        ray_exporter.export(self.dataset.data)

        ds = ray.data.read_json(out_path)
        data_list = ds.take_all()

        self.assertListOfDictEqual(data_list, self.data)

    @TEST_TAG("ray")
    def test_parquet_keep_stats(self):
        import ray

        out_path = osp.join(self.tmp_dir, "outdata.parquet")
        ray_exporter = RayExporter(out_path, keep_stats_in_res_ds=True, keep_hashes_in_res_ds=False)
        ray_exporter.export(self.dataset.data)

        ds = ray.data.read_parquet(out_path)
        data_list = ds.take_all()

        self.assertListEqual(data_list, self._pop_raw_data_keys([HashKeys.hash]))

    @TEST_TAG("ray")
    def test_lance_keep_hashes(self):
        import ray

        out_path = osp.join(self.tmp_dir, "outdata.lance")
        ray_exporter = RayExporter(out_path, keep_stats_in_res_ds=False, keep_hashes_in_res_ds=True)
        ray_exporter.export(self.dataset.data)

        ds = ray.data.read_lance(out_path)
        data_list = ds.take_all()

        self.assertListOfDictEqual(data_list, self._pop_raw_data_keys([Fields.stats]))

    @TEST_TAG("ray")
    def test_webdataset_multi_images(self):
        import io
        from PIL import Image
        import ray
        from data_juicer.core.data.ray_dataset import RayDataset

        data_dir = osp.abspath(osp.join(osp.dirname(osp.realpath(__file__)), "..", "ops", "data"))
        img1_path = osp.join(data_dir, "img1.png")
        img2_path = osp.join(data_dir, "img2.jpg")
        img3_path = osp.join(data_dir, "img3.jpg")

        data = [
            {
                "json": {"text": "hello", "images": [img1_path, img2_path]},
                "jpgs": load_images_byte([img1_path, img2_path]),
            },
            {
                "json": {"text": "world", "images": [img2_path, img3_path]},
                "jpgs": load_images_byte([img2_path, img3_path]),
            },
            {
                "json": {"text": "test", "images": [img1_path, img2_path, img3_path]},
                "jpgs": load_images_byte([img1_path, img2_path, img3_path]),
            },
        ]
        dataset = RayDataset(ray.data.from_items(data))
        out_path = osp.join(self.tmp_dir, "outdata.webdataset")
        ray_exporter = RayExporter(out_path)
        ray_exporter.export(dataset.data)

        ds = RayDataset.read_webdataset(out_path)
        res_list = ds.take_all()

        self.assertEqual(len(res_list), len(data))
        res_list.sort(key=lambda x: x["json"]["text"])
        data.sort(key=lambda x: x["json"]["text"])

        for i in range(len(data)):
            self.assertDictEqual(res_list[i]["json"], data[i]["json"])
            self.assertEqual(res_list[i]["jpgs"], [Image.open(io.BytesIO(v)) for v in data[i]["jpgs"]])

    @TEST_TAG("ray")
    def test_webdataset_multi_videos_frames_bytes(self):
        import io
        from PIL import Image
        import ray
        from data_juicer.core.data.ray_dataset import RayDataset

        data_dir = osp.abspath(osp.join(osp.dirname(osp.realpath(__file__)), "..", "ops", "data"))
        img1_path = osp.join(data_dir, "img1.png")
        img2_path = osp.join(data_dir, "img2.jpg")
        img3_path = osp.join(data_dir, "img3.jpg")

        data = [
            {
                "json": {"text": "hello", "videos": ["video1.mp4", "video2.mp4"]},
                "mp4s": [
                    load_images_byte([img1_path]),  # as video1 frames bytes
                    load_images_byte([img1_path, img2_path]),  # as video2 frames path
                ],
            },
            {
                "json": {"text": "world", "videos": ["video1.mp4"]},
                "mp4s": [load_images_byte([img2_path, img3_path])],  # as video1 frames
            },
        ]
        dataset = RayDataset(ray.data.from_items(data))
        out_path = osp.join(self.tmp_dir, "outdata.webdataset")
        ray_exporter = RayExporter(out_path, export_type="webdataset")
        ray_exporter.export(dataset.data)

        ds = RayDataset.read_webdataset(out_path)
        res_list = ds.take_all()

        self.assertEqual(len(res_list), len(data))
        res_list.sort(key=lambda x: x["json"]["text"])
        data.sort(key=lambda x: x["json"]["text"])

        for i in range(len(data)):
            if len(data[i]["mp4s"]) > 1:
                tgt_mp4s = [[Image.open(io.BytesIO(f_i)) for f_i in v_i] for v_i in data[i]["mp4s"]]
            else:
                tgt_mp4s = [Image.open(io.BytesIO(f_i)) for f_i in data[i]["mp4s"][0]]
            self.assertDictEqual(res_list[i]["json"], data[i]["json"])
            self.assertEqual(res_list[i]["mp4s"], tgt_mp4s)

    @TEST_TAG("ray")
    def test_webdataset_multi_videos_frames_path(self):
        import io
        from PIL import Image
        import ray
        from data_juicer.core.data.ray_dataset import RayDataset

        data_dir = osp.abspath(osp.join(osp.dirname(osp.realpath(__file__)), "..", "ops", "data"))
        img1_path = osp.join(data_dir, "img8.jpg")
        img2_path = osp.join(data_dir, "img9.jpg")
        img3_path = osp.join(data_dir, "img10.jpg")

        data = [
            {
                "json": {"text": "hello", "videos": ["video1.mp4", "video2.mp4"]},
                "mp4s": [[img1_path], [img1_path, img2_path]],  # as video1 frames path  # as video2 frames path
            },
            {
                "json": {"text": "world", "videos": ["video1.mp4"]},
                "mp4s": [[img2_path, img3_path]],  # as video1 frames path
            },
        ]
        dataset = RayDataset(ray.data.from_items(data))
        out_path = osp.join(self.tmp_dir, "outdata.webdataset")
        ray_exporter = RayExporter(out_path, export_type="webdataset")
        ray_exporter.export(dataset.data)

        ds = RayDataset.read_webdataset(out_path)
        res_list = ds.take_all()

        self.assertEqual(len(res_list), len(data))
        res_list.sort(key=lambda x: x["json"]["text"])
        data.sort(key=lambda x: x["json"]["text"])

        for i in range(len(data)):
            if len(data[i]["mp4s"]) > 1:
                tgt_mp4s = [[Image.open(f_i, formats=["jpeg"]) for f_i in v_i] for v_i in data[i]["mp4s"]]
            else:
                tgt_mp4s = [Image.open(f_i, formats=["jpeg"]) for f_i in data[i]["mp4s"][0]]
            self.assertDictEqual(res_list[i]["json"], data[i]["json"])
            self.assertEqual(res_list[i]["mp4s"], tgt_mp4s)

    @TEST_TAG("ray")
    def test_webdataset_multi_audios_path(self):
        import ray
        from data_juicer.core.data.ray_dataset import RayDataset
        from data_juicer.utils.mm_utils import load_audio

        data_dir = osp.abspath(osp.join(osp.dirname(osp.realpath(__file__)), "..", "ops", "data"))
        audio1_path = osp.join(data_dir, "audio1.wav")
        audio2_path = osp.join(data_dir, "audio2.wav")
        audio3_path = osp.join(data_dir, "audio3.ogg")

        data = [
            {
                "json": {
                    "text": "hello",
                },
                "mp3s": [audio1_path],
            },
            {
                "json": {
                    "text": "world",
                },
                "mp3s": [audio2_path, audio3_path],
            },
        ]
        dataset = RayDataset(ray.data.from_items(data))
        out_path = osp.join(self.tmp_dir, "outdata.webdataset")
        ray_exporter = RayExporter(out_path, export_type="webdataset")
        ray_exporter.export(dataset.data)

        ds = RayDataset.read_webdataset(out_path)
        res_list = ds.take_all()

        self.assertEqual(len(res_list), len(data))

        res_list.sort(key=lambda x: x["json"]["text"])
        data.sort(key=lambda x: x["json"]["text"])

        for i in range(len(data)):
            if len(data[i]["mp3s"]) <= 1:
                mp3s_list = [res_list[i]["mp3s"]]
            else:
                mp3s_list = res_list[i]["mp3s"]

            tgt_mp3s = [load_audio(f_i) for f_i in data[i]["mp3s"]]

            self.assertDictEqual(res_list[i]["json"], data[i]["json"])

            for j in range(len(mp3s_list)):
                arr, sampling_rate = mp3s_list[j]
                tgt_arr, tgt_sampling_rate = tgt_mp3s[j]
                import numpy as np

                np.testing.assert_array_equal(arr, tgt_arr)
                self.assertEqual(sampling_rate, tgt_sampling_rate)


class RayExporterEncryptTest(DataJuicerTestCaseBase):
    """Unit tests for RayExporter encryption features (no Ray dependency)."""

    def setUp(self):
        super().setUp()
        self.tmp_dir = tempfile.mkdtemp()
        # Generate a fresh Fernet key for every test
        from cryptography.fernet import Fernet

        self._key = Fernet.generate_key()
        self._key_file = os.path.join(self.tmp_dir, "test.key")
        with open(self._key_file, "wb") as f:
            f.write(self._key)

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # __init__ – encryption flag and Fernet setup
    # ------------------------------------------------------------------

    def test_encrypt_disabled_by_default(self):
        """encrypt_before_export defaults to False; _fernet stays None."""
        exporter = RayExporter(export_path=os.path.join(self.tmp_dir, "out.jsonl"))
        self.assertFalse(exporter.encrypt_before_export)
        self.assertIsNone(exporter._fernet)

    def test_encrypt_enabled_loads_fernet_key(self):
        """encrypt_before_export=True with a valid key file sets _fernet."""
        exporter = RayExporter(
            export_path=os.path.join(self.tmp_dir, "out.jsonl"),
            encrypt_before_export=True,
            encryption_key_path=self._key_file,
        )
        self.assertTrue(exporter.encrypt_before_export)
        self.assertIsNotNone(exporter._fernet)

    def test_s3_path_disables_encryption_with_warning(self):
        """S3 export path silently disables local-file encryption."""
        warnings_seen = []
        from loguru import logger as _logger

        handler_id = _logger.add(
            lambda msg: warnings_seen.append(msg),
            level="WARNING",
            format="{message}",
        )
        try:
            exporter = RayExporter(
                export_path="s3://some-bucket/out.jsonl",
                encrypt_before_export=True,
                encryption_key_path=self._key_file,
            )
        finally:
            _logger.remove(handler_id)
        self.assertFalse(exporter.encrypt_before_export)
        self.assertTrue(
            any("encrypt_before_export" in str(w) and "S3" in str(w) for w in warnings_seen),
            "Expected warning about S3 + encrypt_before_export not found",
        )

    # ------------------------------------------------------------------
    # _export_impl – encrypt_file called for each file in the output dir
    # ------------------------------------------------------------------

    def test_export_impl_encrypts_output_files(self):
        """After _export_impl, all files inside the output dir are encrypted."""
        from cryptography.fernet import Fernet
        from data_juicer.utils.encryption_utils import decrypt_file_to_bytes

        export_dir = os.path.join(self.tmp_dir, "export_out")
        exporter = RayExporter(
            export_path=export_dir,
            export_type="jsonl",
            encrypt_before_export=True,
            encryption_key_path=self._key_file,
        )

        # Write two plain-text sentinel files to simulate Ray output
        os.makedirs(export_dir, exist_ok=True)
        sentinel = b'{"text": "hello"}\n'
        for fname in ("part-0.jsonl", "part-1.jsonl"):
            fpath = os.path.join(export_dir, fname)
            with open(fpath, "wb") as f:
                f.write(sentinel)

        # Mock the underlying Ray write call so _export_impl only runs
        # the post-write encryption step
        mock_dataset = MagicMock()
        mock_dataset.columns.return_value = ["text"]
        mock_dataset.count.return_value = 2
        mock_dataset.size_bytes.return_value = 100
        mock_dataset.drop_columns.return_value = mock_dataset

        with patch.object(RayExporter, "_router") as mock_router:
            mock_write = MagicMock(return_value=None)
            mock_router.return_value = {"jsonl": mock_write}
            exporter._export_impl(mock_dataset, export_dir)

        # Verify every file is now encrypted (not plain text any more)
        fernet = Fernet(self._key)
        for fname in ("part-0.jsonl", "part-1.jsonl"):
            fpath = os.path.join(export_dir, fname)
            raw = open(fpath, "rb").read()
            self.assertNotEqual(raw, sentinel, f"{fname} should have been encrypted")
            # Must be decryptable with the same key
            decrypted = decrypt_file_to_bytes(fpath, fernet)
            self.assertEqual(decrypted, sentinel)

    def test_export_impl_skips_encryption_when_disabled(self):
        """When encrypt_before_export=False, files are left as-is."""
        export_dir = os.path.join(self.tmp_dir, "plain_out")
        exporter = RayExporter(
            export_path=export_dir,
            export_type="jsonl",
            encrypt_before_export=False,
        )

        os.makedirs(export_dir, exist_ok=True)
        sentinel = b'{"text": "world"}\n'
        fpath = os.path.join(export_dir, "part-0.jsonl")
        with open(fpath, "wb") as f:
            f.write(sentinel)

        mock_dataset = MagicMock()
        mock_dataset.columns.return_value = ["text"]
        mock_dataset.count.return_value = 1
        mock_dataset.size_bytes.return_value = 50
        mock_dataset.drop_columns.return_value = mock_dataset

        with patch.object(RayExporter, "_router") as mock_router:
            mock_write = MagicMock(return_value=None)
            mock_router.return_value = {"jsonl": mock_write}
            exporter._export_impl(mock_dataset, export_dir)

        self.assertEqual(open(fpath, "rb").read(), sentinel, "File should not be modified when encryption is off")


class RayExporterEncryptTest(DataJuicerTestCaseBase):
    """Unit tests for RayExporter encryption features (no Ray dependency)."""

    def setUp(self):
        super().setUp()
        self.tmp_dir = tempfile.mkdtemp()
        # Generate a fresh Fernet key for every test
        from cryptography.fernet import Fernet

        self._key = Fernet.generate_key()
        self._key_file = os.path.join(self.tmp_dir, "test.key")
        with open(self._key_file, "wb") as f:
            f.write(self._key)

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # __init__ – encryption flag and Fernet setup
    # ------------------------------------------------------------------

    def test_encrypt_disabled_by_default(self):
        """encrypt_before_export defaults to False; _fernet stays None."""
        exporter = RayExporter(export_path=os.path.join(self.tmp_dir, "out.jsonl"))
        self.assertFalse(exporter.encrypt_before_export)
        self.assertIsNone(exporter._fernet)

    def test_encrypt_enabled_loads_fernet_key(self):
        """encrypt_before_export=True with a valid key file sets _fernet."""
        exporter = RayExporter(
            export_path=os.path.join(self.tmp_dir, "out.jsonl"),
            encrypt_before_export=True,
            encryption_key_path=self._key_file,
        )
        self.assertTrue(exporter.encrypt_before_export)
        self.assertIsNotNone(exporter._fernet)

    def test_s3_path_disables_encryption_with_warning(self):
        """S3 export path silently disables local-file encryption."""
        warnings_seen = []
        from loguru import logger as _logger

        handler_id = _logger.add(
            lambda msg: warnings_seen.append(msg),
            level="WARNING",
            format="{message}",
        )
        try:
            exporter = RayExporter(
                export_path="s3://some-bucket/out.jsonl",
                encrypt_before_export=True,
                encryption_key_path=self._key_file,
            )
        finally:
            _logger.remove(handler_id)
        self.assertFalse(exporter.encrypt_before_export)
        self.assertTrue(
            any("encrypt_before_export" in str(w) and "S3" in str(w) for w in warnings_seen),
            "Expected warning about S3 + encrypt_before_export not found",
        )

    # ------------------------------------------------------------------
    # _export_impl – encrypt_file called for each file in the output dir
    # ------------------------------------------------------------------

    def test_export_impl_encrypts_output_files(self):
        """After _export_impl, all files inside the output dir are encrypted."""
        from cryptography.fernet import Fernet
        from data_juicer.utils.encryption_utils import decrypt_file_to_bytes

        export_dir = os.path.join(self.tmp_dir, "export_out")
        exporter = RayExporter(
            export_path=export_dir,
            export_type="jsonl",
            encrypt_before_export=True,
            encryption_key_path=self._key_file,
        )

        # Write two plain-text sentinel files to simulate Ray output
        os.makedirs(export_dir, exist_ok=True)
        sentinel = b'{"text": "hello"}\n'
        for fname in ("part-0.jsonl", "part-1.jsonl"):
            fpath = os.path.join(export_dir, fname)
            with open(fpath, "wb") as f:
                f.write(sentinel)

        # Mock the underlying Ray write call so _export_impl only runs
        # the post-write encryption step
        mock_dataset = MagicMock()
        mock_dataset.columns.return_value = ["text"]
        mock_dataset.count.return_value = 2
        mock_dataset.size_bytes.return_value = 100
        mock_dataset.drop_columns.return_value = mock_dataset

        with patch.object(RayExporter, "_router") as mock_router:
            mock_write = MagicMock(return_value=None)
            mock_router.return_value = {"jsonl": mock_write}
            exporter._export_impl(mock_dataset, export_dir)

        # Verify every file is now encrypted (not plain text any more)
        fernet = Fernet(self._key)
        for fname in ("part-0.jsonl", "part-1.jsonl"):
            fpath = os.path.join(export_dir, fname)
            raw = open(fpath, "rb").read()
            self.assertNotEqual(raw, sentinel, f"{fname} should have been encrypted")
            # Must be decryptable with the same key
            decrypted = decrypt_file_to_bytes(fpath, fernet)
            self.assertEqual(decrypted, sentinel)

    def test_export_impl_skips_encryption_when_disabled(self):
        """When encrypt_before_export=False, files are left as-is."""
        export_dir = os.path.join(self.tmp_dir, "plain_out")
        exporter = RayExporter(
            export_path=export_dir,
            export_type="jsonl",
            encrypt_before_export=False,
        )

        os.makedirs(export_dir, exist_ok=True)
        sentinel = b'{"text": "world"}\n'
        fpath = os.path.join(export_dir, "part-0.jsonl")
        with open(fpath, "wb") as f:
            f.write(sentinel)

        mock_dataset = MagicMock()
        mock_dataset.columns.return_value = ["text"]
        mock_dataset.count.return_value = 1
        mock_dataset.size_bytes.return_value = 50
        mock_dataset.drop_columns.return_value = mock_dataset

        with patch.object(RayExporter, "_router") as mock_router:
            mock_write = MagicMock(return_value=None)
            mock_router.return_value = {"jsonl": mock_write}
            exporter._export_impl(mock_dataset, export_dir)

        self.assertEqual(open(fpath, "rb").read(), sentinel, "File should not be modified when encryption is off")


class TestRayExporterTableFormats(DataJuicerTestCaseBase):
    def _build_fake_paimon_modules(self, catalog_factory_create, schema_cls=None, table_not_exist_error=None):
        fake_pypaimon = types.ModuleType("pypaimon")
        fake_pypaimon.__path__ = []
        fake_pypaimon.Schema = schema_cls or MagicMock(name="Schema")

        fake_pypaimon_catalog = types.ModuleType("pypaimon.catalog")
        fake_pypaimon_catalog.__path__ = []
        fake_pypaimon_catalog_exception = types.ModuleType("pypaimon.catalog.catalog_exception")
        fake_pypaimon_catalog_exception.TableNotExistException = table_not_exist_error or type(
            "TableNotExistException", (Exception,), {}
        )
        fake_pypaimon_catalog_factory = types.ModuleType("pypaimon.catalog.catalog_factory")
        fake_pypaimon_catalog_factory.CatalogFactory = types.SimpleNamespace(create=catalog_factory_create)

        fake_pypaimon.catalog = fake_pypaimon_catalog
        fake_pypaimon_catalog.catalog_exception = fake_pypaimon_catalog_exception
        fake_pypaimon_catalog.catalog_factory = fake_pypaimon_catalog_factory

        return {
            "pypaimon": fake_pypaimon,
            "pypaimon.catalog": fake_pypaimon_catalog,
            "pypaimon.catalog.catalog_exception": fake_pypaimon_catalog_exception,
            "pypaimon.catalog.catalog_factory": fake_pypaimon_catalog_factory,
        }

    def test_write_paimon_uses_write_ray_when_available(self):
        dataset = MagicMock(name="ray_dataset")

        table_write = MagicMock(name="table_write")
        write_builder = MagicMock(name="write_builder")
        write_builder.new_write.return_value = table_write

        table = MagicMock(name="table")
        table.new_batch_write_builder.return_value = write_builder

        catalog = MagicMock(name="catalog")
        catalog.get_table.return_value = table
        mock_create = MagicMock(return_value=catalog)

        fake_modules = self._build_fake_paimon_modules(mock_create)

        export_extra_args = {
            "catalog_options": {"metastore": "filesystem", "warehouse": "file:///tmp/warehouse"},
            "table_identifier": "db.sample_table",
            "concurrency": 4,
            "ray_remote_args": {"num_cpus": 1},
        }

        with patch.dict(sys.modules, fake_modules):
            RayExporter.write_paimon(dataset, "/tmp/fallback.jsonl", export_extra_args=export_extra_args)

        mock_create.assert_called_once_with(export_extra_args["catalog_options"])
        catalog.get_table.assert_called_once_with("db.sample_table")
        table_write.write_ray.assert_called_once_with(
            dataset,
            concurrency=4,
            ray_remote_args={"num_cpus": 1},
        )
        table_write.close.assert_called_once()
        write_builder.new_commit.assert_not_called()

    def test_write_paimon_falls_back_to_arrow_commit_when_write_ray_unavailable(self):
        import pandas as pd

        dataset = MagicMock(name="ray_dataset")
        dataset.limit.return_value.to_pandas.return_value = pd.DataFrame({"dt": ["2024-01-01"], "value": ["x"]})
        dataset.to_arrow.return_value = "arrow_table"

        table_write = types.SimpleNamespace(
            write_arrow=MagicMock(),
            prepare_commit=MagicMock(return_value=["commit_msg"]),
            close=MagicMock(),
        )
        table_commit = types.SimpleNamespace(commit=MagicMock(), close=MagicMock())
        overwrite_builder = MagicMock(name="overwrite_builder")
        overwrite_builder.new_write.return_value = table_write
        overwrite_builder.new_commit.return_value = table_commit

        write_builder = MagicMock(name="write_builder")
        write_builder.overwrite.return_value = overwrite_builder

        table = MagicMock(name="table")
        table.new_batch_write_builder.return_value = write_builder

        class TableNotExistException(Exception):
            pass

        catalog = MagicMock(name="catalog")
        catalog.get_table.side_effect = [TableNotExistException("missing table"), table]
        mock_create = MagicMock(return_value=catalog)

        schema_cls = MagicMock(name="Schema")
        schema_cls.from_pyarrow_schema = MagicMock(return_value="paimon_schema")

        fake_modules = self._build_fake_paimon_modules(
            mock_create,
            schema_cls=schema_cls,
            table_not_exist_error=TableNotExistException,
        )

        export_extra_args = {
            "catalog_options": {"metastore": "filesystem", "warehouse": "file:///tmp/warehouse"},
            "table_identifier": "db.sample_table",
            "overwrite": True,
            "schema_kwargs": {"partition_keys": ["dt"]},
        }

        with patch.dict(sys.modules, fake_modules):
            RayExporter.write_paimon(dataset, "/tmp/fallback.parquet", export_extra_args=export_extra_args)

        mock_create.assert_called_once_with(export_extra_args["catalog_options"])
        schema_cls.from_pyarrow_schema.assert_called_once()
        schema_call_kwargs = schema_cls.from_pyarrow_schema.call_args.kwargs
        self.assertEqual(schema_call_kwargs["partition_keys"], ["dt"])
        self.assertIn("pa_schema", schema_call_kwargs)
        catalog.create_table.assert_called_once_with(
            "db.sample_table",
            schema="paimon_schema",
            ignore_if_exists=False,
        )
        write_builder.overwrite.assert_called_once_with()
        table_write.write_arrow.assert_called_once_with("arrow_table")
        table_write.prepare_commit.assert_called_once_with()
        table_commit.commit.assert_called_once_with(["commit_msg"])
        table_write.close.assert_called_once()
        table_commit.close.assert_called_once()

    def test_write_paimon_raises_catalog_errors_without_fallback(self):
        dataset = MagicMock(name="ray_dataset")

        catalog = MagicMock(name="catalog")
        catalog.get_table.side_effect = PermissionError("access denied")
        mock_create = MagicMock(return_value=catalog)

        fake_modules = self._build_fake_paimon_modules(mock_create)

        with patch.dict(sys.modules, fake_modules), patch.object(RayExporter, "write_json") as mock_write_json:
            with self.assertRaises(PermissionError):
                RayExporter.write_paimon(
                    dataset,
                    "/tmp/fallback.jsonl",
                    export_extra_args={
                        "catalog_options": {
                            "metastore": "filesystem",
                            "warehouse": "file:///tmp/warehouse",
                        },
                        "table_identifier": "db.sample_table",
                    },
                )

        mock_write_json.assert_not_called()

    def test_write_paimon_raises_create_table_errors_without_fallback(self):
        import pandas as pd

        dataset = MagicMock(name="ray_dataset")
        dataset.limit.return_value.to_pandas.return_value = pd.DataFrame({"dt": ["2024-01-01"], "value": ["x"]})

        class TableNotExistException(Exception):
            pass

        catalog = MagicMock(name="catalog")
        catalog.get_table.side_effect = TableNotExistException("missing table")
        catalog.create_table.side_effect = RuntimeError("create failed")
        mock_create = MagicMock(return_value=catalog)

        schema_cls = MagicMock(name="Schema")
        schema_cls.from_pyarrow_schema = MagicMock(return_value="paimon_schema")

        fake_modules = self._build_fake_paimon_modules(
            mock_create,
            schema_cls=schema_cls,
            table_not_exist_error=TableNotExistException,
        )

        with patch.dict(sys.modules, fake_modules), patch.object(RayExporter, "write_json") as mock_write_json:
            with self.assertRaises(RuntimeError):
                RayExporter.write_paimon(
                    dataset,
                    "/tmp/fallback.jsonl",
                    export_extra_args={
                        "catalog_options": {
                            "metastore": "filesystem",
                            "warehouse": "file:///tmp/warehouse",
                        },
                        "table_identifier": "db.sample_table",
                    },
                )

        schema_cls.from_pyarrow_schema.assert_called_once()
        catalog.create_table.assert_called_once_with(
            "db.sample_table",
            schema="paimon_schema",
            ignore_if_exists=False,
        )
        mock_write_json.assert_not_called()

    def test_write_paimon_raises_write_errors_without_fallback(self):
        dataset = MagicMock(name="ray_dataset")

        table_write = MagicMock(name="table_write")
        table_write.write_ray.side_effect = RuntimeError("write failed")

        write_builder = MagicMock(name="write_builder")
        write_builder.new_write.return_value = table_write

        table = MagicMock(name="table")
        table.new_batch_write_builder.return_value = write_builder

        catalog = MagicMock(name="catalog")
        catalog.get_table.return_value = table
        mock_create = MagicMock(return_value=catalog)

        fake_modules = self._build_fake_paimon_modules(mock_create)

        with patch.dict(sys.modules, fake_modules), patch.object(RayExporter, "write_json") as mock_write_json:
            with self.assertRaises(RuntimeError):
                RayExporter.write_paimon(
                    dataset,
                    "/tmp/fallback.jsonl",
                    export_extra_args={
                        "catalog_options": {
                            "metastore": "filesystem",
                            "warehouse": "file:///tmp/warehouse",
                        },
                        "table_identifier": "db.sample_table",
                    },
                )

        table_write.close.assert_called_once()
        mock_write_json.assert_not_called()

    def test_write_paimon_falls_back_to_json_when_pypaimon_missing(self):
        dataset = MagicMock(name="ray_dataset")
        real_import = __import__

        def raising_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "pypaimon" or name.startswith("pypaimon."):
                raise ImportError("No module named 'pypaimon'")
            return real_import(name, globals, locals, fromlist, level)

        with (
            patch("builtins.__import__", side_effect=raising_import),
            patch.object(RayExporter, "write_json", return_value="json_fallback") as mock_write_json,
        ):
            result = RayExporter.write_paimon(dataset, "/tmp/fallback.jsonl", export_extra_args={})

        mock_write_json.assert_called_once_with(dataset, "/tmp/fallback.jsonl")
        self.assertEqual(result, "json_fallback")

    def _build_fake_iceberg_modules(self, load_catalog, no_such_table_error):
        fake_pyiceberg = types.ModuleType("pyiceberg")
        fake_pyiceberg.__path__ = []

        fake_pyiceberg_catalog = types.ModuleType("pyiceberg.catalog")
        fake_pyiceberg_catalog.load_catalog = load_catalog

        fake_pyiceberg_exceptions = types.ModuleType("pyiceberg.exceptions")
        fake_pyiceberg_exceptions.NoSuchTableError = no_such_table_error

        fake_pyiceberg.catalog = fake_pyiceberg_catalog
        fake_pyiceberg.exceptions = fake_pyiceberg_exceptions

        return {
            "pyiceberg": fake_pyiceberg,
            "pyiceberg.catalog": fake_pyiceberg_catalog,
            "pyiceberg.exceptions": fake_pyiceberg_exceptions,
        }

    def test_write_iceberg_uses_dataset_write_when_table_exists(self):
        dataset = MagicMock(name="ray_dataset")

        catalog = MagicMock(name="catalog")
        catalog.load_table.return_value = object()
        mock_load_catalog = MagicMock(return_value=catalog)

        class NoSuchTableError(Exception):
            pass

        fake_modules = self._build_fake_iceberg_modules(mock_load_catalog, NoSuchTableError)

        export_extra_args = {
            "table_identifier": "db.sample_table",
            "catalog_kwargs": {"name": "demo_catalog", "uri": "http://catalog:8181"},
            "mode": "append",
        }

        with patch.dict(sys.modules, fake_modules):
            RayExporter.write_iceberg(dataset, "/tmp/fallback.jsonl", export_extra_args=export_extra_args)

        mock_load_catalog.assert_called_once_with(**export_extra_args["catalog_kwargs"])
        catalog.load_table.assert_called_once_with("db.sample_table")
        dataset.write_iceberg.assert_called_once_with(
            table_identifier="db.sample_table",
            catalog_kwargs={"name": "demo_catalog", "uri": "http://catalog:8181"},
            mode="append",
        )

    def test_write_iceberg_creates_table_when_missing(self):
        import pandas as pd

        dataset = MagicMock(name="ray_dataset")
        dataset.limit.return_value.to_pandas.return_value = pd.DataFrame({"dt": ["2024-01-01"], "value": ["x"]})

        class NoSuchTableError(Exception):
            pass

        catalog = MagicMock(name="catalog")
        catalog.load_table.side_effect = NoSuchTableError("missing table")
        mock_load_catalog = MagicMock(return_value=catalog)

        fake_modules = self._build_fake_iceberg_modules(mock_load_catalog, NoSuchTableError)

        export_extra_args = {
            "table_identifier": "db.sample_table",
            "catalog_kwargs": {"name": "demo_catalog", "uri": "http://catalog:8181"},
        }

        with patch.dict(sys.modules, fake_modules):
            RayExporter.write_iceberg(dataset, "/tmp/fallback.parquet", export_extra_args=export_extra_args)

        mock_load_catalog.assert_called_once_with(**export_extra_args["catalog_kwargs"])
        catalog.load_table.assert_called_once_with("db.sample_table")
        catalog.create_table.assert_called_once()
        create_table_args = catalog.create_table.call_args.args
        self.assertEqual(create_table_args[0], "db.sample_table")
        self.assertIsNotNone(create_table_args[1])
        dataset.write_iceberg.assert_called_once_with(
            table_identifier="db.sample_table",
            catalog_kwargs={"name": "demo_catalog", "uri": "http://catalog:8181"},
        )

    def test_write_iceberg_falls_back_to_json_when_pyiceberg_missing(self):
        dataset = MagicMock(name="ray_dataset")
        real_import = __import__

        def raising_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "pyiceberg.catalog" or name == "pyiceberg.exceptions" or name.startswith("pyiceberg"):
                raise ImportError("No module named 'pyiceberg'")
            return real_import(name, globals, locals, fromlist, level)

        with (
            patch("builtins.__import__", side_effect=raising_import),
            patch.object(RayExporter, "write_json", return_value="json_fallback") as mock_write_json,
        ):
            result = RayExporter.write_iceberg(dataset, "/tmp/fallback.jsonl", export_extra_args={})

        mock_write_json.assert_called_once_with(dataset, "/tmp/fallback.jsonl")
        self.assertEqual(result, "json_fallback")


if __name__ == "__main__":
    unittest.main()

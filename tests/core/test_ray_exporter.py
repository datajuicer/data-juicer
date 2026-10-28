import copy
import os
import os.path as osp
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from data_juicer.utils.unittest_utils import TEST_TAG, DataJuicerTestCaseBase
from data_juicer.core.ray_exporter import RayExporter
from data_juicer.utils.constant import Fields, HashKeys
from data_juicer.utils.mm_utils import load_images_byte
from .test_exporter import IcebergWriterFixture


class TestRayExporter(DataJuicerTestCaseBase):

    def setUp(self):
        """Set up test data"""
        super().setUp()

        import ray
        from data_juicer.core.data.ray_dataset import RayDataset

        cur_dir = osp.dirname(osp.abspath(__file__))
        self.tmp_dir = f'{cur_dir}/tmp/{self.__class__.__name__}/{self._testMethodName}'
        os.makedirs(self.tmp_dir, exist_ok=True)

        self.data = [
            {'text': 'hello', Fields.stats: {'score': 1}, HashKeys.hash: 'a1'},
            {'text': 'world', Fields.stats: {'score': 2}, HashKeys.hash: 'b2'},
            {'text': 'test', Fields.stats: {'score': 3}, HashKeys.hash: 'c3'}
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

    @TEST_TAG('ray')
    def test_json_not_keep_stats_and_hashes(self):
        import ray

        out_path = osp.join(self.tmp_dir, 'outdata.json')
        ray_exporter = RayExporter(
            out_path,
            keep_stats_in_res_ds=False,
            keep_hashes_in_res_ds=False)
        ray_exporter.export(self.dataset.data)

        ds = ray.data.read_json(out_path)
        data_list = ds.take_all()

        self.assertListOfDictEqual(data_list, self._pop_raw_data_keys([Fields.stats, HashKeys.hash]))

    @TEST_TAG('ray')
    def test_jsonl_keep_stats_and_hashes(self):
        import ray

        out_path = osp.join(self.tmp_dir, 'outdata.jsonl')
        ray_exporter = RayExporter(
            out_path,
            keep_stats_in_res_ds=True,
            keep_hashes_in_res_ds=True)
        ray_exporter.export(self.dataset.data)

        ds = ray.data.read_json(out_path)
        data_list = ds.take_all()

        self.assertListOfDictEqual(data_list, self.data)

    @TEST_TAG('ray')
    def test_parquet_keep_stats(self):
        import ray

        out_path = osp.join(self.tmp_dir, 'outdata.parquet')
        ray_exporter = RayExporter(
            out_path,
            keep_stats_in_res_ds=True,
            keep_hashes_in_res_ds=False)
        ray_exporter.export(self.dataset.data)

        ds = ray.data.read_parquet(out_path)
        data_list = ds.take_all()

        self.assertListEqual(data_list, self._pop_raw_data_keys([HashKeys.hash]))

    @TEST_TAG('ray')
    def test_lance_keep_hashes(self):
        import ray

        out_path = osp.join(self.tmp_dir, 'outdata.lance')
        ray_exporter = RayExporter(
            out_path,
            keep_stats_in_res_ds=False,
            keep_hashes_in_res_ds=True)
        ray_exporter.export(self.dataset.data)

        ds = ray.data.read_lance(out_path)
        data_list = ds.take_all()

        self.assertListOfDictEqual(data_list, self._pop_raw_data_keys([Fields.stats]))

    @TEST_TAG('ray')
    def test_webdataset_multi_images(self):
        import io
        from PIL import Image
        import ray
        from data_juicer.core.data.ray_dataset import RayDataset

        data_dir = osp.abspath(osp.join(osp.dirname(osp.realpath(__file__)), '..', 'ops', 'data'))
        img1_path = osp.join(data_dir, 'img1.png')
        img2_path = osp.join(data_dir, 'img2.jpg')
        img3_path = osp.join(data_dir, 'img3.jpg')

        data = [
            {
                'json': {
                    'text': 'hello',
                    'images': [img1_path, img2_path]
                    },
                'jpgs': load_images_byte([img1_path, img2_path])},
            {
                'json': {
                    'text': 'world',
                    'images': [img2_path, img3_path]
                    },
                'jpgs': load_images_byte([img2_path, img3_path])},
            {
                'json': {
                    'text': 'test',
                    'images': [img1_path, img2_path, img3_path]
                    },
                'jpgs': load_images_byte([img1_path, img2_path, img3_path])}
        ]
        dataset = RayDataset(ray.data.from_items(data))
        out_path = osp.join(self.tmp_dir, 'outdata.webdataset')
        ray_exporter = RayExporter(out_path)
        ray_exporter.export(dataset.data)

        ds = RayDataset.read_webdataset(out_path)
        res_list = ds.take_all()
        
        self.assertEqual(len(res_list), len(data))
        res_list.sort(key=lambda x: x['json']['text'])
        data.sort(key=lambda x: x['json']['text'])

        for i in range(len(data)):
            self.assertDictEqual(res_list[i]['json'], data[i]['json'])
            self.assertEqual(
                res_list[i]['jpgs'],
                [Image.open(io.BytesIO(v)) for v in data[i]['jpgs']]
            )

    @TEST_TAG('ray')
    def test_webdataset_multi_videos_frames_bytes(self):
        import io
        from PIL import Image
        import ray
        from data_juicer.core.data.ray_dataset import RayDataset

        data_dir = osp.abspath(osp.join(osp.dirname(osp.realpath(__file__)), '..', 'ops', 'data'))
        img1_path = osp.join(data_dir, 'img1.png')
        img2_path = osp.join(data_dir, 'img2.jpg')
        img3_path = osp.join(data_dir, 'img3.jpg')

        data = [
            {
                'json': {
                    'text': 'hello',
                    'videos': ['video1.mp4', 'video2.mp4']
                    },
                'mp4s': [
                    load_images_byte([img1_path]),  # as video1 frames bytes
                    load_images_byte([img1_path, img2_path])   # as video2 frames path
                    ]
            },
            {
                'json': {
                    'text': 'world',
                    'videos': ['video1.mp4']
                    },
                'mp4s': [
                    load_images_byte([img2_path, img3_path])  # as video1 frames
                    ]
            }
        ]
        dataset = RayDataset(ray.data.from_items(data))
        out_path = osp.join(self.tmp_dir, 'outdata.webdataset')
        ray_exporter = RayExporter(out_path, export_type='webdataset')
        ray_exporter.export(dataset.data)

        ds = RayDataset.read_webdataset(out_path)
        res_list = ds.take_all()
        
        self.assertEqual(len(res_list), len(data))
        res_list.sort(key=lambda x: x['json']['text'])
        data.sort(key=lambda x: x['json']['text'])
        
        for i in range(len(data)):
            if len(data[i]['mp4s']) > 1:
                tgt_mp4s = [[Image.open(io.BytesIO(f_i)) for f_i in v_i] for v_i in data[i]['mp4s']]
            else:
                tgt_mp4s = [Image.open(io.BytesIO(f_i)) for f_i in data[i]['mp4s'][0]]
            self.assertDictEqual(res_list[i]['json'], data[i]['json'])
            self.assertEqual(res_list[i]['mp4s'], tgt_mp4s)

    @TEST_TAG('ray')
    def test_webdataset_multi_videos_frames_path(self):
        import io
        from PIL import Image
        import ray
        from data_juicer.core.data.ray_dataset import RayDataset

        data_dir = osp.abspath(osp.join(osp.dirname(osp.realpath(__file__)), '..', 'ops', 'data'))
        img1_path = osp.join(data_dir, 'img8.jpg')
        img2_path = osp.join(data_dir, 'img9.jpg')
        img3_path = osp.join(data_dir, 'img10.jpg')

        data = [
            {
                'json': {
                    'text': 'hello',
                    'videos': ['video1.mp4', 'video2.mp4']
                    },
                'mp4s': [
                    [img1_path],  # as video1 frames path
                    [img1_path, img2_path]   # as video2 frames path
                    ]
            },
            {
                'json': {
                    'text': 'world',
                    'videos': ['video1.mp4']
                    },
                'mp4s': [
                    [img2_path, img3_path]  # as video1 frames path
                    ]
            }
        ]
        dataset = RayDataset(ray.data.from_items(data))
        out_path = osp.join(self.tmp_dir, 'outdata.webdataset')
        ray_exporter = RayExporter(out_path, export_type='webdataset')
        ray_exporter.export(dataset.data)

        ds = RayDataset.read_webdataset(out_path)
        res_list = ds.take_all()
        
        self.assertEqual(len(res_list), len(data))
        res_list.sort(key=lambda x: x['json']['text'])
        data.sort(key=lambda x: x['json']['text'])
        
        for i in range(len(data)):
            if len(data[i]['mp4s']) > 1:
                tgt_mp4s = [[Image.open(f_i, formats=['jpeg']) for f_i in v_i] for v_i in data[i]['mp4s']]
            else:
                tgt_mp4s = [Image.open(f_i, formats=['jpeg']) for f_i in data[i]['mp4s'][0]]
            self.assertDictEqual(res_list[i]['json'], data[i]['json'])
            self.assertEqual(res_list[i]['mp4s'], tgt_mp4s)

    @TEST_TAG('ray')
    def test_webdataset_multi_audios_path(self):
        import ray
        from data_juicer.core.data.ray_dataset import RayDataset
        from data_juicer.utils.mm_utils import load_audio

        data_dir = osp.abspath(osp.join(osp.dirname(osp.realpath(__file__)), '..', 'ops', 'data'))
        audio1_path = osp.join(data_dir, 'audio1.wav')
        audio2_path = osp.join(data_dir, 'audio2.wav')
        audio3_path = osp.join(data_dir, 'audio3.ogg')

        data = [
            {
                'json': {
                    'text': 'hello',
                    },
                'mp3s': [audio1_path]
            },
            {
                'json': {
                    'text': 'world',
                    },
                'mp3s': [audio2_path, audio3_path]
            }
        ]
        dataset = RayDataset(ray.data.from_items(data))
        out_path = osp.join(self.tmp_dir, 'outdata.webdataset')
        ray_exporter = RayExporter(out_path, export_type='webdataset')
        ray_exporter.export(dataset.data)

        ds = RayDataset.read_webdataset(out_path)
        res_list = ds.take_all()
        
        self.assertEqual(len(res_list), len(data))

        res_list.sort(key=lambda x: x['json']['text'])
        data.sort(key=lambda x: x['json']['text'])
        
        for i in range(len(data)):
            if len(data[i]['mp3s']) <= 1:
                mp3s_list = [res_list[i]['mp3s']]
            else:
                mp3s_list = res_list[i]['mp3s']

            tgt_mp3s = [load_audio(f_i) for f_i in data[i]['mp3s']]
            
            self.assertDictEqual(res_list[i]['json'], data[i]['json'])

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
        self._key_file = os.path.join(self.tmp_dir, 'test.key')
        with open(self._key_file, 'wb') as f:
            f.write(self._key)

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # __init__ – encryption flag and Fernet setup
    # ------------------------------------------------------------------

    def test_encrypt_disabled_by_default(self):
        """encrypt_before_export defaults to False; _fernet stays None."""
        exporter = RayExporter(export_path=os.path.join(self.tmp_dir, 'out.jsonl'))
        self.assertFalse(exporter.encrypt_before_export)
        self.assertIsNone(exporter._fernet)

    def test_encrypt_enabled_loads_fernet_key(self):
        """encrypt_before_export=True with a valid key file sets _fernet."""
        exporter = RayExporter(
            export_path=os.path.join(self.tmp_dir, 'out.jsonl'),
            encrypt_before_export=True,
            encryption_key_path=self._key_file,
        )
        self.assertTrue(exporter.encrypt_before_export)
        self.assertIsNotNone(exporter._fernet)

    @patch("data_juicer.utils.s3_utils.create_pyarrow_s3_filesystem")
    def test_s3_path_disables_encryption_with_warning(self, mock_create_s3_fs):
        """S3 export path silently disables local-file encryption."""
        warnings_seen = []
        from loguru import logger as _logger
        handler_id = _logger.add(
            lambda msg: warnings_seen.append(str(msg)),
            level='WARNING',
            format='{message}',
        )
        try:
            exporter = RayExporter(
                export_path='s3://some-bucket/out.jsonl',
                encrypt_before_export=True,
                encryption_key_path=self._key_file,
            )
        finally:
            _logger.remove(handler_id)
        self.assertFalse(exporter.encrypt_before_export)
        self.assertTrue(
            any('encrypt_before_export' in str(w) and 's3' in str(w).lower()
                for w in warnings_seen),
            'Expected warning about S3 + encrypt_before_export not found',
        )

    @patch("data_juicer.utils.hdfs_utils.create_pyarrow_hdfs_filesystem")
    def test_hdfs_path_disables_encryption_with_warning(self, mock_create_hdfs_fs):
        """HDFS export path silently disables local-file encryption."""
        warnings_seen = []
        from loguru import logger as _logger
        handler_id = _logger.add(
            lambda msg: warnings_seen.append(str(msg)),
            level='WARNING',
            format='{message}',
        )
        try:
            exporter = RayExporter(
                export_path='hdfs://namenode:8020/user/data/out.jsonl',
                encrypt_before_export=True,
                encryption_key_path=self._key_file,
            )
        finally:
            _logger.remove(handler_id)
        self.assertFalse(exporter.encrypt_before_export)
        self.assertTrue(
            any('encrypt_before_export' in str(w) for w in warnings_seen),
            'Expected warning about HDFS + encrypt_before_export not found',
        )

    @patch("data_juicer.utils.hdfs_utils.create_pyarrow_hdfs_filesystem")
    @patch("data_juicer.core.ray_exporter.is_remote_path")
    def test_hdfs_path_skips_local_makedirs(self, mock_is_remote, mock_create_hdfs_fs):
        """HDFS export_path should skip local os.makedirs via is_remote_path."""
        mock_is_remote.return_value = True

        exporter = RayExporter(
            export_path='hdfs://namenode:8020/user/data',
            export_type='jsonl',
        )

        mock_dataset = MagicMock()
        mock_dataset.columns.return_value = ['text']
        mock_dataset.count.return_value = 1
        mock_dataset.size_bytes.return_value = 50
        mock_dataset.drop_columns.return_value = mock_dataset

        with patch.object(RayExporter, '_router') as mock_router:
            mock_write = MagicMock(return_value=None)
            mock_router.return_value = {'jsonl': mock_write}
            with patch("data_juicer.core.ray_exporter.os.makedirs") as mock_makedirs:
                exporter._export_impl(mock_dataset, 'hdfs://namenode:8020/user/data')

        # os.makedirs should NOT be called because is_remote_path returned True
        mock_makedirs.assert_not_called()


    # ------------------------------------------------------------------
    # _export_impl – encrypt_file called for each file in the output dir
    # ------------------------------------------------------------------

    def test_export_impl_encrypts_output_files(self):
        """After _export_impl, all files inside the output dir are encrypted."""
        from cryptography.fernet import Fernet
        from data_juicer.utils.encryption_utils import decrypt_file_to_bytes

        export_dir = os.path.join(self.tmp_dir, 'export_out')
        exporter = RayExporter(
            export_path=export_dir,
            export_type='jsonl',
            encrypt_before_export=True,
            encryption_key_path=self._key_file,
        )

        # Write two plain-text sentinel files to simulate Ray output
        os.makedirs(export_dir, exist_ok=True)
        sentinel = b'{"text": "hello"}\n'
        for fname in ('part-0.jsonl', 'part-1.jsonl'):
            fpath = os.path.join(export_dir, fname)
            with open(fpath, 'wb') as f:
                f.write(sentinel)

        # Mock the underlying Ray write call so _export_impl only runs
        # the post-write encryption step
        mock_dataset = MagicMock()
        mock_dataset.columns.return_value = ['text']
        mock_dataset.count.return_value = 2
        mock_dataset.size_bytes.return_value = 100
        mock_dataset.drop_columns.return_value = mock_dataset

        with patch.object(RayExporter, '_router') as mock_router:
            mock_write = MagicMock(return_value=None)
            mock_router.return_value = {'jsonl': mock_write}
            exporter._export_impl(mock_dataset, export_dir)

        # Verify every file is now encrypted (not plain text any more)
        fernet = Fernet(self._key)
        for fname in ('part-0.jsonl', 'part-1.jsonl'):
            fpath = os.path.join(export_dir, fname)
            raw = open(fpath, 'rb').read()
            self.assertNotEqual(raw, sentinel,
                                f'{fname} should have been encrypted')
            # Must be decryptable with the same key
            decrypted = decrypt_file_to_bytes(fpath, fernet)
            self.assertEqual(decrypted, sentinel)

    def test_export_impl_skips_encryption_when_disabled(self):
        """When encrypt_before_export=False, files are left as-is."""
        export_dir = os.path.join(self.tmp_dir, 'plain_out')
        exporter = RayExporter(
            export_path=export_dir,
            export_type='jsonl',
            encrypt_before_export=False,
        )

        os.makedirs(export_dir, exist_ok=True)
        sentinel = b'{"text": "world"}\n'
        fpath = os.path.join(export_dir, 'part-0.jsonl')
        with open(fpath, 'wb') as f:
            f.write(sentinel)

        mock_dataset = MagicMock()
        mock_dataset.columns.return_value = ['text']
        mock_dataset.count.return_value = 1
        mock_dataset.size_bytes.return_value = 50
        mock_dataset.drop_columns.return_value = mock_dataset

        with patch.object(RayExporter, '_router') as mock_router:
            mock_write = MagicMock(return_value=None)
            mock_router.return_value = {'jsonl': mock_write}
            exporter._export_impl(mock_dataset, export_dir)

        self.assertEqual(open(fpath, 'rb').read(), sentinel,
                         'File should not be modified when encryption is off')


class RayIcebergExporterConfigTest(IcebergWriterFixture, unittest.TestCase):
    def setUp(self):
        super().setUp()
        version = patch("ray.__version__", "2.53.0")
        version.start()
        self.addCleanup(version.stop)

    def test_invalid_configuration_and_old_ray(self):
        invalid = [
            {"mode": "upsert"}, {"mode": None}, {"table_identifier": ""}, {"catalog_kwargs": None},
            {"catalog_kwargs": {}}, {"catalog_kwargs": {"name": " "}},
            {"catalog_kwargs": {k: v for k, v in self.catalog_kwargs.items() if k != "name"}},
            {"export_path": "file:///tmp/output"},
            {"export_path": None}, {"export_path": ""}, {"export_path": "s3://bucket/output"},
            {"export_path": "hdfs://host/output"}, {"export_path": "https://host/output"},
            {"export_shard_size": 1},
            {"encrypt_before_export": True, "encryption_key_path": "/must/not/read"},
            {"overwrite_filter": "id > 0"}, {"overwrite_kwargs": {"branch": "other"}},
        ]
        with patch("data_juicer.utils.encryption_utils.load_fernet_key") as key_loader, \
                patch("data_juicer.utils.s3_utils.create_pyarrow_s3_filesystem") as filesystem:
            for changes in invalid:
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    RayExporter(**dict(self.options(), **changes))
            key_loader.assert_not_called()
            filesystem.assert_not_called()
        with patch("ray.__version__", "2.52.0"), self.assertRaisesRegex(RuntimeError, "Ray >=2.53.0"):
            RayExporter(**self.options())
        writer = RayExporter(**self.options())
        self.assertEqual(writer.iceberg_config["mode"], "overwrite")
        self.catalog_kwargs["warehouse"] = "changed-after-construction"
        self.assertNotEqual(writer.iceberg_config["catalog_kwargs"]["warehouse"], self.catalog_kwargs["warehouse"])

    def test_forwards_excluded_fields_without_preexecution(self):
        source = MagicMock()
        writer = RayExporter(**self.options(keep_stats_in_res_ds=False))
        with patch.object(writer, "write_iceberg") as write:
            writer.export(source, columns=["stale"])
        write.assert_called_once()
        self.assertEqual(write.call_args.args, (source, self.anchor))
        self.assertSetEqual(
            write.call_args.kwargs.pop("excluded_fields"),
            {Fields.stats, Fields.meta, HashKeys.hash, HashKeys.minhash,
             HashKeys.simhash, HashKeys.imagehash, HashKeys.videohash},
        )
        self.assertEqual(write.call_args.kwargs, writer.iceberg_config)
        source.materialize.assert_not_called()
        source.columns.assert_not_called()

    def test_windows_artifact_constructor(self):
        writer = RayExporter(**dict(self.options(), export_path="C:/tmp/result.jsonl"))
        self.assertEqual(writer.export_path, "C:/tmp/result.jsonl")

    def test_executor_iceberg_uses_configured_stats_flags(self):
        from types import SimpleNamespace
        from data_juicer.core.executor.ray_executor import RayExecutor

        cfg = SimpleNamespace(
            export_path=self.anchor, export_type="iceberg", export_shard_size=0,
            export_extra_args={"table_identifier": self.identifier, "catalog_kwargs": self.catalog_kwargs},
            keep_stats_in_res_ds=False, keep_hashes_in_res_ds=False, work_dir=str(self.root),
            open_tracer=False, min_common_dep_num_to_combine=-1,
        )
        module = "data_juicer.core.executor.ray_executor"
        with patch(f"{module}.ExecutorBase.__init__", lambda obj, config: setattr(obj, "cfg", config)), \
                patch(f"{module}.EventLoggingMixin.__init__", return_value=None), \
                patch(f"{module}.DAGExecutionMixin.__init__", return_value=None), \
                patch(f"{module}.DatasetBuilder"), patch(f"{module}.ray"), \
                patch("data_juicer.utils.ray_utils.initialize_ray"):
            executor = RayExecutor(cfg)
        self.assertFalse(executor.exporter.keep_stats_in_res_ds)
        source = MagicMock()
        with patch.object(executor.exporter, "write_iceberg") as write:
            executor.exporter.export(source)
        self.assertEqual(write.call_args.args, (source, self.anchor))
        self.assertTrue({Fields.stats, Fields.meta}.issubset(write.call_args.kwargs["excluded_fields"]))
        source.materialize.assert_not_called()

class RayIcebergExporterTest(IcebergWriterFixture, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import ray
        from packaging.version import Version

        if Version(ray.__version__) < Version("2.53.0"):
            raise unittest.SkipTest("Iceberg writing requires Ray >=2.53.0")
        cls.owns_ray = not ray.is_initialized()
        if cls.owns_ray:
            # Isolated local instance; never stop another user's Ray cluster.
            cls.ray_temp = tempfile.TemporaryDirectory(prefix="dj_ray_iceberg_")
            ray.init(address="local", num_cpus=2, include_dashboard=False,
                     object_store_memory=128 * 1024 * 1024, _temp_dir=cls.ray_temp.name)
        from ray.data import DataContext

        cls.old_progress = DataContext.get_current().enable_progress_bars
        DataContext.get_current().enable_progress_bars = False

    @classmethod
    def tearDownClass(cls):
        import ray
        from ray.data import DataContext

        DataContext.get_current().enable_progress_bars = cls.old_progress
        if cls.owns_ray:
            ray.shutdown()
            cls.ray_temp.cleanup()

    @TEST_TAG("ray")
    def test_native_append_overwrite_repeated_and_empty(self):
        import pyarrow as pa
        import ray

        arrow = pa.Table.from_pylist(self.rows, schema=self.schema)
        for mode in ("append", "overwrite"):
            with self.subTest(mode=mode):
                self.table.overwrite(pa.Table.from_pylist(self.initial, schema=self.schema))
                writer = RayExporter(**self.options(mode=mode))
                ds = ray.data.from_arrow(arrow)
                writer.export(ds)
                writer.export(ds)
                expected = self.initial + self.rows * 2 if mode == "append" else self.rows
                self.assertCountEqual(self.read_rows(), expected)
                for empty in (ray.data.from_arrow(arrow.slice(0, 0)), ray.data.from_items([])):
                    if mode == "overwrite":
                        self.table.overwrite(pa.Table.from_pylist(self.initial, schema=self.schema))
                    before = self.snapshot()
                    writer.export(empty)
                    if mode == "append":
                        self.assertCountEqual(self.read_rows(), expected)
                        self.assertEqual(self.snapshot(), before)
                    else:
                        self.assertEqual(self.read_rows(), [])
                        self.assertNotEqual(self.snapshot(), before)
                        self.assertEqual(self.table.current_snapshot().parent_snapshot_id, before)
        self.assertFalse(os.path.exists(self.anchor))
        self.assertFalse(os.path.exists(os.path.dirname(self.anchor)))

    @TEST_TAG("ray")
    def test_required_columns_accept_nullable_blocks_without_nulls(self):
        import pyarrow as pa
        import ray

        schema = pa.schema([pa.field("id", pa.int64(), nullable=False), ("text", pa.string())])
        table = self.catalog.create_table("training.required", schema=schema)
        required_schema = table.schema()
        writer = RayExporter(**dict(self.options(), table_identifier="training.required"))
        arrow = pa.Table.from_pylist(self.rows, schema=self.schema)
        self.assertTrue(arrow.schema.field("id").nullable)
        writer.export(ray.data.from_arrow(arrow))
        self.assertCountEqual(table.refresh().scan().to_arrow().to_pylist(), self.rows)
        # Native union-by-name would have relaxed the required id.
        self.assertEqual(table.schema(), required_schema)
        self.assertEqual(table.schema().schema_id, required_schema.schema_id)
        snapshot = table.current_snapshot().snapshot_id
        nulls = pa.Table.from_pylist([{"id": None, "text": "missing"}], schema=self.schema)
        with self.assertRaisesRegex(Exception, "required column id contains null"):
            writer.export(ray.data.from_arrow(nulls))
        self.assertEqual(table.refresh().current_snapshot().snapshot_id, snapshot)

    @TEST_TAG("ray")
    def test_reordered_columns_follow_target_schema(self):
        import pyarrow as pa
        import ray

        arrow = pa.Table.from_pylist(self.rows, schema=self.schema).select(["text", "id"])
        RayExporter(**self.options()).export(ray.data.from_arrow(arrow))
        self.assertCountEqual(self.read_rows(), self.rows)

    @TEST_TAG("ray")
    def test_multiblock_two_workers_multiset_and_file_paths(self):
        from collections import Counter

        import pyarrow as pa
        import ray

        @ray.remote(num_cpus=0)
        class ExecutionCounter:
            def __init__(self):
                self.calls = 0

            def increment(self):
                self.calls += 1

            def count(self):
                return self.calls

        counter = ExecutionCounter.remote()
        self.addCleanup(ray.kill, counter)
        committed = []

        def transform(batch):
            ray.get(counter.increment.remote())
            return batch

        for mode in ("overwrite", "append"):
            ids = [i if mode == "overwrite" else i // 2 for i in range(1024)]
            rows = [{"id": i, "text": f"row-{i}"} for i in ids]
            arrow = pa.Table.from_pylist(rows, schema=self.schema)
            self.table.overwrite(pa.Table.from_pylist(self.initial, schema=self.schema))
            previous_paths = {task.file.file_path for task in self.table.scan().plan_files()}
            writer = RayExporter(**self.options(mode=mode))
            for run in range(2):
                ds = ray.data.from_arrow([arrow.slice(i * 128, 128) for i in range(8)])
                ds = ds.map_batches(transform, batch_format="pyarrow", batch_size=None)
                before = ray.get(counter.count.remote())
                writer.export(ds)
                self.assertEqual(ray.get(counter.count.remote()) - before, 8)
                expected = self.initial + rows * (run + 1) if mode == "append" else rows
                actual = self.read_rows()
                self.assertEqual(
                    Counter((row["id"], row["text"]) for row in actual),
                    Counter((row["id"], row["text"]) for row in expected),
                )
                self.assertEqual(len({row["id"] for row in actual}), 1024 if mode == "overwrite" else 512)
                self.assertEqual(len(actual), 1024 if mode == "overwrite" else 1 + 1024 * (run + 1))
                active_paths = {task.file.file_path for task in self.table.scan().plan_files()}
                new_paths = active_paths - previous_paths
                self.assertEqual(len(new_paths), 8)
                self.assertEqual(len(active_paths), len(list(self.table.scan().plan_files())))
                if mode == "append":
                    self.assertEqual(active_paths, previous_paths | new_paths)
                else:
                    self.assertEqual(active_paths, new_paths)
                committed.append(new_paths)
                previous_paths = active_paths
        self.assertEqual(len(set.union(*committed)), 32)
        self.assertFalse(os.path.exists(self.anchor))

    @TEST_TAG("ray")
    def test_cached_execution_and_public_iceberg_api(self):
        import pyarrow as pa
        import ray
        from ray.data import SaveMode

        @ray.remote(num_cpus=0)
        class Counter:
            def __init__(self):
                self.calls = 0

            def increment(self):
                self.calls += 1
                return self.calls

            def count(self):
                return self.calls

        counter = Counter.remote()
        self.addCleanup(ray.kill, counter)

        def transform(batch):
            value = ray.get(counter.increment.remote())
            return batch.set_column(0, "id", pa.array([value] * len(batch), type=pa.int64()))

        arrow = pa.Table.from_pylist(self.rows, schema=self.schema)
        ds = ray.data.from_arrow(arrow).map_batches(transform, batch_format="pyarrow", batch_size=None)
        native_write = ray.data.Dataset.write_iceberg

        def write(dataset, **kwargs):
            self.assertEqual(kwargs["table_identifier"], self.identifier)
            self.assertEqual(kwargs["mode"], SaveMode.OVERWRITE)
            self.assertEqual(kwargs["catalog_kwargs"], self.catalog_kwargs)
            return native_write(dataset, **kwargs)

        with patch.object(ray.data.Dataset, "write_iceberg", autospec=True, side_effect=write) as routed:
            RayExporter(**self.options()).export(ds)
        routed.assert_called_once()
        self.assertEqual(ray.get(counter.count.remote()), 1)
        self.assertCountEqual(self.read_rows(), [{"id": 1, "text": row["text"]} for row in self.rows])

    @TEST_TAG("ray")
    def test_public_write_single_commit_and_failure_atomicity(self):
        import pyarrow as pa
        import ray
        from pyiceberg.catalog.sql import SqlCatalog

        dataset = ray.data.from_arrow(pa.Table.from_pylist(self.rows, schema=self.schema)).repartition(2)
        writer = RayExporter(**self.options())
        native_commit = SqlCatalog.commit_table
        commits = []

        def record(catalog, *args, **kwargs):
            commits.append(True)
            return native_commit(catalog, *args, **kwargs)

        with patch.object(SqlCatalog, "commit_table", record):
            writer.export(dataset)
        self.assertEqual(len(commits), 1)
        self.assertEqual(self.read_rows(), self.rows)

        snapshot = self.snapshot()
        location = self.table.metadata_location
        content = self.read_rows()
        failed_commits = []

        def fail(*args, **kwargs):
            failed_commits.append(True)
            raise RuntimeError("INJECTED_COMMIT_FAILURE")

        with patch.object(SqlCatalog, "commit_table", fail), self.assertRaises(Exception):
            writer.export(dataset)
        self.assertEqual(len(failed_commits), 1)
        self.assertEqual(self.snapshot(), snapshot)
        self.assertEqual(self.table.metadata_location, location)
        self.assertEqual(self.read_rows(), content)

    @TEST_TAG("ray")
    def test_stats_hashes_use_actual_columns(self):
        import pyarrow as pa
        import ray

        rows = [dict(row, **{Fields.stats: {"score": i}, Fields.meta: {"source": "local"},
                            HashKeys.hash: f"h{i}"}) for i, row in enumerate(self.rows)]
        arrow = pa.Table.from_pylist(rows)
        for keep_stats in (False, True):
            for keep_hashes in (False, True):
                with self.subTest(stats=keep_stats, hashes=keep_hashes):
                    columns = ["id", "text"]
                    if keep_stats:
                        columns += [Fields.stats, Fields.meta]
                    if keep_hashes:
                        columns += [HashKeys.hash]
                    identifier = f"training.kept_{int(keep_stats)}_{int(keep_hashes)}"
                    table = self.catalog.create_table(identifier, schema=arrow.select(columns).schema)
                    writer = RayExporter(**dict(self.options(), table_identifier=identifier),
                                         keep_stats_in_res_ds=keep_stats, keep_hashes_in_res_ds=keep_hashes)
                    # Executor input columns omit columns produced by operators.
                    writer.export(ray.data.from_arrow(arrow), columns=["id", "text"])
                    self.assertCountEqual(table.refresh().scan().to_arrow().to_pylist(),
                                          arrow.select(columns).to_pylist())
        self.assertFalse(os.path.exists(self.anchor))

    @TEST_TAG("ray")
    def test_rich_types_native_roundtrip(self):
        import ray

        arrow = self.rich_arrow()
        writer = RayExporter(**dict(self.options(), table_identifier="training.typed"))
        writer.export(ray.data.from_arrow(arrow))
        result = self.catalog.load_table("training.typed").scan().to_arrow()
        self.assertCountEqual(result.to_pylist(), arrow.to_pylist())

    @TEST_TAG("ray")
    def test_native_read_then_write_existing_table(self):
        import ray

        writer = RayExporter(**self.options())
        for _ in range(2):
            ds = ray.data.read_iceberg(table_identifier=self.identifier, catalog_kwargs=copy.deepcopy(self.catalog_kwargs))
            writer.export(ds)
            self.assertEqual(self.read_rows(), self.initial)

    @TEST_TAG("ray")
    def test_schema_error_precedes_sink_without_snapshot_change(self):
        import pyarrow as pa
        import ray

        before = self.snapshot()
        location = self.table.metadata_location
        schema_id = self.table.schema().schema_id
        writer = RayExporter(**self.options())
        for arrow in (pa.table({"id": ["wrong"], "text": ["x"]}),
                      pa.table({"id": [1], "text": ["x"], "extra": [2]})):
            with self.subTest(schema=arrow.schema), self.assertRaises(ValueError):
                writer.export(ray.data.from_arrow(arrow))
            self.assertEqual(self.snapshot(), before)
            self.assertEqual(self.table.metadata_location, location)
            self.assertEqual(self.table.schema().schema_id, schema_id)
        # A good first block must not hide an incompatible later block.
        mixed = ray.data.from_arrow([
            pa.Table.from_pylist(self.rows, schema=self.schema),
            pa.table({"id": [3], "text": ["bad"], "extra": [2]}),
        ])
        with self.assertRaisesRegex(Exception, "more columns: extra"):
            writer.export(mixed)
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(self.table.metadata_location, location)
        self.assertEqual(self.read_rows(), self.initial)
        self.assertFalse(os.path.exists(self.anchor))

    @TEST_TAG("ray")
    def test_schema_evolution_is_rejected_without_metadata_commit(self):
        import pyarrow as pa
        import ray

        # A column doc can cause evolution even when types are compatible.
        documented = pa.schema([pa.field("id", pa.int64(), metadata={b"doc": b"new doc"}),
                                ("text", pa.string())])
        before = self.snapshot()
        location = self.table.metadata_location
        writer = RayExporter(**self.options())
        ds = ray.data.from_arrow(pa.Table.from_pylist(self.rows, schema=documented))
        with self.assertRaisesRegex(ValueError, "automatic schema evolution"):
            writer.export(ds)
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(self.table.metadata_location, location)

    @TEST_TAG("ray")
    def test_missing_table_even_empty_no_fallback(self):
        import pyarrow as pa
        import ray

        for mode in ("append", "overwrite"):
            writer = RayExporter(**dict(self.options(mode=mode), table_identifier="training.missing"))
            for ds in (ray.data.from_arrow(pa.Table.from_pylist(self.rows, schema=self.schema)),
                       ray.data.from_items([])):
                with self.assertRaisesRegex(ValueError, "target table does not exist"):
                    writer.export(ds)
        self.assertFalse(os.path.exists(self.anchor))
        self.assertFalse(os.path.exists(os.path.dirname(self.anchor)))


if __name__ == '__main__':
    unittest.main()
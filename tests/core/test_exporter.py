import os
import unittest
import jsonlines as jl
from datasets import Dataset
from data_juicer.core import Exporter
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


if __name__ == '__main__':
    unittest.main()

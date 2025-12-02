import os
import unittest
import datasets
from datasets import load_dataset
from loguru import logger
from unittest.mock import patch
from data_juicer.core import Adapter, NestedDataset
from data_juicer.config import get_init_configs
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase
from data_juicer.ops.mapper import FixUnicodeMapper
from data_juicer.ops.filter import PerplexityFilter
from data_juicer.ops.deduplicator import DocumentDeduplicator

class AdapterTest(DataJuicerTestCaseBase):
    test_file = 'text_only_2.3k.jsonl'

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        # test dataset
        download_link = f'http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/data_juicer/unittest_data/{cls.test_file}'
        os.system(f'wget {download_link}')

    @classmethod
    def tearDownClass(cls, hf_model_name=None) -> None:
        # remove test dataset
        os.system(f'rm -f {cls.test_file}')

        super().tearDownClass(hf_model_name)

    def setUp(self) -> None:
        super().setUp()
        # tmp dir
        self.tmp_dir = 'tmp/test_adapter/'
        os.makedirs(self.tmp_dir, exist_ok=True)

    def tearDown(self) -> None:
        super().tearDown()
        if os.path.exists(self.tmp_dir):
            os.system(f'rm -rf {self.tmp_dir}')

    def test_take_batch(self):
        ds = load_dataset('json', data_files=self.test_file, split='train')
        logger.info(f'Length of test dataset: [{len(ds)}]')
        batch_sizes = [100, 1000, 3000]
        test_config = {}

        for bs in batch_sizes:
            tgt_len = min(bs, len(ds))
            test_config['batch_size'] = bs
            res_ds = Adapter.take_batch(ds, test_config)
            logger.info(f'Require [{bs}] and got [{len(res_ds)}].')
            self.assertEqual(len(res_ds), tgt_len)

    def test_batch_size_strategy(self):
        test_analysis_res = [
            {
                'resource_analysis': {
                    'CPU util.': {
                        'max': 0.5,
                    },
                    'GPU util.': {
                        'max': 0.8,
                    },
                    'Mem. util.': {
                        'max': 0.3,
                    }
                }
            },
            {
                'resource_analysis': {
                    'CPU util.': {
                        'max': 0.2,
                    },
                    'GPU util.': {
                        'max': 1.0,
                    },
                    'Mem. util.': {
                        'max': 0.1,
                    }
                }
            },
        ]

        adapter = Adapter({'batch_size': 1})
        adapter.idle_resources = {
            'CPU util.': 0,
            'GPU util.': 0,
            'Mem. util.': 0,
        }

        # basic test
        tgt_bs_1 = [2, 5]
        bs_res = adapter.batch_size_strategy(test_analysis_res,
                                             base_bs=1,
                                             util_th=1.0)
        self.assertEqual(bs_res, tgt_bs_1)

        # lower util threshold
        tgt_bs_2 = [1, 3]
        bs_res = adapter.batch_size_strategy(test_analysis_res,
                                             base_bs=1,
                                             util_th=0.7)
        self.assertEqual(bs_res, tgt_bs_2)

        # larger base batch size
        adapter.cfg['batch_size'] = 10
        tgt_bs_3 = [18, 45]
        bs_res = adapter.batch_size_strategy(test_analysis_res,
                                             base_bs=10,
                                             util_th=0.9)
        self.assertEqual(bs_res, tgt_bs_3)

        # out of resource
        tgt_bs_4 = [2, 5]
        bs_res = adapter.batch_size_strategy(test_analysis_res,
                                             base_bs=10,
                                             util_th=0.1)
        self.assertEqual(bs_res, tgt_bs_4)

        # out of resource 2
        adapter.cfg['batch_size'] = 1
        tgt_bs_4 = [1, 1]
        bs_res = adapter.batch_size_strategy(test_analysis_res,
                                             base_bs=1,
                                             util_th=0.1)
        self.assertEqual(bs_res, tgt_bs_4)

    @patch('data_juicer.core.adapter.Monitor.monitor_func')
    def test_execute_and_probe(self, mock_monitor_func):
        mock_monitor_func.return_value = None, {
            'resource': [{
                'CPU util.': {
                    'max': 0.5,
                },
            }],
            'sampling_interval': 0.5,
            'time': 0.54,
        }

        datasets.disable_caching()
        # basic test
        ds = load_dataset('json', data_files=self.test_file, split='train').take(100)
        ops = [
            FixUnicodeMapper(num_proc=1),
            PerplexityFilter(num_proc=1),
            DocumentDeduplicator(num_proc=1),
        ]  # use some batched OPs later

        resource_util_list = Adapter.execute_and_probe(ds, ops)
        self.assertEqual(len(resource_util_list), len(ops))

        # finer-grained test
        # reinitialize the OPs to avoid warm start.
        mock_monitor_func.return_value = None, {
            'resource': [{
                'CPU util.': {
                    'max': 0.5,
                },
            }, {
                'CPU util.': {
                    'max': 0.4,
                },
            }
            ],
            'sampling_interval': 0.2,
            'time': 0.55,
        }
        ops = [
            FixUnicodeMapper(num_proc=1),
            PerplexityFilter(num_proc=1),
            DocumentDeduplicator(num_proc=1),
        ]
        resource_util_list2 = Adapter.execute_and_probe(ds, ops, sample_interval=0.2)
        logger.info(f'interval=\t0.5\t0.2')
        for item1, item2 in zip(resource_util_list, resource_util_list2):
            logger.info(f'         \t{len(item1["resource"])}\t{len(item2["resource"])}')
            self.assertLessEqual(len(item1['resource']), len(item2['resource']))

        datasets.enable_caching()

    def test_execute_and_probe_on_empty_ops(self):
        datasets.disable_caching()
        # basic test
        ds = load_dataset('json', data_files=self.test_file, split='train').take(100)
        ops = []  # use some batched OPs later

        resource_util_list = Adapter.execute_and_probe(ds, ops)
        self.assertEqual(resource_util_list, [])

        datasets.enable_caching()

    def test_probe_small_batch(self):
        datasets.disable_caching()
        # basic test
        ds = load_dataset('json', data_files=self.test_file, split='train')
        ops = [
            FixUnicodeMapper(num_proc=1),
            PerplexityFilter(num_proc=1),
            DocumentDeduplicator(num_proc=1),
        ]  # use some batched OPs later

        adapter = Adapter({'batch_size': 100})
        resource_util_analysis_res, probe_bs = adapter.probe_small_batch(ds, ops)
        logger.info(f'Probe on batch size is [{probe_bs}].')
        self.assertEqual(len(resource_util_analysis_res), len(ops))
        for item in resource_util_analysis_res:
            self.assertIn('resource_analysis', item)

        datasets.enable_caching()

    def test_adapt_workloads(self):
        datasets.disable_caching()
        # basic test
        ds = load_dataset('json', data_files=self.test_file, split='train')
        ops = [
            FixUnicodeMapper(num_proc=1),
            PerplexityFilter(num_proc=1),
            DocumentDeduplicator(num_proc=1),
        ]  # use some batched OPs later

        adapter = Adapter({'batch_size': 100})
        adapted_batch_sizes = adapter.adapt_workloads(ds, ops)
        self.assertEqual(len(adapted_batch_sizes), len(ops))
        logger.info(adapted_batch_sizes)

        datasets.enable_caching()

    def test_adapt_workloads_multiprocessing(self):
        datasets.disable_caching()
        # basic test
        ds = load_dataset('json', data_files=self.test_file, split='train')
        ops = [
            FixUnicodeMapper(num_proc=4),
            PerplexityFilter(num_proc=4),
            DocumentDeduplicator(num_proc=4),
        ]  # use some batched OPs later

        adapter = Adapter({'batch_size': 100})
        adapted_batch_sizes = adapter.adapt_workloads(ds, ops)
        self.assertEqual(len(adapted_batch_sizes), len(ops))
        logger.info(adapted_batch_sizes)

        datasets.enable_caching()

    def test_insight_mining(self):
        simple_cfg = get_init_configs({
            'work_dir': os.path.join(self.tmp_dir, 'test_insight_mining'),
            'open_insight_mining': True,
            'np': 2,
            'op_list_to_mine': [
                'language_id_score_filter',
                'perplexity_filter',
            ],
        })
        ds = NestedDataset(load_dataset('json', data_files=self.test_file, split='train'))

        adapter = Adapter(simple_cfg)
        adapter.analyze_small_batch(ds, '0_original')
        adapter.analyze_small_batch(NestedDataset(ds.shuffle(seed=42)), '1_op')
        adapter.insight_mining()

        # check the result files
        self.assertTrue(os.path.exists(os.path.join(simple_cfg.work_dir, 'insight_mining', '0_original', '0_original_stats.jsonl')))
        self.assertTrue(os.path.exists(os.path.join(simple_cfg.work_dir, 'insight_mining', '1_op', '1_op_stats.jsonl')))
        self.assertTrue(os.path.exists(os.path.join(simple_cfg.work_dir, 'insight_mining', 'insight_mining.json')))


if __name__ == '__main__':
    unittest.main()

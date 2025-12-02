import os
import unittest
from unittest.mock import patch
from jsonargparse import Namespace
from contextlib import redirect_stdout
from io import StringIO
from data_juicer.config import init_configs
from data_juicer.core.data.dataset_builder import (rewrite_cli_datapath, 
                                                   parse_cli_datapath,
                                                   DatasetBuilder)
from data_juicer.core.data.config_validator import ConfigValidationError
from data_juicer.utils.unittest_utils import (DataJuicerTestCaseBase, TEST_TAG)
from data_juicer.core.data.load_strategy import RayLocalJsonDataLoadStrategy
import tempfile


WORK_DIR = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
             

class DatasetBuilderTest(DataJuicerTestCaseBase):

    def setUp(self):
        """Setup basic configuration for tests"""
        super().setUp()
        self.base_cfg = Namespace()
        self.base_cfg.dataset_path = None
        self.executor_type = 'default'

        self._original_cwd = os.getcwd()

        # Get the directory where this test file is located
        test_file_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(test_file_dir)

    def tearDown(self):
        super().tearDown()
        os.chdir(self._original_cwd)

    def test_rewrite_cli_datapath_local_single_file(self):
        dataset_path = os.path.join(WORK_DIR, "test_data/sample.txt")
        ans = rewrite_cli_datapath(dataset_path)
        self.assertEqual(
            {'configs': [
                {'path': dataset_path, 'type': 'local', 'weight': 1.0}]},
             ans)

    def test_rewrite_cli_datapath_local_directory(self):
        dataset_path = os.path.join(WORK_DIR, "")
        ans = rewrite_cli_datapath(dataset_path)
        self.assertEqual(
            {'configs': [
                {'path': dataset_path, 'type': 'local', 'weight': 1.0}]},
             ans)

    def test_rewrite_cli_datapath_hf(self):
        dataset_path = "hf-internal-testing/librispeech_asr_dummy"
        ans = rewrite_cli_datapath(dataset_path)
        self.assertEqual(
            {'configs': [
                {'path': 'hf-internal-testing/librispeech_asr_dummy',
                'split': 'train',
                'type': 'huggingface'}]},
             ans)

    def test_rewrite_cli_datapath_local_wrong_files(self):
        dataset_path = os.path.join(WORK_DIR, "missingDir")
        self.assertRaisesRegex(ValueError, "Unable to load the dataset",
                               rewrite_cli_datapath, dataset_path)

    def test_rewrite_cli_datapath_with_weights(self):
        dataset_path = "0.5 ./test_data/sample.jsonl ./test_data/sample.txt"
        ans = rewrite_cli_datapath(dataset_path)
        self.assertEqual(
            {'configs': [
                {'path': './test_data/sample.jsonl', 'type': 'local', 'weight': 0.5},
                {'path': './test_data/sample.txt', 'type': 'local', 'weight': 1.0}]},
            ans)

    @patch('os.path.isdir')
    @patch('os.path.isfile')
    def test_rewrite_cli_datapath_local_files(self, mock_isfile, mock_isdir):
        # Mock os.path.isdir and os.path.isfile to simulate local files
        mock_isfile.side_effect = lambda x: x.endswith('.jsonl')
        mock_isdir.side_effect = lambda x: x.endswith('_dir')

        dataset_path = "1.0 ds1.jsonl 2.0 ds2_dir 3.0 ds3.jsonl"
        expected = {
            'configs': [
                {'type': 'local', 'path': 'ds1.jsonl', 'weight': 1.0},
                {'type': 'local', 'path': 'ds2_dir', 'weight': 2.0},
                {'type': 'local', 'path': 'ds3.jsonl', 'weight': 3.0}
            ]
        }
        result = rewrite_cli_datapath(dataset_path)
        self.assertEqual(result, expected)

    def test_rewrite_cli_datapath_huggingface(self):
        dataset_path = "1.0 huggingface/dataset"
        expected = {
            'configs': [
                {'type': 'huggingface', 'path': 'huggingface/dataset', 'split': 'train'}
            ]
        }
        result = rewrite_cli_datapath(dataset_path)
        self.assertEqual(result, expected)

    def test_rewrite_cli_datapath_invalid(self):
        dataset_path = "1.0 ./invalid_path"
        with self.assertRaises(ValueError):
            rewrite_cli_datapath(dataset_path)

    def test_rewrite_cli_datapath_with_max_samples(self):
        """Test rewriting CLI datapath with max_sample_num"""
        dataset_path = "./test_data/sample.txt"
        max_sample_num = 1000
        
        result = rewrite_cli_datapath(dataset_path, max_sample_num)
        
        expected = {
            'configs': [{
                'type': 'local',
                'path': './test_data/sample.txt',
                'weight': 1.0
            }],
            'max_sample_num': 1000
        }
        self.assertEqual(result, expected)

    def test_rewrite_cli_datapath_without_max_samples(self):
        """Test rewriting CLI datapath without max_sample_num"""
        dataset_path = "./test_data/sample.txt"
        
        result = rewrite_cli_datapath(dataset_path)
        
        expected = {
            'configs': [{
                'type': 'local',
                'path': './test_data/sample.txt',
                'weight': 1.0
            }]
        }
        self.assertEqual(result, expected)

    def test_parse_cli_datapath(self):
        dataset_path = "1.0 ds1.jsonl 2.0 ds2_dir 3.0 ds3.jsonl"
        expected_paths = ['ds1.jsonl', 'ds2_dir', 'ds3.jsonl']
        expected_weights = [1.0, 2.0, 3.0]
        paths, weights = parse_cli_datapath(dataset_path)
        self.assertEqual(paths, expected_paths)
        self.assertEqual(weights, expected_weights)

    def test_parse_cli_datapath_default_weight(self):
        dataset_path = "ds1.jsonl ds2_dir 2.0 ds3.jsonl"
        expected_paths = ['ds1.jsonl', 'ds2_dir', 'ds3.jsonl']
        expected_weights = [1.0, 1.0, 2.0]
        paths, weights = parse_cli_datapath(dataset_path)
        self.assertEqual(paths, expected_paths)
        self.assertEqual(weights, expected_weights)


    def test_parse_cli_datapath_edge_cases(self):
        # Test various edge cases
        test_cases = [
            # Empty string
            ("", [], []),
            # Single path
            ("file.txt", ['file.txt'], [1.0]),
            # Multiple spaces between items
            ("file1.txt     file2.txt", ['file1.txt', 'file2.txt'], [1.0, 1.0]),
            # Tab characters
            ("file1.txt\tfile2.txt", ['file1.txt', 'file2.txt'], [1.0, 1.0]),
            # Paths with spaces in them (quoted)
            ('"my file.txt" 1.5 "other file.txt"',
            ['my file.txt', 'other file.txt'],
            [1.0, 1.5]),
        ]
        
        for input_path, expected_paths, expected_weights in test_cases:
            paths, weights = parse_cli_datapath(input_path)
            self.assertEqual(paths, expected_paths, 
                            f"Failed paths for input: {input_path}")
            self.assertEqual(weights, expected_weights, 
                            f"Failed weights for input: {input_path}")

    def test_parse_cli_datapath_valid_weights(self):
        # Test various valid weight formats
        test_cases = [
            ("1.0 file.txt", ['file.txt'], [1.0]),
            ("1.5 file1.txt 2.0 file2.txt", 
            ['file1.txt', 'file2.txt'], 
            [1.5, 2.0]),
            ("0.5 file1.txt file2.txt 1.5 file3.txt",
            ['file1.txt', 'file2.txt', 'file3.txt'],
            [0.5, 1.0, 1.5]),
            # Test integer weights
            ("1 file.txt", ['file.txt'], [1.0]),
            ("2 file1.txt 3 file2.txt",
            ['file1.txt', 'file2.txt'],
            [2.0, 3.0]),
        ]
        
        for input_path, expected_paths, expected_weights in test_cases:
            paths, weights = parse_cli_datapath(input_path)
            self.assertEqual(paths, expected_paths, 
                            f"Failed paths for input: {input_path}")
            self.assertEqual(weights, expected_weights, 
                            f"Failed weights for input: {input_path}")

    def test_parse_cli_datapath_special_characters(self):
        # Test paths with special characters
        test_cases = [
            # Paths with hyphens and underscores
            ("my-file_1.txt", ['my-file_1.txt'], [1.0]),
            # Paths with dots
            ("path/to/file.with.dots.txt", ['path/to/file.with.dots.txt'], [1.0]),
            # Paths with special characters
            ("file#1.txt", ['file#1.txt'], [1.0]),
            # Mixed case with weight
            ("1.0 Path/To/File.TXT", ['Path/To/File.TXT'], [1.0]),
            # Multiple paths with special characters
            ("2.0 file#1.txt 3.0 path/to/file-2.txt",
            ['file#1.txt', 'path/to/file-2.txt'],
            [2.0, 3.0]),
        ]
        
        for input_path, expected_paths, expected_weights in test_cases:
            paths, weights = parse_cli_datapath(input_path)
            self.assertEqual(paths, expected_paths, 
                            f"Failed paths for input: {input_path}")
            self.assertEqual(weights, expected_weights, 
                            f"Failed weights for input: {input_path}")

    def test_parse_cli_datapath_multiple_datasets(self):
        # Test multiple datasets with various weight combinations
        test_cases = [
            # Multiple datasets with all weights specified
            ("0.5 data1.txt 1.5 data2.txt 2.0 data3.txt",
            ['data1.txt', 'data2.txt', 'data3.txt'],
            [0.5, 1.5, 2.0]),
            # Mix of weighted and unweighted datasets
            ("data1.txt 1.5 data2.txt data3.txt",
            ['data1.txt', 'data2.txt', 'data3.txt'],
            [1.0, 1.5, 1.0]),
            # Multiple datasets with same weight
            ("2.0 data1.txt 2.0 data2.txt 2.0 data3.txt",
            ['data1.txt', 'data2.txt', 'data3.txt'],
            [2.0, 2.0, 2.0]),
        ]
    
        for input_path, expected_paths, expected_weights in test_cases:
            paths, weights = parse_cli_datapath(input_path)
            self.assertEqual(paths, expected_paths, 
                            f"Failed paths for input: {input_path}")
            self.assertEqual(weights, expected_weights, 
                            f"Failed weights for input: {input_path}")
        
    def test_builder_single_dataset_config(self):
        """Test handling of single dataset configuration"""
        # Setup single dataset config
        self.base_cfg.dataset = {
            'configs': [
                {
                    'type': 'local',
                    'path': 'test.jsonl'
                }
            ]
        }
        
        builder = DatasetBuilder(self.base_cfg, self.executor_type)
        
        # Verify config was converted to list
        self.assertIsInstance(builder.load_strategies, list)
        self.assertEqual(len(builder.load_strategies), 1)
        
        # Verify config content preserved
        strategy = builder.load_strategies[0]
        self.assertEqual(strategy.ds_config['type'], 'local')
        self.assertEqual(strategy.ds_config['path'], 'test.jsonl')

    def test_builder_multiple_dataset_config(self):
        """Test handling of multiple dataset configurations"""
        # Setup multiple dataset config
        self.base_cfg.dataset = {
            'configs': [
                {
                    'type': 'local',
                    'path': 'test1.jsonl'
                },
                {
                    'type': 'local',
                    'path': 'test2.jsonl'
                }
            ]
        }
        
        builder = DatasetBuilder(self.base_cfg, self.executor_type)
        
        # Verify list handling
        self.assertEqual(len(builder.load_strategies), 2)
        
        # Verify each config
        self.assertEqual(builder.load_strategies[0].ds_config['path'], 'test1.jsonl')
        self.assertEqual(builder.load_strategies[1].ds_config['path'], 'test2.jsonl')

    def test_builder_none_dataset_config(self):
        """Test handling when both dataset and dataset_path are None"""
        self.base_cfg.dataset = None
        
        with self.assertRaises(ValueError) as context:
            builder = DatasetBuilder(self.base_cfg, self.executor_type)
            builder.load_dataset()
        
        self.assertIn('Unable to load dataset', str(context.exception))

    def test_builder_mixed_dataset_types(self):
        """Test validation of mixed dataset types"""
        self.base_cfg.dataset = {
            'configs': [
                {
                    'type': 'local',
                    'path': 'test1.jsonl'
                },
                {
                    'type': 'remote',
                    'source': 'some_source'
                }
            ]
        }
        
        with self.assertRaises(ConfigValidationError) as context:
            DatasetBuilder(self.base_cfg, self.executor_type)
        
        self.assertIn('Mixture of diff types', str(context.exception))

    def test_builder_multiple_remote_datasets(self):
        """Test validation of multiple remote datasets"""
        self.base_cfg.dataset = {
            'configs': [
                {
                    'type': 'remote',
                    'source': 'source1'
                },
                {
                    'type': 'remote',
                    'source': 'source2'
                }
            ]
        }
        
        with self.assertRaises(ConfigValidationError) as context:
            DatasetBuilder(self.base_cfg, self.executor_type)
        
        self.assertIn('Multiple remote datasets', str(context.exception))

    def test_builder_empty_dataset_config(self):
        """Test handling of empty dataset configuration"""
        self.base_cfg.dataset = {
            'configs': []
        }
        
        with self.assertRaises(ConfigValidationError) as context:
            DatasetBuilder(self.base_cfg, self.executor_type)
        
        self.assertIn('non-empty list', str(context.exception))

    def test_builder_empty_dataset(self):
        """Test handling of empty dataset configuration"""
        cfg = Namespace()
        cfg.dataset = {'non-configs': 'non-configs-value'}

        with self.assertRaises(ConfigValidationError) as context:
            DatasetBuilder(cfg, self.executor_type)

        self.assertIn('should have a "configs" key', str(context.exception))

    def test_builder_invalid_dataset_config(self):
        """Test validation of multiple remote datasets"""
        self.base_cfg.dataset = {
            'configs': [
                'path1', 'path2'
            ]
        }

        with self.assertRaises(ConfigValidationError) as context:
            DatasetBuilder(self.base_cfg, self.executor_type)

        self.assertIn('should be dictionaries', str(context.exception))

    def test_builder_no_matching_strategy(self):
        """Test validation of multiple remote datasets"""
        self.base_cfg.dataset = {
            'configs': [
                {
                    'type': 'non-existing type',
                    'source': 'non-existing source',
                }
            ]
        }

        with self.assertRaises(ValueError) as context:
            DatasetBuilder(self.base_cfg, self.executor_type)

        self.assertIn('No data load strategy found', str(context.exception))

    def test_builder_invalid_dataset_config_type(self):
        """Test handling of invalid dataset configuration type"""
        self.base_cfg.dataset = "invalid_string_config"
        
        with self.assertRaises(ConfigValidationError) as context:
            DatasetBuilder(self.base_cfg, self.executor_type)
        
        self.assertIn('Dataset config should be a dictionary', 
                      str(context.exception))

    def test_builder_ondisk_config(self):
        test_config_file = os.path.join(WORK_DIR,
                                        'test_data/test_config.yaml')
        out = StringIO()
        with redirect_stdout(out):
            cfg = init_configs(args=f'--config {test_config_file}'.split())
            self.assertIsInstance(cfg, Namespace)
            self.assertEqual(cfg.project_name, 'dataset-local-json')
            self.assertEqual(cfg.dataset,
                             {'configs': [{'path': 'sample.jsonl', 'type': 'local'}]})
            self.assertEqual(not cfg.dataset_path, True)

    def test_builder_ondisk_config_list(self):
        test_config_file = os.path.join(WORK_DIR,
                                        'test_data/test_config_list.yaml')
        out = StringIO()
        with redirect_stdout(out):
            cfg = init_configs(args=f'--config {test_config_file}'.split())
            self.assertIsInstance(cfg, Namespace)
            self.assertEqual(cfg.project_name, 'dataset-local-list')
            self.assertEqual(cfg.dataset,
                             {'configs': [
                                {'path': 'sample.jsonl', 'type': 'local'},
                                {'path': 'sample.txt', 'type': 'local'}
                                ]})
            self.assertEqual(not cfg.dataset_path, True)

    def test_builder_with_max_samples(self):
        """Test DatasetBuilder with max_sample_num"""
        self.base_cfg.dataset = {
            'configs': [{
                'type': 'local',
                'path': 'test.jsonl',
                'weight': 1.0
            }],
            'max_sample_num': 1000
        }
        
        builder = DatasetBuilder(self.base_cfg, self.executor_type)
        
        self.assertEqual(len(builder.load_strategies), 1)
        self.assertEqual(builder.max_sample_num, 1000)

    def test_builder_without_max_samples(self):
        """Test DatasetBuilder without max_sample_num"""
        self.base_cfg.dataset = {
            'configs': [{
                'type': 'local',
                'path': 'test.jsonl',
                'weight': 1.0
            }]
        }
        
        builder = DatasetBuilder(self.base_cfg, self.executor_type)
        
        self.assertEqual(len(builder.load_strategies), 1)
        self.assertIsNone(builder.max_sample_num)

    def test_mixed_dataset_configs(self):
        """Test handling of mixed dataset configurations"""
        self.base_cfg.dataset = {
            'configs': [
                {
                    'type': 'local',
                    'path': 'test1.jsonl',
                    'weight': 1.0
                },
                {
                    'type': 'local',
                    'path': 'test2.jsonl',
                    'weight': 2.0
                }
            ],
            'max_sample_num': 500
        }
        
        builder = DatasetBuilder(self.base_cfg, self.executor_type)
        
        self.assertEqual(len(builder.load_strategies), 2)
        self.assertEqual(builder.max_sample_num, 500)
        self.assertEqual(
            builder.load_strategies[0].ds_config['weight'],
            1.0
        )
        self.assertEqual(
            builder.load_strategies[1].ds_config['weight'],
            2.0
        )

    def test_invalid_max_sample_num(self):
        """Test handling of invalid max_sample_num"""
        invalid_values = [-1, 0, "100", None]
        
        for value in invalid_values:
            self.base_cfg.dataset = {
                'configs': [{
                    'type': 'local',
                    'path': 'test.jsonl',
                    'weight': 1.0
                }],
                'max_sample_num': value
            }
            
            with self.assertRaises(ConfigValidationError) as context:
                DatasetBuilder(self.base_cfg, self.executor_type)
            self.assertIn('should be a positive integer', 
                          str(context.exception))

    @TEST_TAG('ray')
    def test_builder_ray_config(self):
        """Test loading Ray configuration from YAML"""
        test_config_file = os.path.join(WORK_DIR, 'test_data', 'test_config_ray.yaml')

        cfg = init_configs(args=f'--config {test_config_file}'.split())
        
        # Verify basic config
        self.assertIsInstance(cfg, Namespace)
        self.assertEqual(cfg.project_name, 'ray-demo-new-config')
        self.assertEqual(cfg.executor_type, 'ray')
        self.assertEqual(cfg.ray_address, 'auto')
        
        # Verify dataset config
        self.assertEqual(cfg.dataset, {
            'configs': [{
                'type': 'local',
                'path': './test_data/sample.jsonl'
            }]
        })
        
        # Create builder and verify
        builder = DatasetBuilder(cfg, executor_type=cfg.executor_type)
        self.assertEqual(len(builder.load_strategies), 1)
        self.assertIsInstance(builder.load_strategies[0], RayLocalJsonDataLoadStrategy)

        # Load dataset and verify schema
        dataset = builder.load_dataset()
        schema = dataset.schema()
        
        # Verify expected columns exist
        self.assertIn('text', schema.columns)
        
        # Verify schema types
        self.assertEqual(schema.column_types['text'], str) 
            

    def test_dataset_path_and_dataset_priority(self):
        """Test priority between dataset_path and dataset configuration"""
        # Create test files
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create two different test files
            path1 = os.path.join(tmp_dir, 'test1.jsonl')
            path2 = os.path.join(tmp_dir, 'test2.jsonl')
            
            # Write different content to each file
            with open(path1, 'w', encoding='utf-8') as f:
                f.write('{"text": "content from dataset_path"}\n')
            
            with open(path2, 'w', encoding='utf-8') as f:
                f.write('{"text": "content from dataset config"}\n')
            
            # Test Case 1: Only dataset_path
            cfg1 = Namespace()
            cfg1.dataset_path = path1
            cfg1.dataset = None
            
            builder1 = DatasetBuilder(cfg1, self.executor_type)
            self.assertEqual(len(builder1.load_strategies), 1)
            self.assertEqual(
                builder1.load_strategies[0].ds_config['path'],
                path1
            )
            
            # Test Case 2: Only dataset config
            cfg2 = Namespace()
            cfg2.dataset_path = None
            cfg2.dataset = {
                'configs': [{
                    'type': 'local',
                    'path': path2
                }]
            }
            
            builder2 = DatasetBuilder(cfg2, self.executor_type)
            self.assertEqual(len(builder2.load_strategies), 1)
            self.assertEqual(
                builder2.load_strategies[0].ds_config['path'],
                path2
            )
            
            # Test Case 3: Both present - dataset config should take priority
            cfg3 = Namespace()
            cfg3.dataset_path = path1
            cfg3.dataset = {
                'configs': [{
                    'type': 'local',
                    'path': path2
                }]
            }
            
            builder3 = DatasetBuilder(cfg3, self.executor_type)
            self.assertEqual(len(builder3.load_strategies), 1)
            self.assertEqual(
                builder3.load_strategies[0].ds_config['path'],
                path1,
                "dataset_path should take priority over dataset config"
            )

    def test_generated_dataset_config(self):
        cfg = Namespace()
        cfg.generated_dataset_config = {
            'type': 'EmptyFormatter',
            'length': 10,
            'feature_keys': ['text']
        }

        builder = DatasetBuilder(cfg)
        ds = builder.load_dataset()
        self.assertEqual(len(ds), 10)
        self.assertTrue(ds.contain_column('text'))

    def test_validators(self):
        cfg = Namespace()
        cfg.dataset = {
            'configs': [
                {
                    'type': 'local',
                    'path': 'test_data/sample.jsonl',
                    'weight': 1.0
                }
            ],
            'max_sample_num': 3
        }
        cfg.validators = [
            {
                'type': 'code',
            },
            {
                'type': 'required_fields',
                'required_fields': ['text']
            }
        ]

        builder = DatasetBuilder(cfg)
        self.assertEqual(len(builder.validators), 2)

        ds = builder.load_dataset()
        self.assertEqual(len(ds), 3)

    def test_invalid_validators(self):
        cfg = Namespace()
        cfg.generated_dataset_config = {
            'type': 'EmptyFormatter',
            'length': 10,
            'feature_keys': ['text']
        }
        # no type
        cfg.validators = [
            {
                'useless_key': 'useless_value'
            }
        ]

        with self.assertRaises(ValueError) as context:
            DatasetBuilder(cfg)

        self.assertIn('must have a "type" key', str(context.exception))

        # invalid type
        cfg.validators = [
            {
                'type': 'xxx'
            }
        ]

        with self.assertRaises(ValueError) as context:
            DatasetBuilder(cfg)

        self.assertIn('No data validator found', str(context.exception))


if __name__ == '__main__':
    unittest.main()

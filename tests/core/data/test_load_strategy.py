import unittest
from unittest.mock import patch, MagicMock
from data_juicer.core.data.load_strategy import (
    DataLoadStrategyRegistry, DataLoadStrategy, StrategyKey,
    DefaultLocalDataLoadStrategy,
    RayLocalJsonDataLoadStrategy,
    DefaultS3DataLoadStrategy,
    RayS3DataLoadStrategy
)
from jsonargparse import Namespace
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase, TEST_TAG
from data_juicer.config import get_default_cfg
import os
import os.path as osp
import json
import shutil
import uuid

WORK_DIR = os.path.dirname(os.path.abspath(__file__))

class MockStrategy(DataLoadStrategy):
    def load_data(self):
        pass

class DataLoadStrategyRegistryTest(DataJuicerTestCaseBase):
    @classmethod
    def setUpClass(cls):
        """Class-level setup run once before all tests"""
        super().setUpClass()
        # Save original strategies
        cls._original_strategies = DataLoadStrategyRegistry._strategies.copy()

    @classmethod
    def tearDownClass(cls):
        """Class-level cleanup run once after all tests"""
        # Restore original strategies
        DataLoadStrategyRegistry._strategies = cls._original_strategies
        super().tearDownClass()

    def setUp(self):
        """Instance-level setup run before each test"""
        super().setUp()
        # Clear strategies before each test
        DataLoadStrategyRegistry._strategies = {}

    def tearDown(self):
        """Instance-level cleanup"""
        # Reset strategies after each test
        DataLoadStrategyRegistry._strategies = {}
        super().tearDown()

    def test_exact_match(self):
        # Register a specific strategy
        DataLoadStrategyRegistry._strategies = {}
        @DataLoadStrategyRegistry.register("default", 'local', 'json')
        class TestStrategy(MockStrategy):
            pass

        # Test exact match
        strategy = DataLoadStrategyRegistry.get_strategy_class(
            "default", 'local', 'json')
        self.assertEqual(strategy, TestStrategy)

        # Test no match
        strategy = DataLoadStrategyRegistry.get_strategy_class(
            "default", 'local', 'csv')
        self.assertIsNone(strategy)

    def test_wildcard_matching(self):
        # Register strategies with different wildcard patterns
        DataLoadStrategyRegistry._strategies = {}
        @DataLoadStrategyRegistry.register("default", 'local', '*')
        class AllFilesStrategy(MockStrategy):
            pass

        @DataLoadStrategyRegistry.register("default", '*', '*')
        class AllLocalStrategy(MockStrategy):
            pass

        @DataLoadStrategyRegistry.register("*", '*', '*')
        class FallbackStrategy(MockStrategy):
            pass

        # Test specific matches
        strategy = DataLoadStrategyRegistry.get_strategy_class(
            "default", 'local', 'json')
        self.assertEqual(strategy, AllFilesStrategy)  # Should match most specific wildcard

        strategy = DataLoadStrategyRegistry.get_strategy_class(
            "default", 'remote', 'json')
        self.assertEqual(strategy, AllLocalStrategy)  # Should match second level wildcard

        strategy = DataLoadStrategyRegistry.get_strategy_class(
            "ray", 'remote', 'json')
        self.assertEqual(strategy, FallbackStrategy)  # Should match most general wildcard

    def test_specificity_priority(self):
        DataLoadStrategyRegistry._strategies = {}

        @DataLoadStrategyRegistry.register("*", '*', '*')
        class GeneralStrategy(MockStrategy):
            pass

        @DataLoadStrategyRegistry.register("default", '*', '*')
        class LocalStrategy(MockStrategy):
            pass

        @DataLoadStrategyRegistry.register("default", 'local', '*')
        class LocalOndiskStrategy(MockStrategy):
            pass

        @DataLoadStrategyRegistry.register("default", 'local', 'json')
        class ExactStrategy(MockStrategy):
            pass

        # Test matching priority
        strategy = DataLoadStrategyRegistry.get_strategy_class(
            "default", 'local', 'json')
        self.assertEqual(strategy, ExactStrategy)  # Should match exact first

        strategy = DataLoadStrategyRegistry.get_strategy_class(
            "default", 'local', 'csv')
        self.assertEqual(strategy, LocalOndiskStrategy)  # Should match one wildcard

        strategy = DataLoadStrategyRegistry.get_strategy_class(
            "default", 'remote', 'json')
        self.assertEqual(strategy, LocalStrategy)  # Should match two wildcards

        strategy = DataLoadStrategyRegistry.get_strategy_class(
            "ray", 'remote', 'json')
        self.assertEqual(strategy, GeneralStrategy)  # Should match general wildcard

    def test_pattern_matching(self):
        @DataLoadStrategyRegistry.register(
            "default", 'local', '*.json')
        class JsonStrategy(MockStrategy):
            pass

        @DataLoadStrategyRegistry.register(
            "default", 'local', 'data_[0-9]*')
        class NumberedDataStrategy(MockStrategy):
            pass

        # Test pattern matching
        strategy = DataLoadStrategyRegistry.get_strategy_class(
            "default", 'local', 'test.json')
        self.assertEqual(strategy, JsonStrategy)

        strategy = DataLoadStrategyRegistry.get_strategy_class(
            "default", 'local', 'data_123')
        self.assertEqual(strategy, NumberedDataStrategy)

        strategy = DataLoadStrategyRegistry.get_strategy_class(
            "default", 'local', 'test.csv')
        self.assertIsNone(strategy)

    def test_strategy_key_matches(self):
        DataLoadStrategyRegistry._strategies = {}

        # Test StrategyKey matching directly
        wildcard_key = StrategyKey("*", 'local', '*.json')
        specific_key = StrategyKey("default", 'local', 'test.json')
        
        # Exact keys don't match wildcards
        self.assertTrue(wildcard_key.matches(specific_key))
        self.assertFalse(specific_key.matches(wildcard_key))  

        # Test pattern matching
        pattern_key = StrategyKey("default", '*', 'data_[0-9]*')
        match_key = StrategyKey("default", 'local', 'data_123')
        no_match_key = StrategyKey("default", 'local', 'data_abc')
        
        self.assertTrue(pattern_key.matches(match_key))
        self.assertFalse(pattern_key.matches(no_match_key))

    def test_load_strategy_default_config(self):
        """Test load strategy with minimal config"""
        DataLoadStrategyRegistry._strategies = {}

        # Create minimal config
        minimal_cfg = Namespace(
            path='test/path'
        )
        
        ds_config = {
            'path': 'test/path'
        }
        
        strategy = DefaultLocalDataLoadStrategy(ds_config, minimal_cfg)
        
        # Verify defaults are used
        assert getattr(strategy.cfg, 'text_keys', ['text']) == ['text']
        assert getattr(strategy.cfg, 'suffixes', None) is None
        assert getattr(strategy.cfg, 'add_suffix', False) is False

    def test_load_strategy_full_config(self):
        """Test load strategy with full config"""
        DataLoadStrategyRegistry._strategies = {}

        # Create config with all options
        full_cfg = Namespace(
            path='test/path',
            text_keys=['content', 'title'],
            suffixes=['.txt', '.md'],
            add_suffix=True
        )
        
        ds_config = {
            'path': 'test/path'
        }
        
        strategy = DefaultLocalDataLoadStrategy(ds_config, full_cfg)
        
        # Verify all config values are used
        assert strategy.cfg.text_keys == ['content', 'title']
        assert strategy.cfg.suffixes == ['.txt', '.md']
        assert strategy.cfg.add_suffix is True

    def test_load_strategy_partial_config(self):
        """Test load strategy with partial config"""
        DataLoadStrategyRegistry._strategies = {}

        # Create config with some options
        partial_cfg = Namespace(
            path='test/path',
            text_keys=['content'],
            # suffixes and add_suffix omitted
        )
        
        ds_config = {
            'path': 'test/path'
        }
        
        strategy = DefaultLocalDataLoadStrategy(ds_config, partial_cfg)
        
        # Verify mix of specified and default values
        assert strategy.cfg.text_keys == ['content']
        assert getattr(strategy.cfg, 'suffixes', None) is None
        assert getattr(strategy.cfg, 'add_suffix', False) is False

    def test_load_strategy_empty_config(self):
        """Test load strategy with empty config"""
        DataLoadStrategyRegistry._strategies = {}
        
        # Create empty config
        empty_cfg = Namespace()
        
        ds_config = {
            'path': 'test/path'
        }
        
        strategy = DefaultLocalDataLoadStrategy(ds_config, empty_cfg)
        
        # Verify all defaults are used
        assert getattr(strategy.cfg, 'text_keys', ['text']) == ['text']
        assert getattr(strategy.cfg, 'suffixes', None) is None
        assert getattr(strategy.cfg, 'add_suffix', False) is False
        

class TestRayLocalJsonDataLoadStrategy(DataJuicerTestCaseBase):
    def setUp(self):
        """Instance-level setup run before each test"""
        super().setUp()

        cur_dir = osp.dirname(osp.abspath(__file__))
        self.tmp_dir = osp.join(cur_dir, f'tmp_{uuid.uuid4().hex}')
        os.makedirs(self.tmp_dir, exist_ok=True)

        self.cfg = get_default_cfg()
        self.cfg.ray_address = 'local'
        self.cfg.executor_type = 'ray'
        self.cfg.work_dir = self.tmp_dir

        self.test_data = [
            {'text': 'hello world'},
            {'text': 'hello world again'}
        ]

    def tearDown(self):
        if osp.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

        super().tearDown()


    @TEST_TAG('ray')
    def test_absolute_path_resolution(self):
        """Test loading from absolute path"""
        abs_path = os.path.join(WORK_DIR, 'test_data', 'sample.jsonl')
    
        # Now test the strategy
        strategy = RayLocalJsonDataLoadStrategy({
            'path': abs_path
        }, self.cfg)
        
        dataset = strategy.load_data()
        result = list(dataset.get(2))
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['text'], "Today is Sunday and it's a happy day!")
        self.assertEqual(result[1]['text'], "Today is Monday and it's a happy day!")

    @TEST_TAG('ray')
    def test_relative_path_resolution(self):
        """Test loading from relative path"""
        rel_path = './tests/core/data/test_data/sample.jsonl'
    
        # Now test the strategy
        strategy = RayLocalJsonDataLoadStrategy({
            'path': rel_path
        }, self.cfg)
        
        dataset = strategy.load_data()
        result = list(dataset.get(2))
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['text'], "Today is Sunday and it's a happy day!")
        self.assertEqual(result[1]['text'], "Today is Monday and it's a happy day!")

    @TEST_TAG('ray')
    def test_workdir_resolution(self):
        """Test path resolution for work_dir"""
        test_filename = 'test_resolution.jsonl'
        
        # Create test file in work_dir
        work_path = osp.join(self.cfg.work_dir, test_filename)
        with open(work_path, 'w', encoding='utf-8', newline='\n') as f:
            for item in self.test_data:
                f.write(json.dumps(item, ensure_ascii=False).rstrip() + '\n')
    
        strategy = RayLocalJsonDataLoadStrategy({
            'path': test_filename  # relative to work_dir
        }, self.cfg)
        
        dataset = strategy.load_data()
        result = list(dataset.get(2))
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['text'], 'hello world')

    @TEST_TAG('ray')
    def test_read_parquet(self):
        """Test read parquet"""
        rel_path = './tests/core/data/test_data/parquet/sample.parquet'
        strategy = RayLocalJsonDataLoadStrategy({
            'path': rel_path
        }, self.cfg)

        dataset = strategy.load_data()
        result = list(dataset.get(2))
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['text'], "Today is Sunday and it's a happy day!")
        self.assertEqual(result[1]['text'], "Today is Monday and it's a happy day!")

        rel_path = './tests/core/data/test_data/parquet'
        strategy = RayLocalJsonDataLoadStrategy({
            'path': rel_path
        }, self.cfg)

        dataset = strategy.load_data()
        result = list(dataset.get(2))
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['text'], "Today is Sunday and it's a happy day!")
        self.assertEqual(result[1]['text'], "Today is Monday and it's a happy day!")


class TestDefaultS3DataLoadStrategy(DataJuicerTestCaseBase):
    """Test cases for DefaultS3DataLoadStrategy"""

    def setUp(self):
        """Instance-level setup run before each test"""
        super().setUp()
        self.cfg = Namespace()
        self.cfg.text_keys = ["text"]

    def test_strategy_registration(self):
        """Test that DefaultS3DataLoadStrategy is registered correctly"""
        strategy_class = DataLoadStrategyRegistry.get_strategy_class(
            executor_type="default", data_type="remote", data_source="s3"
        )
        self.assertIsNotNone(strategy_class)
        self.assertEqual(strategy_class, DefaultS3DataLoadStrategy)

    def test_config_validation_valid_path(self):
        """Test config validation with valid S3 path"""
        ds_config = {
            "type": "remote",
            "source": "s3",
            "path": "s3://bucket-name/path/to/file.jsonl"
        }
        
        # Should not raise an error
        strategy = DefaultS3DataLoadStrategy(ds_config, self.cfg)
        self.assertEqual(strategy.ds_config["path"], "s3://bucket-name/path/to/file.jsonl")

    def test_config_validation_invalid_path(self):
        """Test config validation with invalid S3 path"""
        from data_juicer.utils.s3_utils import validate_s3_path
        
        ds_config = {
            "type": "remote",
            "source": "s3",
            "path": "https://bucket-name/path/to/file.jsonl"  # Not s3://
        }
        
        # The custom validator returns False but doesn't raise, so validation passes during init
        # But validate_s3_path will raise ValueError during load_data
        strategy = DefaultS3DataLoadStrategy(ds_config, self.cfg)
        
        # Verify that validate_s3_path raises ValueError for invalid path
        # This is what gets called in load_data()
        with self.assertRaises(ValueError) as ctx:
            validate_s3_path(ds_config["path"])
        self.assertIn("s3://", str(ctx.exception).lower())

    def test_config_validation_optional_fields(self):
        """Test config validation with optional fields"""
        ds_config = {
            "type": "remote",
            "source": "s3",
            "path": "s3://bucket-name/path/to/file.jsonl",
            "aws_access_key_id": "test_key",
            "aws_secret_access_key": "test_secret",
            "aws_session_token": "test_token",
            "aws_region": "us-east-1",
            "endpoint_url": "https://s3.amazonaws.com"
        }
        
        # Should not raise an error
        strategy = DefaultS3DataLoadStrategy(ds_config, self.cfg)
        self.assertEqual(strategy.ds_config["aws_access_key_id"], "test_key")
        self.assertEqual(strategy.ds_config["aws_secret_access_key"], "test_secret")
        self.assertEqual(strategy.ds_config["aws_session_token"], "test_token")
        self.assertEqual(strategy.ds_config["aws_region"], "us-east-1")
        self.assertEqual(strategy.ds_config["endpoint_url"], "https://s3.amazonaws.com")

    def test_path_validation(self):
        """Test S3 path validation"""
        from data_juicer.utils.s3_utils import validate_s3_path
        
        # Valid paths
        valid_paths = [
            "s3://bucket/file.jsonl",
            "s3://bucket/path/to/file.jsonl",
            "s3://my-bucket-name/data/file.json"
        ]
        for path in valid_paths:
            try:
                validate_s3_path(path)
            except ValueError:
                self.fail(f"validate_s3_path raised ValueError for valid path: {path}")
        
        # Invalid paths
        invalid_paths = [
            "https://bucket/file.jsonl",
            "file://bucket/file.jsonl",
            "/local/path/file.jsonl",
            "bucket/file.jsonl"
        ]
        for path in invalid_paths:
            with self.assertRaises(ValueError):
                validate_s3_path(path)

    @patch('data_juicer.core.data.load_strategy.datasets.load_dataset')
    @patch('data_juicer.utils.s3_utils.get_aws_credentials')
    def test_load_data_with_credentials(self, mock_get_credentials, mock_load_dataset):
        """Test load_data with credentials"""
        from datasets import Dataset
        
        # Mock credentials
        mock_get_credentials.return_value = ("test_key", "test_secret", "test_token", "us-east-1")
        
        # Create a proper Dataset object for the mock to return
        test_dataset = Dataset.from_dict({"text": ["Hello", "World"]})
        mock_load_dataset.return_value = test_dataset
        
        ds_config = {
            "type": "remote",
            "source": "s3",
            "path": "s3://bucket-name/path/to/file.jsonl",
            "aws_access_key_id": "test_key",
            "aws_secret_access_key": "test_secret"
        }
        
        strategy = DefaultS3DataLoadStrategy(ds_config, self.cfg)
        
        # Mock unify_format to return the dataset as-is
        with patch('data_juicer.core.data.load_strategy.unify_format') as mock_unify:
            mock_unify.return_value = test_dataset
            result = strategy.load_data()
            
            # Verify load_dataset was called with correct arguments
            mock_load_dataset.assert_called_once()
            call_args = mock_load_dataset.call_args
            # Check that data_files is passed (either as positional or keyword)
            # datasets.load_dataset(data_format, data_files=path, storage_options=...)
            self.assertIn('data_files', call_args[1] or call_args[0])
            if 'data_files' in call_args[1]:
                self.assertEqual(call_args[1]['data_files'], "s3://bucket-name/path/to/file.jsonl")
            self.assertIn('storage_options', call_args[1])
            storage_options = call_args[1]['storage_options']
            self.assertEqual(storage_options['key'], "test_key")
            self.assertEqual(storage_options['secret'], "test_secret")

    @patch('data_juicer.core.data.load_strategy.datasets.load_dataset')
    @patch('data_juicer.utils.s3_utils.get_aws_credentials')
    def test_load_data_without_credentials(self, mock_get_credentials, mock_load_dataset):
        """Test load_data without credentials (uses default credential chain)"""
        from datasets import Dataset
        
        # Mock no credentials
        mock_get_credentials.return_value = (None, None, None, None)
        
        # Create a proper Dataset object for the mock to return
        test_dataset = Dataset.from_dict({"text": ["Hello", "World"]})
        mock_load_dataset.return_value = test_dataset
        
        ds_config = {
            "type": "remote",
            "source": "s3",
            "path": "s3://bucket-name/path/to/file.jsonl"
        }
        
        strategy = DefaultS3DataLoadStrategy(ds_config, self.cfg)
        
        # Mock unify_format to return the dataset as-is
        with patch('data_juicer.core.data.load_strategy.unify_format') as mock_unify:
            mock_unify.return_value = test_dataset
            _ = strategy.load_data()
            
            # Verify load_dataset was called
            mock_load_dataset.assert_called_once()
            call_args = mock_load_dataset.call_args
            storage_options = call_args[1]['storage_options']
            # With no credentials, storage_options should be empty (or minimal)
            # This allows s3fs to use default credential chain (IAM role, ~/.aws/credentials)
            # Anonymous access is NOT automatically enabled
            self.assertNotIn('key', storage_options)
            self.assertNotIn('secret', storage_options)
            self.assertNotIn('token', storage_options)
            self.assertNotIn('anon', storage_options)


class TestRayS3DataLoadStrategy(DataJuicerTestCaseBase):
    """Test cases for RayS3DataLoadStrategy"""

    def setUp(self):
        """Instance-level setup run before each test"""
        super().setUp()
        self.cfg = get_default_cfg()
        self.cfg.text_keys = ["text"]

    def test_strategy_registration(self):
        """Test that RayS3DataLoadStrategy is registered correctly"""
        strategy_class = DataLoadStrategyRegistry.get_strategy_class(
            executor_type="ray", data_type="remote", data_source="s3"
        )
        self.assertIsNotNone(strategy_class)
        self.assertEqual(strategy_class, RayS3DataLoadStrategy)

    def test_config_validation_valid_path(self):
        """Test config validation with valid S3 path"""
        ds_config = {
            "type": "remote",
            "source": "s3",
            "path": "s3://bucket-name/path/to/file.jsonl"
        }
        
        # Should not raise an error
        strategy = RayS3DataLoadStrategy(ds_config, self.cfg)
        self.assertEqual(strategy.ds_config["path"], "s3://bucket-name/path/to/file.jsonl")

    def test_config_validation_invalid_path(self):
        """Test config validation with invalid S3 path"""
        from data_juicer.utils.s3_utils import validate_s3_path
        
        ds_config = {
            "type": "remote",
            "source": "s3",
            "path": "https://bucket-name/path/to/file.jsonl"  # Not s3://
        }
        
        # Verify that validate_s3_path raises ValueError for invalid path
        # This is what gets called in load_data()
        with self.assertRaises(ValueError) as ctx:
            validate_s3_path(ds_config["path"])
        self.assertIn("s3://", str(ctx.exception).lower())

    def test_config_validation_optional_fields(self):
        """Test config validation with optional fields"""
        ds_config = {
            "type": "remote",
            "source": "s3",
            "path": "s3://bucket-name/path/to/file.jsonl",
            "aws_access_key_id": "test_key",
            "aws_secret_access_key": "test_secret",
            "aws_session_token": "test_token",
            "aws_region": "us-east-1",
            "endpoint_url": "https://s3.amazonaws.com"
        }
        
        # Should not raise an error
        strategy = RayS3DataLoadStrategy(ds_config, self.cfg)
        self.assertEqual(strategy.ds_config["aws_access_key_id"], "test_key")
        self.assertEqual(strategy.ds_config["aws_secret_access_key"], "test_secret")
        self.assertEqual(strategy.ds_config["aws_session_token"], "test_token")
        self.assertEqual(strategy.ds_config["aws_region"], "us-east-1")
        self.assertEqual(strategy.ds_config["endpoint_url"], "https://s3.amazonaws.com")


if __name__ == '__main__':
    unittest.main()
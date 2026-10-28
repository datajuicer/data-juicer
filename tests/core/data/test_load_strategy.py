import json
import os
import os.path as osp
import shutil
import tempfile
import unittest
import uuid
from unittest.mock import patch, MagicMock

from jsonargparse import Namespace

from data_juicer.config import get_default_cfg
from data_juicer.core.data.load_strategy import (
    DataLoadStrategy,
    DataLoadStrategyRegistry,
    DefaultArxivDataLoadStrategy,
    DefaultCommonCrawlDataLoadStrategy,
    DefaultHuggingfaceDataLoadStrategy,
    DefaultLocalDataLoadStrategy,
    DefaultModelScopeDataLoadStrategy,
    DefaultS3DataLoadStrategy,
    DefaultWikiDataLoadStrategy,
    RayLocalJsonDataLoadStrategy,
    RayS3DataLoadStrategy,
    StrategyKey,
    DefaultHdfsDataLoadStrategy,
    RayHdfsDataLoadStrategy,
)
from data_juicer.core.data.config_validator import ConfigValidationError
from data_juicer.utils.constant import Fields
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase, TEST_TAG

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

    def test_load_strategy_full_config(self):
        """Test load strategy with full config"""
        DataLoadStrategyRegistry._strategies = {}

        # Create config with all options
        full_cfg = Namespace(
            path='test/path',
            text_keys=['content', 'title'],
            suffixes=['.txt', '.md']
        )
        
        ds_config = {
            'path': 'test/path'
        }
        
        strategy = DefaultLocalDataLoadStrategy(ds_config, full_cfg)
        
        # Verify all config values are used
        assert strategy.cfg.text_keys == ['content', 'title']
        assert strategy.cfg.suffixes == ['.txt', '.md']

    def test_load_strategy_partial_config(self):
        """Test load strategy with partial config"""
        DataLoadStrategyRegistry._strategies = {}

        # Create config with some options
        partial_cfg = Namespace(
            path='test/path',
            text_keys=['content'],
            # suffixes omitted
        )
        
        ds_config = {
            'path': 'test/path'
        }
        
        strategy = DefaultLocalDataLoadStrategy(ds_config, partial_cfg)
        
        # Verify mix of specified and default values
        assert strategy.cfg.text_keys == ['content']
        assert getattr(strategy.cfg, 'suffixes', None) is None

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

    def test_local_strategy_forwards_load_dataset_kwargs(self):
        """Test that extra kwargs passed to load_data reach datasets.load_dataset.

        Passes a ``features`` kwarg that adds an extra column not present in the
        source file.  If kwargs are forwarded correctly, the loaded dataset will
        contain that column; if not, it won't.
        """
        from datasets import Features, Value

        DataLoadStrategyRegistry._strategies = {}

        sample_path = osp.join(WORK_DIR, "test_data", "sample.jsonl")
        cfg = Namespace(text_keys=["text"], suffixes=None, process=[])
        ds_config = {"type": "local", "path": sample_path}

        extra_features = Features({"text": Value("string"), "extra": Value("string")})

        strategy = DefaultLocalDataLoadStrategy(ds_config, cfg)
        ds = strategy.load_data(num_proc=1, features=extra_features)

        self.assertIn("extra", ds.features)

    def test_registry_uses_defaults_and_specific_patterns(self):
        saved_strategies = DataLoadStrategyRegistry._strategies.copy()
        self.addCleanup(
            setattr, DataLoadStrategyRegistry, '_strategies', saved_strategies)
        DataLoadStrategyRegistry._strategies = {}

        class ConcreteDataLoadStrategy(DataLoadStrategy):
            def load_data(self, **kwargs):
                return kwargs

        @DataLoadStrategyRegistry.register("*", "local", "*.jsonl")
        class JsonLinesStrategy(ConcreteDataLoadStrategy):
            pass

        @DataLoadStrategyRegistry.register("ray", "local", "records.json?")
        class QuestionPatternStrategy(ConcreteDataLoadStrategy):
            pass

        @DataLoadStrategyRegistry.register("ray", "local", "records.jsonl")
        class ExactJsonLinesStrategy(ConcreteDataLoadStrategy):
            pass

        @DataLoadStrategyRegistry.register("default", "local", "*")
        class LocalFileStrategy(ConcreteDataLoadStrategy):
            pass

        self.assertTrue(
            StrategyKey("*", "local", "*.jsonl").matches(
                StrategyKey("ray", "local", "records.jsonl")
            )
        )
        self.assertIs(
            DataLoadStrategyRegistry.get_strategy_class("ray", "local", "records.json1"),
            QuestionPatternStrategy,
        )
        self.assertIs(
            DataLoadStrategyRegistry.get_strategy_class("ray", "local", "records.jsonl"),
            ExactJsonLinesStrategy,
        )
        self.assertIs(
            DataLoadStrategyRegistry.get_strategy_class(None, "local", "records.jsonl"),
            JsonLinesStrategy,
        )
        self.assertIs(
            DataLoadStrategyRegistry.get_strategy_class("default", "local", "records.csv"),
            LocalFileStrategy,
        )
        self.assertIsNone(
            DataLoadStrategyRegistry.get_strategy_class("ray", "remote", "records.csv")
        )

    def test_default_local_strategy_loads_jsonl_with_suffix_filter_enabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "records.jsonl")
            with open(data_path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"text": "alpha"}) + "\n")
                f.write(json.dumps({"text": "beta"}) + "\n")

            cfg = Namespace(
                text_keys=["text"], suffixes=None, process=[{"suffix_filter": {}}]
            )
            strategy = DefaultLocalDataLoadStrategy({"path": data_path}, cfg)
            dataset = strategy.load_data(num_proc=1)

        rows = dataset.to_list()
        self.assertEqual([row["text"] for row in rows], ["alpha", "beta"])
        self.assertIn(Fields.suffix, dataset.features)
        self.assertEqual({row[Fields.suffix] for row in rows}, {".jsonl"})

    @patch("data_juicer.core.data.load_strategy.datasets.load_dataset")
    def test_huggingface_strategy_forwards_load_dataset_kwargs(self, mock_load_dataset):
        """Test that extra kwargs passed to load_data reach datasets.load_dataset.

        The HuggingFace strategy calls ``datasets.load_dataset(path, ...)``
        which requires a real hub dataset, so we mock it and assert the
        ``features`` kwarg is present in the call.
        """
        from datasets import Features, Value

        DataLoadStrategyRegistry._strategies = {}

        cfg = Namespace(text_keys=["text"])
        ds_config = {"type": "huggingface", "path": "dummy/dataset"}

        mock_dataset = MagicMock()
        mock_load_dataset.return_value = mock_dataset

        extra_features = Features({"text": Value("string"), "extra": Value("string")})

        strategy = DefaultHuggingfaceDataLoadStrategy(ds_config, cfg)

        with patch("data_juicer.core.data.load_strategy.unify_format") as mock_unify:
            mock_unify.return_value = mock_dataset
            strategy.load_data(num_proc=1, features=extra_features)

        self.assertEqual(mock_load_dataset.call_args.kwargs.get("features"), extra_features)

    @patch("data_juicer.core.data.load_strategy.datasets.load_dataset")
    def test_huggingface_strategy_does_not_forward_limit(self, mock_load_dataset):
        """``limit`` is not a ``datasets.load_dataset`` argument, so it must not be forwarded."""
        from datasets import Dataset

        DataLoadStrategyRegistry._strategies = {}

        cfg = Namespace(text_keys=["text"])
        mock_load_dataset.return_value = Dataset.from_dict({"text": ["a"]})

        for ds_config in (
            {"type": "huggingface", "path": "dummy/dataset"},
            {"type": "huggingface", "path": "dummy/dataset", "limit": 2},
        ):
            strategy = DefaultHuggingfaceDataLoadStrategy(ds_config, cfg)
            with patch("data_juicer.core.data.load_strategy.unify_format"):
                strategy.load_data(num_proc=1)
            self.assertNotIn("limit", mock_load_dataset.call_args.kwargs)

    @patch("data_juicer.core.data.load_strategy.datasets.load_dataset")
    def test_huggingface_strategy_limit_caps_rows(self, mock_load_dataset):
        """``limit`` keeps only that many rows after loading."""
        from datasets import Dataset

        DataLoadStrategyRegistry._strategies = {}

        cfg = Namespace(text_keys=["text"])
        mock_load_dataset.return_value = Dataset.from_dict({"text": ["a", "b", "c"]})

        for limit, expected in ((0, []), (2, ["a", "b"]), (10, ["a", "b", "c"])):
            strategy = DefaultHuggingfaceDataLoadStrategy(
                {"type": "huggingface", "path": "dummy/dataset", "limit": limit}, cfg
            )
            with patch("data_juicer.core.data.load_strategy.unify_format", side_effect=lambda ds, **kw: ds):
                ds = strategy.load_data(num_proc=1)
            self.assertEqual(ds["text"], expected)

    @patch("data_juicer.core.data.load_strategy.datasets.load_dataset")
    def test_huggingface_strategy_limit_caps_dataset_dict(self, mock_load_dataset):
        """``limit`` is applied per split when no ``split`` is configured."""
        from datasets import Dataset, DatasetDict

        DataLoadStrategyRegistry._strategies = {}

        cfg = Namespace(text_keys=["text"])
        mock_load_dataset.return_value = DatasetDict(
            {"train": Dataset.from_dict({"text": ["a", "b", "c"]}), "test": Dataset.from_dict({"text": ["d"]})}
        )
        strategy = DefaultHuggingfaceDataLoadStrategy({"type": "huggingface", "path": "dummy/dataset", "limit": 2}, cfg)
        with patch("data_juicer.core.data.load_strategy.unify_format", side_effect=lambda ds, **kw: ds):
            ds = strategy.load_data(num_proc=1)
        self.assertEqual(ds["train"]["text"], ["a", "b"])
        self.assertEqual(ds["test"]["text"], ["d"])

    def test_huggingface_strategy_limit_validation(self):
        from data_juicer.core.data.config_validator import ConfigValidationError

        DataLoadStrategyRegistry._strategies = {}

        cfg = Namespace(text_keys=["text"])
        for limit in (-1, "10", True):
            with self.assertRaises(ConfigValidationError):
                DefaultHuggingfaceDataLoadStrategy(
                    {"type": "huggingface", "path": "dummy/dataset", "limit": limit}, cfg
                )
        for limit in (0, None):
            DefaultHuggingfaceDataLoadStrategy({"type": "huggingface", "path": "dummy/dataset", "limit": limit}, cfg)

    def test_huggingface_strategy_limit_caps_iterable(self):
        """``limit`` also applies to streaming datasets and dataset dicts."""
        from datasets import Dataset, IterableDatasetDict

        it = Dataset.from_dict({"text": ["a", "b", "c"]}).to_iterable_dataset()
        self.assertEqual([r["text"] for r in DefaultHuggingfaceDataLoadStrategy._apply_limit(it, 2)], ["a", "b"])
        out = DefaultHuggingfaceDataLoadStrategy._apply_limit(IterableDatasetDict({"train": it}), 1)
        self.assertEqual([r["text"] for r in out["train"]], ["a"])


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


class UnimplementedRemoteStrategiesTest(DataJuicerTestCaseBase):
    def test_unimplemented_remote_strategies_raise(self):
        cfg = Namespace(text_keys=["text"])
        cases = [
            (DefaultModelScopeDataLoadStrategy, {"path": "modelscope_name"}),
            (DefaultArxivDataLoadStrategy, {"path": "2026.00001"}),
            (DefaultWikiDataLoadStrategy, {"path": "enwiki"}),
            (
                DefaultCommonCrawlDataLoadStrategy,
                {"start_snapshot": "2023-06", "end_snapshot": "2023-14"},
            ),
        ]

        for strategy_cls, ds_config in cases:
            with self.subTest(strategy=strategy_cls.__name__):
                strategy = strategy_cls(ds_config, cfg)
                with self.assertRaises(NotImplementedError):
                    strategy.load_data()


class LocalAndS3PathValidationTest(DataJuicerTestCaseBase):
    def test_local_and_s3_strategies_reject_missing_or_invalid_paths(self):
        with tempfile.TemporaryDirectory() as work_dir:
            cfg = Namespace(work_dir=work_dir, text_keys=["text"])
            missing = RayLocalJsonDataLoadStrategy(
                {"path": "missing/records.jsonl"}, cfg
            )
            with self.assertRaises(FileNotFoundError):
                missing.load_data()

            default_s3 = DefaultS3DataLoadStrategy(
                {"path": "https://bucket/records.jsonl"}, cfg
            )
            with self.assertRaises(ValueError):
                default_s3.load_data()

            ray_s3 = RayS3DataLoadStrategy(
                {"path": "https://bucket/records.jsonl"}, cfg
            )
            with self.assertRaises(ValueError):
                ray_s3.load_data()


class TestDefaultHdfsDataLoadStrategy(DataJuicerTestCaseBase):
    """Test cases for DefaultHdfsDataLoadStrategy"""

    def setUp(self):
        super().setUp()
        self.cfg = get_default_cfg()
        self.cfg.text_keys = ["text"]

    def test_strategy_registration(self):
        """Test that DefaultHdfsDataLoadStrategy is registered correctly"""
        strategy_class = DataLoadStrategyRegistry.get_strategy_class(
            executor_type="default", data_type="remote", data_source="hdfs"
        )
        self.assertIsNotNone(strategy_class)
        self.assertEqual(strategy_class, DefaultHdfsDataLoadStrategy)

    def test_config_validation_valid_path(self):
        """Test config validation with valid HDFS path"""
        ds_config = {
            "type": "remote",
            "source": "hdfs",
            "path": "hdfs://namenode:8020/user/data/file.jsonl"
        }
        strategy = DefaultHdfsDataLoadStrategy(ds_config, self.cfg)
        self.assertEqual(strategy.ds_config["path"], "hdfs://namenode:8020/user/data/file.jsonl")

    def test_config_validation_invalid_path(self):
        """Test config validation with invalid HDFS path"""
        from data_juicer.utils.hdfs_utils import validate_hdfs_path

        ds_config = {
            "type": "remote",
            "source": "hdfs",
            "path": "s3://bucket/file.jsonl"  # Not hdfs://
        }

        with self.assertRaises(ValueError) as ctx:
            validate_hdfs_path(ds_config["path"])
        self.assertIn("hdfs://", str(ctx.exception))

    def test_config_validation_optional_fields(self):
        """Test config validation with optional HDFS fields"""
        ds_config = {
            "type": "remote",
            "source": "hdfs",
            "path": "hdfs://namenode:8020/user/data/file.jsonl",
            "hdfs_host": "namenode",
            "hdfs_port": 8020,
            "hdfs_user": "data_juicer",
            "hdfs_kerb_ticket": "/tmp/krb5cc",
            "hdfs_extra_conf": {"dfs.replication": "3"},
        }
        strategy = DefaultHdfsDataLoadStrategy(ds_config, self.cfg)
        self.assertEqual(strategy.ds_config["hdfs_host"], "namenode")
        self.assertEqual(strategy.ds_config["hdfs_port"], 8020)
        self.assertEqual(strategy.ds_config["hdfs_user"], "data_juicer")
        self.assertEqual(strategy.ds_config["hdfs_kerb_ticket"], "/tmp/krb5cc")
        self.assertEqual(strategy.ds_config["hdfs_extra_conf"], {"dfs.replication": "3"})

    def test_config_validation_missing_required_path(self):
        """Test that missing required path field raises error"""
        ds_config = {
            "type": "remote",
            "source": "hdfs",
        }
        with self.assertRaises((ConfigValidationError, ValueError, KeyError)):
            DefaultHdfsDataLoadStrategy(ds_config, self.cfg)


class TestRayHdfsDataLoadStrategy(DataJuicerTestCaseBase):
    """Test cases for RayHdfsDataLoadStrategy"""

    def setUp(self):
        super().setUp()
        self.cfg = get_default_cfg()
        self.cfg.text_keys = ["text"]

    def test_strategy_registration(self):
        """Test that RayHdfsDataLoadStrategy is registered correctly"""
        strategy_class = DataLoadStrategyRegistry.get_strategy_class(
            executor_type="ray", data_type="remote", data_source="hdfs"
        )
        self.assertIsNotNone(strategy_class)
        self.assertEqual(strategy_class, RayHdfsDataLoadStrategy)

    def test_config_validation_valid_path(self):
        """Test config validation with valid HDFS path"""
        ds_config = {
            "type": "remote",
            "source": "hdfs",
            "path": "hdfs://namenode:8020/user/data/file.jsonl"
        }
        strategy = RayHdfsDataLoadStrategy(ds_config, self.cfg)
        self.assertEqual(strategy.ds_config["path"], "hdfs://namenode:8020/user/data/file.jsonl")

    def test_config_validation_invalid_path(self):
        """Test config validation with invalid HDFS path"""
        from data_juicer.utils.hdfs_utils import validate_hdfs_path

        ds_config = {
            "type": "remote",
            "source": "hdfs",
            "path": "/local/path/file.jsonl"
        }

        with self.assertRaises(ValueError) as ctx:
            validate_hdfs_path(ds_config["path"])
        self.assertIn("hdfs://", str(ctx.exception))

    def test_config_validation_optional_fields(self):
        """Test config validation with optional HDFS fields"""
        ds_config = {
            "type": "remote",
            "source": "hdfs",
            "path": "hdfs://namenode:8020/user/data/file.jsonl",
            "format": "jsonl",
            "hdfs_host": "namenode",
            "hdfs_port": 8020,
            "hdfs_user": "data_juicer",
        }
        strategy = RayHdfsDataLoadStrategy(ds_config, self.cfg)
        self.assertEqual(strategy.ds_config["hdfs_host"], "namenode")
        self.assertEqual(strategy.ds_config["hdfs_port"], 8020)
        self.assertEqual(strategy.ds_config["format"], "jsonl")




class IcebergFixture:
    """Real local tables shared by Local and Ray integration tests."""

    def setUp(self):
        super().setUp()
        import importlib.util

        if importlib.util.find_spec("pyiceberg") is None:
            self.skipTest("Install py-data-juicer[iceberg] to run Iceberg integration tests")
        from pathlib import Path
        import pyarrow as pa
        from pyiceberg.catalog import load_catalog

        # CI workers share the repository mount, but not the head's /tmp.
        temp = tempfile.TemporaryDirectory(prefix="tmp_iceberg_", dir=WORK_DIR)
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.catalog_kwargs = {
            "name": "local_test",
            "type": "sql",
            "uri": f"sqlite:///{self.root / 'catalog.db'}",
            "warehouse": self.root.as_uri(),
            "py-io-impl": "pyiceberg.io.pyarrow.PyArrowFileIO",
            # Native properties must survive unchanged, even for a local catalog.
            "s3.endpoint": "http://127.0.0.1:1",
            "s3.region": "us-east-1",
        }
        self.catalog = load_catalog(**self.catalog_kwargs)
        self.catalog.create_namespace("training")
        self.arrow_schema = pa.schema([("id", pa.int64()), ("body", pa.string()), ("language", pa.string())])
        self.table = self.catalog.create_table("training.documents", schema=self.arrow_schema)
        self.initial = [
            {"id": 1, "body": "A long training document", "language": "en"},
            {"id": 2, "body": "短", "language": "zh"},
        ]
        self.added = [{"id": 3, "body": "另一篇足够长的训练文章", "language": "zh"}]
        self.table.append(pa.Table.from_pylist(self.initial, schema=self.arrow_schema))
        self.snapshot_id = self.table.current_snapshot().snapshot_id
        self.table.append(pa.Table.from_pylist(self.added, schema=self.arrow_schema))
        self.ds_config = {
            "type": "remote",
            "source": "iceberg",
            "table_identifier": "training.documents",
            "catalog_kwargs": self.catalog_kwargs,
        }
        self.cfg = Namespace(
            dataset={"configs": [self.ds_config]},
            text_keys=["body"],
            text_key="body",
            np=1,
        )

    def create_typed_table(self):
        import datetime
        from decimal import Decimal

        import pyarrow as pa

        schema = pa.schema(
            [
                ("id", pa.int64()),
                ("body", pa.string()),
                ("score", pa.float64()),
                ("price", pa.decimal128(10, 2)),
                ("blob", pa.binary()),
                ("created", pa.timestamp("us")),
                ("tags", pa.list_(pa.string())),
                ("meta", pa.struct([("source", pa.string()), ("page", pa.int32())])),
                ("note", pa.string()),
            ]
        )
        rows = [
            {
                "id": 1,
                "body": "typed row one",
                "score": 0.5,
                "price": Decimal("12.34"),
                "blob": b"\x00\xffbin",
                "created": datetime.datetime(2026, 1, 2, 3, 4, 5, 678901),
                "tags": ["a", "b"],
                "meta": {"source": "web", "page": 7},
                "note": None,
            },
            {
                "id": 2,
                "body": "typed row two",
                "score": None,
                "price": None,
                "blob": None,
                "created": None,
                "tags": [],
                "meta": None,
                "note": "present",
            },
        ]
        table = self.catalog.create_table("training.typed", schema=schema)
        table.append(pa.Table.from_pylist(rows, schema=schema))
        self.ds_config["table_identifier"] = "training.typed"
        return table.scan().to_arrow()

    def load(self, executor="default"):
        from data_juicer.core.data.dataset_builder import DatasetBuilder

        return DatasetBuilder(self.cfg, executor_type=executor).load_dataset()

    def create_media_table(self):
        import pyarrow as pa
        from PIL import Image

        media = self.root / "media"
        media.mkdir()
        Image.new("RGB", (16, 8)).save(media / "img.png")
        schema = pa.schema([("body", pa.string()), ("images", pa.list_(pa.string()))])
        table = self.catalog.create_table("training.media", schema=schema)
        table.append(pa.Table.from_pylist([{"body": "hello", "images": ["img.png"]}], schema=schema))
        self.ds_config.update(table_identifier="training.media", path=str(media / "source.jsonl"))
        return media

    @staticmethod
    def sorted_rows(rows):
        return sorted(rows, key=lambda row: row["id"])


class TestDefaultIcebergDataLoadStrategy(IcebergFixture, DataJuicerTestCaseBase):
    def test_media_anchor_reaches_image_operator_without_mutating_config(self):
        from copy import deepcopy

        from data_juicer.ops.filter.image_shape_filter import ImageShapeFilter

        media = self.create_media_table()
        run_dir = self.root / "run"
        run_dir.mkdir()
        old_cwd = os.getcwd()
        self.addCleanup(os.chdir, old_cwd)
        os.chdir(run_dir)
        # The file anchor need not exist; the image itself must exist.
        self.assertFalse((media / "source.jsonl").exists())
        for anchor in (str(media / "source.jsonl"), str(media)):
            self.ds_config["path"] = anchor
            before = deepcopy(self.cfg)
            dataset = self.load()
            self.assertEqual(dataset[0]["images"], [str(media / "img.png")])
            result = dataset.process(
                ImageShapeFilter(text_key="body", min_width=16, max_width=16,
                                 min_height=8, max_height=8, num_proc=1, skip_op_error=False),
                open_monitor=False,
            )
            self.assertEqual(len(result), 1)
            self.assertEqual(self.cfg, before)

    def test_builder_reads_latest_without_mutating_configuration(self):
        from copy import deepcopy

        before = deepcopy(self.ds_config)
        for _ in range(2):
            self.assertEqual(self.sorted_rows(self.load().to_list()), self.initial + self.added)
        self.assertEqual(self.ds_config, before)

    def test_snapshot_filter_and_projection(self):
        # The current table contains id=3, but the selected old snapshot must not.
        self.assertEqual(len(self.load()), 3)
        self.ds_config.update(
            snapshot_id=self.snapshot_id, row_filter="language == 'zh'", selected_fields=["id", "body"]
        )
        self.assertEqual(self.load().to_list(), [{"id": 2, "body": "短"}])

    def test_typed_columns_match_pyiceberg_scan(self):
        expected = self.create_typed_table()
        result = self.load()
        self.assertEqual(self.sorted_rows(result.to_list()), self.sorted_rows(expected.to_pylist()))
        actual_types = {field.name: field.type for field in result.data.table.schema}
        for field in expected.schema:
            with self.subTest(column=field.name):
                self.assertTrue(actual_types[field.name].equals(field.type), (actual_types[field.name], field.type))

    def test_empty_table_preserves_schema(self):
        self.catalog.create_table("training.empty", schema=self.arrow_schema)
        self.ds_config["table_identifier"] = "training.empty"
        result = self.load()
        self.assertEqual(result.to_list(), [])
        self.assertEqual(result.column_names, ["id", "body", "language"])

    def test_normalization_uses_configured_text_key(self):
        import pyarrow as pa

        self.table.append(pa.Table.from_pylist([{"id": 4, "body": None, "language": "zh"}], schema=self.arrow_schema))
        self.assertEqual(self.sorted_rows(self.load().to_list()), self.initial + self.added)

    def test_missing_table_preserves_original_error(self):
        from pyiceberg.exceptions import NoSuchTableError

        self.ds_config["table_identifier"] = "training.missing"
        with self.assertRaisesRegex(RuntimeError, "training.missing") as raised:
            self.load()
        self.assertIsInstance(raised.exception.__cause__, NoSuchTableError)

    def test_invalid_snapshot_is_not_silently_replaced_by_latest(self):
        self.ds_config["snapshot_id"] = 1
        with self.assertRaisesRegex(RuntimeError, "training.documents") as raised:
            self.load()
        self.assertIsInstance(raised.exception.__cause__, ValueError)

    def test_load_filter_export_round_trip(self):
        from data_juicer.core.exporter import Exporter
        from data_juicer.ops.filter.text_length_filter import TextLengthFilter

        dataset = self.load()
        result = dataset.process(TextLengthFilter(min_len=5, text_key="body", num_proc=1), open_monitor=False)
        path = self.root / "result.jsonl"
        Exporter(str(path), export_stats=False, keep_stats_in_res_ds=False).export(result)
        exported = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(self.sorted_rows(exported), [self.initial[0], self.added[0]])
        # Reading/processing must not write back to the source table.
        self.assertEqual(self.sorted_rows(self.table.scan().to_arrow().to_pylist()), self.initial + self.added)


class TestIcebergConfiguration(DataJuicerTestCaseBase):
    def test_unsupported_ray_version_fails_before_io(self):
        from unittest.mock import patch

        from data_juicer.core.data.load_strategy import RayIcebergDataLoadStrategy

        strategy = RayIcebergDataLoadStrategy(
            {"table_identifier": "training.documents", "catalog_kwargs": {}}, Namespace()
        )
        with patch("ray.__version__", "2.51.0"), patch("ray.data.read_iceberg") as reader:
            with self.assertRaisesRegex(RuntimeError, "requires Ray >=2.52.0"):
                strategy.load_data()
        reader.assert_not_called()

    def test_invalid_configuration_fails_before_io(self):
        from data_juicer.core.data.load_strategy import DefaultIcebergDataLoadStrategy, RayIcebergDataLoadStrategy

        valid = {"table_identifier": "training.documents", "catalog_kwargs": {}}
        invalid = [
            {},
            {"table_identifier": "training.documents"},
            dict(valid, table_identifier=" "),
            dict(valid, catalog_kwargs=[]),
            dict(valid, snapshot_id=True),
            dict(valid, snapshot_id=-1),
            dict(valid, snapshot_id="1"),
            dict(valid, row_filter=""),
            dict(valid, selected_fields=[]),
            dict(valid, selected_fields=[1]),
            dict(valid, path=""),
        ]
        for strategy in (DefaultIcebergDataLoadStrategy, RayIcebergDataLoadStrategy):
            for config in invalid:
                with self.subTest(strategy=strategy.__name__, config=config):
                    with self.assertRaises(ConfigValidationError):
                        strategy(config, Namespace())


class TestRayIcebergDataLoadStrategy(IcebergFixture, DataJuicerTestCaseBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        import ray

        cls.started_ray = not ray.is_initialized()
        if cls.started_ray:
            address = os.environ.get("RAY_ADDRESS", "local")
            options = {"num_cpus": 2} if address == "local" else {}
            ray.init(address=address, include_dashboard=False, **options)

    @classmethod
    def tearDownClass(cls):
        import ray

        if cls.started_ray:
            ray.shutdown()
        super().tearDownClass()

    @TEST_TAG("ray")
    def test_builder_reads_latest_without_mutating_configuration(self):
        from copy import deepcopy

        before = deepcopy(self.ds_config)
        self.cfg.override_num_blocks = 2
        result = self.load("ray")
        self.assertEqual(self.sorted_rows(result.data.take_all()), self.initial + self.added)
        self.assertEqual(self.ds_config, before)
        self.assertEqual(self.sorted_rows(self.load("ray").data.take_all()), self.initial + self.added)

    @TEST_TAG("ray")
    def test_snapshot_filter_and_projection(self):
        self.assertEqual(len(self.load("ray").data.take_all()), 3)
        self.ds_config.update(
            snapshot_id=self.snapshot_id, row_filter="language == 'zh'", selected_fields=["id", "body"]
        )
        self.assertEqual(self.load("ray").data.take_all(), [{"id": 2, "body": "短"}])

    @TEST_TAG("ray")
    def test_typed_columns_match_pyiceberg_scan(self):
        expected = self.create_typed_table()
        result = self.load("ray").data
        self.assertEqual(self.sorted_rows(result.take_all()), self.sorted_rows(expected.to_pylist()))
        actual_types = dict(zip(result.schema().names, result.schema().types))
        for field in expected.schema:
            with self.subTest(column=field.name):
                # Ray's Iceberg reader returns 64-bit offset variants (large_string etc.).
                actual = self.logical_type(actual_types[field.name])
                self.assertTrue(actual.equals(field.type), (actual_types[field.name], field.type))

    @classmethod
    def logical_type(cls, arrow_type):
        import pyarrow as pa

        if pa.types.is_large_string(arrow_type):
            return pa.string()
        if pa.types.is_large_binary(arrow_type):
            return pa.binary()
        if pa.types.is_large_list(arrow_type) or pa.types.is_list(arrow_type):
            return pa.list_(cls.logical_type(arrow_type.value_type))
        if pa.types.is_struct(arrow_type):
            return pa.struct([pa.field(f.name, cls.logical_type(f.type), f.nullable) for f in arrow_type])
        return arrow_type

    @TEST_TAG("ray")
    def test_empty_table(self):
        self.catalog.create_table("training.empty", schema=self.arrow_schema)
        self.ds_config["table_identifier"] = "training.empty"
        self.assertEqual(self.load("ray").data.take_all(), [])

    @TEST_TAG("ray")
    def test_empty_table_with_media_anchor(self):
        self.catalog.create_table("training.empty", schema=self.arrow_schema)
        self.ds_config.update(table_identifier="training.empty", path=str(self.root / "source.jsonl"))
        self.assertEqual(self.load("ray").data.take_all(), [])

    @TEST_TAG("ray")
    def test_missing_table_surfaces_error(self):
        from pyiceberg.exceptions import NoSuchTableError

        self.ds_config["table_identifier"] = "training.missing"
        with self.assertRaisesRegex(RuntimeError, "training.missing") as raised:
            self.load("ray").data.take_all()
        self.assertIsInstance(raised.exception.__cause__, NoSuchTableError)

    @TEST_TAG("ray")
    def test_invalid_snapshot_is_not_silently_replaced_by_latest(self):
        self.ds_config["snapshot_id"] = 1
        with self.assertRaisesRegex(RuntimeError, "training.documents") as raised:
            self.load("ray").data.take_all()
        self.assertIsInstance(raised.exception.__cause__, ValueError)


if __name__ == '__main__':
    unittest.main()

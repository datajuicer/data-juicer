import copy
import importlib.util
import json
import os
import os.path as osp
import shutil
import sys
import types
import unittest
from unittest.mock import patch, MagicMock

from data_juicer.core.data.load_strategy import (
    DataLoadStrategyRegistry,
    DataLoadStrategy,
    StrategyKey,
    DefaultLocalDataLoadStrategy,
    DefaultHuggingfaceDataLoadStrategy,
    RayHuggingfaceDataLoadStrategy,
    DefaultModelScopeDataLoadStrategy,
    DefaultHDFSDataLoadStrategy,
    DefaultArxivDataLoadStrategy,
    DefaultWikiDataLoadStrategy,
    DefaultCommonCrawlDataLoadStrategy,
    DefaultIcebergDataLoadStrategy,
    RayLocalJsonDataLoadStrategy,
    RayHDFSDataLoadStrategy,
    RayDeltaDataLoadStrategy,
    RayHudiDataLoadStrategy,
    RayIcebergDataLoadStrategy,
    RayPaimonDataLoadStrategy,
    DefaultS3DataLoadStrategy,
    RayS3DataLoadStrategy,
)
from data_juicer.core.ray_exporter import RayExporter
from data_juicer.config import get_default_cfg
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase, TEST_TAG
from jsonargparse import Namespace
import uuid

WORK_DIR = os.path.dirname(os.path.abspath(__file__))


def build_fake_package(name, *submodules):
    package = types.ModuleType(name)
    package.__path__ = []
    created_submodules = {}
    modules = {name: package}
    for submodule in submodules:
        module = types.ModuleType(f"{name}.{submodule}")
        setattr(package, submodule, module)
        created_submodules[submodule] = module
        modules[f"{name}.{submodule}"] = module
    return package, created_submodules, modules


def build_fake_ray_dataset_module(ray_dataset_cls=None, read_json_stream=None):
    module = types.ModuleType("data_juicer.core.data.ray_dataset")
    if ray_dataset_cls is not None:
        module.RayDataset = ray_dataset_cls
    if read_json_stream is not None:
        module.read_json_stream = read_json_stream
    return module


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

        @DataLoadStrategyRegistry.register("default", "local", "json")
        class TestStrategy(MockStrategy):
            pass

        # Test exact match
        strategy = DataLoadStrategyRegistry.get_strategy_class("default", "local", "json")
        self.assertEqual(strategy, TestStrategy)

        # Test no match
        strategy = DataLoadStrategyRegistry.get_strategy_class("default", "local", "csv")
        self.assertIsNone(strategy)

    def test_wildcard_matching(self):
        # Register strategies with different wildcard patterns
        DataLoadStrategyRegistry._strategies = {}

        @DataLoadStrategyRegistry.register("default", "local", "*")
        class AllFilesStrategy(MockStrategy):
            pass

        @DataLoadStrategyRegistry.register("default", "*", "*")
        class AllLocalStrategy(MockStrategy):
            pass

        @DataLoadStrategyRegistry.register("*", "*", "*")
        class FallbackStrategy(MockStrategy):
            pass

        # Test specific matches
        strategy = DataLoadStrategyRegistry.get_strategy_class("default", "local", "json")
        self.assertEqual(strategy, AllFilesStrategy)  # Should match most specific wildcard

        strategy = DataLoadStrategyRegistry.get_strategy_class("default", "remote", "json")
        self.assertEqual(strategy, AllLocalStrategy)  # Should match second level wildcard

        strategy = DataLoadStrategyRegistry.get_strategy_class("ray", "remote", "json")
        self.assertEqual(strategy, FallbackStrategy)  # Should match most general wildcard

    def test_specificity_priority(self):
        DataLoadStrategyRegistry._strategies = {}

        @DataLoadStrategyRegistry.register("*", "*", "*")
        class GeneralStrategy(MockStrategy):
            pass

        @DataLoadStrategyRegistry.register("default", "*", "*")
        class LocalStrategy(MockStrategy):
            pass

        @DataLoadStrategyRegistry.register("default", "local", "*")
        class LocalOndiskStrategy(MockStrategy):
            pass

        @DataLoadStrategyRegistry.register("default", "local", "json")
        class ExactStrategy(MockStrategy):
            pass

        # Test matching priority
        strategy = DataLoadStrategyRegistry.get_strategy_class("default", "local", "json")
        self.assertEqual(strategy, ExactStrategy)  # Should match exact first

        strategy = DataLoadStrategyRegistry.get_strategy_class("default", "local", "csv")
        self.assertEqual(strategy, LocalOndiskStrategy)  # Should match one wildcard

        strategy = DataLoadStrategyRegistry.get_strategy_class("default", "remote", "json")
        self.assertEqual(strategy, LocalStrategy)  # Should match two wildcards

        strategy = DataLoadStrategyRegistry.get_strategy_class("ray", "remote", "json")
        self.assertEqual(strategy, GeneralStrategy)  # Should match general wildcard

    def test_pattern_matching(self):
        @DataLoadStrategyRegistry.register("default", "local", "*.json")
        class JsonStrategy(MockStrategy):
            pass

        @DataLoadStrategyRegistry.register("default", "local", "data_[0-9]*")
        class NumberedDataStrategy(MockStrategy):
            pass

        # Test pattern matching
        strategy = DataLoadStrategyRegistry.get_strategy_class("default", "local", "test.json")
        self.assertEqual(strategy, JsonStrategy)

        strategy = DataLoadStrategyRegistry.get_strategy_class("default", "local", "data_123")
        self.assertEqual(strategy, NumberedDataStrategy)

        strategy = DataLoadStrategyRegistry.get_strategy_class("default", "local", "test.csv")
        self.assertIsNone(strategy)

    def test_strategy_key_matches(self):
        DataLoadStrategyRegistry._strategies = {}

        # Test StrategyKey matching directly
        wildcard_key = StrategyKey("*", "local", "*.json")
        specific_key = StrategyKey("default", "local", "test.json")

        # Exact keys don't match wildcards
        self.assertTrue(wildcard_key.matches(specific_key))
        self.assertFalse(specific_key.matches(wildcard_key))

        # Test pattern matching
        pattern_key = StrategyKey("default", "*", "data_[0-9]*")
        match_key = StrategyKey("default", "local", "data_123")
        no_match_key = StrategyKey("default", "local", "data_abc")

        self.assertTrue(pattern_key.matches(match_key))
        self.assertFalse(pattern_key.matches(no_match_key))

    def test_load_strategy_default_config(self):
        """Test load strategy with minimal config"""
        DataLoadStrategyRegistry._strategies = {}

        # Create minimal config
        minimal_cfg = Namespace(path="test/path")

        ds_config = {"path": "test/path"}

        strategy = DefaultLocalDataLoadStrategy(ds_config, minimal_cfg)

        # Verify defaults are used
        assert getattr(strategy.cfg, "text_keys", ["text"]) == ["text"]
        assert getattr(strategy.cfg, "suffixes", None) is None
        assert getattr(strategy.cfg, "add_suffix", False) is False

    def test_load_strategy_full_config(self):
        """Test load strategy with full config"""
        DataLoadStrategyRegistry._strategies = {}

        # Create config with all options
        full_cfg = Namespace(
            path="test/path", text_keys=["content", "title"], suffixes=[".txt", ".md"], add_suffix=True
        )

        ds_config = {"path": "test/path"}

        strategy = DefaultLocalDataLoadStrategy(ds_config, full_cfg)

        # Verify all config values are used
        assert strategy.cfg.text_keys == ["content", "title"]
        assert strategy.cfg.suffixes == [".txt", ".md"]
        assert strategy.cfg.add_suffix is True

    def test_load_strategy_partial_config(self):
        """Test load strategy with partial config"""
        DataLoadStrategyRegistry._strategies = {}

        # Create config with some options
        partial_cfg = Namespace(
            path="test/path",
            text_keys=["content"],
            # suffixes and add_suffix omitted
        )

        ds_config = {"path": "test/path"}

        strategy = DefaultLocalDataLoadStrategy(ds_config, partial_cfg)

        # Verify mix of specified and default values
        assert strategy.cfg.text_keys == ["content"]
        assert getattr(strategy.cfg, "suffixes", None) is None
        assert getattr(strategy.cfg, "add_suffix", False) is False

    def test_load_strategy_empty_config(self):
        """Test load strategy with empty config"""
        DataLoadStrategyRegistry._strategies = {}

        # Create empty config
        empty_cfg = Namespace()

        ds_config = {"path": "test/path"}

        strategy = DefaultLocalDataLoadStrategy(ds_config, empty_cfg)

        # Verify all defaults are used
        assert getattr(strategy.cfg, "text_keys", ["text"]) == ["text"]
        assert getattr(strategy.cfg, "suffixes", None) is None
        assert getattr(strategy.cfg, "add_suffix", False) is False

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


class TestRayLocalJsonDataLoadStrategy(DataJuicerTestCaseBase):
    def setUp(self):
        """Instance-level setup run before each test"""
        super().setUp()

        cur_dir = osp.dirname(osp.abspath(__file__))
        self.tmp_dir = osp.join(cur_dir, f"tmp_{uuid.uuid4().hex}")
        os.makedirs(self.tmp_dir, exist_ok=True)

        self.cfg = get_default_cfg()
        self.cfg.ray_address = "local"
        self.cfg.executor_type = "ray"
        self.cfg.work_dir = self.tmp_dir

        self.test_data = [{"text": "hello world"}, {"text": "hello world again"}]

    def tearDown(self):
        if osp.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

        super().tearDown()

    @TEST_TAG("ray")
    def test_absolute_path_resolution(self):
        """Test loading from absolute path"""
        abs_path = os.path.join(WORK_DIR, "test_data", "sample.jsonl")

        # Now test the strategy
        strategy = RayLocalJsonDataLoadStrategy({"path": abs_path}, self.cfg)

        dataset = strategy.load_data()
        result = list(dataset.get(2))

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["text"], "Today is Sunday and it's a happy day!")
        self.assertEqual(result[1]["text"], "Today is Monday and it's a happy day!")

    @TEST_TAG("ray")
    def test_relative_path_resolution(self):
        """Test loading from relative path"""
        rel_path = "./tests/core/data/test_data/sample.jsonl"

        # Now test the strategy
        strategy = RayLocalJsonDataLoadStrategy({"path": rel_path}, self.cfg)

        dataset = strategy.load_data()
        result = list(dataset.get(2))

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["text"], "Today is Sunday and it's a happy day!")
        self.assertEqual(result[1]["text"], "Today is Monday and it's a happy day!")

    @TEST_TAG("ray")
    def test_workdir_resolution(self):
        """Test path resolution for work_dir"""
        test_filename = "test_resolution.jsonl"

        # Create test file in work_dir
        work_path = osp.join(self.cfg.work_dir, test_filename)
        with open(work_path, "w", encoding="utf-8", newline="\n") as f:
            for item in self.test_data:
                f.write(json.dumps(item, ensure_ascii=False).rstrip() + "\n")

        strategy = RayLocalJsonDataLoadStrategy({"path": test_filename}, self.cfg)  # relative to work_dir

        dataset = strategy.load_data()
        result = list(dataset.get(2))
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["text"], "hello world")

    @TEST_TAG("ray")
    def test_read_parquet(self):
        """Test read parquet"""
        rel_path = "./tests/core/data/test_data/parquet/sample.parquet"
        strategy = RayLocalJsonDataLoadStrategy({"path": rel_path}, self.cfg)

        dataset = strategy.load_data()
        result = list(dataset.get(2))

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["text"], "Today is Sunday and it's a happy day!")
        self.assertEqual(result[1]["text"], "Today is Monday and it's a happy day!")

        rel_path = "./tests/core/data/test_data/parquet"
        strategy = RayLocalJsonDataLoadStrategy({"path": rel_path}, self.cfg)

        dataset = strategy.load_data()
        result = list(dataset.get(2))

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["text"], "Today is Sunday and it's a happy day!")
        self.assertEqual(result[1]["text"], "Today is Monday and it's a happy day!")


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
        ds_config = {"type": "remote", "source": "s3", "path": "s3://bucket-name/path/to/file.jsonl"}

        # Should not raise an error
        strategy = DefaultS3DataLoadStrategy(ds_config, self.cfg)
        self.assertEqual(strategy.ds_config["path"], "s3://bucket-name/path/to/file.jsonl")

    def test_config_validation_invalid_path(self):
        """Test config validation with invalid S3 path"""
        from data_juicer.utils.s3_utils import validate_s3_path

        ds_config = {"type": "remote", "source": "s3", "path": "https://bucket-name/path/to/file.jsonl"}  # Not s3://

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
            "endpoint_url": "https://s3.amazonaws.com",
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
        valid_paths = ["s3://bucket/file.jsonl", "s3://bucket/path/to/file.jsonl", "s3://my-bucket-name/data/file.json"]
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
            "bucket/file.jsonl",
        ]
        for path in invalid_paths:
            with self.assertRaises(ValueError):
                validate_s3_path(path)

    @patch("data_juicer.core.data.load_strategy.datasets.load_dataset")
    @patch("data_juicer.utils.s3_utils.get_aws_credentials")
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
            "aws_secret_access_key": "test_secret",
        }

        strategy = DefaultS3DataLoadStrategy(ds_config, self.cfg)

        # Mock unify_format to return the dataset as-is
        with patch("data_juicer.core.data.load_strategy.unify_format") as mock_unify:
            mock_unify.return_value = test_dataset
            result = strategy.load_data()

            # Verify load_dataset was called with correct arguments
            mock_load_dataset.assert_called_once()
            call_args = mock_load_dataset.call_args
            # Check that data_files is passed (either as positional or keyword)
            # datasets.load_dataset(data_format, data_files=path, storage_options=...)
            self.assertIn("data_files", call_args[1] or call_args[0])
            if "data_files" in call_args[1]:
                self.assertEqual(call_args[1]["data_files"], "s3://bucket-name/path/to/file.jsonl")
            self.assertIn("storage_options", call_args[1])
            storage_options = call_args[1]["storage_options"]
            self.assertEqual(storage_options["key"], "test_key")
            self.assertEqual(storage_options["secret"], "test_secret")

    @patch("data_juicer.core.data.load_strategy.datasets.load_dataset")
    @patch("data_juicer.utils.s3_utils.get_aws_credentials")
    def test_load_data_without_credentials(self, mock_get_credentials, mock_load_dataset):
        """Test load_data without credentials (uses default credential chain)"""
        from datasets import Dataset

        # Mock no credentials
        mock_get_credentials.return_value = (None, None, None, None)

        # Create a proper Dataset object for the mock to return
        test_dataset = Dataset.from_dict({"text": ["Hello", "World"]})
        mock_load_dataset.return_value = test_dataset

        ds_config = {"type": "remote", "source": "s3", "path": "s3://bucket-name/path/to/file.jsonl"}

        strategy = DefaultS3DataLoadStrategy(ds_config, self.cfg)

        # Mock unify_format to return the dataset as-is
        with patch("data_juicer.core.data.load_strategy.unify_format") as mock_unify:
            mock_unify.return_value = test_dataset
            _ = strategy.load_data()

            # Verify load_dataset was called
            mock_load_dataset.assert_called_once()
            call_args = mock_load_dataset.call_args
            storage_options = call_args[1]["storage_options"]
            # With no credentials, storage_options should be empty (or minimal)
            # This allows s3fs to use default credential chain (IAM role, ~/.aws/credentials)
            # Anonymous access is NOT automatically enabled
            self.assertNotIn("key", storage_options)
            self.assertNotIn("secret", storage_options)
            self.assertNotIn("token", storage_options)
            self.assertNotIn("anon", storage_options)

    @patch("data_juicer.core.data.load_strategy.datasets.load_dataset")
    @patch("data_juicer.utils.s3_utils.get_aws_credentials")
    def test_load_data_with_anon_access(self, mock_get_credentials, mock_load_dataset):
        from datasets import Dataset

        mock_get_credentials.return_value = (None, None, None, None)
        test_dataset = Dataset.from_dict({"text": ["Hello"]})
        mock_load_dataset.return_value = test_dataset

        ds_config = {
            "type": "remote",
            "source": "s3",
            "path": "s3://public-bucket/path/to/file.jsonl",
            "anon": True,
        }

        strategy = DefaultS3DataLoadStrategy(ds_config, self.cfg)

        with patch("data_juicer.core.data.load_strategy.unify_format", return_value=test_dataset):
            strategy.load_data()

        storage_options = mock_load_dataset.call_args.kwargs["storage_options"]
        self.assertTrue(storage_options["anon"])

    @patch("data_juicer.core.data.load_strategy.datasets.load_dataset")
    @patch("data_juicer.utils.s3_utils.get_aws_credentials")
    def test_load_data_concatenates_dataset_dict(self, mock_get_credentials, mock_load_dataset):
        from datasets import Dataset, DatasetDict

        mock_get_credentials.return_value = (None, None, None, None)
        dataset_dict = DatasetDict(
            {
                "train": Dataset.from_dict({"text": ["Hello"]}),
                "validation": Dataset.from_dict({"text": ["World"]}),
            }
        )
        mock_load_dataset.return_value = dataset_dict

        ds_config = {
            "type": "remote",
            "source": "s3",
            "path": "s3://bucket-name/path/to/file.jsonl",
        }

        strategy = DefaultS3DataLoadStrategy(ds_config, self.cfg)

        with (
            patch("data_juicer.core.data.NestedDataset", side_effect=lambda ds: ds),
            patch("data_juicer.core.data.load_strategy.unify_format", side_effect=lambda ds, **kwargs: ds),
        ):
            result = strategy.load_data()

        self.assertEqual(len(result), 2)

    @patch("data_juicer.core.data.load_strategy.datasets.load_dataset")
    @patch("data_juicer.utils.s3_utils.get_aws_credentials")
    def test_load_data_wraps_exceptions(self, mock_get_credentials, mock_load_dataset):
        mock_get_credentials.return_value = (None, None, None, None)
        mock_load_dataset.side_effect = ValueError("cannot read s3 object")

        ds_config = {
            "type": "remote",
            "source": "s3",
            "path": "s3://bucket-name/path/to/file.jsonl",
        }

        strategy = DefaultS3DataLoadStrategy(ds_config, self.cfg)

        with self.assertRaises(RuntimeError) as ctx:
            strategy.load_data()

        self.assertIn("Failed to load dataset from S3 path s3://bucket-name/path/to/file.jsonl", str(ctx.exception))
        self.assertIn("cannot read s3 object", str(ctx.exception))


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
        ds_config = {"type": "remote", "source": "s3", "path": "s3://bucket-name/path/to/file.jsonl"}

        # Should not raise an error
        strategy = RayS3DataLoadStrategy(ds_config, self.cfg)
        self.assertEqual(strategy.ds_config["path"], "s3://bucket-name/path/to/file.jsonl")

    def test_config_validation_invalid_path(self):
        """Test config validation with invalid S3 path"""
        from data_juicer.utils.s3_utils import validate_s3_path

        ds_config = {"type": "remote", "source": "s3", "path": "https://bucket-name/path/to/file.jsonl"}  # Not s3://

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
            "endpoint_url": "https://s3.amazonaws.com",
        }

        # Should not raise an error
        strategy = RayS3DataLoadStrategy(ds_config, self.cfg)
        self.assertEqual(strategy.ds_config["aws_access_key_id"], "test_key")
        self.assertEqual(strategy.ds_config["aws_secret_access_key"], "test_secret")
        self.assertEqual(strategy.ds_config["aws_session_token"], "test_token")
        self.assertEqual(strategy.ds_config["aws_region"], "us-east-1")
        self.assertEqual(strategy.ds_config["endpoint_url"], "https://s3.amazonaws.com")


class TestUnimplementedLoadStrategies(DataJuicerTestCaseBase):
    def test_ray_huggingface_load_data_not_implemented(self):
        strategy = RayHuggingfaceDataLoadStrategy({"path": "dummy/dataset"}, Namespace())
        with self.assertRaises(NotImplementedError):
            strategy.load_data()

    def test_default_modelscope_load_data_not_implemented(self):
        strategy = DefaultModelScopeDataLoadStrategy({"path": "dummy/path"}, Namespace())
        with self.assertRaises(NotImplementedError):
            strategy.load_data()

    def test_default_arxiv_load_data_not_implemented(self):
        strategy = DefaultArxivDataLoadStrategy({"path": "dummy/path"}, Namespace())
        with self.assertRaises(NotImplementedError):
            strategy.load_data()

    def test_default_wiki_load_data_not_implemented(self):
        strategy = DefaultWikiDataLoadStrategy({"path": "dummy/path"}, Namespace())
        with self.assertRaises(NotImplementedError):
            strategy.load_data()

    def test_default_commoncrawl_load_data_not_implemented(self):
        strategy = DefaultCommonCrawlDataLoadStrategy(
            {"start_snapshot": "2024-10", "end_snapshot": "2024-11"},
            Namespace(),
        )
        with self.assertRaises(NotImplementedError):
            strategy.load_data()


class TestDefaultHDFSDataLoadStrategy(DataJuicerTestCaseBase):
    """Test cases for DefaultHDFSDataLoadStrategy"""

    def setUp(self):
        super().setUp()
        self.cfg = Namespace(text_keys=["text"])

    def test_strategy_registration(self):
        strategy_class = DataLoadStrategyRegistry.get_strategy_class(
            executor_type="default", data_type="remote", data_source="hdfs"
        )
        self.assertIsNotNone(strategy_class)
        self.assertEqual(strategy_class, DefaultHDFSDataLoadStrategy)

    def test_create_hdfs_fs_casts_port_to_int(self):
        _, pyarrow_modules, fake_modules = build_fake_package("pyarrow", "fs")
        mock_hadoop_fs = MagicMock(return_value="fake_hdfs_fs")
        pyarrow_modules["fs"].HadoopFileSystem = mock_hadoop_fs

        ds_config = {
            "type": "remote",
            "source": "hdfs",
            "path": "hdfs://namenode:9000/data/sample.jsonl",
            "host": "namenode",
            "port": "9000",
            "user": "tester",
            "kerb_ticket": "/tmp/krb5cc",
            "extra_conf": {"dfs.replication": "1"},
        }
        strategy = DefaultHDFSDataLoadStrategy(ds_config, self.cfg)

        with patch.dict(sys.modules, fake_modules):
            result = strategy._create_hdfs_fs()

        self.assertEqual(result, "fake_hdfs_fs")
        mock_hadoop_fs.assert_called_once_with(
            host="namenode",
            port=9000,
            user="tester",
            kerb_ticket="/tmp/krb5cc",
            extra_conf={"dfs.replication": "1"},
        )

    def test_load_json_uses_hdfs_stream_and_unify_format(self):
        _, pyarrow_modules, fake_modules = build_fake_package("pyarrow", "json", "csv", "parquet")
        arrow_table = object()
        pyarrow_modules["json"].read_json = MagicMock(return_value=arrow_table)

        hdfs_stream = object()
        mock_stream_ctx = MagicMock()
        mock_stream_ctx.__enter__.return_value = hdfs_stream
        mock_stream_ctx.__exit__.return_value = False
        mock_hdfs = MagicMock()
        mock_hdfs.open_input_stream.return_value = mock_stream_ctx

        hf_dataset = MagicMock(name="hf_dataset")
        nested_dataset = MagicMock(name="nested_dataset")
        unified_dataset = MagicMock(name="unified_dataset")
        ds_config = {
            "type": "remote",
            "source": "hdfs",
            "path": "hdfs://namenode:9000/data/sample.jsonl",
        }
        strategy = DefaultHDFSDataLoadStrategy(ds_config, self.cfg)

        with (
            patch.dict(sys.modules, fake_modules),
            patch.object(strategy, "_create_hdfs_fs", return_value=mock_hdfs),
            patch("data_juicer.core.data.load_strategy.datasets.Dataset", return_value=hf_dataset) as mock_dataset,
            patch("data_juicer.core.data.NestedDataset", return_value=nested_dataset) as mock_nested,
            patch("data_juicer.core.data.load_strategy.unify_format", return_value=unified_dataset) as mock_unify,
        ):
            result = strategy.load_data(num_proc=3)

        mock_hdfs.open_input_stream.assert_called_once_with("/data/sample.jsonl")
        pyarrow_modules["json"].read_json.assert_called_once_with(hdfs_stream)
        mock_dataset.assert_called_once_with(arrow_table)
        mock_nested.assert_called_once_with(hf_dataset)
        mock_unify.assert_called_once_with(
            nested_dataset,
            text_keys=["text"],
            num_proc=3,
            global_cfg=self.cfg,
        )
        self.assertIs(result, unified_dataset)

    def test_load_text_uses_single_text_column_read_options(self):
        _, pyarrow_modules, fake_modules = build_fake_package("pyarrow", "json", "csv", "parquet")
        arrow_table = object()
        pyarrow_modules["csv"].ReadOptions = MagicMock(side_effect=lambda **kwargs: kwargs)
        pyarrow_modules["csv"].ParseOptions = MagicMock(side_effect=lambda **kwargs: kwargs)
        pyarrow_modules["csv"].read_csv = MagicMock(return_value=arrow_table)

        hdfs_stream = object()
        mock_stream_ctx = MagicMock()
        mock_stream_ctx.__enter__.return_value = hdfs_stream
        mock_stream_ctx.__exit__.return_value = False
        mock_hdfs = MagicMock()
        mock_hdfs.open_input_stream.return_value = mock_stream_ctx

        ds_config = {
            "type": "remote",
            "source": "hdfs",
            "path": "hdfs://namenode:9000/data/sample.txt",
        }
        strategy = DefaultHDFSDataLoadStrategy(ds_config, self.cfg)

        with (
            patch.dict(sys.modules, fake_modules),
            patch.object(strategy, "_create_hdfs_fs", return_value=mock_hdfs),
            patch("data_juicer.core.data.load_strategy.datasets.Dataset", return_value=MagicMock()),
            patch("data_juicer.core.data.NestedDataset", return_value=MagicMock()),
            patch("data_juicer.core.data.load_strategy.unify_format", return_value=MagicMock()),
        ):
            strategy.load_data()

        pyarrow_modules["csv"].ReadOptions.assert_called_once_with(column_names=["text"])
        pyarrow_modules["csv"].ParseOptions.assert_called_once_with(delimiter="\0", quote_char=False)
        pyarrow_modules["csv"].read_csv.assert_called_once_with(
            hdfs_stream,
            read_options={"column_names": ["text"]},
            parse_options={"delimiter": "\0", "quote_char": False},
        )

    def test_load_parquet_uses_hdfs_input_file(self):
        _, pyarrow_modules, fake_modules = build_fake_package("pyarrow", "json", "csv", "parquet")

        parquet_file = object()
        mock_file_ctx = MagicMock()
        mock_file_ctx.__enter__.return_value = parquet_file
        mock_file_ctx.__exit__.return_value = False
        mock_hdfs = MagicMock()
        mock_hdfs.open_input_file.return_value = mock_file_ctx

        arrow_table = object()
        pyarrow_modules["parquet"].read_table = MagicMock(return_value=arrow_table)
        hf_dataset = MagicMock(name="hf_dataset")
        nested_dataset = MagicMock(name="nested_dataset")
        unified_dataset = MagicMock(name="unified_dataset")
        ds_config = {
            "type": "remote",
            "source": "hdfs",
            "path": "hdfs://namenode:9000/data/sample.parquet",
        }
        strategy = DefaultHDFSDataLoadStrategy(ds_config, self.cfg)

        with (
            patch.dict(sys.modules, fake_modules),
            patch.object(strategy, "_create_hdfs_fs", return_value=mock_hdfs),
            patch("data_juicer.core.data.load_strategy.datasets.Dataset", return_value=hf_dataset) as mock_dataset,
            patch("data_juicer.core.data.NestedDataset", return_value=nested_dataset) as mock_nested,
            patch("data_juicer.core.data.load_strategy.unify_format", return_value=unified_dataset) as mock_unify,
        ):
            result = strategy.load_data(num_proc=2)

        mock_hdfs.open_input_file.assert_called_once_with("/data/sample.parquet")
        mock_hdfs.open_input_stream.assert_not_called()
        pyarrow_modules["parquet"].read_table.assert_called_once_with(parquet_file)
        mock_dataset.assert_called_once_with(arrow_table)
        mock_nested.assert_called_once_with(hf_dataset)
        mock_unify.assert_called_once_with(
            nested_dataset,
            text_keys=["text"],
            num_proc=2,
            global_cfg=self.cfg,
        )
        self.assertIs(result, unified_dataset)

    def test_load_data_wraps_reader_errors(self):
        _, pyarrow_modules, fake_modules = build_fake_package("pyarrow", "json", "csv", "parquet")
        pyarrow_modules["json"].read_json = MagicMock(side_effect=ValueError("broken json"))

        hdfs_stream = object()
        mock_stream_ctx = MagicMock()
        mock_stream_ctx.__enter__.return_value = hdfs_stream
        mock_stream_ctx.__exit__.return_value = False
        mock_hdfs = MagicMock()
        mock_hdfs.open_input_stream.return_value = mock_stream_ctx

        ds_config = {
            "type": "remote",
            "source": "hdfs",
            "path": "hdfs://namenode:9000/data/bad.jsonl",
        }
        strategy = DefaultHDFSDataLoadStrategy(ds_config, self.cfg)

        with patch.dict(sys.modules, fake_modules), patch.object(strategy, "_create_hdfs_fs", return_value=mock_hdfs):
            with self.assertRaises(RuntimeError) as ctx:
                strategy.load_data()

        self.assertIn("hdfs://namenode:9000/data/bad.jsonl", str(ctx.exception))
        self.assertIn("broken json", str(ctx.exception))


class TestRayHDFSDataLoadStrategy(DataJuicerTestCaseBase):
    """Test cases for RayHDFSDataLoadStrategy"""

    def setUp(self):
        super().setUp()
        self.cfg = Namespace(text_keys=["text"])

    def test_strategy_registration(self):
        strategy_class = DataLoadStrategyRegistry.get_strategy_class(
            executor_type="ray", data_type="remote", data_source="hdfs"
        )
        self.assertIsNotNone(strategy_class)
        self.assertEqual(strategy_class, RayHDFSDataLoadStrategy)


class TestDefaultIcebergDataLoadStrategy(DataJuicerTestCaseBase):
    """Test cases for DefaultIcebergDataLoadStrategy"""

    def setUp(self):
        super().setUp()
        self.cfg = Namespace(text_keys=["text"])

    def test_strategy_registration(self):
        strategy_class = DataLoadStrategyRegistry.get_strategy_class(
            executor_type="default", data_type="remote", data_source="iceberg"
        )
        self.assertIsNotNone(strategy_class)
        self.assertEqual(strategy_class, DefaultIcebergDataLoadStrategy)

    @patch("data_juicer.core.data.load_strategy.get_aws_credentials")
    def test_load_data_uses_catalog_and_rehydrates_s3_credentials(self, mock_get_credentials):
        _, pyiceberg_modules, fake_modules = build_fake_package("pyiceberg", "catalog")
        mock_get_credentials.return_value = (
            "resolved_key",
            "resolved_secret",
            "resolved_token",
            "resolved_region",
        )

        arrow_table = object()
        hf_dataset = MagicMock(name="hf_dataset")
        nested_dataset = MagicMock(name="nested_dataset")
        unified_dataset = MagicMock(name="unified_dataset")

        mock_table = MagicMock()
        mock_table.scan.return_value.to_arrow.return_value = arrow_table
        mock_catalog = MagicMock()
        mock_catalog.load_table.return_value = mock_table
        mock_load_catalog = MagicMock(return_value=mock_catalog)
        pyiceberg_modules["catalog"].load_catalog = mock_load_catalog

        ds_config = {
            "type": "remote",
            "source": "iceberg",
            "table_identifier": "db.sample_table",
            "catalog_kwargs": {
                "name": "demo_catalog",
                "uri": "http://catalog:8181",
                "s3.access_key_id": "cfg_key",
                "s3.secret_access_key": "cfg_secret",
                "s3.session_token": "cfg_token",
                "s3.region": "cfg_region",
                "s3.endpoint": "http://minio:9000",
            },
        }
        strategy = DefaultIcebergDataLoadStrategy(copy.deepcopy(ds_config), self.cfg)

        with (
            patch.dict(sys.modules, fake_modules),
            patch("data_juicer.core.data.load_strategy.datasets.Dataset", return_value=hf_dataset) as mock_dataset,
            patch("data_juicer.core.data.NestedDataset", return_value=nested_dataset) as mock_nested,
            patch("data_juicer.core.data.load_strategy.unify_format", return_value=unified_dataset) as mock_unify,
        ):
            result = strategy.load_data(num_proc=4)

        mock_get_credentials.assert_called_once_with(
            {
                "aws_access_key_id": "cfg_key",
                "aws_secret_access_key": "cfg_secret",
                "aws_session_token": "cfg_token",
                "aws_region": "cfg_region",
                "endpoint_url": "http://minio:9000",
            }
        )
        mock_load_catalog.assert_called_once_with(
            **{
                "name": "demo_catalog",
                "uri": "http://catalog:8181",
                "s3.access_key_id": "resolved_key",
                "s3.secret_access_key": "resolved_secret",
                "s3.session_token": "resolved_token",
                "s3.region": "resolved_region",
            }
        )
        mock_catalog.load_table.assert_called_once_with("db.sample_table")
        mock_dataset.assert_called_once_with(arrow_table)
        mock_nested.assert_called_once_with(hf_dataset)
        mock_unify.assert_called_once_with(
            nested_dataset,
            text_keys=["text"],
            num_proc=4,
            global_cfg=self.cfg,
        )
        self.assertIs(result, unified_dataset)

    def test_load_data_raises_runtime_error_when_pyiceberg_missing(self):
        real_import = __import__

        def raising_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "pyiceberg.catalog" or name.startswith("pyiceberg"):
                raise ImportError("No module named 'pyiceberg'")
            return real_import(name, globals, locals, fromlist, level)

        ds_config = {
            "type": "remote",
            "source": "iceberg",
            "table_identifier": "db.sample_table",
            "catalog_kwargs": {"name": "demo_catalog"},
        }
        strategy = DefaultIcebergDataLoadStrategy(ds_config, self.cfg)

        with patch("builtins.__import__", side_effect=raising_import):
            with self.assertRaises(RuntimeError) as ctx:
                strategy.load_data()

        self.assertIn("pyiceberg is not installed", str(ctx.exception))

    @patch("data_juicer.core.data.load_strategy.get_aws_credentials", return_value=(None, None, None, None))
    def test_load_data_wraps_catalog_errors(self, _mock_get_credentials):
        _, pyiceberg_modules, fake_modules = build_fake_package("pyiceberg", "catalog")
        pyiceberg_modules["catalog"].load_catalog = MagicMock(side_effect=ValueError("catalog unavailable"))

        ds_config = {
            "type": "remote",
            "source": "iceberg",
            "table_identifier": "db.sample_table",
            "catalog_kwargs": {"name": "demo_catalog"},
        }
        strategy = DefaultIcebergDataLoadStrategy(ds_config, self.cfg)

        with patch.dict(sys.modules, fake_modules):
            with self.assertRaises(RuntimeError) as ctx:
                strategy.load_data()

        self.assertIn("Failed to load Iceberg table db.sample_table", str(ctx.exception))
        self.assertIn("catalog unavailable", str(ctx.exception))


class TestRayIcebergDataLoadStrategy(DataJuicerTestCaseBase):
    """Test cases for RayIcebergDataLoadStrategy"""

    def setUp(self):
        super().setUp()
        self.cfg = Namespace(text_keys=["text"])

    def test_strategy_registration(self):
        strategy_class = DataLoadStrategyRegistry.get_strategy_class(
            executor_type="ray", data_type="remote", data_source="iceberg"
        )
        self.assertIsNotNone(strategy_class)
        self.assertEqual(strategy_class, RayIcebergDataLoadStrategy)

    @patch("data_juicer.utils.model_utils.filter_arguments")
    @patch("data_juicer.core.data.load_strategy.get_aws_credentials")
    def test_load_data_filters_args_and_calls_ray_read_iceberg(self, mock_get_credentials, mock_filter_arguments):
        mock_get_credentials.return_value = (
            "resolved_key",
            "resolved_secret",
            "resolved_token",
            "resolved_region",
        )

        raw_dataset = object()
        wrapped_dataset = object()
        mock_read_iceberg = MagicMock(return_value=raw_dataset)
        fake_ray = types.ModuleType("ray")
        fake_ray.data = types.SimpleNamespace(read_iceberg=mock_read_iceberg)

        mock_ray_dataset = MagicMock(return_value=wrapped_dataset)
        fake_ray_dataset_module = build_fake_ray_dataset_module(ray_dataset_cls=mock_ray_dataset)

        expected_catalog_kwargs = {
            "name": "demo_catalog",
            "uri": "http://catalog:8181",
            "s3.access_key_id": "resolved_key",
            "s3.secret_access_key": "resolved_secret",
            "s3.session_token": "resolved_token",
            "s3.region": "resolved_region",
        }
        mock_filter_arguments.return_value = {
            "table_identifier": "db.sample_table",
            "catalog_kwargs": expected_catalog_kwargs,
        }

        ds_config = {
            "type": "remote",
            "source": "iceberg",
            "table_identifier": "db.sample_table",
            "path": "warehouse/sample_table",
            "catalog_kwargs": {
                "name": "demo_catalog",
                "uri": "http://catalog:8181",
                "s3.access_key_id": "cfg_key",
                "s3.secret_access_key": "cfg_secret",
                "s3.session_token": "cfg_token",
                "s3.region": "cfg_region",
                "s3.endpoint": "http://minio:9000",
            },
        }
        strategy = RayIcebergDataLoadStrategy(copy.deepcopy(ds_config), self.cfg)

        with patch.dict(
            sys.modules,
            {
                "ray": fake_ray,
                "data_juicer.core.data.ray_dataset": fake_ray_dataset_module,
            },
        ):
            result = strategy.load_data()

        mock_get_credentials.assert_called_once_with(
            {
                "aws_access_key_id": "cfg_key",
                "aws_secret_access_key": "cfg_secret",
                "aws_session_token": "cfg_token",
                "aws_region": "cfg_region",
                "endpoint_url": "http://minio:9000",
            }
        )
        self.assertEqual(strategy.ds_config["catalog_kwargs"], expected_catalog_kwargs)
        mock_filter_arguments.assert_called_once_with(fake_ray.data.read_iceberg, strategy.ds_config)
        mock_read_iceberg.assert_called_once_with(**mock_filter_arguments.return_value)
        mock_ray_dataset.assert_called_once_with(
            raw_dataset,
            dataset_path="warehouse/sample_table",
            cfg=self.cfg,
        )
        self.assertIs(result, wrapped_dataset)


class TestRayPaimonDataLoadStrategy(DataJuicerTestCaseBase):
    def setUp(self):
        super().setUp()
        self.cfg = Namespace(text_keys=["text"])

    def test_strategy_registration(self):
        strategy_class = DataLoadStrategyRegistry.get_strategy_class(
            executor_type="ray", data_type="remote", data_source="paimon"
        )
        self.assertIsNotNone(strategy_class)
        self.assertEqual(strategy_class, RayPaimonDataLoadStrategy)

    def test_load_data_creates_catalog_and_returns_ray_dataset(self):
        raw_dataset = object()
        wrapped_dataset = object()
        splits = [object(), object()]

        mock_to_ray = MagicMock(return_value=raw_dataset)
        mock_table_read = MagicMock()
        mock_table_read.to_ray = mock_to_ray

        mock_scan_plan = MagicMock()
        mock_scan_plan.splits.return_value = splits
        mock_table_scan = MagicMock()
        mock_table_scan.plan.return_value = mock_scan_plan

        mock_read_builder = MagicMock()
        mock_read_builder.new_scan.return_value = mock_table_scan
        mock_read_builder.new_read.return_value = mock_table_read

        mock_table = MagicMock()
        mock_table.new_read_builder.return_value = mock_read_builder

        mock_catalog = MagicMock()
        mock_catalog.get_table.return_value = mock_table

        mock_catalog_create = MagicMock(return_value=mock_catalog)

        fake_pypaimon = types.ModuleType("pypaimon")
        fake_pypaimon.__path__ = []
        fake_pypaimon_catalog = types.ModuleType("pypaimon.catalog")
        fake_pypaimon_catalog.__path__ = []
        fake_pypaimon_catalog_factory = types.ModuleType("pypaimon.catalog.catalog_factory")
        fake_pypaimon_catalog_factory.CatalogFactory = types.SimpleNamespace(create=mock_catalog_create)
        fake_pypaimon.catalog = fake_pypaimon_catalog
        fake_pypaimon_catalog.catalog_factory = fake_pypaimon_catalog_factory

        mock_ray_dataset = MagicMock(return_value=wrapped_dataset)
        fake_ray_dataset_module = build_fake_ray_dataset_module(ray_dataset_cls=mock_ray_dataset)

        ds_config = {
            "type": "remote",
            "source": "paimon",
            "table_identifier": "db.sample_table",
            "path": "warehouse/sample_table",
            "catalog_options": {
                "metastore": "filesystem",
                "warehouse": "oss://bucket/path",
            },
        }
        strategy = RayPaimonDataLoadStrategy(copy.deepcopy(ds_config), self.cfg)

        with patch.dict(
            sys.modules,
            {
                "pypaimon": fake_pypaimon,
                "pypaimon.catalog": fake_pypaimon_catalog,
                "pypaimon.catalog.catalog_factory": fake_pypaimon_catalog_factory,
                "data_juicer.core.data.ray_dataset": fake_ray_dataset_module,
            },
        ):
            result = strategy.load_data()

        mock_catalog_create.assert_called_once_with(ds_config["catalog_options"])
        mock_catalog.get_table.assert_called_once_with("db.sample_table")
        mock_to_ray.assert_called_once_with(splits)
        mock_ray_dataset.assert_called_once_with(
            raw_dataset,
            dataset_path="warehouse/sample_table",
            cfg=self.cfg,
        )
        self.assertIs(result, wrapped_dataset)

    def test_load_data_raises_runtime_error_when_pypaimon_missing(self):
        real_import = __import__

        def raising_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "pypaimon.catalog.catalog_factory" or name.startswith("pypaimon"):
                raise ImportError("No module named 'pypaimon'")
            return real_import(name, globals, locals, fromlist, level)

        ds_config = {
            "type": "remote",
            "source": "paimon",
            "table_identifier": "db.sample_table",
            "catalog_options": {"metastore": "filesystem"},
        }
        strategy = RayPaimonDataLoadStrategy(ds_config, self.cfg)

        with patch("builtins.__import__", side_effect=raising_import):
            with self.assertRaises(RuntimeError) as ctx:
                strategy.load_data()

        self.assertIn("pypaimon is not installed", str(ctx.exception))

    def test_load_data_wraps_catalog_errors(self):
        mock_catalog_create = MagicMock(side_effect=ValueError("catalog unavailable"))

        fake_pypaimon = types.ModuleType("pypaimon")
        fake_pypaimon.__path__ = []
        fake_pypaimon_catalog = types.ModuleType("pypaimon.catalog")
        fake_pypaimon_catalog.__path__ = []
        fake_pypaimon_catalog_factory = types.ModuleType("pypaimon.catalog.catalog_factory")
        fake_pypaimon_catalog_factory.CatalogFactory = types.SimpleNamespace(create=mock_catalog_create)
        fake_pypaimon.catalog = fake_pypaimon_catalog
        fake_pypaimon_catalog.catalog_factory = fake_pypaimon_catalog_factory

        fake_ray_dataset_module = build_fake_ray_dataset_module(ray_dataset_cls=MagicMock())

        ds_config = {
            "type": "remote",
            "source": "paimon",
            "table_identifier": "db.sample_table",
            "catalog_options": {"metastore": "filesystem"},
        }
        strategy = RayPaimonDataLoadStrategy(ds_config, self.cfg)

        with patch.dict(
            sys.modules,
            {
                "pypaimon": fake_pypaimon,
                "pypaimon.catalog": fake_pypaimon_catalog,
                "pypaimon.catalog.catalog_factory": fake_pypaimon_catalog_factory,
                "data_juicer.core.data.ray_dataset": fake_ray_dataset_module,
            },
        ):
            with self.assertRaises(RuntimeError) as ctx:
                strategy.load_data()

        self.assertIn("Failed to load Paimon table db.sample_table in Ray", str(ctx.exception))
        self.assertIn("catalog unavailable", str(ctx.exception))


class TestTableFormatsLoadStrategy(DataJuicerTestCaseBase):
    def setUp(self):
        super().setUp()
        self.cfg = Namespace(text_keys=["text"])
        self.tmp_dir = osp.join(WORK_DIR, "tmp", self.__class__.__name__, self._testMethodName)
        os.makedirs(self.tmp_dir, exist_ok=True)
        self.records = [
            {
                "text": "alpha sample",
                "doc_id": 1,
                "lang": "zh",
                "source_rank": 10,
                "quality_score": 0.98,
                "has_image": True,
                "payload": b"\x00\x01alpha-payload",
                "tags": ["news", "alpha"],
                "payloads": [
                    b"\x00\x01alpha-part-1",
                    b"alpha-part-2",
                ],
            },
            {
                "text": "beta sample",
                "doc_id": 2,
                "lang": "en",
                "source_rank": 20,
                "quality_score": 0.87,
                "has_image": False,
                "payload": b"beta-\xff-payload",
                "tags": ["review"],
                "payloads": [
                    b"beta-part-1",
                ],
            },
            {
                "text": "gamma sample",
                "doc_id": 3,
                "lang": "fr",
                "source_rank": 30,
                "quality_score": 0.91,
                "has_image": True,
                "payload": b"gamma-payload-\x10\x11",
                "tags": ["summary", "gamma", "archive"],
                "payloads": [
                    b"gamma-part-1",
                    b"\x10\x11gamma-part-2",
                ],
            },
        ]

    def tearDown(self):
        if osp.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)
        super().tearDown()

    def _build_ray_dataset(self):
        import ray

        return ray.data.from_items(copy.deepcopy(self.records))

    @staticmethod
    def _get_paimon_column_stats_for_complex_types(record_batch, column_name):
        import pyarrow as pa
        import pyarrow.compute as pc

        column_array = record_batch.column(column_name)
        column_type = column_array.type
        if (
            pa.types.is_list(column_type)
            or pa.types.is_large_list(column_type)
            or pa.types.is_map(column_type)
            or pa.types.is_struct(column_type)
        ):
            return {
                "min_values": None,
                "max_values": None,
                "null_counts": column_array.null_count,
            }
        if column_array.null_count == len(column_array):
            return {
                "min_values": None,
                "max_values": None,
                "null_counts": column_array.null_count,
            }
        return {
            "min_values": pc.min(column_array).as_py(),
            "max_values": pc.max(column_array).as_py(),
            "null_counts": column_array.null_count,
        }

    @TEST_TAG("ray")
    def test_default_iceberg_load_data_reads_local_catalog_table(self):
        if importlib.util.find_spec("pyiceberg") is None:
            self.skipTest("pyiceberg is required for local Iceberg integration tests")

        from pyiceberg.catalog import load_catalog

        warehouse = osp.join(self.tmp_dir, "iceberg_warehouse")
        os.makedirs(warehouse, exist_ok=True)
        catalog_kwargs = {
            "name": "local",
            "type": "sql",
            "uri": f"sqlite:///{osp.join(self.tmp_dir, 'iceberg_catalog.db')}",
            "warehouse": f"file://{warehouse}",
        }
        table_identifier = "default.integration_documents"

        catalog = load_catalog(**catalog_kwargs)
        catalog.create_namespace_if_not_exists("default")

        exporter = RayExporter(
            export_path=osp.join(self.tmp_dir, "iceberg_fallback.jsonl"),
            export_type="iceberg",
            table_identifier=table_identifier,
            catalog_kwargs=copy.deepcopy(catalog_kwargs),
        )
        exporter.export(self._build_ray_dataset())

        strategy = DefaultIcebergDataLoadStrategy(
            {
                "type": "remote",
                "source": "iceberg",
                "table_identifier": table_identifier,
                "catalog_kwargs": copy.deepcopy(catalog_kwargs),
            },
            self.cfg,
        )

        loaded = strategy.load_data()

        self.assertDatasetEqual(loaded.to_list(), self.records)

    @TEST_TAG("ray")
    def test_ray_iceberg_load_data_reads_local_catalog_table(self):
        if importlib.util.find_spec("pyiceberg") is None:
            self.skipTest("pyiceberg is required for local Iceberg integration tests")

        from pyiceberg.catalog import load_catalog

        warehouse = osp.join(self.tmp_dir, "iceberg_warehouse")
        os.makedirs(warehouse, exist_ok=True)
        catalog_kwargs = {
            "name": "local",
            "type": "sql",
            "uri": f"sqlite:///{osp.join(self.tmp_dir, 'iceberg_catalog.db')}",
            "warehouse": f"file://{warehouse}",
        }
        table_identifier = "default.integration_documents"

        catalog = load_catalog(**catalog_kwargs)
        catalog.create_namespace_if_not_exists("default")

        exporter = RayExporter(
            export_path=osp.join(self.tmp_dir, "iceberg_fallback.jsonl"),
            export_type="iceberg",
            table_identifier=table_identifier,
            catalog_kwargs=copy.deepcopy(catalog_kwargs),
        )
        exporter.export(self._build_ray_dataset())

        strategy = RayIcebergDataLoadStrategy(
            {
                "type": "remote",
                "source": "iceberg",
                "table_identifier": table_identifier,
                "catalog_kwargs": copy.deepcopy(catalog_kwargs),
            },
            self.cfg,
        )

        loaded = strategy.load_data()

        self.assertDatasetEqual(loaded.to_list(), self.records)

    @TEST_TAG("ray")
    def test_ray_paimon_load_data_reads_local_catalog_table(self):
        if importlib.util.find_spec("pypaimon") is None:
            self.skipTest("pypaimon is required for local Paimon integration tests")

        from pypaimon.catalog.catalog_factory import CatalogFactory
        from pypaimon.write.writer.data_writer import DataWriter

        warehouse = osp.join(self.tmp_dir, "paimon_warehouse")
        os.makedirs(warehouse, exist_ok=True)
        catalog_options = {
            "warehouse": f"file://{warehouse}",
        }
        table_identifier = "default.integration_documents"

        catalog = CatalogFactory.create(copy.deepcopy(catalog_options))
        catalog.create_database("default", ignore_if_exists=True)

        exporter = RayExporter(
            export_path=osp.join(self.tmp_dir, "paimon_fallback.jsonl"),
            export_type="paimon",
            table_identifier=table_identifier,
            catalog_options=copy.deepcopy(catalog_options),
        )
        with patch.object(
            DataWriter,
            "_get_column_stats",
            side_effect=self._get_paimon_column_stats_for_complex_types,
            autospec=False,
        ):
            exporter.export(self._build_ray_dataset())

        strategy = RayPaimonDataLoadStrategy(
            {
                "type": "remote",
                "source": "paimon",
                "table_identifier": table_identifier,
                "catalog_options": copy.deepcopy(catalog_options),
            },
            self.cfg,
        )

        loaded = strategy.load_data()

        self.assertDatasetEqual(loaded.to_list(), self.records)


class TestRayDeltaDataLoadStrategy(DataJuicerTestCaseBase):
    def setUp(self):
        super().setUp()
        self.cfg = Namespace(text_keys=["text"])

    def test_strategy_registration(self):
        strategy_class = DataLoadStrategyRegistry.get_strategy_class(
            executor_type="ray", data_type="remote", data_source="delta"
        )
        self.assertIsNotNone(strategy_class)
        self.assertEqual(strategy_class, RayDeltaDataLoadStrategy)

    @patch("data_juicer.utils.model_utils.filter_arguments")
    @patch("data_juicer.core.data.load_strategy.create_pyarrow_s3_filesystem")
    def test_load_data_calls_read_delta_with_filesystem(self, mock_create_fs, mock_filter_arguments):
        raw_dataset = object()
        wrapped_dataset = object()
        s3_fs = object()
        mock_create_fs.return_value = s3_fs

        fake_ray, ray_modules, fake_ray_modules = build_fake_package("ray", "data")
        mock_read_delta = MagicMock(return_value=raw_dataset)
        ray_modules["data"].read_delta = mock_read_delta

        mock_ray_dataset = MagicMock(return_value=wrapped_dataset)
        fake_ray_dataset_module = build_fake_ray_dataset_module(ray_dataset_cls=mock_ray_dataset)

        def filter_side_effect(func, args_dict):
            self.assertIs(func, mock_read_delta)
            self.assertEqual(args_dict["path"], "s3://bucket/table")
            self.assertIs(args_dict["filesystem"], s3_fs)
            return {"path": args_dict["path"], "filesystem": args_dict["filesystem"]}

        mock_filter_arguments.side_effect = filter_side_effect

        ds_config = {
            "type": "remote",
            "source": "delta",
            "path": "s3://bucket/table",
            "endpoint_url": "http://minio:9000",
        }
        strategy = RayDeltaDataLoadStrategy(ds_config, self.cfg)

        with patch.dict(
            sys.modules,
            {
                **fake_ray_modules,
                "data_juicer.core.data.ray_dataset": fake_ray_dataset_module,
            },
        ):
            result = strategy.load_data()

        mock_create_fs.assert_called_once_with(ds_config)
        mock_read_delta.assert_called_once_with(path="s3://bucket/table", filesystem=s3_fs)
        mock_ray_dataset.assert_called_once_with(raw_dataset, dataset_path="s3://bucket/table", cfg=self.cfg)
        self.assertIs(result, wrapped_dataset)

    @patch("data_juicer.utils.model_utils.filter_arguments", return_value={"path": "s3://bucket/table"})
    @patch("data_juicer.core.data.load_strategy.create_pyarrow_s3_filesystem", return_value=object())
    def test_load_data_wraps_read_delta_errors(self, _mock_create_fs, _mock_filter_arguments):
        fake_ray, ray_modules, fake_ray_modules = build_fake_package("ray", "data")
        ray_modules["data"].read_delta = MagicMock(side_effect=ValueError("delta unavailable"))

        fake_ray_dataset_module = build_fake_ray_dataset_module(ray_dataset_cls=MagicMock())

        ds_config = {
            "type": "remote",
            "source": "delta",
            "path": "s3://bucket/table",
        }
        strategy = RayDeltaDataLoadStrategy(ds_config, self.cfg)

        with patch.dict(
            sys.modules,
            {
                **fake_ray_modules,
                "data_juicer.core.data.ray_dataset": fake_ray_dataset_module,
            },
        ):
            with self.assertRaises(RuntimeError) as ctx:
                strategy.load_data()

        self.assertIn("Failed to load Delta Lake table from path s3://bucket/table in Ray", str(ctx.exception))
        self.assertIn("delta unavailable", str(ctx.exception))


class TestRayHudiDataLoadStrategy(DataJuicerTestCaseBase):
    def setUp(self):
        super().setUp()
        self.cfg = Namespace(text_keys=["text"])

    def test_strategy_registration(self):
        strategy_class = DataLoadStrategyRegistry.get_strategy_class(
            executor_type="ray", data_type="remote", data_source="hudi"
        )
        self.assertIsNotNone(strategy_class)
        self.assertEqual(strategy_class, RayHudiDataLoadStrategy)

    @patch("data_juicer.utils.model_utils.filter_arguments")
    def test_load_data_calls_read_hudi(self, mock_filter_arguments):
        raw_dataset = object()
        wrapped_dataset = object()

        fake_ray, ray_modules, fake_ray_modules = build_fake_package("ray", "data")
        mock_read_hudi = MagicMock(return_value=raw_dataset)
        ray_modules["data"].read_hudi = mock_read_hudi

        mock_ray_dataset = MagicMock(return_value=wrapped_dataset)
        fake_ray_dataset_module = build_fake_ray_dataset_module(ray_dataset_cls=mock_ray_dataset)

        mock_filter_arguments.return_value = {"table_uri": "s3://bucket/hudi_table"}

        ds_config = {
            "type": "remote",
            "source": "hudi",
            "table_uri": "s3://bucket/hudi_table",
            "path": "warehouse/hudi_table",
        }
        strategy = RayHudiDataLoadStrategy(ds_config, self.cfg)

        with patch.dict(
            sys.modules,
            {
                **fake_ray_modules,
                "data_juicer.core.data.ray_dataset": fake_ray_dataset_module,
            },
        ):
            result = strategy.load_data()

        mock_filter_arguments.assert_called_once_with(mock_read_hudi, ds_config)
        mock_read_hudi.assert_called_once_with(table_uri="s3://bucket/hudi_table")
        mock_ray_dataset.assert_called_once_with(raw_dataset, dataset_path="warehouse/hudi_table", cfg=self.cfg)
        self.assertIs(result, wrapped_dataset)

    @patch("data_juicer.utils.model_utils.filter_arguments", return_value={"table_uri": "s3://bucket/hudi_table"})
    def test_load_data_wraps_read_hudi_errors(self, _mock_filter_arguments):
        fake_ray, ray_modules, fake_ray_modules = build_fake_package("ray", "data")
        ray_modules["data"].read_hudi = MagicMock(side_effect=ValueError("hudi unavailable"))

        fake_ray_dataset_module = build_fake_ray_dataset_module(ray_dataset_cls=MagicMock())

        ds_config = {
            "type": "remote",
            "source": "hudi",
            "table_uri": "s3://bucket/hudi_table",
        }
        strategy = RayHudiDataLoadStrategy(ds_config, self.cfg)

        with patch.dict(
            sys.modules,
            {
                **fake_ray_modules,
                "data_juicer.core.data.ray_dataset": fake_ray_dataset_module,
            },
        ):
            with self.assertRaises(RuntimeError) as ctx:
                strategy.load_data()

        self.assertIn("Failed to load Hudi table from s3://bucket/hudi_table in Ray", str(ctx.exception))
        self.assertIn("hudi unavailable", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

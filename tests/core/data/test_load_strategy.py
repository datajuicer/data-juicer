import unittest
from data_juicer.core.data.load_strategy import (
    DataLoadStrategyRegistry, DataLoadStrategy, StrategyKey,
    DefaultLocalDataLoadStrategy,
    RayLocalJsonDataLoadStrategy
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


if __name__ == '__main__':
    unittest.main()
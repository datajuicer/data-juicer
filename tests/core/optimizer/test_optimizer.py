import unittest
from typing import List, Dict, Any
from unittest.mock import Mock, patch

from data_juicer.core.pipeline_ast import PipelineAST, OpNode, OpType
from data_juicer.core.optimizer.optimizer import PipelineOptimizer
from data_juicer.core.optimizer.strategy import OptimizationStrategy
from data_juicer.core.optimizer.filter_fusion_strategy import FilterFusionStrategy

class MockStrategy(OptimizationStrategy):
    """Mock strategy for testing."""
    
    def __init__(self, name: str = "mock_strategy"):
        super().__init__(name)
        self.optimize_called = False
    
    def optimize(self, ast: PipelineAST) -> PipelineAST:
        self.optimize_called = True
        return ast

class TestPipelineOptimizer(unittest.TestCase):
    def setUp(self):
        self.ast = PipelineAST()
        self.config = {
            'process': [
                {
                    'name': 'clean_copyright_mapper',
                    'type': 'mapper',
                    'config': {'key': 'value1'}
                },
                {
                    'name': 'language_id_score_filter',
                    'type': 'filter',
                    'config': {'key': 'value2'}
                },
                {
                    'name': 'alphanumeric_filter',
                    'type': 'filter',
                    'config': {'key': 'value3'}
                }
            ]
        }
        self.ast.build_from_config(self.config)
    
    def test_init_default_strategies(self):
        """Test initialization with default strategies."""
        optimizer = PipelineOptimizer()
        self.assertEqual(len(optimizer.strategies), 1)
        self.assertIsInstance(optimizer.strategies[0], FilterFusionStrategy)
    
    def test_init_custom_strategies(self):
        """Test initialization with custom strategies."""
        strategies = [MockStrategy("strategy1"), MockStrategy("strategy2")]
        optimizer = PipelineOptimizer(strategies)
        self.assertEqual(len(optimizer.strategies), 2)
        self.assertEqual(optimizer.strategies, strategies)
    
    def test_add_strategy(self):
        """Test adding a new strategy."""
        optimizer = PipelineOptimizer()
        strategy = MockStrategy()
        optimizer.add_strategy(strategy)
        self.assertEqual(len(optimizer.strategies), 2)
        self.assertIn(strategy, optimizer.strategies)
    
    def test_remove_strategy(self):
        """Test removing a strategy by name."""
        optimizer = PipelineOptimizer()
        strategy = MockStrategy("test_strategy")
        optimizer.add_strategy(strategy)
        optimizer.remove_strategy("test_strategy")
        self.assertEqual(len(optimizer.strategies), 1)
        self.assertNotIn(strategy, optimizer.strategies)
    
    def test_optimize_empty_pipeline(self):
        """Test optimization of an empty pipeline."""
        optimizer = PipelineOptimizer()
        empty_ast = PipelineAST()
        optimized_ast = optimizer.optimize(empty_ast)
        self.assertIsNone(optimized_ast.root)
    
    def test_optimize_with_multiple_strategies(self):
        """Test optimization with multiple strategies."""
        strategy1 = MockStrategy("strategy1")
        strategy2 = MockStrategy("strategy2")
        optimizer = PipelineOptimizer([strategy1, strategy2])
        
        optimized_ast = optimizer.optimize(self.ast)
        
        self.assertTrue(strategy1.optimize_called)
        self.assertTrue(strategy2.optimize_called)
        self.assertIsNotNone(optimized_ast.root)
    
    def test_get_strategy(self):
        """Test getting a strategy by name."""
        strategy = MockStrategy("test_strategy")
        optimizer = PipelineOptimizer([strategy])
        
        found_strategy = optimizer.get_strategy("test_strategy")
        self.assertEqual(found_strategy, strategy)
        
        not_found = optimizer.get_strategy("nonexistent")
        self.assertIsNone(not_found)
    
    def test_get_strategy_names(self):
        """Test getting names of all strategies."""
        strategies = [
            MockStrategy("strategy1"),
            MockStrategy("strategy2")
        ]
        optimizer = PipelineOptimizer(strategies)
        
        names = optimizer.get_strategy_names()
        self.assertEqual(names, ["strategy1", "strategy2"])
    
    def test_clear_strategies(self):
        """Test clearing all strategies."""
        optimizer = PipelineOptimizer()
        optimizer.clear_strategies()
        self.assertEqual(len(optimizer.strategies), 0)

if __name__ == '__main__':
    unittest.main() 
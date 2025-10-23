import unittest
from unittest.mock import Mock, patch

from data_juicer.core.optimizer.filter_fusion_strategy import FilterFusionStrategy
from data_juicer.core.pipeline_ast import PipelineAST, OpNode, OpType

class TestFilterFusionStrategy(unittest.TestCase):
    def setUp(self):
        self.strategy = FilterFusionStrategy()
        self.ast = PipelineAST()
        
        # Sample probe results
        self.probe_results = {
            'language_id_score_filter': {'speed': 0.5},
            'clean_copyright_mapper': {'speed': 0.3},
            'alphanumeric_filter': {'speed': 0.4}
        }
        
        # Create a sample pipeline configuration
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
    
    def test_optimize_single_filter(self):
        """Test optimization with a single filter."""
        # Build AST with single filter
        config = {
            'process': [
                {
                    'name': 'language_id_score_filter',
                    'type': 'filter',
                    'config': {'key': 'value'}
                }
            ]
        }
        self.ast.build_from_config(config)
        
        # Apply optimization
        optimized_ast = self.strategy.optimize(self.ast)
        
        # Verify the filter remains unchanged
        chain = self.strategy._get_operation_chain(optimized_ast.root)
        self.assertEqual(len(chain), 1)
        self.assertEqual(chain[0].name, 'language_id_score_filter')
    
    def test_optimize_multiple_filters(self):
        """Test optimization with multiple filters."""
        # Build AST with multiple filters
        self.ast.build_from_config(self.config)
        
        # Apply optimization
        optimized_ast = self.strategy.optimize(self.ast)
        
        # Verify filters are fused
        chain = self.strategy._get_operation_chain(optimized_ast.root)
        self.assertEqual(len(chain), 2)  # mapper + fused filters
        
        # Check that the fused node contains both filters
        fused_node = chain[1]
        self.assertTrue(fused_node.name.startswith('fused_'))
        self.assertEqual(len(fused_node.original_ops), 2)
    
    def test_optimize_with_probe_results(self):
        """Test optimization with probe results for speed-based sorting."""
        strategy = FilterFusionStrategy(probe_results=self.probe_results)
        self.ast.build_from_config(self.config)
        
        # Apply optimization
        optimized_ast = strategy.optimize(self.ast)
        
        # Verify filters are fused and sorted by speed
        chain = strategy._get_operation_chain(optimized_ast.root)
        fused_node = chain[1]
        
        # Check that filters are sorted by speed
        original_ops = fused_node.original_ops
        self.assertEqual(original_ops[0].name, 'language_id_score_filter')  # speed: 0.5
        self.assertEqual(original_ops[1].name, 'alphanumeric_filter')  # speed: 0.4
    
    def test_optimize_empty_pipeline(self):
        """Test optimization with an empty pipeline."""
        optimized_ast = self.strategy.optimize(self.ast)
        self.assertIsNone(optimized_ast.root)
    
    def test_create_fused_filter_node(self):
        """Test creation of a fused filter node."""
        # Create sample filter nodes
        filter1 = OpNode('filter1', OpType.FILTER, {'key1': 'value1'})
        filter2 = OpNode('filter2', OpType.FILTER, {'key2': 'value2'})
        
        # Create fused node
        fused_node = self.strategy._create_fused_filter_node(
            'fused_filters',
            [filter1, filter2]
        )
        
        # Verify fused node properties
        self.assertEqual(fused_node.name, 'fused_filters')
        self.assertEqual(fused_node.op_type, OpType.FILTER)
        self.assertEqual(fused_node.config, {'key1': 'value1', 'key2': 'value2'})
        self.assertEqual(fused_node.original_ops, [filter1, filter2])

if __name__ == '__main__':
    unittest.main() 
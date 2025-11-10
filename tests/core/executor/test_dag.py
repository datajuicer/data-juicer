#!/usr/bin/env python3
"""
Tests for DAG Execution functionality.

This module tests the AST-based pipeline parsing and DAG execution planning
capabilities of the Data-Juicer system.
"""

import os
import tempfile
import unittest

from data_juicer.core.pipeline_ast import PipelineAST, OpType
from data_juicer.core.pipeline_dag import PipelineDAG, DAGNodeStatus
from data_juicer.core.executor.dag_execution_strategies import (
    NonPartitionedDAGStrategy, 
    PartitionedDAGStrategy,
    is_global_operation
)


class TestPipelineAST(unittest.TestCase):
    """Test AST parsing functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.ast = PipelineAST()
        self.sample_config = {
            "process": [
                {"text_length_filter": {"min_len": 10, "max_len": 1000}},
                {"character_repetition_filter": {"repetition_ratio": 0.2}},
                {"words_num_filter": {"min_num": 5, "max_num": 1000}},
                {"language_id_score_filter": {"lang": "en", "min_score": 0.8}},
                {"document_deduplicator": {"method": "exact"}},
                {"text_cleaning_mapper": {"text_key": "text"}},
                {"text_splitter_mapper": {"text_key": "text", "max_length": 512}},
            ]
        }

    def test_ast_build_from_config(self):
        """Test building AST from configuration."""
        self.ast.build_from_config(self.sample_config)
        
        self.assertIsNotNone(self.ast.root)
        self.assertEqual(self.ast.root.op_type, OpType.ROOT)
        # Root should have 1 child (first operation), creating a skewed tree
        self.assertEqual(len(self.ast.root.children), 1)
        
        # Verify the skewed tree has exactly 7 layers (7 operations)
        layer_count = 0
        current_node = self.ast.root
        while current_node.children:
            current_node = current_node.children[0]  # Take the first (and only) child
            layer_count += 1
        
        self.assertEqual(layer_count, 7, f"Expected 7 layers in skewed tree, got {layer_count}")

    def test_ast_operation_type_detection(self):
        """Test operation type detection."""
        self.ast.build_from_config(self.sample_config)
        
        # Check that operation types are correctly detected in the skewed tree
        # Traverse the tree to collect all operation types
        op_types = []
        current_node = self.ast.root
        while current_node.children:
            current_node = current_node.children[0]  # Take the first (and only) child
            op_types.append(current_node.op_type)
        
        expected_types = [
            OpType.FILTER,      # text_length_filter
            OpType.FILTER,      # character_repetition_filter
            OpType.FILTER,      # words_num_filter
            OpType.FILTER,      # language_id_score_filter
            OpType.DEDUPLICATOR, # document_deduplicator
            OpType.MAPPER,      # text_cleaning_mapper
            OpType.MAPPER,      # text_splitter_mapper
        ]
        
        self.assertEqual(op_types, expected_types)
        
        # Verify we have exactly 7 operations in the chain
        self.assertEqual(len(op_types), 7, f"Expected 7 operations, got {len(op_types)}")
        
        # Print the tree structure for verification
        print(f"\nSkewed tree structure:")
        print(f"Root has {len(self.ast.root.children)} child(ren)")
        
        current_node = self.ast.root
        layer = 0
        while current_node.children:
            current_node = current_node.children[0]
            layer += 1
            print(f"Layer {layer}: {current_node.name} ({current_node.op_type.value})")
        
        print(f"Total layers: {layer}")

    def test_ast_skewed_tree_structure(self):
        """Test that AST creates a proper skewed tree with exactly 7 layers."""
        self.ast.build_from_config(self.sample_config)
        
        # Verify root has exactly 1 child
        self.assertEqual(len(self.ast.root.children), 1, "Root should have exactly 1 child")
        
        # Traverse the skewed tree and count layers
        layers = []
        current_node = self.ast.root
        layer_count = 0
        
        while current_node.children:
            current_node = current_node.children[0]  # Take the first (and only) child
            layer_count += 1
            layers.append({
                'layer': layer_count,
                'name': current_node.name,
                'type': current_node.op_type.value
            })
        
        # Verify we have exactly 7 layers
        self.assertEqual(layer_count, 7, f"Expected 7 layers, got {layer_count}")
        
        # Verify each layer has the expected operation
        expected_operations = [
            "text_length_filter",
            "character_repetition_filter", 
            "words_num_filter",
            "language_id_score_filter",
            "document_deduplicator",
            "text_cleaning_mapper",
            "text_splitter_mapper"
        ]
        
        for i, (layer_info, expected_name) in enumerate(zip(layers, expected_operations)):
            self.assertEqual(layer_info['name'], expected_name, 
                           f"Layer {i+1} should be {expected_name}, got {layer_info['name']}")
            self.assertEqual(layer_info['layer'], i+1, 
                           f"Layer number should be {i+1}, got {layer_info['layer']}")
        
        # Print detailed structure for verification
        print(f"\nDetailed skewed tree structure:")
        print(f"Root (layer 0): {self.ast.root.name} ({self.ast.root.op_type.value})")
        for layer_info in layers:
            print(f"Layer {layer_info['layer']}: {layer_info['name']} ({layer_info['type']})")
        
        print(f"✅ Verified: Skewed tree has exactly {layer_count} layers")

    def test_ast_visualization(self):
        """Test AST visualization."""
        self.ast.build_from_config(self.sample_config)
        viz = self.ast.visualize()
        
        self.assertIsInstance(viz, str)
        self.assertIn("root", viz)
        self.assertIn("text_length_filter", viz)


class TestPipelineDAG(unittest.TestCase):
    """Test DAG execution planning functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.dag = PipelineDAG(self.temp_dir)
        self.ast = PipelineAST()
        self.sample_config = {
            "process": [
                {"text_length_filter": {"min_len": 10, "max_len": 1000}},
                {"character_repetition_filter": {"repetition_ratio": 0.2}},
                {"words_num_filter": {"min_num": 5, "max_num": 1000}},
                {"language_id_score_filter": {"lang": "en", "min_score": 0.8}},
                {"document_deduplicator": {"method": "exact"}},
                {"text_cleaning_mapper": {"text_key": "text"}},
                {"text_splitter_mapper": {"text_key": "text", "max_length": 512}},
            ]
        }

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_dag_build_from_ast(self):
        """Test building DAG from AST."""
        self.ast.build_from_config(self.sample_config)
        self.dag.build_from_ast(self.ast)
        
        self.assertGreater(len(self.dag.nodes), 0)
        self.assertGreater(len(self.dag.execution_plan), 0)

    def test_dag_execution_plan_save_load(self):
        """Test saving and loading execution plans."""
        self.ast.build_from_config(self.sample_config)
        self.dag.build_from_ast(self.ast)
        
        # Save execution plan
        plan_path = self.dag.save_execution_plan()
        self.assertTrue(os.path.exists(plan_path))
        
        # Load execution plan
        new_dag = PipelineDAG(self.temp_dir)
        success = new_dag.load_execution_plan()
        self.assertTrue(success)
        self.assertEqual(len(new_dag.nodes), len(self.dag.nodes))

    def test_dag_visualization(self):
        """Test DAG visualization."""
        self.ast.build_from_config(self.sample_config)
        self.dag.build_from_ast(self.ast)
        
        viz = self.dag.visualize()
        self.assertIsInstance(viz, str)
        self.assertIn("DAG Execution Plan", viz)

    def test_dag_node_status_management(self):
        """Test DAG node status management."""
        self.ast.build_from_config(self.sample_config)
        self.dag.build_from_ast(self.ast)
        
        # Get first node
        first_node_id = list(self.dag.nodes.keys())[0]
        
        # Test status transitions
        self.dag.mark_node_started(first_node_id)
        self.assertEqual(self.dag.nodes[first_node_id].status, DAGNodeStatus.RUNNING)
        
        self.dag.mark_node_completed(first_node_id, 1.5)
        self.assertEqual(self.dag.nodes[first_node_id].status, DAGNodeStatus.COMPLETED)
        self.assertEqual(self.dag.nodes[first_node_id].actual_duration, 1.5)

    def test_dag_execution_summary(self):
        """Test DAG execution summary generation."""
        self.ast.build_from_config(self.sample_config)
        self.dag.build_from_ast(self.ast)
        
        summary = self.dag.get_execution_summary()
        
        self.assertIn("total_nodes", summary)
        self.assertIn("completed_nodes", summary)
        self.assertIn("pending_nodes", summary)
        self.assertIn("parallel_groups_count", summary)


class TestDAGExecutionStrategies(unittest.TestCase):
    """Test DAG execution strategies."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock operations
        class MockOperation:
            def __init__(self, name):
                self._name = name
        
        self.operations = [
            MockOperation("text_length_filter"),
            MockOperation("character_repetition_filter"),
            MockOperation("document_deduplicator"),
            MockOperation("text_cleaning_mapper"),
        ]

    def test_non_partitioned_strategy(self):
        """Test non-partitioned execution strategy."""
        strategy = NonPartitionedDAGStrategy()
        
        # Generate nodes
        nodes = strategy.generate_dag_nodes(self.operations)
        self.assertEqual(len(nodes), 4)
        
        # Test node ID generation
        node_id = strategy.get_dag_node_id("text_length_filter", 0)
        self.assertEqual(node_id, "op_001_text_length_filter")
        
        # Test dependency building
        strategy.build_dependencies(nodes, self.operations)
        self.assertGreater(len(nodes["op_002_character_repetition_filter"]["dependencies"]), 0)

    def test_partitioned_strategy(self):
        """Test partitioned execution strategy."""
        strategy = PartitionedDAGStrategy(num_partitions=2)
        
        # Generate nodes
        nodes = strategy.generate_dag_nodes(self.operations)
        self.assertGreater(len(nodes), 4)  # Should have partition-specific nodes
        
        # Test node ID generation
        node_id = strategy.get_dag_node_id("text_length_filter", 0, partition_id=1)
        self.assertEqual(node_id, "op_001_text_length_filter_partition_1")

    def test_global_operation_detection(self):
        """Test global operation detection."""
        class MockDeduplicator:
            def __init__(self):
                self._name = "document_deduplicator"
        
        class MockFilter:
            def __init__(self):
                self._name = "text_length_filter"
        
        deduplicator = MockDeduplicator()
        filter_op = MockFilter()
        
        self.assertTrue(is_global_operation(deduplicator))
        self.assertFalse(is_global_operation(filter_op))


if __name__ == "__main__":
    unittest.main() 
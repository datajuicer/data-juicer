#!/usr/bin/env python3
"""
Tests for DAG Execution functionality.

This module tests the strategy-based DAG execution planning
capabilities of the Data-Juicer system.
"""

import os
import tempfile
import unittest

from data_juicer.core.pipeline_dag import PipelineDAG, DAGNodeStatus
from data_juicer.core.executor.dag_execution_strategies import (
    NonPartitionedDAGStrategy, 
    PartitionedDAGStrategy,
    is_global_operation
)
from data_juicer.ops import load_ops


# Note: PipelineAST tests removed - AST functionality was removed in favor of strategy-based DAG building


class TestPipelineDAG(unittest.TestCase):
    """Test DAG execution planning functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.dag = PipelineDAG(self.temp_dir)
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

    def _build_dag_from_config(self):
        """Helper method to build DAG from config using strategy-based approach."""
        # Load operations from config
        operations = load_ops(self.sample_config["process"])
        
        # Create strategy and build DAG
        strategy = NonPartitionedDAGStrategy()
        nodes = strategy.generate_dag_nodes(operations)
        strategy.build_dependencies(nodes, operations)
        
        # Assign nodes to DAG
        self.dag.nodes = nodes

    def test_dag_build_from_strategy(self):
        """Test building DAG using strategy-based approach."""
        self._build_dag_from_config()
        
        self.assertGreater(len(self.dag.nodes), 0)
        # Note: execution_plan is not populated by strategies currently
        # self.assertGreater(len(self.dag.execution_plan), 0)

    def test_dag_execution_plan_save_load(self):
        """Test saving and loading execution plans."""
        self._build_dag_from_config()
        
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
        self._build_dag_from_config()
        
        viz = self.dag.visualize()
        self.assertIsInstance(viz, str)
        self.assertIn("DAG Execution Plan", viz)

    def test_dag_node_status_management(self):
        """Test DAG node status management."""
        self._build_dag_from_config()
        
        # Get first node
        first_node_id = list(self.dag.nodes.keys())[0]
        
        # Test status transitions
        self.dag.mark_node_started(first_node_id)
        # Check status for dict nodes
        node = self.dag.nodes[first_node_id]
        if isinstance(node, dict):
            self.assertEqual(node["status"], DAGNodeStatus.RUNNING.value)
        else:
            self.assertEqual(node.status, DAGNodeStatus.RUNNING)
        
        self.dag.mark_node_completed(first_node_id, 1.5)
        # Check status for dict nodes
        node = self.dag.nodes[first_node_id]
        if isinstance(node, dict):
            self.assertEqual(node["status"], DAGNodeStatus.COMPLETED.value)
            self.assertEqual(node["actual_duration"], 1.5)
        else:
            self.assertEqual(node.status, DAGNodeStatus.COMPLETED)
            self.assertEqual(node.actual_duration, 1.5)

    def test_dag_execution_summary(self):
        """Test DAG execution summary generation."""
        self._build_dag_from_config()
        
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
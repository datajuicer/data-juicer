#!/usr/bin/env python3
"""
Tests for DAG Execution functionality.

This module tests the strategy-based DAG execution planning
capabilities of the Data-Juicer system.
"""

import os
import tempfile
import unittest

from data_juicer.core.executor.pipeline_dag import PipelineDAG, DAGNodeStatus
from data_juicer.core.executor.dag_execution_strategies import (
    NonPartitionedDAGStrategy, 
    PartitionedDAGStrategy,
    is_global_operation
)
from data_juicer.ops import load_ops
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


# Note: PipelineAST tests removed - AST functionality was removed in favor of strategy-based DAG building


class TestPipelineDAG(DataJuicerTestCaseBase):
    """Test DAG execution planning functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.dag = PipelineDAG(self.temp_dir)
        self.sample_config = {
            "process": [
                {"text_length_filter": {"min_len": 10, "max_len": 1000}},
                {"character_repetition_filter": {"rep_len": 3}},
                {"words_num_filter": {"min_num": 5, "max_num": 1000}},
                {"language_id_score_filter": {"lang": "en", "min_score": 0.8}},
                {"document_deduplicator": {}},
                {"clean_email_mapper": {}},
                {"clean_links_mapper": {}},
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
        self.assertIn("completion_percentage", summary)


class TestDAGExecutionStrategies(DataJuicerTestCaseBase):
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


def _make_node(node_id, operation_name="op", dependencies=None,
               partition_id=None, execution_order=0):
    """Helper to create a DAG node dict."""
    return {
        "node_id": node_id,
        "operation_name": operation_name,
        "node_type": "operation",
        "partition_id": partition_id,
        "config": {},
        "dependencies": dependencies or [],
        "execution_order": execution_order,
        "estimated_duration": 0.0,
        "metadata": {},
        "status": DAGNodeStatus.PENDING.value,
        "actual_duration": None,
        "start_time": None,
        "end_time": None,
        "error_message": None,
    }


class PipelineDAGFailureTest(DataJuicerTestCaseBase):

    def setUp(self):
        super().setUp()
        self.tmp_dir = tempfile.mkdtemp()
        self.dag = PipelineDAG(self.tmp_dir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        super().tearDown()

    def test_mark_node_failed_with_start_time(self):
        self.dag.nodes["n1"] = _make_node("n1")
        self.dag.mark_node_started("n1")
        start_time = self.dag.nodes["n1"]["start_time"]

        import time
        time.sleep(0.02)
        self.dag.mark_node_failed("n1", "something broke")

        node = self.dag.nodes["n1"]
        self.assertEqual(node["status"], DAGNodeStatus.FAILED.value)
        self.assertEqual(node["error_message"], "something broke")
        self.assertGreater(node["actual_duration"], 0)
        self.assertEqual(node["start_time"], start_time)

    def test_mark_node_failed_without_start_time(self):
        """When a node fails before being started, duration should be 0."""
        self.dag.nodes["n1"] = _make_node("n1")
        self.dag.mark_node_failed("n1", "failed early")

        node = self.dag.nodes["n1"]
        self.assertEqual(node["status"], DAGNodeStatus.FAILED.value)
        self.assertIsNotNone(node["actual_duration"])
        self.assertEqual(node["actual_duration"], 0.0)

    def test_mark_node_failed_nonexistent_node(self):
        self.dag.mark_node_failed("nonexistent", "error")
        self.assertNotIn("nonexistent", self.dag.nodes)


class PipelineDAGReadyNodesTest(DataJuicerTestCaseBase):

    def setUp(self):
        super().setUp()
        self.tmp_dir = tempfile.mkdtemp()
        self.dag = PipelineDAG(self.tmp_dir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        super().tearDown()

    def test_ready_nodes_no_dependencies(self):
        self.dag.nodes["a"] = _make_node("a", "op_a")
        self.dag.nodes["b"] = _make_node("b", "op_b")
        ready = self.dag.get_ready_nodes()
        self.assertEqual(set(ready), {"a", "b"})

    def test_ready_nodes_with_dependencies(self):
        self.dag.nodes["a"] = _make_node("a", "op_a")
        self.dag.nodes["b"] = _make_node("b", "op_b", dependencies=["a"])

        ready = self.dag.get_ready_nodes()
        self.assertEqual(ready, ["a"])

        self.dag.mark_node_started("a")
        self.dag.mark_node_completed("a")
        ready = self.dag.get_ready_nodes()
        self.assertEqual(ready, ["b"])

    def test_ready_nodes_skips_non_pending(self):
        self.dag.nodes["a"] = _make_node("a", "op_a")
        self.dag.mark_node_started("a")

        ready = self.dag.get_ready_nodes()
        self.assertEqual(ready, [])

    def test_ready_nodes_multiple_deps(self):
        self.dag.nodes["a"] = _make_node("a", "op_a", execution_order=0)
        self.dag.nodes["b"] = _make_node("b", "op_b", execution_order=1)
        self.dag.nodes["c"] = _make_node("c", "op_c",
                                         dependencies=["a", "b"],
                                         execution_order=2)

        self.assertNotIn("c", self.dag.get_ready_nodes())

        self.dag.mark_node_started("a")
        self.dag.mark_node_completed("a")
        self.assertNotIn("c", self.dag.get_ready_nodes())

        self.dag.mark_node_started("b")
        self.dag.mark_node_completed("b")
        self.assertIn("c", self.dag.get_ready_nodes())

    def test_ready_nodes_dep_failed(self):
        self.dag.nodes["a"] = _make_node("a", "op_a")
        self.dag.nodes["b"] = _make_node("b", "op_b", dependencies=["a"])

        self.dag.mark_node_failed("a", "crash")
        ready = self.dag.get_ready_nodes()
        self.assertNotIn("b", ready)


class PipelineDAGStatusTest(DataJuicerTestCaseBase):

    def setUp(self):
        super().setUp()
        self.tmp_dir = tempfile.mkdtemp()
        self.dag = PipelineDAG(self.tmp_dir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        super().tearDown()

    def test_unknown_node_returns_pending(self):
        status = self.dag.get_node_status("nonexistent")
        self.assertEqual(status, DAGNodeStatus.PENDING)

    def test_status_transitions(self):
        self.dag.nodes["n1"] = _make_node("n1")

        self.assertEqual(self.dag.get_node_status("n1"), DAGNodeStatus.PENDING)

        self.dag.mark_node_started("n1")
        self.assertEqual(self.dag.get_node_status("n1"), DAGNodeStatus.RUNNING)

        self.dag.mark_node_completed("n1")
        self.assertEqual(self.dag.get_node_status("n1"),
                         DAGNodeStatus.COMPLETED)


class PipelineDAGVisualizeTest(DataJuicerTestCaseBase):

    def setUp(self):
        super().setUp()
        self.tmp_dir = tempfile.mkdtemp()
        self.dag = PipelineDAG(self.tmp_dir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        super().tearDown()

    def test_empty_dag(self):
        result = self.dag.visualize()
        self.assertEqual(result, "Empty DAG")

    def test_visualize_with_partitions(self):
        self.dag.nodes["n1"] = _make_node("n1", "clean_mapper",
                                          partition_id=0, execution_order=0)
        self.dag.nodes["n2"] = _make_node("n2", "clean_mapper",
                                          partition_id=1, execution_order=1)

        output = self.dag.visualize()
        self.assertIn("partition 0", output)
        self.assertIn("partition 1", output)
        self.assertIn("clean_mapper", output)

    def test_visualize_with_dependencies(self):
        self.dag.nodes["a"] = _make_node("a", "step_a", execution_order=0)
        self.dag.nodes["b"] = _make_node("b", "step_b",
                                         dependencies=["a"],
                                         execution_order=1)

        output = self.dag.visualize()
        self.assertIn("Dependencies:", output)
        self.assertIn("step_b <- step_a", output)

    def test_visualize_status_icons(self):
        self.dag.nodes["a"] = _make_node("a", "op_a", execution_order=0)
        self.dag.mark_node_started("a")

        self.dag.nodes["b"] = _make_node("b", "op_b", execution_order=1)
        self.dag.mark_node_started("b")
        self.dag.mark_node_completed("b")

        self.dag.nodes["c"] = _make_node("c", "op_c", execution_order=2)
        self.dag.mark_node_failed("c", "err")

        output = self.dag.visualize()
        self.assertIn("[~]", output)
        self.assertIn("[x]", output)
        self.assertIn("[!]", output)

    def test_visualize_no_partition(self):
        self.dag.nodes["a"] = _make_node("a", "op_a", execution_order=0)
        output = self.dag.visualize()
        self.assertNotIn("partition", output)


class PipelineDAGLoadPlanTest(DataJuicerTestCaseBase):

    def setUp(self):
        super().setUp()
        self.tmp_dir = tempfile.mkdtemp()
        self.dag = PipelineDAG(self.tmp_dir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        super().tearDown()

    def test_load_nonexistent_file(self):
        result = self.dag.load_execution_plan("does_not_exist.json")
        self.assertFalse(result)

    def test_save_and_load_roundtrip(self):
        self.dag.nodes["n1"] = _make_node("n1", "op_a", execution_order=0)
        self.dag.nodes["n2"] = _make_node("n2", "op_b",
                                          dependencies=["n1"],
                                          execution_order=1)

        self.dag.save_execution_plan()

        new_dag = PipelineDAG(self.tmp_dir)
        loaded = new_dag.load_execution_plan()
        self.assertTrue(loaded)
        self.assertIn("n1", new_dag.nodes)
        self.assertIn("n2", new_dag.nodes)
        self.assertEqual(new_dag.nodes["n2"]["dependencies"], ["n1"])
        self.assertEqual(new_dag.nodes["n1"]["status"],
                         DAGNodeStatus.PENDING.value)

    def test_load_corrupted_file(self):
        plan_path = os.path.join(self.tmp_dir, "dag_execution_plan.json")
        with open(plan_path, "w") as f:
            f.write("not valid json{{{")

        result = self.dag.load_execution_plan()
        self.assertFalse(result)


class PipelineDAGCompletionTest(DataJuicerTestCaseBase):

    def setUp(self):
        super().setUp()
        self.tmp_dir = tempfile.mkdtemp()
        self.dag = PipelineDAG(self.tmp_dir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        super().tearDown()

    def test_mark_completed_auto_duration(self):
        import time
        self.dag.nodes["n1"] = _make_node("n1")
        self.dag.mark_node_started("n1")
        time.sleep(0.02)
        self.dag.mark_node_completed("n1")

        node = self.dag.nodes["n1"]
        self.assertGreater(node["actual_duration"], 0)

    def test_mark_completed_explicit_duration(self):
        self.dag.nodes["n1"] = _make_node("n1")
        self.dag.mark_node_started("n1")
        self.dag.mark_node_completed("n1", duration=42.0)

        self.assertEqual(self.dag.nodes["n1"]["actual_duration"], 42.0)

    def test_mark_completed_without_start(self):
        """Complete a node that was never started - duration should be 0."""
        self.dag.nodes["n1"] = _make_node("n1")
        self.dag.mark_node_completed("n1")

        node = self.dag.nodes["n1"]
        self.assertEqual(node["status"], DAGNodeStatus.COMPLETED.value)
        self.assertEqual(node["actual_duration"], 0.0)

    def test_execution_summary_mixed_states(self):
        self.dag.nodes["a"] = _make_node("a", execution_order=0)
        self.dag.nodes["b"] = _make_node("b", execution_order=1)
        self.dag.nodes["c"] = _make_node("c", execution_order=2)
        self.dag.nodes["d"] = _make_node("d", execution_order=3)

        self.dag.mark_node_started("a")
        self.dag.mark_node_completed("a", duration=1.0)

        self.dag.mark_node_started("b")
        self.dag.mark_node_failed("b", "err")

        self.dag.mark_node_started("c")

        summary = self.dag.get_execution_summary()
        self.assertEqual(summary["total_nodes"], 4)
        self.assertEqual(summary["completed_nodes"], 1)
        self.assertEqual(summary["failed_nodes"], 1)
        self.assertEqual(summary["running_nodes"], 1)
        self.assertEqual(summary["pending_nodes"], 1)
        self.assertAlmostEqual(summary["completion_percentage"], 25.0)


if __name__ == "__main__":
    unittest.main() 
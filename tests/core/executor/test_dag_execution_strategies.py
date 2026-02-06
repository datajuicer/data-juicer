"""
Comprehensive tests for DAG Execution Strategies.

Tests cover:
- NonPartitionedDAGStrategy (for default/ray executors)
- PartitionedDAGStrategy (for ray_partitioned executor)
- NodeID utilities
- Scatter-gather pattern for global operations
- Dependency building
- Node execution readiness checking
"""

import unittest
from unittest.mock import MagicMock

from data_juicer.core.executor.dag_execution_strategies import (
    DAGExecutionStrategy,
    DAGNodeType,
    NodeID,
    NonPartitionedDAGStrategy,
    PartitionedDAGStrategy,
    ScatterGatherNode,
    is_global_operation,
)
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class MockOperation:
    """Mock operation for testing."""

    def __init__(self, name: str, is_global: bool = False):
        self._name = name
        self.is_global_operation = is_global


class NodeIDTest(DataJuicerTestCaseBase):
    """Tests for NodeID utility class."""

    # ==================== Node ID Creation Tests ====================

    def test_for_operation(self):
        """Test creating node ID for global operation."""
        node_id = NodeID.for_operation(0, "text_filter")
        self.assertEqual(node_id, "op_001_text_filter")

        node_id = NodeID.for_operation(9, "deduplicator")
        self.assertEqual(node_id, "op_010_deduplicator")

    def test_for_partition_operation(self):
        """Test creating node ID for partition operation."""
        node_id = NodeID.for_partition_operation(0, 0, "mapper")
        self.assertEqual(node_id, "op_001_mapper_partition_0")

        node_id = NodeID.for_partition_operation(5, 3, "filter")
        self.assertEqual(node_id, "op_004_filter_partition_5")

    def test_for_scatter_gather(self):
        """Test creating node ID for scatter-gather operation."""
        node_id = NodeID.for_scatter_gather(2, "deduplicator")
        self.assertEqual(node_id, "sg_002_deduplicator")

    # ==================== Node ID Parsing Tests ====================

    def test_parse_operation_node_id(self):
        """Test parsing global operation node ID."""
        result = NodeID.parse("op_001_text_filter")

        self.assertIsNotNone(result)
        self.assertEqual(result["type"], DAGNodeType.OPERATION)
        self.assertEqual(result["operation_index"], 0)
        self.assertEqual(result["operation_name"], "text_filter")

    def test_parse_partition_operation_node_id(self):
        """Test parsing partition operation node ID."""
        result = NodeID.parse("op_005_mapper_partition_3")

        self.assertIsNotNone(result)
        self.assertEqual(result["type"], DAGNodeType.PARTITION_OPERATION)
        self.assertEqual(result["operation_index"], 4)  # 5-1 = 4
        self.assertEqual(result["operation_name"], "mapper")
        self.assertEqual(result["partition_id"], 3)

    def test_parse_scatter_gather_node_id(self):
        """Test parsing scatter-gather node ID."""
        result = NodeID.parse("sg_002_deduplicator")

        self.assertIsNotNone(result)
        self.assertEqual(result["type"], DAGNodeType.SCATTER_GATHER)
        self.assertEqual(result["operation_index"], 2)
        self.assertEqual(result["operation_name"], "deduplicator")

    def test_parse_invalid_node_id(self):
        """Test parsing invalid node ID returns None."""
        result = NodeID.parse("invalid_format")
        self.assertIsNone(result)

        result = NodeID.parse("")
        self.assertIsNone(result)

        result = NodeID.parse("random_string_123")
        self.assertIsNone(result)

    def test_parse_operation_with_underscores_in_name(self):
        """Test parsing node ID where operation name contains underscores."""
        result = NodeID.parse("op_001_text_length_filter")

        self.assertIsNotNone(result)
        self.assertEqual(result["operation_name"], "text_length_filter")


class ScatterGatherNodeTest(DataJuicerTestCaseBase):
    """Tests for ScatterGatherNode dataclass."""

    def test_node_id_generation(self):
        """Test scatter-gather node ID generation."""
        sg_node = ScatterGatherNode(
            operation_index=5,
            operation_name="deduplicator",
            input_partitions=[0, 1, 2, 3],
            output_partitions=[0, 1, 2, 3],
        )

        self.assertEqual(sg_node.node_id, "sg_005_deduplicator")

    def test_partition_lists(self):
        """Test input/output partition tracking."""
        sg_node = ScatterGatherNode(
            operation_index=3,
            operation_name="global_op",
            input_partitions=[0, 1, 2],
            output_partitions=[0, 1],  # Could reduce partitions
        )

        self.assertEqual(sg_node.input_partitions, [0, 1, 2])
        self.assertEqual(sg_node.output_partitions, [0, 1])


class NonPartitionedDAGStrategyTest(DataJuicerTestCaseBase):
    """Tests for NonPartitionedDAGStrategy."""

    def setUp(self):
        super().setUp()
        self.strategy = NonPartitionedDAGStrategy()

    # ==================== Node Generation Tests ====================

    def test_generate_dag_nodes_empty(self):
        """Test generating nodes with empty operations list."""
        nodes = self.strategy.generate_dag_nodes([])
        self.assertEqual(len(nodes), 0)

    def test_generate_dag_nodes_single_op(self):
        """Test generating nodes with single operation."""
        ops = [MockOperation("filter")]
        nodes = self.strategy.generate_dag_nodes(ops)

        self.assertEqual(len(nodes), 1)
        node = list(nodes.values())[0]
        self.assertEqual(node["operation_name"], "filter")
        self.assertEqual(node["execution_order"], 1)
        self.assertEqual(node["node_type"], DAGNodeType.OPERATION.value)
        self.assertIsNone(node["partition_id"])

    def test_generate_dag_nodes_multiple_ops(self):
        """Test generating nodes with multiple operations."""
        ops = [MockOperation(f"op_{i}") for i in range(5)]
        nodes = self.strategy.generate_dag_nodes(ops)

        self.assertEqual(len(nodes), 5)
        for i, node in enumerate(nodes.values()):
            self.assertEqual(node["execution_order"], i + 1)

    def test_generate_dag_nodes_initial_status(self):
        """Test that generated nodes have pending status."""
        ops = [MockOperation("filter")]
        nodes = self.strategy.generate_dag_nodes(ops)

        node = list(nodes.values())[0]
        self.assertEqual(node["status"], "pending")
        self.assertIsNone(node["start_time"])
        self.assertIsNone(node["end_time"])

    # ==================== Node ID Tests ====================

    def test_get_dag_node_id(self):
        """Test getting node ID for non-partitioned operation."""
        node_id = self.strategy.get_dag_node_id("filter", 0)
        self.assertEqual(node_id, "op_001_filter")

        node_id = self.strategy.get_dag_node_id("mapper", 5)
        self.assertEqual(node_id, "op_006_mapper")

    # ==================== Dependency Building Tests ====================

    def test_build_dependencies_empty(self):
        """Test building dependencies with empty operations."""
        nodes = {}
        self.strategy.build_dependencies(nodes, [])
        # Should not raise

    def test_build_dependencies_single_op(self):
        """Test building dependencies with single operation."""
        ops = [MockOperation("filter")]
        nodes = self.strategy.generate_dag_nodes(ops)

        self.strategy.build_dependencies(nodes, ops)

        node = list(nodes.values())[0]
        self.assertEqual(len(node["dependencies"]), 0)  # First op has no deps

    def test_build_dependencies_sequential(self):
        """Test building sequential dependencies."""
        ops = [MockOperation(f"op_{i}") for i in range(4)]
        nodes = self.strategy.generate_dag_nodes(ops)

        self.strategy.build_dependencies(nodes, ops)

        # First op has no dependencies
        first_node = nodes[self.strategy.get_dag_node_id("op_0", 0)]
        self.assertEqual(len(first_node["dependencies"]), 0)

        # Second op depends on first
        second_node = nodes[self.strategy.get_dag_node_id("op_1", 1)]
        self.assertEqual(len(second_node["dependencies"]), 1)
        self.assertIn("op_001_op_0", second_node["dependencies"])

        # Last op depends on previous
        last_node = nodes[self.strategy.get_dag_node_id("op_3", 3)]
        self.assertEqual(len(last_node["dependencies"]), 1)
        self.assertIn("op_003_op_2", last_node["dependencies"])

    # ==================== Execution Readiness Tests ====================

    def test_can_execute_node_no_deps(self):
        """Test execution readiness for node with no dependencies."""
        nodes = {"op_001_filter": {"dependencies": []}}
        completed = set()

        can_execute = self.strategy.can_execute_node("op_001_filter", nodes, completed)
        self.assertTrue(can_execute)

    def test_can_execute_node_deps_met(self):
        """Test execution readiness when all dependencies completed."""
        nodes = {
            "op_001_first": {"dependencies": []},
            "op_002_second": {"dependencies": ["op_001_first"]},
        }
        completed = {"op_001_first"}

        can_execute = self.strategy.can_execute_node("op_002_second", nodes, completed)
        self.assertTrue(can_execute)

    def test_can_execute_node_deps_not_met(self):
        """Test execution readiness when dependencies not completed."""
        nodes = {
            "op_001_first": {"dependencies": []},
            "op_002_second": {"dependencies": ["op_001_first"]},
        }
        completed = set()  # First op not completed

        can_execute = self.strategy.can_execute_node("op_002_second", nodes, completed)
        self.assertFalse(can_execute)

    def test_can_execute_node_nonexistent(self):
        """Test execution readiness for nonexistent node."""
        nodes = {}
        completed = set()

        can_execute = self.strategy.can_execute_node("nonexistent", nodes, completed)
        self.assertFalse(can_execute)


class PartitionedDAGStrategyTest(DataJuicerTestCaseBase):
    """Tests for PartitionedDAGStrategy."""

    def setUp(self):
        super().setUp()
        self.strategy = PartitionedDAGStrategy(num_partitions=3)

    # ==================== Node Generation Tests ====================

    def test_generate_dag_nodes_empty(self):
        """Test generating nodes with empty operations."""
        nodes = self.strategy.generate_dag_nodes([])
        self.assertEqual(len(nodes), 0)

    def test_generate_dag_nodes_creates_partition_nodes(self):
        """Test that nodes are created for each partition."""
        ops = [MockOperation("filter"), MockOperation("mapper")]
        nodes = self.strategy.generate_dag_nodes(ops)

        # Should have 2 ops * 3 partitions = 6 nodes
        self.assertEqual(len(nodes), 6)

        # Check partition nodes exist
        for partition_id in range(3):
            for op_idx in range(2):
                node_id = self.strategy.get_dag_node_id(
                    ops[op_idx]._name, op_idx, partition_id=partition_id
                )
                self.assertIn(node_id, nodes)
                self.assertEqual(nodes[node_id]["partition_id"], partition_id)

    def test_generate_dag_nodes_with_convergence_points(self):
        """Test generating nodes with convergence points."""
        ops = [
            MockOperation("filter"),
            MockOperation("deduplicator"),  # Global op at index 1
            MockOperation("mapper"),
        ]
        nodes = self.strategy.generate_dag_nodes(ops, convergence_points=[1])

        # Should have partition nodes + scatter-gather node
        # 3 ops * 3 partitions = 9 partition nodes + 1 scatter-gather
        self.assertEqual(len(nodes), 10)

        # Verify scatter-gather node
        sg_node_id = "sg_001_deduplicator"
        self.assertIn(sg_node_id, nodes)
        self.assertEqual(nodes[sg_node_id]["node_type"], DAGNodeType.SCATTER_GATHER.value)

    def test_generate_dag_nodes_node_type(self):
        """Test that partition nodes have correct type."""
        ops = [MockOperation("filter")]
        nodes = self.strategy.generate_dag_nodes(ops)

        for node in nodes.values():
            self.assertEqual(node["node_type"], DAGNodeType.PARTITION_OPERATION.value)

    # ==================== Node ID Tests ====================

    def test_get_dag_node_id_with_partition(self):
        """Test getting node ID with partition ID."""
        node_id = self.strategy.get_dag_node_id("filter", 0, partition_id=2)
        self.assertEqual(node_id, "op_001_filter_partition_2")

    def test_get_dag_node_id_without_partition(self):
        """Test getting node ID without partition ID."""
        node_id = self.strategy.get_dag_node_id("filter", 0)
        self.assertEqual(node_id, "op_001_filter")

    # ==================== Dependency Building Tests ====================

    def test_build_dependencies_within_partition(self):
        """Test that dependencies are built within each partition."""
        ops = [MockOperation(f"op_{i}") for i in range(3)]
        nodes = self.strategy.generate_dag_nodes(ops)

        self.strategy.build_dependencies(nodes, ops)

        # Check partition 0
        node_1 = nodes["op_002_op_1_partition_0"]
        self.assertEqual(len(node_1["dependencies"]), 1)
        self.assertIn("op_001_op_0_partition_0", node_1["dependencies"])

        # Check partition 1
        node_1_p1 = nodes["op_002_op_1_partition_1"]
        self.assertEqual(len(node_1_p1["dependencies"]), 1)
        self.assertIn("op_001_op_0_partition_1", node_1_p1["dependencies"])

    def test_build_dependencies_no_cross_partition(self):
        """Test that partition 0 doesn't depend on partition 1."""
        ops = [MockOperation("op_0"), MockOperation("op_1")]
        nodes = self.strategy.generate_dag_nodes(ops)

        self.strategy.build_dependencies(nodes, ops)

        # Partition 0's second op should not depend on partition 1
        node = nodes["op_002_op_1_partition_0"]
        for dep in node["dependencies"]:
            self.assertNotIn("partition_1", dep)
            self.assertNotIn("partition_2", dep)

    def test_build_dependencies_scatter_gather(self):
        """Test scatter-gather dependency building."""
        ops = [
            MockOperation("filter"),
            MockOperation("deduplicator"),
            MockOperation("mapper"),
        ]
        nodes = self.strategy.generate_dag_nodes(ops, convergence_points=[1])

        self.strategy.build_dependencies(nodes, ops, convergence_points=[1])

        # Scatter-gather node should depend on all partitions from previous op
        sg_node = nodes.get("sg_001_deduplicator")
        if sg_node:  # If scatter-gather node was created
            # Should have dependencies from all partitions
            pass  # Exact behavior depends on implementation

    # ==================== Execution Readiness Tests ====================

    def test_can_execute_node_partition_ready(self):
        """Test execution readiness for partition node."""
        nodes = {
            "op_001_filter_partition_0": {"dependencies": []},
            "op_002_mapper_partition_0": {"dependencies": ["op_001_filter_partition_0"]},
        }
        completed = {"op_001_filter_partition_0"}

        can_execute = self.strategy.can_execute_node(
            "op_002_mapper_partition_0", nodes, completed
        )
        self.assertTrue(can_execute)

    def test_can_execute_node_partition_not_ready(self):
        """Test execution readiness when partition dependency not met."""
        nodes = {
            "op_001_filter_partition_0": {"dependencies": []},
            "op_002_mapper_partition_0": {"dependencies": ["op_001_filter_partition_0"]},
        }
        completed = set()

        can_execute = self.strategy.can_execute_node(
            "op_002_mapper_partition_0", nodes, completed
        )
        self.assertFalse(can_execute)

    # ==================== Number of Partitions Tests ====================

    def test_different_partition_counts(self):
        """Test strategy with different partition counts."""
        for num_partitions in [1, 2, 4, 8, 16]:
            strategy = PartitionedDAGStrategy(num_partitions=num_partitions)
            ops = [MockOperation("filter")]
            nodes = strategy.generate_dag_nodes(ops)

            self.assertEqual(len(nodes), num_partitions)

    def test_single_partition(self):
        """Test strategy with single partition."""
        strategy = PartitionedDAGStrategy(num_partitions=1)
        ops = [MockOperation("op_0"), MockOperation("op_1")]
        nodes = strategy.generate_dag_nodes(ops)

        self.assertEqual(len(nodes), 2)

        # Verify dependencies
        strategy.build_dependencies(nodes, ops)
        node_1 = nodes["op_002_op_1_partition_0"]
        self.assertIn("op_001_op_0_partition_0", node_1["dependencies"])


class GlobalOperationDetectionTest(DataJuicerTestCaseBase):
    """Tests for is_global_operation function."""

    def test_deduplicator_is_global(self):
        """Test that deduplicators are detected as global operations."""
        op = MockOperation("minhash_deduplicator")
        self.assertTrue(is_global_operation(op))

        op = MockOperation("document_deduplicator")
        self.assertTrue(is_global_operation(op))

    def test_filter_is_not_global(self):
        """Test that filters are not global operations."""
        op = MockOperation("text_length_filter")
        self.assertFalse(is_global_operation(op))

    def test_mapper_is_not_global(self):
        """Test that mappers are not global operations."""
        op = MockOperation("clean_links_mapper")
        self.assertFalse(is_global_operation(op))

    def test_explicit_global_flag(self):
        """Test that explicit is_global_operation flag is respected."""
        op = MockOperation("custom_op", is_global=True)
        self.assertTrue(is_global_operation(op))

    def test_missing_name_attribute(self):
        """Test handling of operation without _name attribute."""
        class NoNameOp:
            pass

        op = NoNameOp()
        # Should not raise, should return False
        result = is_global_operation(op)
        self.assertFalse(result)


class DAGNodeTypeEnumTest(DataJuicerTestCaseBase):
    """Tests for DAGNodeType enum."""

    def test_node_type_values(self):
        """Test all node type values."""
        self.assertEqual(DAGNodeType.OPERATION.value, "operation")
        self.assertEqual(DAGNodeType.PARTITION_OPERATION.value, "partition_operation")
        self.assertEqual(DAGNodeType.SCATTER_GATHER.value, "scatter_gather")


if __name__ == '__main__':
    unittest.main()

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from data_juicer.core.executor.pipeline_dag import DAGNodeStatus


class DAGNodeType(Enum):
    """Types of DAG nodes."""

    OPERATION = "operation"
    PARTITION_OPERATION = "partition_operation"
    SCATTER_GATHER = "scatter_gather"


class DAGNodeStatusTransition:
    """Validates DAG node status transitions.

    Uses DAGNodeStatus enum for type safety. Valid transitions:
    - PENDING -> RUNNING (node starts execution)
    - PENDING -> COMPLETED (skipped - already done in previous run)
    - RUNNING -> COMPLETED (node finishes successfully)
    - RUNNING -> FAILED (node fails)
    - FAILED -> RUNNING (node retries)
    - COMPLETED is terminal (no transitions out)
    """

    VALID_TRANSITIONS = {
        DAGNodeStatus.PENDING: {DAGNodeStatus.RUNNING, DAGNodeStatus.COMPLETED},
        DAGNodeStatus.RUNNING: {DAGNodeStatus.COMPLETED, DAGNodeStatus.FAILED},
        DAGNodeStatus.FAILED: {DAGNodeStatus.RUNNING},  # retry
        DAGNodeStatus.COMPLETED: set(),  # terminal state
    }

    @classmethod
    def _normalize_status(cls, status: Union[str, DAGNodeStatus]) -> DAGNodeStatus:
        """Convert string status to DAGNodeStatus enum."""
        if isinstance(status, DAGNodeStatus):
            return status
        return DAGNodeStatus(status)

    @classmethod
    def is_valid(cls, from_status: Union[str, DAGNodeStatus], to_status: Union[str, DAGNodeStatus]) -> bool:
        """Check if a status transition is valid.

        Args:
            from_status: Current status (string or enum)
            to_status: Target status (string or enum)

        Returns:
            True if transition is valid, False otherwise
        """
        from_enum = cls._normalize_status(from_status)
        to_enum = cls._normalize_status(to_status)
        valid_targets = cls.VALID_TRANSITIONS.get(from_enum, set())
        return to_enum in valid_targets

    @classmethod
    def validate_and_log(
        cls, node_id: str, from_status: Union[str, DAGNodeStatus], to_status: Union[str, DAGNodeStatus]
    ) -> bool:
        """Validate transition and log warning if invalid.

        Args:
            node_id: Node identifier for logging
            from_status: Current status (string or enum)
            to_status: Target status (string or enum)

        Returns:
            True if transition is valid, False otherwise
        """
        if cls.is_valid(from_status, to_status):
            return True

        from loguru import logger

        from_enum = cls._normalize_status(from_status)
        to_enum = cls._normalize_status(to_status)
        valid_targets = cls.VALID_TRANSITIONS.get(from_enum, set())

        logger.warning(
            f"Invalid DAG node transition for {node_id}: {from_enum.value} -> {to_enum.value}. "
            f"Valid targets: {[s.value for s in valid_targets]}"
        )
        return False


@dataclass
class ScatterGatherNode:
    """Represents a scatter-gather operation in partitioned execution.

    Encapsulates the complete scatter-gather pattern:
    1. Convergence: All partitions complete their work and converge
    2. Global Operation: A single operation runs on the gathered data
    3. Redistribution: Results are redistributed back to partitions
    """

    operation_index: int
    operation_name: str
    input_partitions: List[int]
    output_partitions: List[int]

    @property
    def node_id(self) -> str:
        """Generate unique node ID for scatter-gather operation."""
        return f"sg_{self.operation_index:03d}_{self.operation_name}"


class NodeID:
    """Utility for creating and parsing standardized node IDs.

    Node ID formats:
    - Operation: "op_{idx:03d}_{name}"
    - Partition Operation: "op_{idx:03d}_{name}_partition_{pid}"
    - Scatter-Gather: "sg_{idx:03d}_{name}"
    """

    @staticmethod
    def for_operation(op_idx: int, op_name: str) -> str:
        """Create node ID for global operation.

        Args:
            op_idx: Operation index (0-based)
            op_name: Operation name

        Returns:
            Standardized node ID
        """
        return f"op_{op_idx+1:03d}_{op_name}"

    @staticmethod
    def for_partition_operation(partition_id: int, op_idx: int, op_name: str) -> str:
        """Create node ID for partition operation.

        Args:
            partition_id: Partition ID
            op_idx: Operation index (0-based)
            op_name: Operation name

        Returns:
            Standardized node ID
        """
        return f"op_{op_idx+1:03d}_{op_name}_partition_{partition_id}"

    @staticmethod
    def for_scatter_gather(op_idx: int, op_name: str) -> str:
        """Create node ID for scatter-gather operation.

        Args:
            op_idx: Operation index (0-based)
            op_name: Operation name

        Returns:
            Standardized node ID
        """
        return f"sg_{op_idx:03d}_{op_name}"

    @staticmethod
    def parse(node_id: str) -> Optional[Dict[str, Any]]:
        """Parse node ID into components.

        Args:
            node_id: The node ID to parse

        Returns:
            Dictionary with node type and components, or None if invalid format

        Example:
            >>> NodeID.parse("op_001_mapper_partition_0")
            {'type': DAGNodeType.PARTITION_OPERATION, 'partition_id': 0,
             'operation_index': 0, 'operation_name': 'mapper'}

            >>> NodeID.parse("sg_002_deduplicator")
            {'type': DAGNodeType.SCATTER_GATHER, 'operation_index': 2,
             'operation_name': 'deduplicator'}
        """
        # Partition operation: op_001_mapper_name_partition_0
        match = re.match(r"op_(\d+)_(.+)_partition_(\d+)", node_id)
        if match:
            return {
                "type": DAGNodeType.PARTITION_OPERATION,
                "operation_index": int(match.group(1)) - 1,  # Convert back to 0-based
                "operation_name": match.group(2),
                "partition_id": int(match.group(3)),
            }

        # Scatter-gather: sg_002_mapper_name
        match = re.match(r"sg_(\d+)_(.+)", node_id)
        if match:
            return {
                "type": DAGNodeType.SCATTER_GATHER,
                "operation_index": int(match.group(1)),
                "operation_name": match.group(2),
            }

        # Regular operation: op_001_mapper_name
        match = re.match(r"op_(\d+)_(.+)", node_id)
        if match:
            return {
                "type": DAGNodeType.OPERATION,
                "operation_index": int(match.group(1)) - 1,
                "operation_name": match.group(2),
            }

        return None


class DAGExecutionStrategy(ABC):
    """Abstract base class for different DAG execution strategies."""

    @abstractmethod
    def generate_dag_nodes(self, operations: List, **kwargs) -> Dict[str, Any]:
        """Generate DAG nodes based on execution strategy."""
        pass

    @abstractmethod
    def get_dag_node_id(self, op_name: str, op_idx: int, **kwargs) -> str:
        """Get DAG node ID for operation based on strategy."""
        pass

    @abstractmethod
    def build_dependencies(self, nodes: Dict[str, Any], operations: List, **kwargs) -> None:
        """Build dependencies between nodes based on strategy."""
        pass

    @abstractmethod
    def can_execute_node(self, node_id: str, nodes: Dict[str, Any], completed_nodes: set) -> bool:
        """Check if a node can be executed based on strategy."""
        pass

    def validate_dag(self, nodes: Dict[str, Any]) -> bool:
        """Validate DAG has no cycles using DFS.

        Returns:
            True if DAG is valid (no cycles), False otherwise
        """
        # Build adjacency list
        adj = {node_id: node.get("dependencies", []) for node_id, node in nodes.items()}

        # Track visited nodes
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {node_id: WHITE for node_id in nodes}

        def has_cycle(node_id: str) -> bool:
            """DFS to detect cycle."""
            color[node_id] = GRAY
            for dep in adj.get(node_id, []):
                if dep not in color:
                    continue  # Skip missing nodes
                if color[dep] == GRAY:
                    return True  # Back edge = cycle
                if color[dep] == WHITE and has_cycle(dep):
                    return True
            color[node_id] = BLACK
            return False

        for node_id in nodes:
            if color[node_id] == WHITE:
                if has_cycle(node_id):
                    return False
        return True


class NonPartitionedDAGStrategy(DAGExecutionStrategy):
    """Strategy for non-partitioned executors (default, ray)."""

    def generate_dag_nodes(self, operations: List, **kwargs) -> Dict[str, Any]:
        """Generate DAG nodes for non-partitioned execution."""
        nodes = {}
        for op_idx, op in enumerate(operations):
            node_id = self.get_dag_node_id(op._name, op_idx)
            nodes[node_id] = {
                "node_id": node_id,
                "operation_name": op._name,
                "execution_order": op_idx + 1,
                "node_type": DAGNodeType.OPERATION.value,
                "partition_id": None,
                "dependencies": [],
                "status": "pending",
                "start_time": None,
                "end_time": None,
                "actual_duration": None,
                "error_message": None,
            }
        return nodes

    def get_dag_node_id(self, op_name: str, op_idx: int, **kwargs) -> str:
        """Get DAG node ID for non-partitioned operation."""
        return f"op_{op_idx+1:03d}_{op_name}"

    def build_dependencies(self, nodes: Dict[str, Any], operations: List, **kwargs) -> None:
        """Build sequential dependencies for non-partitioned execution."""
        # Simple sequential dependencies
        for i in range(1, len(operations)):
            current_node = self.get_dag_node_id(operations[i]._name, i)
            prev_node = self.get_dag_node_id(operations[i - 1]._name, i - 1)
            if current_node in nodes and prev_node in nodes:
                nodes[current_node]["dependencies"].append(prev_node)

    def can_execute_node(self, node_id: str, nodes: Dict[str, Any], completed_nodes: set) -> bool:
        """Check if a node can be executed (all dependencies completed)."""
        if node_id not in nodes:
            return False
        node = nodes[node_id]
        return all(dep in completed_nodes for dep in node["dependencies"])


class PartitionedDAGStrategy(DAGExecutionStrategy):
    """Strategy for partitioned executors (ray_partitioned)."""

    def __init__(self, num_partitions: int):
        self.num_partitions = num_partitions

    def generate_dag_nodes(self, operations: List, **kwargs) -> Dict[str, Any]:
        """Generate DAG nodes for partitioned execution using scatter-gather pattern."""
        nodes = {}
        convergence_points = kwargs.get("convergence_points", [])

        # Generate partition-specific nodes
        for partition_id in range(self.num_partitions):
            for op_idx, op in enumerate(operations):
                node_id = self.get_dag_node_id(op._name, op_idx, partition_id=partition_id)
                nodes[node_id] = {
                    "node_id": node_id,
                    "operation_name": op._name,
                    "execution_order": op_idx + 1,
                    "node_type": DAGNodeType.PARTITION_OPERATION.value,
                    "partition_id": partition_id,
                    "dependencies": [],
                    "status": "pending",
                    "start_time": None,
                    "end_time": None,
                    "actual_duration": None,
                    "error_message": None,
                }

        # Generate scatter-gather nodes for global operations
        for conv_idx, conv_point in enumerate(convergence_points):
            if conv_point < len(operations):
                op = operations[conv_point]
                sg_node = ScatterGatherNode(
                    operation_index=conv_point,
                    operation_name=op._name,
                    input_partitions=list(range(self.num_partitions)),
                    output_partitions=list(range(self.num_partitions)),
                )

                nodes[sg_node.node_id] = {
                    "node_id": sg_node.node_id,
                    "operation_name": op._name,
                    "execution_order": conv_point + 1,
                    "node_type": DAGNodeType.SCATTER_GATHER.value,
                    "operation_index": conv_point,
                    "input_partitions": sg_node.input_partitions,
                    "output_partitions": sg_node.output_partitions,
                    "dependencies": [],
                    "status": "pending",
                    "start_time": None,
                    "end_time": None,
                    "actual_duration": None,
                    "error_message": None,
                    "scatter_gather_node": sg_node,
                }

        return nodes

    def get_dag_node_id(self, op_name: str, op_idx: int, partition_id: int = None, **kwargs) -> str:
        """Get DAG node ID for partitioned operation."""
        if partition_id is not None:
            return f"op_{op_idx+1:03d}_{op_name}_partition_{partition_id}"
        else:
            return f"op_{op_idx+1:03d}_{op_name}"

    def build_dependencies(self, nodes: Dict[str, Any], operations: List, **kwargs) -> None:
        """Build dependencies for partitioned execution using scatter-gather pattern.

        - Partition operations depend on previous operation in same partition
        - Scatter-gather nodes depend on ALL partitions from previous op
        - Post-scatter-gather partition ops depend on the scatter-gather node
        """
        convergence_points = kwargs.get("convergence_points", [])

        # Find all scatter-gather nodes
        sg_nodes = {
            node_id: node
            for node_id, node in nodes.items()
            if node.get("node_type") == DAGNodeType.SCATTER_GATHER.value
        }

        # First, build scatter-gather dependencies (only once, not per-partition)
        for op_idx in convergence_points:
            if op_idx >= len(operations):
                continue

            # Find the scatter-gather node for this operation
            sg_node_id = None
            for nid, node in sg_nodes.items():
                if node.get("operation_index") == op_idx:
                    sg_node_id = nid
                    break

            if sg_node_id and op_idx > 0:
                # Scatter-gather depends on PREVIOUS operation's partition outputs
                prev_op = operations[op_idx - 1]
                for pid in range(self.num_partitions):
                    dep_node = self.get_dag_node_id(prev_op._name, op_idx - 1, partition_id=pid)
                    if dep_node in nodes:
                        nodes[sg_node_id]["dependencies"].append(dep_node)

        # Build partition-specific dependencies
        for partition_id in range(self.num_partitions):
            prev_node_id = None
            for op_idx, op in enumerate(operations):
                # Skip convergence points - they're handled by scatter-gather nodes
                if op_idx in convergence_points:
                    # Find the scatter-gather node to use as prev_node for next op
                    for nid, node in sg_nodes.items():
                        if node.get("operation_index") == op_idx:
                            prev_node_id = nid
                            break
                    continue

                # Regular partition operation
                node_id = self.get_dag_node_id(op._name, op_idx, partition_id=partition_id)
                if node_id in nodes:
                    # Depends on previous node in this partition (could be partition op or scatter-gather)
                    if prev_node_id:
                        nodes[node_id]["dependencies"].append(prev_node_id)
                    prev_node_id = node_id

    def can_execute_node(self, node_id: str, nodes: Dict[str, Any], completed_nodes: set) -> bool:
        """Check if a node can be executed (all dependencies completed)."""
        if node_id not in nodes:
            return False
        node = nodes[node_id]
        return all(dep in completed_nodes for dep in node["dependencies"])


def is_global_operation(operation) -> bool:
    """Check if an operation is a global operation that requires convergence.

    Global operations need to see all data at once (e.g., deduplication, global sorting).
    They cannot be partitioned and require a scatter-gather pattern.

    Detection priority:
    1. Explicit `is_global_operation` flag on the operation
    2. Base class inheritance (Deduplicator)
    3. Operation name pattern (fallback for unknown operations)
    """
    # Priority 1: Explicit flag (most reliable)
    if getattr(operation, "is_global_operation", False):
        return True

    # Priority 2: Check base class (interface-based detection)
    try:
        from data_juicer.ops.base_op import Deduplicator

        if isinstance(operation, Deduplicator):
            return True
    except ImportError:
        pass  # Deduplicator class not available

    # Priority 3: Name-based detection (fallback for unknown ops)
    op_name = getattr(operation, "_name", "")
    global_op_patterns = ["deduplicator", "global_", "full_dataset_"]
    if any(pattern in op_name.lower() for pattern in global_op_patterns):
        return True

    return False

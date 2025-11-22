from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List


class DAGNodeType(Enum):
    """Types of DAG nodes."""

    OPERATION = "operation"
    PARTITION_OPERATION = "partition_operation"
    CONVERGENCE_POINT = "convergence_point"
    GLOBAL_OPERATION = "global_operation"
    REDISTRIBUTION = "redistribution"


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
        """Generate DAG nodes for partitioned execution."""
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

        # Generate convergence points
        for conv_idx, conv_point in enumerate(convergence_points):
            conv_node_id = f"convergence_point_{conv_idx}"
            nodes[conv_node_id] = {
                "node_id": conv_node_id,
                "node_type": DAGNodeType.CONVERGENCE_POINT.value,
                "convergence_idx": conv_idx,
                "operation_idx": conv_point,
                "dependencies": [],
                "status": "pending",
                "start_time": None,
                "end_time": None,
                "actual_duration": None,
                "error_message": None,
            }

        # Generate global operation nodes
        for conv_idx, conv_point in enumerate(convergence_points):
            if conv_point < len(operations):
                op = operations[conv_point]
                global_node_id = self.get_dag_node_id(op._name, conv_point)
                nodes[global_node_id] = {
                    "node_id": global_node_id,
                    "operation_name": op._name,
                    "execution_order": conv_point + 1,
                    "node_type": DAGNodeType.GLOBAL_OPERATION.value,
                    "partition_id": None,
                    "dependencies": [],
                    "status": "pending",
                    "start_time": None,
                    "end_time": None,
                    "actual_duration": None,
                    "error_message": None,
                }

        # Generate redistribution points
        for conv_idx, conv_point in enumerate(convergence_points):
            redist_node_id = f"redistribution_point_{conv_idx}"
            nodes[redist_node_id] = {
                "node_id": redist_node_id,
                "node_type": DAGNodeType.REDISTRIBUTION.value,
                "redistribution_idx": conv_idx,
                "dependencies": [],
                "status": "pending",
                "start_time": None,
                "end_time": None,
                "actual_duration": None,
                "error_message": None,
            }

        return nodes

    def get_dag_node_id(self, op_name: str, op_idx: int, partition_id: int = None, **kwargs) -> str:
        """Get DAG node ID for partitioned operation."""
        if partition_id is not None:
            return f"op_{op_idx+1:03d}_{op_name}_partition_{partition_id}"
        else:
            # For global operations
            return f"op_{op_idx+1:03d}_{op_name}_global"

    def build_dependencies(self, nodes: Dict[str, Any], operations: List, **kwargs) -> None:
        """Build dependencies for partitioned execution."""
        convergence_points = kwargs.get("convergence_points", [])

        # Build partition-specific dependencies (within each partition)
        for partition_id in range(self.num_partitions):
            for i in range(1, len(operations)):
                current_node = self.get_dag_node_id(operations[i]._name, i, partition_id=partition_id)
                prev_node = self.get_dag_node_id(operations[i - 1]._name, i - 1, partition_id=partition_id)
                if current_node in nodes and prev_node in nodes:
                    nodes[current_node]["dependencies"].append(prev_node)

        # Build convergence dependencies (all partitions converge)
        for conv_idx, conv_point in enumerate(convergence_points):
            conv_node_id = f"convergence_point_{conv_idx}"
            if conv_node_id in nodes:
                for partition_id in range(self.num_partitions):
                    dep_node = self.get_dag_node_id(operations[conv_point]._name, conv_point, partition_id=partition_id)
                    if dep_node in nodes:
                        nodes[conv_node_id]["dependencies"].append(dep_node)

        # Build global operation dependencies (after convergence)
        for conv_idx, conv_point in enumerate(convergence_points):
            conv_node_id = f"convergence_point_{conv_idx}"
            global_node_id = self.get_dag_node_id(operations[conv_point]._name, conv_point)
            if global_node_id in nodes and conv_node_id in nodes:
                nodes[global_node_id]["dependencies"].append(conv_node_id)

        # Build redistribution dependencies (after global operation)
        for conv_idx, conv_point in enumerate(convergence_points):
            global_node_id = self.get_dag_node_id(operations[conv_point]._name, conv_point)
            redist_node_id = f"redistribution_point_{conv_idx}"
            if redist_node_id in nodes and global_node_id in nodes:
                nodes[redist_node_id]["dependencies"].append(global_node_id)

        # Build post-redistribution dependencies (partitions resume independently)
        for conv_idx, conv_point in enumerate(convergence_points):
            redist_node_id = f"redistribution_point_{conv_idx}"
            if redist_node_id in nodes:
                for partition_id in range(self.num_partitions):
                    for i in range(conv_point + 1, len(operations)):
                        post_node = self.get_dag_node_id(operations[i]._name, i, partition_id=partition_id)
                        if post_node in nodes:
                            nodes[post_node]["dependencies"].append(redist_node_id)

    def can_execute_node(self, node_id: str, nodes: Dict[str, Any], completed_nodes: set) -> bool:
        """Check if a node can be executed (all dependencies completed)."""
        if node_id not in nodes:
            return False
        node = nodes[node_id]
        return all(dep in completed_nodes for dep in node["dependencies"])


def is_global_operation(operation) -> bool:
    """Check if an operation is a global operation that requires convergence."""
    # Deduplicators are typically global operations
    if hasattr(operation, "_name") and "deduplicator" in operation._name:
        return True

    # Check for explicit global operation flag
    if hasattr(operation, "is_global_operation") and operation.is_global_operation:
        return True

    return False

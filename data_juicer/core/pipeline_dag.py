"""
Pipeline DAG Representation for Data-Juicer Pipelines

This module provides Pipeline DAG (Directed Acyclic Graph) representation and planning
capabilities for tracking execution state, dependencies, and monitoring.
"""

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from loguru import logger


class DAGNodeStatus(Enum):
    """Status of a DAG node during execution."""

    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class DAGEdgeType(Enum):
    """Types of edges in the DAG."""

    SEQUENTIAL = "sequential"  # Standard sequential dependency
    PARALLEL = "parallel"  # Can run in parallel
    CONDITIONAL = "conditional"  # Conditional dependency


@dataclass
class DAGNode:
    """Node in the execution DAG.

    Note: This is kept for backward compatibility, but strategies typically use dict nodes.
    """

    node_id: str
    op_name: str
    node_type: str  # Changed from op_type: OpType to node_type: str
    config: Dict[str, Any]
    status: DAGNodeStatus = DAGNodeStatus.PENDING
    dependencies: Set[str] = field(default_factory=set)
    dependents: Set[str] = field(default_factory=set)
    execution_order: int = -1
    estimated_duration: float = 0.0
    actual_duration: float = 0.0
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "node_id": self.node_id,
            "op_name": self.op_name,
            "node_type": self.node_type,
            "config": self.config,
            "status": self.status.value,
            "dependencies": list(self.dependencies),
            "dependents": list(self.dependents),
            "execution_order": self.execution_order,
            "estimated_duration": self.estimated_duration,
            "actual_duration": self.actual_duration,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }


@dataclass
class DAGEdge:
    """Edge in the execution DAG."""

    source_id: str
    target_id: str
    edge_type: DAGEdgeType = DAGEdgeType.SEQUENTIAL
    condition: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type.value,
            "condition": self.condition,
            "metadata": self.metadata,
        }


class PipelineDAG:
    """Pipeline DAG representation and execution planner."""

    def __init__(self, work_dir: str):
        """Initialize the Pipeline DAG.

        Args:
            work_dir: Working directory for storing DAG execution plans and logs
        """
        self.work_dir = Path(work_dir)
        # Remove the separate dag_execution subdirectory - save directly in work_dir
        # self.dag_dir = self.work_dir / "dag_execution"
        # self.dag_dir.mkdir(parents=True, exist_ok=True)
        self.dag_dir = self.work_dir  # Use work_dir directly

        # DAG structure - support both DAGNode objects and dict nodes from strategies
        self.nodes: Dict[str, Any] = {}
        self.edges: List[DAGEdge] = []  # Not currently populated by strategies
        self.execution_plan: List[str] = []  # Not currently populated by strategies
        self.parallel_groups: List[List[str]] = []  # Not currently populated by strategies

    def save_execution_plan(self, filename: str = "dag_execution_plan.json") -> str:
        """Save the execution plan to file.

        Args:
            filename: Name of the file to save the plan

        Returns:
            Path to the saved file
        """
        # Save only static DAG structure, not execution state
        static_nodes = {}
        for node_id, node in self.nodes.items():
            # Handle both DAGNode objects and dict nodes from strategies
            if hasattr(node, "to_dict"):
                # DAGNode object
                static_node_data = {
                    "node_id": node.node_id,
                    "op_name": node.op_name,
                    "op_type": node.op_type.value,
                    "config": node.config,
                    "dependencies": list(node.dependencies),
                    "dependents": list(node.dependents),
                    "execution_order": node.execution_order,
                    "estimated_duration": node.estimated_duration,
                    "metadata": node.metadata,
                }
            else:
                # Dict node from strategy
                static_node_data = {
                    "node_id": node["node_id"],
                    "op_name": node.get("operation_name", ""),
                    "op_type": node.get("node_type", "operation"),
                    "config": node.get("config", {}),
                    "dependencies": node.get("dependencies", []),
                    "dependents": node.get("dependents", []),
                    "execution_order": node.get("execution_order", 0),
                    "estimated_duration": node.get("estimated_duration", 0.0),
                    "metadata": node.get("metadata", {}),
                }
            static_nodes[node_id] = static_node_data

        plan_data = {
            "nodes": static_nodes,
            "edges": [edge.to_dict() for edge in self.edges],
            "execution_plan": self.execution_plan,
            "parallel_groups": self.parallel_groups,
            "metadata": {
                "created_at": time.time(),
                "total_nodes": len(self.nodes),
                "total_edges": len(self.edges),
                "parallel_groups_count": len(self.parallel_groups),
            },
        }

        plan_path = self.dag_dir / filename
        with open(plan_path, "w") as f:
            json.dump(plan_data, f, indent=2, default=str)

        logger.info(f"Execution plan saved to: {plan_path}")
        return str(plan_path)

    def load_execution_plan(self, filename: str = "dag_execution_plan.json") -> bool:
        """Load execution plan from file.

        Args:
            filename: Name of the file to load the plan from

        Returns:
            True if loaded successfully, False otherwise
        """
        plan_path = self.dag_dir / filename
        if not plan_path.exists():
            logger.warning(f"Execution plan file not found: {plan_path}")
            return False

        try:
            with open(plan_path, "r") as f:
                plan_data = json.load(f)

            # Reconstruct nodes (static structure only)
            self.nodes.clear()
            for node_id, node_data in plan_data["nodes"].items():
                # Keep as dict to match strategy format
                self.nodes[node_id] = {
                    "node_id": node_data["node_id"],
                    "operation_name": node_data.get("op_name", node_data.get("operation_name", "")),
                    "node_type": node_data.get("node_type", node_data.get("op_type", "operation")),
                    "config": node_data.get("config", {}),
                    "status": "pending",  # Always start with pending status
                    "dependencies": node_data.get("dependencies", []),
                    "dependents": node_data.get("dependents", []),
                    "execution_order": node_data.get("execution_order", 0),
                    "estimated_duration": node_data.get("estimated_duration", 0.0),
                    "actual_duration": 0.0,  # Reset execution state
                    "start_time": None,  # Reset execution state
                    "end_time": None,  # Reset execution state
                    "error_message": None,  # Reset execution state
                    "metadata": node_data.get("metadata", {}),
                }

            # Reconstruct edges
            self.edges.clear()
            for edge_data in plan_data["edges"]:
                edge = DAGEdge(
                    source_id=edge_data["source_id"],
                    target_id=edge_data["target_id"],
                    edge_type=DAGEdgeType(edge_data["edge_type"]),
                    condition=edge_data["condition"],
                    metadata=edge_data["metadata"],
                )
                self.edges.append(edge)

            # Load execution plan and parallel groups
            self.execution_plan = plan_data["execution_plan"]
            self.parallel_groups = plan_data["parallel_groups"]

            logger.info(f"Execution plan loaded from: {plan_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load execution plan: {e}")
            return False

    def visualize(self) -> str:
        """Generate a string representation of the DAG for visualization."""
        if not self.nodes:
            return "Empty DAG"

        lines = ["DAG Execution Plan:"]
        lines.append("=" * 50)

        # Show execution order (if available, otherwise show all nodes)
        if self.execution_plan:
            lines.append("Execution Order:")
            for i, node_id in enumerate(self.execution_plan):
                node = self.nodes[node_id]
                status = DAGNodeStatus(node.get("status", "pending")) if isinstance(node, dict) else node.status
                op_name = node.get("operation_name", "unknown") if isinstance(node, dict) else node.op_name
                op_type = (
                    node.get("node_type", "operation")
                    if isinstance(node, dict)
                    else getattr(node, "node_type", "operation")
                )

                status_icon = {
                    DAGNodeStatus.PENDING: "⏳",
                    DAGNodeStatus.READY: "✅",
                    DAGNodeStatus.RUNNING: "🔄",
                    DAGNodeStatus.COMPLETED: "✅",
                    DAGNodeStatus.FAILED: "❌",
                    DAGNodeStatus.SKIPPED: "⏭️",
                }.get(status, "❓")

                lines.append(f"  {i+1:2d}. {status_icon} {op_name} ({op_type})")
        else:
            # No execution plan, show all nodes
            lines.append("Nodes:")
            for i, (node_id, node) in enumerate(self.nodes.items()):
                status = DAGNodeStatus(node.get("status", "pending")) if isinstance(node, dict) else node.status
                op_name = node.get("operation_name", "unknown") if isinstance(node, dict) else node.op_name
                op_type = (
                    node.get("node_type", "operation")
                    if isinstance(node, dict)
                    else getattr(node, "node_type", "operation")
                )

                status_icon = {
                    DAGNodeStatus.PENDING: "⏳",
                    DAGNodeStatus.READY: "✅",
                    DAGNodeStatus.RUNNING: "🔄",
                    DAGNodeStatus.COMPLETED: "✅",
                    DAGNodeStatus.FAILED: "❌",
                    DAGNodeStatus.SKIPPED: "⏭️",
                }.get(status, "❓")

                lines.append(f"  {i+1:2d}. {status_icon} {op_name} ({op_type})")

        # Show parallel groups
        if self.parallel_groups:
            lines.append("\nParallel Groups:")
            for i, group in enumerate(self.parallel_groups):
                group_names = []
                for node_id in group:
                    node = self.nodes[node_id]
                    if hasattr(node, "op_name"):
                        group_names.append(node.op_name)
                    else:
                        group_names.append(node.get("operation_name", "unknown"))
                lines.append(f"  Group {i+1}: {', '.join(group_names)}")

        # Show dependencies
        lines.append("\nDependencies:")
        for node_id, node in self.nodes.items():
            dependencies = node.get("dependencies", []) if isinstance(node, dict) else getattr(node, "dependencies", [])
            op_name = (
                node.get("operation_name", "unknown") if isinstance(node, dict) else getattr(node, "op_name", "unknown")
            )

            if dependencies:
                dep_names = []
                for dep_id in dependencies:
                    dep_node = self.nodes.get(dep_id, {})
                    dep_name = (
                        dep_node.get("operation_name", "unknown")
                        if isinstance(dep_node, dict)
                        else getattr(dep_node, "op_name", "unknown")
                    )
                    dep_names.append(dep_name)
                lines.append(f"  {op_name} depends on: {', '.join(dep_names)}")

        return "\n".join(lines)

    def get_ready_nodes(self) -> List[str]:
        """Get list of nodes that are ready to execute (all dependencies completed)."""
        ready_nodes = []
        for node_id, node in self.nodes.items():
            # Handle both DAGNode objects and dict nodes
            if hasattr(node, "status"):
                status = node.status
                dependencies = node.dependencies
            else:
                status = DAGNodeStatus(node.get("status", "pending"))
                dependencies = node.get("dependencies", [])

            if status == DAGNodeStatus.PENDING:
                # Check if all dependencies are completed
                all_deps_completed = all(
                    self._get_node_status(dep_id) == DAGNodeStatus.COMPLETED for dep_id in dependencies
                )
                if all_deps_completed:
                    ready_nodes.append(node_id)
        return ready_nodes

    def _get_node_status(self, node: Any) -> DAGNodeStatus:
        """Get status of a node, handling both DAGNode objects and dict nodes.

        Args:
            node: Can be a node_id (str), DAGNode object, or dict representation of a node.

        Returns:
            DAGNodeStatus of the node.
        """
        if isinstance(node, str):
            # Argument `node` is node_id
            node = self.nodes[node]
        if hasattr(node, "status"):
            return node.status
        elif isinstance(node, dict):
            return DAGNodeStatus(node.get("status", "pending"))
        else:
            return DAGNodeStatus.PENDING

    def mark_node_started(self, node_id: str) -> None:
        """Mark a node as started."""
        if node_id in self.nodes:
            node = self.nodes[node_id]
            current_time = time.time()
            if hasattr(node, "status"):
                node.status = DAGNodeStatus.RUNNING
                node.start_time = current_time
            elif isinstance(node, dict):
                node["status"] = DAGNodeStatus.RUNNING.value
                node["start_time"] = current_time

    def mark_node_completed(self, node_id: str, duration: float = None) -> None:
        """Mark a node as completed."""
        if node_id in self.nodes:
            node = self.nodes[node_id]
            current_time = time.time()
            if hasattr(node, "status"):
                node.status = DAGNodeStatus.COMPLETED
                node.end_time = current_time
                if duration is not None:
                    node.actual_duration = duration
                else:
                    node.actual_duration = current_time - (node.start_time or current_time)
            elif isinstance(node, dict):
                node["status"] = DAGNodeStatus.COMPLETED.value
                node["end_time"] = current_time
                if duration is not None:
                    node["actual_duration"] = duration
                else:
                    node["actual_duration"] = current_time - (node.get("start_time", current_time))

    def mark_node_failed(self, node_id: str, error_message: str) -> None:
        """Mark a node as failed."""
        if node_id in self.nodes:
            node = self.nodes[node_id]
            current_time = time.time()
            if hasattr(node, "status"):
                node.status = DAGNodeStatus.FAILED
                node.end_time = current_time
                node.error_message = error_message
                node.actual_duration = current_time - (node.start_time or current_time)
            elif isinstance(node, dict):
                node["status"] = DAGNodeStatus.FAILED.value
                node["end_time"] = current_time
                node["error_message"] = error_message
                node["actual_duration"] = current_time - (node.get("start_time", current_time))

    def get_execution_summary(self) -> Dict[str, Any]:
        """Get execution summary statistics."""
        total_nodes = len(self.nodes)

        def get_node_duration(node):
            if hasattr(node, "actual_duration"):
                duration = node.actual_duration
                return duration if duration is not None else 0
            elif isinstance(node, dict):
                duration = node.get("actual_duration")
                return duration if duration is not None else 0
            else:
                return 0

        completed_nodes = sum(
            1 for node in self.nodes.values() if self._get_node_status(node) == DAGNodeStatus.COMPLETED
        )
        failed_nodes = sum(1 for node in self.nodes.values() if self._get_node_status(node) == DAGNodeStatus.FAILED)
        running_nodes = sum(1 for node in self.nodes.values() if self._get_node_status(node) == DAGNodeStatus.RUNNING)
        pending_nodes = sum(1 for node in self.nodes.values() if self._get_node_status(node) == DAGNodeStatus.PENDING)

        total_duration = sum(get_node_duration(node) for node in self.nodes.values())

        return {
            "total_nodes": total_nodes,
            "completed_nodes": completed_nodes,
            "failed_nodes": failed_nodes,
            "running_nodes": running_nodes,
            "pending_nodes": pending_nodes,
            "completion_percentage": (completed_nodes / total_nodes * 100) if total_nodes > 0 else 0,
            "total_duration": total_duration,
            "parallel_groups_count": len(self.parallel_groups),
        }

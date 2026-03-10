"""
Pipeline DAG Representation for Data-Juicer Pipelines

This module provides Pipeline DAG (Directed Acyclic Graph) representation
for tracking execution state, dependencies, and monitoring.

Refactored to:
- Live in core/executor/ where it's actually used
- Use dict nodes consistently (matching strategy output)
- Share status enum with DAGNodeStatusTransition
"""

import json
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List

from loguru import logger


class DAGNodeStatus(Enum):
    """Status of a DAG node during execution.

    State machine transitions (enforced by DAGNodeStatusTransition):
    - pending -> running (node starts execution)
    - pending -> completed (skipped - already done in previous run)
    - running -> completed (node finishes successfully)
    - running -> failed (node fails)
    - failed -> running (node retries)
    - completed is terminal (no transitions out)
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class PipelineDAG:
    """Pipeline DAG representation and execution state tracker.

    Stores DAG nodes as dicts (matching strategy output format).
    Provides methods for state management, serialization, and visualization.
    """

    def __init__(self, work_dir: str):
        """Initialize the Pipeline DAG.

        Args:
            work_dir: Working directory for storing DAG execution plans
        """
        self.work_dir = Path(work_dir)
        self.dag_dir = self.work_dir  # Save directly in work_dir

        # Pipeline nodes - dicts from strategies
        # Dependencies are stored in nodes themselves: node["dependencies"]
        self.nodes: Dict[str, Dict[str, Any]] = {}

        # Reserved for future DAG enhancements:
        # - edges: explicit edge objects for complex dependencies
        # - execution_plan: optimized execution order
        # - parallel_groups: ops that can run concurrently
        self.edges: List[Any] = []
        self.execution_plan: List[str] = []
        self.parallel_groups: List[List[str]] = []

    def save_execution_plan(self, filename: str = "dag_execution_plan.json") -> str:
        """Save the execution plan to file.

        Args:
            filename: Name of the file to save the plan

        Returns:
            Path to the saved file
        """
        static_nodes = {}
        for node_id, node in self.nodes.items():
            static_nodes[node_id] = {
                "node_id": node["node_id"],
                "operation_name": node.get("operation_name", ""),
                "node_type": node.get("node_type", "operation"),
                "partition_id": node.get("partition_id"),
                "config": node.get("config", {}),
                "dependencies": node.get("dependencies", []),
                "execution_order": node.get("execution_order", 0),
                "estimated_duration": node.get("estimated_duration", 0.0),
                "metadata": node.get("metadata", {}),
            }

        plan_data = {
            "nodes": static_nodes,
            "metadata": {
                "created_at": time.time(),
                "total_nodes": len(self.nodes),
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

            self.nodes.clear()
            for node_id, node_data in plan_data["nodes"].items():
                self.nodes[node_id] = {
                    "node_id": node_data["node_id"],
                    "operation_name": node_data.get("operation_name", ""),
                    "node_type": node_data.get("node_type", "operation"),
                    "partition_id": node_data.get("partition_id"),
                    "config": node_data.get("config", {}),
                    "dependencies": node_data.get("dependencies", []),
                    "execution_order": node_data.get("execution_order", 0),
                    "estimated_duration": node_data.get("estimated_duration", 0.0),
                    "metadata": node_data.get("metadata", {}),
                    # Reset execution state
                    "status": DAGNodeStatus.PENDING.value,
                    "actual_duration": None,
                    "start_time": None,
                    "end_time": None,
                    "error_message": None,
                }

            logger.info(f"Execution plan loaded from: {plan_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load execution plan: {e}")
            return False

    def mark_node_started(self, node_id: str) -> None:
        """Mark a node as started (running)."""
        if node_id in self.nodes:
            node = self.nodes[node_id]
            node["status"] = DAGNodeStatus.RUNNING.value
            node["start_time"] = time.time()

    def mark_node_completed(self, node_id: str, duration: float = None) -> None:
        """Mark a node as completed."""
        if node_id in self.nodes:
            node = self.nodes[node_id]
            current_time = time.time()
            node["status"] = DAGNodeStatus.COMPLETED.value
            node["end_time"] = current_time
            if duration is not None:
                node["actual_duration"] = duration
            else:
                start = node.get("start_time") or current_time
                node["actual_duration"] = current_time - start

    def mark_node_failed(self, node_id: str, error_message: str) -> None:
        """Mark a node as failed."""
        if node_id in self.nodes:
            node = self.nodes[node_id]
            current_time = time.time()
            node["status"] = DAGNodeStatus.FAILED.value
            node["end_time"] = current_time
            node["error_message"] = error_message
            start = node.get("start_time") or current_time
            node["actual_duration"] = current_time - start

    def get_node_status(self, node_id: str) -> DAGNodeStatus:
        """Get status of a node by ID.

        Args:
            node_id: The node identifier

        Returns:
            DAGNodeStatus of the node
        """
        if node_id not in self.nodes:
            return DAGNodeStatus.PENDING
        status_str = self.nodes[node_id].get("status", "pending")
        return DAGNodeStatus(status_str)

    def get_ready_nodes(self) -> List[str]:
        """Get list of nodes ready to execute (all dependencies completed)."""
        ready_nodes = []
        for node_id, node in self.nodes.items():
            if node.get("status", "pending") != DAGNodeStatus.PENDING.value:
                continue

            dependencies = node.get("dependencies", [])
            all_deps_completed = all(self.get_node_status(dep_id) == DAGNodeStatus.COMPLETED for dep_id in dependencies)
            if all_deps_completed:
                ready_nodes.append(node_id)

        return ready_nodes

    def get_execution_summary(self) -> Dict[str, Any]:
        """Get execution summary statistics."""
        total_nodes = len(self.nodes)

        completed = sum(1 for n in self.nodes.values() if n.get("status") == DAGNodeStatus.COMPLETED.value)
        failed = sum(1 for n in self.nodes.values() if n.get("status") == DAGNodeStatus.FAILED.value)
        running = sum(1 for n in self.nodes.values() if n.get("status") == DAGNodeStatus.RUNNING.value)
        pending = sum(1 for n in self.nodes.values() if n.get("status", "pending") == DAGNodeStatus.PENDING.value)

        total_duration = sum(n.get("actual_duration") or 0 for n in self.nodes.values())

        return {
            "total_nodes": total_nodes,
            "completed_nodes": completed,
            "failed_nodes": failed,
            "running_nodes": running,
            "pending_nodes": pending,
            "completion_percentage": (completed / total_nodes * 100) if total_nodes > 0 else 0,
            "total_duration": total_duration,
        }

    def visualize(self) -> str:
        """Generate a string representation of the DAG for visualization."""
        if not self.nodes:
            return "Empty DAG"

        lines = ["DAG Execution Plan:"]
        lines.append("=" * 50)

        status_icons = {
            DAGNodeStatus.PENDING.value: "[ ]",
            DAGNodeStatus.RUNNING.value: "[~]",
            DAGNodeStatus.COMPLETED.value: "[x]",
            DAGNodeStatus.FAILED.value: "[!]",
        }

        # Sort by execution order
        sorted_nodes = sorted(self.nodes.items(), key=lambda x: x[1].get("execution_order", 0))

        lines.append("\nNodes:")
        for i, (node_id, node) in enumerate(sorted_nodes):
            status = node.get("status", "pending")
            op_name = node.get("operation_name", "unknown")
            node_type = node.get("node_type", "operation")
            partition_id = node.get("partition_id")

            icon = status_icons.get(status, "[?]")
            partition_info = f" (partition {partition_id})" if partition_id is not None else ""

            lines.append(f"  {i+1:2d}. {icon} {op_name}{partition_info} [{node_type}]")

        # Show dependencies
        lines.append("\nDependencies:")
        for node_id, node in sorted_nodes:
            dependencies = node.get("dependencies", [])
            if dependencies:
                op_name = node.get("operation_name", "unknown")
                dep_names = []
                for dep_id in dependencies:
                    dep_node = self.nodes.get(dep_id, {})
                    dep_names.append(dep_node.get("operation_name", dep_id))
                lines.append(f"  {op_name} <- {', '.join(dep_names)}")

        return "\n".join(lines)

"""
DAG Execution Mixin for Data-Juicer Executors

This mixin provides AST-based pipeline parsing and DAG execution planning
that can be integrated into existing executors to provide intelligent
pipeline analysis and execution monitoring.
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from data_juicer.core.executor.dag_execution_strategies import (
    DAGExecutionStrategy,
    NonPartitionedDAGStrategy,
    PartitionedDAGStrategy,
    is_global_operation,
)
from data_juicer.core.executor.event_logging_mixin import EventType
from data_juicer.core.pipeline_ast import PipelineAST
from data_juicer.core.pipeline_dag import DAGNodeStatus, PipelineDAG


class DAGExecutionMixin:
    """
    Mixin that provides DAG-based execution planning and monitoring.

    This mixin can be integrated into any executor to provide:
    - AST-based pipeline parsing
    - DAG execution planning
    - Execution monitoring tied to DAG nodes
    - Event logging with DAG context
    """

    def __init__(self):
        """Initialize the DAG execution mixin."""
        self.pipeline_dag: Optional[PipelineDAG] = None
        self.pipeline_ast: Optional[PipelineAST] = None
        self.dag_initialized = False
        self.current_dag_node: Optional[str] = None
        self.dag_execution_start_time: Optional[float] = None
        self.dag_execution_strategy: Optional[DAGExecutionStrategy] = None

    def _initialize_dag_execution(self, cfg) -> None:
        """Initialize DAG execution planning with appropriate strategy."""
        if self.dag_initialized:
            return

        logger.info("Initializing DAG execution planning...")

        # Determine execution strategy based on executor type
        self.dag_execution_strategy = self._create_execution_strategy(cfg)

        # Generate DAG using strategy
        self._generate_dag_with_strategy(cfg)

        self.dag_initialized = True
        self.dag_execution_start_time = time.time()

        logger.info(
            f"DAG execution planning initialized: {len(self.pipeline_dag.nodes)} nodes, {len(self.pipeline_dag.edges)} edges"
        )

    def _create_execution_strategy(self, cfg) -> DAGExecutionStrategy:
        """Create the appropriate execution strategy based on executor type."""
        if self._is_partitioned_executor():
            return self._create_partitioned_strategy(cfg)
        else:
            return self._create_non_partitioned_strategy(cfg)

    def _is_partitioned_executor(self) -> bool:
        """Determine if this is a partitioned executor."""
        return hasattr(self, "executor_type") and self.executor_type == "ray_partitioned"

    def _create_partitioned_strategy(self, cfg) -> DAGExecutionStrategy:
        """Create partitioned execution strategy."""
        # Partition count should be determined by the executor, not the DAG mixin
        # Get it from the executor's attribute if available, otherwise use a default
        if hasattr(self, "num_partitions"):
            num_partitions = self.num_partitions
        else:
            # Last resort: use a default (shouldn't happen in practice)
            logger.error("Partition count not found in executor")
            raise ValueError("Partition count not found in executor")

        return PartitionedDAGStrategy(num_partitions)

    def _create_non_partitioned_strategy(self, cfg) -> DAGExecutionStrategy:
        """Create non-partitioned execution strategy."""
        return NonPartitionedDAGStrategy()

    def _generate_dag_with_strategy(self, cfg) -> None:
        """Generate DAG using the selected strategy."""
        # Create pipeline AST
        self.pipeline_ast = PipelineAST()
        config = {"process": cfg.process}
        self.pipeline_ast.build_from_config(config)

        # Get operations from AST
        operations = self._get_operations_from_config(cfg)

        # Get strategy-specific parameters
        strategy_kwargs = self._get_strategy_kwargs(cfg)

        # Generate nodes using strategy
        nodes = self.dag_execution_strategy.generate_dag_nodes(operations, **strategy_kwargs)

        # Build dependencies using strategy
        self.dag_execution_strategy.build_dependencies(nodes, operations, **strategy_kwargs)

        # Create PipelineDAG instance
        self.pipeline_dag = PipelineDAG(cfg.work_dir)
        self.pipeline_dag.nodes = nodes

        # Log DAG initialization
        if hasattr(self, "log_dag_build_start"):
            ast_info = {
                "config_source": "process_config",
                "build_start_time": time.time(),
                "node_count": len(self.pipeline_ast.root.children) if self.pipeline_ast.root else 0,
                "depth": self._calculate_ast_depth(self.pipeline_ast.root) if self.pipeline_ast.root else 0,
                "operation_types": (
                    self._extract_operation_types(self.pipeline_ast.root) if self.pipeline_ast.root else []
                ),
            }
            self.log_dag_build_start(ast_info)

        if hasattr(self, "log_dag_build_complete"):
            dag_info = {
                "node_count": len(self.pipeline_dag.nodes),
                "edge_count": len(self.pipeline_dag.edges),
                "parallel_groups_count": len(self.pipeline_dag.parallel_groups),
                "execution_plan_length": len(self.pipeline_dag.execution_plan),
                "build_duration": time.time() - (self.dag_execution_start_time or time.time()),
            }
            self.log_dag_build_complete(dag_info)

        # Save execution plan
        if self.pipeline_dag:
            plan_path = self.pipeline_dag.save_execution_plan()
            if hasattr(self, "log_dag_execution_plan_saved"):
                dag_info = {
                    "node_count": len(self.pipeline_dag.nodes),
                    "edge_count": len(self.pipeline_dag.edges),
                    "parallel_groups_count": len(self.pipeline_dag.parallel_groups),
                }
                self.log_dag_execution_plan_saved(plan_path, dag_info)

    def _get_operations_from_config(self, cfg) -> List:
        """Get operations from configuration - can be overridden by executors."""
        # Default implementation - create operation instances
        operations = []
        for op_config in cfg.process:
            op_name = list(op_config.keys())[0]
            op_args = op_config[op_name] or {}

            # Import and instantiate operation
            from data_juicer.ops import OPERATORS

            try:
                op_class = OPERATORS.modules[op_name]
                operation = op_class(**op_args)
                operations.append(operation)
            except KeyError:
                # If operation not found, create a mock operation for DAG planning
                logger.warning(f"Operation {op_name} not found in OPERATORS registry, creating mock for DAG planning")

                class MockOperation:
                    def __init__(self, name, **kwargs):
                        self._name = name
                        self.config = kwargs

                operation = MockOperation(op_name, **op_args)
                operations.append(operation)

        return operations

    def _get_strategy_kwargs(self, cfg) -> Dict[str, Any]:
        """Get strategy-specific parameters - can be overridden by executors."""
        kwargs = {}

        if self._is_partitioned_executor():
            kwargs["convergence_points"] = self._detect_convergence_points(cfg)

        return kwargs

    def _detect_convergence_points(self, cfg) -> List[int]:
        """Detect convergence points - can be overridden by executors."""
        operations = self._get_operations_from_config(cfg)
        convergence_points = []

        for op_idx, op in enumerate(operations):
            # Detect global operations (deduplicators, etc.)
            if is_global_operation(op):
                convergence_points.append(op_idx)

            # Detect manual convergence points
            if hasattr(op, "converge_after") and op.converge_after:
                convergence_points.append(op_idx)

        return convergence_points

    def _get_dag_node_for_operation(self, op_name: str, op_idx: int, **kwargs) -> Optional[str]:
        """Get the DAG node ID for a given operation using strategy."""
        if not self.dag_execution_strategy:
            return None

        return self.dag_execution_strategy.get_dag_node_id(op_name, op_idx, **kwargs)

    def _mark_dag_node_started(self, node_id: str) -> None:
        """Mark a DAG node as started."""
        if not self.pipeline_dag or node_id not in self.pipeline_dag.nodes:
            return

        node = self.pipeline_dag.nodes[node_id]
        self.pipeline_dag.mark_node_started(node_id)
        self.current_dag_node = node_id

        # Log DAG node start
        if hasattr(self, "log_dag_node_start"):
            node_info = {
                "op_name": node.get("op_name") or node.get("operation_name", ""),
                "op_type": node.get("op_type") or node.get("node_type", "operation"),
                "execution_order": node.get("execution_order", 0),
            }
            self.log_dag_node_start(node_id, node_info)

    def _mark_dag_node_completed(self, node_id: str, duration: float = None) -> None:
        """Mark a DAG node as completed."""
        if not self.pipeline_dag or node_id not in self.pipeline_dag.nodes:
            return

        node = self.pipeline_dag.nodes[node_id]
        self.pipeline_dag.mark_node_completed(node_id, duration)

        # Log DAG node completion
        if hasattr(self, "log_dag_node_complete"):
            node_info = {
                "op_name": node.get("op_name") or node.get("operation_name", ""),
                "op_type": node.get("op_type") or node.get("node_type", "operation"),
                "execution_order": node.get("execution_order", 0),
            }
            self.log_dag_node_complete(node_id, node_info, duration or 0)

        self.current_dag_node = None

    def _mark_dag_node_failed(self, node_id: str, error_message: str, duration: float = 0) -> None:
        """Mark a DAG node as failed."""
        if not self.pipeline_dag or node_id not in self.pipeline_dag.nodes:
            return

        node = self.pipeline_dag.nodes[node_id]
        self.pipeline_dag.mark_node_failed(node_id, error_message)

        # Log DAG node failure
        if hasattr(self, "log_dag_node_failed"):
            node_info = {
                "op_name": node.get("op_name") or node.get("operation_name", ""),
                "op_type": node.get("op_type") or node.get("node_type", "operation"),
                "execution_order": node.get("execution_order", 0),
            }
            self.log_dag_node_failed(node_id, node_info, error_message, duration)

        self.current_dag_node = None

    def _log_operation_with_dag_context(self, op_name: str, op_idx: int, event_type: str, **kwargs) -> None:
        """Log an operation event with DAG context."""
        # Get the corresponding DAG node
        node_id = self._get_dag_node_for_operation(op_name, op_idx)

        # Add DAG node ID to metadata if found
        if "metadata" not in kwargs:
            kwargs["metadata"] = {}

        if node_id:
            kwargs["metadata"]["dag_node_id"] = node_id
        else:
            # Log warning if DAG node not found
            logger.warning(f"DAG node not found for operation {op_name} (idx {op_idx})")

        # Call the original logging method with correct parameters
        if event_type == "op_start" and hasattr(self, "log_op_start"):
            self.log_op_start(0, op_name, op_idx, kwargs.get("metadata", {}))
        elif event_type == "op_complete" and hasattr(self, "log_op_complete"):
            self.log_op_complete(
                0,
                op_name,
                op_idx,
                kwargs.get("duration", 0),
                kwargs.get("checkpoint_path"),
                kwargs.get("input_rows", 0),
                kwargs.get("output_rows", 0),
            )
        elif event_type == "op_failed" and hasattr(self, "log_op_failed"):
            self.log_op_failed(0, op_name, op_idx, kwargs.get("error", "Unknown error"), kwargs.get("retry_count", 0))

    def log_op_start(self, partition_id, operation_name, operation_idx, op_args, **kwargs):
        """Override to add DAG context to operation start events."""
        # Get the corresponding DAG node
        node_id = self._get_dag_node_for_operation(operation_name, operation_idx, partition_id=partition_id)

        # Create metadata with DAG context
        if "metadata" not in kwargs:
            kwargs["metadata"] = {}
        if node_id:
            kwargs["metadata"]["dag_node_id"] = node_id
        else:
            logger.warning(f"DAG node not found for operation {operation_name} (idx {operation_idx})")

        # Call the parent method with metadata
        super().log_op_start(partition_id, operation_name, operation_idx, op_args, **kwargs)

    def log_op_complete(
        self, partition_id, operation_name, operation_idx, duration, checkpoint_path, input_rows, output_rows, **kwargs
    ):
        """Override to add DAG context to operation complete events."""
        # Get the corresponding DAG node
        node_id = self._get_dag_node_for_operation(operation_name, operation_idx, partition_id=partition_id)

        # Create metadata with DAG context
        if "metadata" not in kwargs:
            kwargs["metadata"] = {}
        if node_id:
            kwargs["metadata"]["dag_node_id"] = node_id
        else:
            logger.warning(f"DAG node not found for operation {operation_name} (idx {operation_idx})")

        # Call the parent method with metadata
        super().log_op_complete(
            partition_id,
            operation_name,
            operation_idx,
            duration,
            checkpoint_path,
            input_rows,
            output_rows,
            **kwargs,
        )

    def log_op_failed(self, partition_id, operation_name, operation_idx, error_message, retry_count, **kwargs):
        """Override to add DAG context to operation failed events."""
        # Get the corresponding DAG node
        node_id = self._get_dag_node_for_operation(operation_name, operation_idx, partition_id=partition_id)

        # Create metadata with DAG context
        if "metadata" not in kwargs:
            kwargs["metadata"] = {}
        if node_id:
            kwargs["metadata"]["dag_node_id"] = node_id
        else:
            logger.warning(f"DAG node not found for operation {operation_name} (idx {operation_idx})")

        # Call the parent method with metadata
        super().log_op_failed(partition_id, operation_name, operation_idx, error_message, retry_count, **kwargs)

    def _execute_operations_with_dag_monitoring(self, dataset, ops: List) -> None:
        """Execute operations with DAG monitoring."""
        if not self.pipeline_dag:
            logger.warning("Pipeline DAG not initialized, falling back to normal execution")
            dataset.process(ops)
            return

        # Log operation start events for all operations
        for op_idx, op in enumerate(ops):
            op_name = op._name
            node_id = self._get_dag_node_for_operation(op_name, op_idx)

            if node_id:
                # Mark DAG node as started
                self._mark_dag_node_started(node_id)

                # Log operation start with DAG context
                self._log_operation_with_dag_context(op_name, op_idx, "op_start")
            else:
                # Log operation start without DAG context
                logger.warning(f"DAG node not found for operation {op_name}, logging without DAG context")
                if hasattr(self, "log_op_start"):
                    self.log_op_start(0, op_name, op_idx, {})

        # Execute all operations normally (this is what actually processes the data)
        dataset.process(ops)

        # Log operation completion events for all operations
        for op_idx, op in enumerate(ops):
            op_name = op._name
            node_id = self._get_dag_node_for_operation(op_name, op_idx)

            if node_id:
                # Mark DAG node as completed
                self._mark_dag_node_completed(node_id, 0.0)  # Duration will be updated from events

                # Log operation completion with DAG context
                self._log_operation_with_dag_context(
                    op_name, op_idx, "op_complete", duration=0.0, input_rows=0, output_rows=0
                )
            else:
                # Log operation completion without DAG context
                if hasattr(self, "log_op_complete"):
                    self.log_op_complete(0, op_name, op_idx, 0.0, None, 0, 0)

    def _calculate_ast_depth(self, node) -> int:
        """Calculate the depth of an AST node."""
        if not node or not node.children:
            return 0

        max_depth = 0
        for child in node.children:
            child_depth = self._calculate_ast_depth(child)
            max_depth = max(max_depth, child_depth)

        return max_depth + 1

    def _extract_operation_types(self, node) -> List[str]:
        """Extract operation types from AST node."""
        types = set()

        if node and node.op_type.value != "root":
            types.add(node.op_type.value)

        if node and node.children:
            for child in node.children:
                types.update(self._extract_operation_types(child))

        return list(types)

    def get_dag_execution_status(self) -> Dict[str, Any]:
        """Get DAG execution status."""
        if not self.pipeline_dag:
            return {"status": "not_initialized"}

        summary = self.pipeline_dag.get_execution_summary()

        return {
            "status": "running" if summary["pending_nodes"] > 0 else "completed",
            "summary": summary,
            "execution_plan_length": len(self.pipeline_dag.execution_plan),
            "parallel_groups_count": len(self.pipeline_dag.parallel_groups),
            "dag_execution_start_time": self.dag_execution_start_time,
        }

    def visualize_dag_execution_plan(self) -> str:
        """Get visualization of the DAG execution plan."""
        if not self.pipeline_dag:
            return "Pipeline DAG not initialized"

        return self.pipeline_dag.visualize()

    def get_dag_execution_plan_path(self) -> str:
        """Get the path to the saved DAG execution plan."""
        if not self.pipeline_dag:
            # If pipeline_dag is not initialized, try to construct the path from work_dir
            if hasattr(self, "cfg") and hasattr(self.cfg, "work_dir"):
                return str(Path(self.cfg.work_dir) / "dag_execution_plan.json")
            return ""

        # DAG execution plan is now saved directly in the work directory
        return str(self.pipeline_dag.dag_dir / "dag_execution_plan.json")

    def reconstruct_dag_state_from_events(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Reconstruct DAG execution state from event logs.

        Args:
            job_id: The job ID to analyze

        Returns:
            Dictionary containing reconstructed DAG state and resumption information
        """
        if not hasattr(self, "event_logger") or not self.event_logger:
            logger.warning("Event logger not available for DAG state reconstruction")
            return None

        # Get DAG-related events
        dag_events = self.event_logger.get_events(
            event_type=[
                EventType.DAG_BUILD_START,
                EventType.DAG_BUILD_COMPLETE,
                EventType.DAG_NODE_START,
                EventType.DAG_NODE_COMPLETE,
                EventType.DAG_NODE_FAILED,
                EventType.DAG_EXECUTION_PLAN_SAVED,
                EventType.OP_START,
                EventType.OP_COMPLETE,
                EventType.OP_FAILED,
            ]
        )

        # Load the saved DAG execution plan
        dag_plan_path = self.get_dag_execution_plan_path()
        if not os.path.exists(dag_plan_path):
            logger.warning(f"DAG execution plan not found: {dag_plan_path}")
            return None

        try:
            with open(dag_plan_path, "r") as f:
                dag_plan = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load DAG execution plan: {e}")
            return None

        # Reconstruct DAG node states from events
        node_states = {}
        for node_id, node_data in dag_plan.get("nodes", {}).items():
            node_states[node_id] = {
                "node_id": node_id,
                "op_name": node_data.get("op_name"),
                "op_type": node_data.get("op_type"),
                "status": DAGNodeStatus.PENDING.value,
                "execution_order": node_data.get("execution_order", -1),
                "dependencies": node_data.get("dependencies", []),
                "dependents": node_data.get("dependents", []),
                "start_time": None,
                "end_time": None,
                "actual_duration": 0.0,
                "error_message": None,
            }

        # Update node states based on events
        for event in dag_events:
            event_data = event.__dict__ if hasattr(event, "__dict__") else event

            # Handle DAG node events
            if event_data.get("event_type") == EventType.DAG_NODE_START.value:
                node_id = event_data.get("metadata", {}).get("dag_node_id")
                if node_id and node_id in node_states:
                    node_states[node_id]["status"] = DAGNodeStatus.RUNNING.value
                    node_states[node_id]["start_time"] = event_data.get("timestamp")

            elif event_data.get("event_type") == EventType.DAG_NODE_COMPLETE.value:
                node_id = event_data.get("metadata", {}).get("dag_node_id")
                if node_id and node_id in node_states:
                    node_states[node_id]["status"] = DAGNodeStatus.COMPLETED.value
                    node_states[node_id]["end_time"] = event_data.get("timestamp")
                    node_states[node_id]["actual_duration"] = event_data.get("duration", 0.0)

            elif event_data.get("event_type") == EventType.DAG_NODE_FAILED.value:
                node_id = event_data.get("metadata", {}).get("dag_node_id")
                if node_id and node_id in node_states:
                    node_states[node_id]["status"] = DAGNodeStatus.FAILED.value
                    node_states[node_id]["end_time"] = event_data.get("timestamp")
                    node_states[node_id]["actual_duration"] = event_data.get("duration", 0.0)
                    node_states[node_id]["error_message"] = event_data.get("error_message")

            # Handle operation events with DAG context
            elif event_data.get("event_type") in [
                EventType.OP_START.value,
                EventType.OP_COMPLETE.value,
                EventType.OP_FAILED.value,
            ]:
                dag_context = event_data.get("metadata", {}).get("dag_context", {})
                node_id = dag_context.get("dag_node_id")
                if node_id and node_id in node_states:
                    if event_data.get("event_type") == EventType.OP_START.value:
                        node_states[node_id]["status"] = DAGNodeStatus.RUNNING.value
                        node_states[node_id]["start_time"] = event_data.get("timestamp")
                    elif event_data.get("event_type") == EventType.OP_COMPLETE.value:
                        node_states[node_id]["status"] = DAGNodeStatus.COMPLETED.value
                        node_states[node_id]["end_time"] = event_data.get("timestamp")
                        node_states[node_id]["actual_duration"] = event_data.get("duration", 0.0)
                    elif event_data.get("event_type") == EventType.OP_FAILED.value:
                        node_states[node_id]["status"] = DAGNodeStatus.FAILED.value
                        node_states[node_id]["end_time"] = event_data.get("timestamp")
                        node_states[node_id]["actual_duration"] = event_data.get("duration", 0.0)
                        node_states[node_id]["error_message"] = event_data.get("error_message")

        # Calculate completion statistics
        total_nodes = len(node_states)
        completed_nodes = sum(1 for node in node_states.values() if node["status"] == DAGNodeStatus.COMPLETED.value)
        failed_nodes = sum(1 for node in node_states.values() if node["status"] == DAGNodeStatus.FAILED.value)
        running_nodes = sum(1 for node in node_states.values() if node["status"] == DAGNodeStatus.RUNNING.value)
        pending_nodes = sum(1 for node in node_states.values() if node["status"] == DAGNodeStatus.PENDING.value)

        # Determine which nodes are ready to execute
        ready_nodes = []
        for node_id, node_state in node_states.items():
            if node_state["status"] == DAGNodeStatus.PENDING.value:
                # Check if all dependencies are completed
                all_deps_completed = all(
                    node_states[dep_id]["status"] == DAGNodeStatus.COMPLETED.value
                    for dep_id in node_state["dependencies"]
                    if dep_id in node_states
                )
                if all_deps_completed:
                    ready_nodes.append(node_id)

        # Determine resumption strategy
        can_resume = True
        resume_from_node = None

        if failed_nodes > 0:
            # Find the first failed node to resume from
            failed_node_ids = [
                node_id for node_id, state in node_states.items() if state["status"] == DAGNodeStatus.FAILED.value
            ]
            if failed_node_ids:
                # Sort by execution order and take the first
                failed_node_ids.sort(key=lambda x: node_states[x]["execution_order"])
                resume_from_node = failed_node_ids[0]
        elif running_nodes > 0:
            # Find the first running node to resume from
            running_node_ids = [
                node_id for node_id, state in node_states.items() if state["status"] == DAGNodeStatus.RUNNING.value
            ]
            if running_node_ids:
                running_node_ids.sort(key=lambda x: node_states[x]["execution_order"])
                resume_from_node = running_node_ids[0]
        elif ready_nodes:
            # Start from the first ready node
            ready_nodes.sort(key=lambda x: node_states[x]["execution_order"])
            resume_from_node = ready_nodes[0]
        elif completed_nodes == total_nodes:
            can_resume = False  # All nodes completed

        return {
            "job_id": job_id,
            "dag_plan_path": dag_plan_path,
            "node_states": node_states,
            "statistics": {
                "total_nodes": total_nodes,
                "completed_nodes": completed_nodes,
                "failed_nodes": failed_nodes,
                "running_nodes": running_nodes,
                "pending_nodes": pending_nodes,
                "ready_nodes": len(ready_nodes),
                "completion_percentage": (completed_nodes / total_nodes * 100) if total_nodes > 0 else 0,
            },
            "resumption": {
                "can_resume": can_resume,
                "resume_from_node": resume_from_node,
                "ready_nodes": ready_nodes,
                "failed_nodes": [
                    node_id for node_id, state in node_states.items() if state["status"] == DAGNodeStatus.FAILED.value
                ],
                "running_nodes": [
                    node_id for node_id, state in node_states.items() if state["status"] == DAGNodeStatus.RUNNING.value
                ],
            },
            "execution_plan": dag_plan.get("execution_plan", []),
            "parallel_groups": dag_plan.get("parallel_groups", []),
        }

    def resume_dag_execution(self, job_id: str, dataset, ops: List) -> bool:
        """
        Resume DAG execution from the last known state.

        Args:
            job_id: The job ID to resume
            dataset: The dataset to process
            ops: List of operations to execute

        Returns:
            True if resumption was successful, False otherwise
        """
        # Reconstruct DAG state from events
        dag_state = self.reconstruct_dag_state_from_events(job_id)
        if not dag_state:
            logger.error("Failed to reconstruct DAG state for resumption")
            return False

        if not dag_state["resumption"]["can_resume"]:
            logger.info("No resumption needed - all nodes completed")
            return True

        # Load the DAG execution plan
        if not self.pipeline_dag:
            logger.error("Pipeline DAG not initialized")
            return False

        dag_plan_path = dag_state["dag_plan_path"]
        if not self.pipeline_dag.load_execution_plan(dag_plan_path):
            logger.error("Failed to load DAG execution plan for resumption")
            return False

        # Restore node states
        for node_id, node_state in dag_state["node_states"].items():
            if node_id in self.pipeline_dag.nodes:
                node = self.pipeline_dag.nodes[node_id]
                node.status = DAGNodeStatus(node_state["status"])
                node.start_time = node_state["start_time"]
                node.end_time = node_state["end_time"]
                node.actual_duration = node_state["actual_duration"]
                node.error_message = node_state["error_message"]

        logger.info(f"Resuming DAG execution from node: {dag_state['resumption']['resume_from_node']}")
        logger.info(f"Statistics: {dag_state['statistics']}")

        # Execute remaining operations
        resume_from_node = dag_state["resumption"]["resume_from_node"]
        if resume_from_node:
            # Find the operation index for this node
            node_state = dag_state["node_states"][resume_from_node]
            execution_order = node_state["execution_order"]

            # Execute operations starting from the resume point
            for op_idx, op in enumerate(ops):
                if op_idx >= execution_order:
                    op_name = op._name
                    node_id = self._get_dag_node_for_operation(op_name, op_idx)

                    if node_id:
                        # Check if this node was already completed
                        if node_id in dag_state["node_states"]:
                            node_status = dag_state["node_states"][node_id]["status"]
                            if node_status == DAGNodeStatus.COMPLETED.value:
                                logger.info(f"Skipping completed node: {node_id}")
                                continue

                        # Execute the operation with DAG monitoring
                        self._mark_dag_node_started(node_id)
                        self._log_operation_with_dag_context(op_name, op_idx, "op_start")

                        start_time = time.time()
                        try:
                            dataset.process([op])
                            duration = time.time() - start_time
                            self._mark_dag_node_completed(node_id, duration)
                            self._log_operation_with_dag_context(
                                op_name, op_idx, "op_complete", duration=duration, input_rows=0, output_rows=0
                            )
                        except Exception as e:
                            duration = time.time() - start_time
                            error_message = str(e)
                            self._mark_dag_node_failed(node_id, error_message, duration)
                            self._log_operation_with_dag_context(
                                op_name, op_idx, "op_failed", error=error_message, duration=duration
                            )
                            raise

        return True

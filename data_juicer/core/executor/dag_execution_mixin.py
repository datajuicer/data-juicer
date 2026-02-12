"""
DAG Execution Mixin for Data-Juicer Executors

This mixin provides DAG execution planning and monitoring that can be integrated
into existing executors to provide intelligent pipeline analysis and execution tracking.
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from data_juicer.core.executor.dag_execution_strategies import (
    DAGExecutionStrategy,
    DAGNodeStatusTransition,
    NonPartitionedDAGStrategy,
    PartitionedDAGStrategy,
    is_global_operation,
)
from data_juicer.core.executor.event_logging_mixin import EventType
from data_juicer.core.executor.pipeline_dag import DAGNodeStatus, PipelineDAG


class DAGExecutionMixin:
    """
    Mixin that provides DAG-based execution planning and monitoring.

    This mixin can be integrated into any executor to provide:
    - DAG execution planning
    - Execution monitoring tied to DAG nodes
    - Event logging with DAG context
    """

    def __init__(self):
        """Initialize the DAG execution mixin."""
        self.pipeline_dag: Optional[PipelineDAG] = None
        self.dag_initialized = False
        self.current_dag_node: Optional[str] = None
        self.dag_execution_start_time: Optional[float] = None
        self.dag_execution_strategy: Optional[DAGExecutionStrategy] = None
        self._dag_ops: Optional[List] = None  # Cached operations for DAG planning

    def _initialize_dag_execution(self, cfg, ops: List = None) -> None:
        """Initialize DAG execution planning with appropriate strategy.

        Args:
            cfg: Configuration object
            ops: Optional list of already-loaded operations. If provided, avoids
                 redundant operation loading. If None, operations will be loaded
                 from cfg.process.

        Note: For standalone mode (default executor), DAG execution can be disabled
        by setting cfg.use_dag = False. DAG execution is primarily useful for
        distributed/partitioned executors where execution planning and monitoring
        provide significant value.
        """
        if self.dag_initialized:
            return

        # Check if DAG execution is enabled (default: True for distributed executors, False for standalone)
        use_dag = getattr(cfg, "use_dag", None)
        if use_dag is None:
            # Default: enable for partitioned executors, disable for standalone (default executor)
            use_dag = self._is_partitioned_executor() or getattr(self, "executor_type", "default") != "default"

        if not use_dag:
            logger.info("DAG execution disabled for standalone mode")
            self.dag_initialized = True  # Mark as initialized to skip future attempts
            return

        logger.info("Initializing DAG execution planning...")

        # Store ops for reuse (avoid redundant loading)
        self._dag_ops = ops

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
        return getattr(self, "executor_type", None) == "ray_partitioned"

    def _create_partitioned_strategy(self, cfg) -> DAGExecutionStrategy:
        """Create partitioned execution strategy."""
        # Partition count should be determined by the executor, not the DAG mixin
        # Get it from the executor's attribute if available, otherwise use a default
        num_partitions = getattr(self, "num_partitions", None)
        if num_partitions is None:
            # Last resort: use a default (shouldn't happen in practice)
            logger.error("Partition count not found in executor")
            raise ValueError("Partition count not found in executor")

        return PartitionedDAGStrategy(num_partitions)

    def _create_non_partitioned_strategy(self, cfg) -> DAGExecutionStrategy:
        """Create non-partitioned execution strategy."""
        return NonPartitionedDAGStrategy()

    def _generate_dag_with_strategy(self, cfg) -> None:
        """Generate DAG using the selected strategy."""
        # Get operations directly from config
        operations = self._get_operations_from_config(cfg)

        # Get strategy-specific parameters
        strategy_kwargs = self._get_strategy_kwargs(cfg)

        # Generate nodes using strategy
        nodes = self.dag_execution_strategy.generate_dag_nodes(operations, **strategy_kwargs)

        # Build dependencies using strategy
        self.dag_execution_strategy.build_dependencies(nodes, operations, **strategy_kwargs)

        # Validate DAG has no cycles
        if not self.dag_execution_strategy.validate_dag(nodes):
            logger.error("DAG validation failed: cycle detected in dependencies")
            raise ValueError("Invalid DAG: cycle detected in dependencies")

        # Create PipelineDAG instance
        self.pipeline_dag = PipelineDAG(cfg.work_dir)
        self.pipeline_dag.nodes = nodes

        # Log DAG initialization
        if log_method := getattr(self, "log_dag_build_start", None):
            ast_info = {
                "config_source": "process_config",
                "build_start_time": time.time(),
                "node_count": len(operations),
                "depth": len(operations),  # AST is linear, so depth equals number of operations
                "operation_types": self._extract_operation_types_from_ops(operations),
            }
            log_method(ast_info)

        if log_method := getattr(self, "log_dag_build_complete", None):
            dag_info = {
                "node_count": len(self.pipeline_dag.nodes),
                "edge_count": len(self.pipeline_dag.edges),
                "parallel_groups_count": len(self.pipeline_dag.parallel_groups),
                "execution_plan_length": len(self.pipeline_dag.execution_plan),
                "build_duration": time.time() - (self.dag_execution_start_time or time.time()),
            }
            log_method(dag_info)

        # Save execution plan
        if self.pipeline_dag:
            plan_path = self.pipeline_dag.save_execution_plan()
            if log_method := getattr(self, "log_dag_execution_plan_saved", None):
                dag_info = {
                    "node_count": len(self.pipeline_dag.nodes),
                    "edge_count": len(self.pipeline_dag.edges),
                    "parallel_groups_count": len(self.pipeline_dag.parallel_groups),
                }
                log_method(plan_path, dag_info)

    def _get_operations_from_config(self, cfg) -> List:
        """Get operations for DAG planning.

        Returns cached operations if available (passed to _initialize_dag_execution),
        otherwise loads from configuration.
        """
        # Use cached ops if available (avoids redundant loading)
        if hasattr(self, "_dag_ops") and self._dag_ops is not None:
            return self._dag_ops

        # Fallback: load from configuration
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
            if getattr(op, "converge_after", False):
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

        # Validate state transition
        current_status = node.get("status", "pending")
        DAGNodeStatusTransition.validate_and_log(node_id, current_status, "running")

        self.pipeline_dag.mark_node_started(node_id)
        self.current_dag_node = node_id

        # Log DAG node start
        if log_method := getattr(self, "log_dag_node_start", None):
            node_info = {
                "op_name": node.get("op_name") or node.get("operation_name", ""),
                "op_type": node.get("op_type") or node.get("node_type", "operation"),
                "execution_order": node.get("execution_order", 0),
            }
            log_method(node_id, node_info)

    def _mark_dag_node_completed(self, node_id: str, duration: float = None) -> None:
        """Mark a DAG node as completed."""
        if not self.pipeline_dag or node_id not in self.pipeline_dag.nodes:
            return

        node = self.pipeline_dag.nodes[node_id]

        # Validate state transition
        current_status = node.get("status", "pending")
        DAGNodeStatusTransition.validate_and_log(node_id, current_status, "completed")

        self.pipeline_dag.mark_node_completed(node_id, duration)

        # Log DAG node completion
        if log_method := getattr(self, "log_dag_node_complete", None):
            node_info = {
                "op_name": node.get("op_name") or node.get("operation_name", ""),
                "op_type": node.get("op_type") or node.get("node_type", "operation"),
                "execution_order": node.get("execution_order", 0),
            }
            log_method(node_id, node_info, duration or 0)

        self.current_dag_node = None

    def _mark_dag_node_failed(self, node_id: str, error_message: str, duration: float = 0) -> None:
        """Mark a DAG node as failed."""
        if not self.pipeline_dag or node_id not in self.pipeline_dag.nodes:
            return

        node = self.pipeline_dag.nodes[node_id]

        # Validate state transition
        current_status = node.get("status", "pending")
        DAGNodeStatusTransition.validate_and_log(node_id, current_status, "failed")

        self.pipeline_dag.mark_node_failed(node_id, error_message)

        # Log DAG node failure
        if log_method := getattr(self, "log_dag_node_failed", None):
            node_info = {
                "op_name": node.get("op_name") or node.get("operation_name", ""),
                "op_type": node.get("op_type") or node.get("node_type", "operation"),
                "execution_order": node.get("execution_order", 0),
            }
            log_method(node_id, node_info, error_message, duration)

        self.current_dag_node = None

    def _log_operation_with_dag_context(
        self, op_name: str, op_idx: int, event_type: str, partition_id: int = 0, **kwargs
    ) -> None:
        """Log an operation event with DAG context.

        Args:
            op_name: Operation name
            op_idx: Operation index
            event_type: Type of event ("op_start", "op_complete", "op_failed")
            partition_id: Partition ID for partitioned executors (default: 0)
            **kwargs: Additional arguments for logging
        """
        # Get the corresponding DAG node
        node_id = self._get_dag_node_for_operation(op_name, op_idx, partition_id=partition_id)

        # Add DAG node ID to metadata if found
        if "metadata" not in kwargs:
            kwargs["metadata"] = {}

        if node_id:
            kwargs["metadata"]["dag_node_id"] = node_id
        else:
            # Log warning if DAG node not found
            logger.warning(f"DAG node not found for operation {op_name} (idx {op_idx})")

        # Call the original logging method with correct parameters
        if event_type == "op_start" and (log_method := getattr(self, "log_op_start", None)):
            log_method(partition_id, op_name, op_idx, kwargs.get("metadata", {}))
        elif event_type == "op_complete" and (log_method := getattr(self, "log_op_complete", None)):
            log_method(
                partition_id,
                op_name,
                op_idx,
                kwargs.get("duration", 0),
                kwargs.get("checkpoint_path"),
                kwargs.get("input_rows", 0),
                kwargs.get("output_rows", 0),
            )
        elif event_type == "op_failed" and (log_method := getattr(self, "log_op_failed", None)):
            log_method(
                partition_id, op_name, op_idx, kwargs.get("error", "Unknown error"), kwargs.get("retry_count", 0)
            )

    def _pre_execute_operations_with_dag_monitoring(self, ops: List, partition_id: int = 0) -> None:
        """Log operation start events with DAG monitoring before execution.

        This method should be called before dataset.process() to log operation start events.
        Each executor can then call dataset.process() with its own specific parameters.

        Args:
            ops: List of operations that will be executed
            partition_id: Partition ID for partitioned executors (default: 0)
        """
        if not self.pipeline_dag:
            return

        # Log operation start events for all operations
        for op_idx, op in enumerate(ops):
            op_name = op._name
            node_id = self._get_dag_node_for_operation(op_name, op_idx, partition_id=partition_id)

            if node_id:
                # Mark DAG node as started
                self._mark_dag_node_started(node_id)

                # Log operation start with DAG context
                self._log_operation_with_dag_context(op_name, op_idx, "op_start", partition_id=partition_id)
            else:
                # Log operation start without DAG context
                logger.warning(f"DAG node not found for operation {op_name}, logging without DAG context")
                if log_method := getattr(self, "log_op_start", None):
                    log_method(partition_id, op_name, op_idx, {})

    def _post_execute_operations_with_dag_monitoring(
        self, ops: List, partition_id: int = 0, metrics: dict = None
    ) -> None:
        """Log operation completion events with DAG monitoring after execution.

        This method should be called after dataset.process() to log operation completion events.

        Args:
            ops: List of operations that were executed
            partition_id: Partition ID for partitioned executors (default: 0)
            metrics: Optional dict with real execution metrics:
                {
                    'duration': float,
                    'input_rows': int,
                    'output_rows': int,
                    'per_op_metrics': List[dict]  # Optional per-op breakdown
                }
        """
        if not self.pipeline_dag:
            return

        # Default metrics if not provided
        if metrics is None:
            metrics = {"duration": 0.0, "input_rows": 0, "output_rows": 0}

        # Check if we have per-op metrics
        per_op_metrics = metrics.get("per_op_metrics", [])

        # Log operation completion events for all operations
        for op_idx, op in enumerate(ops):
            op_name = op._name
            node_id = self._get_dag_node_for_operation(op_name, op_idx, partition_id=partition_id)

            # Get metrics for this specific op if available
            if per_op_metrics and op_idx < len(per_op_metrics):
                op_metrics = per_op_metrics[op_idx]
            else:
                # We materialize per group, not per op, so we can't measure intermediate row counts
                # Only show what we actually know:
                # - First op: input to group
                # - Last op: output from group
                # - Middle ops: no row counts (unknown)
                num_ops = len(ops)
                op_metrics = {
                    "duration": metrics["duration"] / num_ops if num_ops > 0 else 0.0,
                }

                # Only show input rows for first op in group
                if op_idx == 0 and metrics.get("input_rows"):
                    op_metrics["input_rows"] = metrics["input_rows"]

                # Only show output rows for last op in group
                if op_idx == len(ops) - 1 and metrics.get("output_rows"):
                    op_metrics["output_rows"] = metrics["output_rows"]

            if node_id:
                # Mark DAG node as completed with real duration
                self._mark_dag_node_completed(node_id, op_metrics["duration"])

                # Log operation completion with DAG context
                self._log_operation_with_dag_context(
                    op_name,
                    op_idx,
                    "op_complete",
                    partition_id=partition_id,
                    duration=op_metrics["duration"],
                    input_rows=op_metrics.get("input_rows"),
                    output_rows=op_metrics.get("output_rows"),
                )
            else:
                # Log operation completion without DAG context
                if log_method := getattr(self, "log_op_complete", None):
                    log_method(
                        partition_id,
                        op_name,
                        op_idx,
                        op_metrics["duration"],
                        None,
                        op_metrics.get("input_rows"),
                        op_metrics.get("output_rows"),
                    )

    def _extract_operation_types_from_ops(self, operations: List) -> List[str]:
        """Extract operation types from operations list."""
        types = set()
        for op in operations:
            # Determine op type from operation name or class
            op_name = getattr(op, "_name", "")
            if op_name.endswith("_filter"):
                types.add("filter")
            elif op_name.endswith("_mapper"):
                types.add("mapper")
            elif op_name.endswith("_deduplicator"):
                types.add("deduplicator")
            elif op_name.endswith("_selector"):
                types.add("selector")
            elif op_name.endswith("_grouper"):
                types.add("grouper")
            elif op_name.endswith("_aggregator"):
                types.add("aggregator")
            else:
                # Try to infer from class hierarchy
                from data_juicer.ops.base_op import Filter, Mapper

                if isinstance(op, Filter):
                    types.add("filter")
                elif isinstance(op, Mapper):
                    types.add("mapper")
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
            work_dir = getattr(getattr(self, "cfg", None), "work_dir", None)
            if work_dir:
                return str(Path(work_dir) / "dag_execution_plan.json")
            return ""

        # DAG execution plan is now saved directly in the work directory
        return str(self.pipeline_dag.dag_dir / "dag_execution_plan.json")

    def reconstruct_dag_state_from_events(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Reconstruct DAG execution state from event logs.

        This method has been decomposed into smaller, focused methods for better
        maintainability and testability.

        Args:
            job_id: The job ID to analyze

        Returns:
            Dictionary containing reconstructed DAG state and resumption information
        """
        # Step 1: Validate event logger availability
        if not getattr(self, "event_logger", None):
            logger.warning("Event logger not available for DAG state reconstruction")
            return None

        # Step 2: Load DAG events and execution plan
        dag_events = self._load_dag_events()
        dag_plan = self._load_dag_execution_plan()
        if not dag_plan:
            return None

        # Step 3: Reconstruct node states from plan and events
        node_states = self._initialize_node_states_from_plan(dag_plan)
        self._update_node_states_from_events(node_states, dag_events)

        # Step 4: Calculate statistics
        statistics = self._calculate_dag_statistics(node_states)

        # Step 5: Determine ready nodes
        ready_nodes = self._find_ready_nodes(node_states)

        # Step 6: Determine resumption strategy
        resumption_info = self._determine_resumption_strategy(node_states, ready_nodes, statistics)

        return {
            "job_id": job_id,
            "dag_plan_path": self.get_dag_execution_plan_path(),
            "node_states": node_states,
            "statistics": statistics,
            "resumption": resumption_info,
            "execution_plan": dag_plan.get("execution_plan", []),
            "parallel_groups": dag_plan.get("parallel_groups", []),
        }

    def _load_dag_events(self) -> List[Any]:
        """Load DAG-related events from the event logger.

        Returns:
            List of DAG-related events
        """
        return self.event_logger.get_events(
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

    def _load_dag_execution_plan(self) -> Optional[Dict[str, Any]]:
        """Load the saved DAG execution plan.

        Returns:
            DAG execution plan dictionary, or None if loading fails
        """
        dag_plan_path = self.get_dag_execution_plan_path()
        if not os.path.exists(dag_plan_path):
            logger.warning(f"DAG execution plan not found: {dag_plan_path}")
            return None

        try:
            with open(dag_plan_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load DAG execution plan: {e}")
            return None

    def _initialize_node_states_from_plan(self, dag_plan: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Initialize node states from the DAG execution plan.

        Args:
            dag_plan: The loaded DAG execution plan

        Returns:
            Dictionary mapping node_id to initial node state
        """
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
        return node_states

    def _update_node_states_from_events(self, node_states: Dict[str, Dict[str, Any]], dag_events: List[Any]) -> None:
        """Update node states based on events.

        Args:
            node_states: Dictionary of node states to update (modified in-place)
            dag_events: List of DAG-related events
        """
        for event in dag_events:
            event_data = getattr(event, "__dict__", event)

            # Handle DAG node events
            if event_data.get("event_type") == EventType.DAG_NODE_START.value:
                self._handle_dag_node_start_event(event_data, node_states)
            elif event_data.get("event_type") == EventType.DAG_NODE_COMPLETE.value:
                self._handle_dag_node_complete_event(event_data, node_states)
            elif event_data.get("event_type") == EventType.DAG_NODE_FAILED.value:
                self._handle_dag_node_failed_event(event_data, node_states)
            # Handle operation events with DAG context
            elif event_data.get("event_type") in [
                EventType.OP_START.value,
                EventType.OP_COMPLETE.value,
                EventType.OP_FAILED.value,
            ]:
                self._handle_operation_event(event_data, node_states)

    def _handle_dag_node_start_event(self, event_data: Dict[str, Any], node_states: Dict[str, Dict[str, Any]]) -> None:
        """Handle DAG_NODE_START event."""
        node_id = event_data.get("metadata", {}).get("dag_node_id")
        if node_id and node_id in node_states:
            node_states[node_id]["status"] = DAGNodeStatus.RUNNING.value
            node_states[node_id]["start_time"] = event_data.get("timestamp")

    def _handle_dag_node_complete_event(
        self, event_data: Dict[str, Any], node_states: Dict[str, Dict[str, Any]]
    ) -> None:
        """Handle DAG_NODE_COMPLETE event."""
        node_id = event_data.get("metadata", {}).get("dag_node_id")
        if node_id and node_id in node_states:
            node_states[node_id]["status"] = DAGNodeStatus.COMPLETED.value
            node_states[node_id]["end_time"] = event_data.get("timestamp")
            node_states[node_id]["actual_duration"] = event_data.get("duration", 0.0)

    def _handle_dag_node_failed_event(self, event_data: Dict[str, Any], node_states: Dict[str, Dict[str, Any]]) -> None:
        """Handle DAG_NODE_FAILED event."""
        node_id = event_data.get("metadata", {}).get("dag_node_id")
        if node_id and node_id in node_states:
            node_states[node_id]["status"] = DAGNodeStatus.FAILED.value
            node_states[node_id]["end_time"] = event_data.get("timestamp")
            node_states[node_id]["actual_duration"] = event_data.get("duration", 0.0)
            node_states[node_id]["error_message"] = event_data.get("error_message")

    def _handle_operation_event(self, event_data: Dict[str, Any], node_states: Dict[str, Dict[str, Any]]) -> None:
        """Handle operation events (OP_START, OP_COMPLETE, OP_FAILED) with DAG context."""
        dag_context = event_data.get("metadata", {}).get("dag_context", {})
        node_id = dag_context.get("dag_node_id")
        if not node_id or node_id not in node_states:
            return

        event_type = event_data.get("event_type")
        if event_type == EventType.OP_START.value:
            node_states[node_id]["status"] = DAGNodeStatus.RUNNING.value
            node_states[node_id]["start_time"] = event_data.get("timestamp")
        elif event_type == EventType.OP_COMPLETE.value:
            node_states[node_id]["status"] = DAGNodeStatus.COMPLETED.value
            node_states[node_id]["end_time"] = event_data.get("timestamp")
            node_states[node_id]["actual_duration"] = event_data.get("duration", 0.0)
        elif event_type == EventType.OP_FAILED.value:
            node_states[node_id]["status"] = DAGNodeStatus.FAILED.value
            node_states[node_id]["end_time"] = event_data.get("timestamp")
            node_states[node_id]["actual_duration"] = event_data.get("duration", 0.0)
            node_states[node_id]["error_message"] = event_data.get("error_message")

    def _calculate_dag_statistics(self, node_states: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate DAG execution statistics.

        Args:
            node_states: Dictionary of node states

        Returns:
            Dictionary with statistics
        """
        total_nodes = len(node_states)
        completed_nodes = sum(1 for node in node_states.values() if node["status"] == DAGNodeStatus.COMPLETED.value)
        failed_nodes = sum(1 for node in node_states.values() if node["status"] == DAGNodeStatus.FAILED.value)
        running_nodes = sum(1 for node in node_states.values() if node["status"] == DAGNodeStatus.RUNNING.value)
        pending_nodes = sum(1 for node in node_states.values() if node["status"] == DAGNodeStatus.PENDING.value)

        return {
            "total_nodes": total_nodes,
            "completed_nodes": completed_nodes,
            "failed_nodes": failed_nodes,
            "running_nodes": running_nodes,
            "pending_nodes": pending_nodes,
            "ready_nodes": 0,  # Will be set by caller
            "completion_percentage": (completed_nodes / total_nodes * 100) if total_nodes > 0 else 0,
        }

    def _find_ready_nodes(self, node_states: Dict[str, Dict[str, Any]]) -> List[str]:
        """Find nodes that are ready to execute (all dependencies completed).

        Args:
            node_states: Dictionary of node states

        Returns:
            List of node IDs that are ready to execute
        """
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
        return ready_nodes

    def _determine_resumption_strategy(
        self, node_states: Dict[str, Dict[str, Any]], ready_nodes: List[str], statistics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Determine the resumption strategy based on current DAG state.

        Args:
            node_states: Dictionary of node states
            ready_nodes: List of ready node IDs
            statistics: DAG statistics

        Returns:
            Dictionary with resumption information
        """
        can_resume = True
        resume_from_node = None

        # Priority 1: Resume from failed nodes
        if statistics["failed_nodes"] > 0:
            failed_node_ids = [
                node_id for node_id, state in node_states.items() if state["status"] == DAGNodeStatus.FAILED.value
            ]
            if failed_node_ids:
                failed_node_ids.sort(key=lambda x: node_states[x]["execution_order"])
                resume_from_node = failed_node_ids[0]

        # Priority 2: Resume from running nodes
        elif statistics["running_nodes"] > 0:
            running_node_ids = [
                node_id for node_id, state in node_states.items() if state["status"] == DAGNodeStatus.RUNNING.value
            ]
            if running_node_ids:
                running_node_ids.sort(key=lambda x: node_states[x]["execution_order"])
                resume_from_node = running_node_ids[0]

        # Priority 3: Start from ready nodes
        elif ready_nodes:
            ready_nodes_sorted = sorted(ready_nodes, key=lambda x: node_states[x]["execution_order"])
            resume_from_node = ready_nodes_sorted[0]

        # All nodes completed - cannot resume
        elif statistics["completed_nodes"] == statistics["total_nodes"]:
            can_resume = False

        return {
            "can_resume": can_resume,
            "resume_from_node": resume_from_node,
            "ready_nodes": ready_nodes,
            "failed_nodes": [
                node_id for node_id, state in node_states.items() if state["status"] == DAGNodeStatus.FAILED.value
            ],
            "running_nodes": [
                node_id for node_id, state in node_states.items() if state["status"] == DAGNodeStatus.RUNNING.value
            ],
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

        # Restore node states (nodes are dicts, not objects)
        for node_id, node_state in dag_state["node_states"].items():
            if node_id in self.pipeline_dag.nodes:
                node = self.pipeline_dag.nodes[node_id]
                node["status"] = node_state["status"]
                node["start_time"] = node_state["start_time"]
                node["end_time"] = node_state["end_time"]
                node["actual_duration"] = node_state["actual_duration"]
                node["error_message"] = node_state["error_message"]

        logger.info(f"Resuming DAG execution from node: {dag_state['resumption']['resume_from_node']}")
        logger.info(f"Statistics: {dag_state['statistics']}")

        # Execute remaining operations
        resume_from_node = dag_state["resumption"]["resume_from_node"]
        if resume_from_node:
            # Find the operation index for this node
            node_state = dag_state["node_states"][resume_from_node]
            execution_order = node_state["execution_order"]

            # Collect remaining operations to execute (batch for efficiency)
            remaining_ops = []
            remaining_op_info = []  # (op_idx, op_name, node_id)

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

                        remaining_ops.append(op)
                        remaining_op_info.append((op_idx, op_name, node_id))

            if not remaining_ops:
                logger.info("No remaining operations to execute")
                return True

            # Mark all nodes as started
            for op_idx, op_name, node_id in remaining_op_info:
                self._mark_dag_node_started(node_id)
                self._log_operation_with_dag_context(op_name, op_idx, "op_start")

            # Execute all remaining operations in one batch for efficiency
            # This allows Ray to optimize the execution plan across operations
            start_time = time.time()
            try:
                dataset.process(remaining_ops)
                total_duration = time.time() - start_time

                # Estimate per-operation duration (evenly distributed)
                per_op_duration = total_duration / len(remaining_ops)

                # Mark all nodes as completed
                for op_idx, op_name, node_id in remaining_op_info:
                    self._mark_dag_node_completed(node_id, per_op_duration)
                    self._log_operation_with_dag_context(
                        op_name, op_idx, "op_complete", duration=per_op_duration, input_rows=0, output_rows=0
                    )

                logger.info(f"Resumed execution: {len(remaining_ops)} operations in {total_duration:.2f}s")

            except Exception as e:
                duration = time.time() - start_time
                error_message = str(e)
                # Mark remaining nodes as failed (we don't know exactly which one failed)
                for op_idx, op_name, node_id in remaining_op_info:
                    node = self.pipeline_dag.nodes.get(node_id)
                    if node and node.status != DAGNodeStatus.COMPLETED:
                        self._mark_dag_node_failed(node_id, error_message, duration)
                        self._log_operation_with_dag_context(
                            op_name, op_idx, "op_failed", error=error_message, duration=duration
                        )
                raise

        return True

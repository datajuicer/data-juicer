#!/usr/bin/env python3
"""
Event Logging Mixin for Data-Juicer Executors

This module provides comprehensive event logging capabilities that can be used
by any executor (default, ray, partitioned, etc.) to track operations,
performance, and errors in real-time.

Features:
1. Real-time event logging with configurable levels
2. Event filtering and querying
3. Performance metrics tracking
4. Error tracking with stack traces
5. Status reporting and monitoring
6. Log rotation and cleanup
"""

import json
import os
import re
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional
from uuid import uuid4

from loguru import logger


class EventType(Enum):
    """Types of events that can be logged."""

    JOB_START = "job_start"
    JOB_COMPLETE = "job_complete"
    JOB_FAILED = "job_failed"
    JOB_RESTART = "job_restart"  # New: Job restart event
    PARTITION_START = "partition_start"
    PARTITION_COMPLETE = "partition_complete"
    PARTITION_FAILED = "partition_failed"
    PARTITION_RESUME = "partition_resume"  # New: Partition resume event
    OP_START = "op_start"
    OP_COMPLETE = "op_complete"
    OP_FAILED = "op_failed"
    CHECKPOINT_SAVE = "checkpoint_save"
    CHECKPOINT_LOAD = "checkpoint_load"
    PROCESSING_START = "processing_start"
    PROCESSING_COMPLETE = "processing_complete"
    PROCESSING_ERROR = "processing_error"
    # DAG-specific events
    DAG_BUILD_START = "dag_build_start"
    DAG_BUILD_COMPLETE = "dag_build_complete"
    DAG_NODE_READY = "dag_node_ready"
    DAG_NODE_START = "dag_node_start"
    DAG_NODE_COMPLETE = "dag_node_complete"
    DAG_NODE_FAILED = "dag_node_failed"
    DAG_PARALLEL_GROUP_START = "dag_parallel_group_start"
    DAG_PARALLEL_GROUP_COMPLETE = "dag_parallel_group_complete"
    DAG_EXECUTION_PLAN_SAVED = "dag_execution_plan_saved"
    DAG_EXECUTION_PLAN_LOADED = "dag_execution_plan_loaded"


@dataclass
class Event:
    """Event data structure."""

    event_type: EventType
    timestamp: float
    message: str
    event_id: Optional[str] = None
    job_id: Optional[str] = None
    partition_id: Optional[int] = None
    operation_name: Optional[str] = None
    operation_idx: Optional[int] = None
    status: Optional[str] = None
    duration: Optional[float] = None
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None
    retry_count: Optional[int] = None
    checkpoint_path: Optional[str] = None
    op_args: Optional[Dict[str, Any]] = None
    input_rows: Optional[int] = None
    output_rows: Optional[int] = None
    output_path: Optional[str] = None
    partition_meta: Optional[Dict[str, Any]] = None
    config: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    total_partitions: Optional[int] = None
    successful_partitions: Optional[int] = None
    failed_partitions: Optional[int] = None
    job_duration: Optional[float] = None
    completion_time: Optional[float] = None
    failure_time: Optional[float] = None
    error_type: Optional[str] = None
    # Process and thread tracking
    process_id: Optional[int] = None
    thread_id: Optional[int] = None


class EventLogger:
    """Event logging system with real-time capabilities and JSONL event log for resumability."""

    def __init__(self, log_dir: str, job_id: Optional[str] = None, work_dir: Optional[str] = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        # Use provided job_id or generate a simple timestamp-based one
        self.job_id = job_id or f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}-{uuid4().hex[:6]}"
        self.events: deque = deque(maxlen=10000)
        self.event_lock = threading.Lock()

        # Use work_dir for JSONL file if provided, otherwise use log_dir
        self.jsonl_dir = Path(work_dir) if work_dir else self.log_dir
        self.jsonl_dir.mkdir(parents=True, exist_ok=True)

        # Create timestamped events file
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.jsonl_file = self.jsonl_dir / f"events_{timestamp}.jsonl"

    def log_event(self, event: Event):
        """Log an event (to memory, loguru, and JSONL for resumability)."""
        with self.event_lock:
            event.job_id = self.job_id
            self.events.append(event)
            # Log to file (loguru)
            log_message = self._format_event_for_logging(event)
            logger.info(log_message)
            # Write to JSONL for resumability
            with open(self.jsonl_file, "a") as f:
                f.write(
                    json.dumps(
                        {k: (v.value if isinstance(v, Enum) else v) for k, v in event.__dict__.items() if v is not None}
                    )
                    + "\n"
                )

    def find_latest_events_file(self, work_dir: str) -> Optional[Path]:
        """Find the latest events file in the work directory."""
        events_dir = Path(work_dir)
        if not events_dir.exists():
            return None

        # Find all events files with timestamp pattern
        events_files = list(events_dir.glob("events_*.jsonl"))
        if not events_files:
            return None

        # Sort by modification time and return the latest
        latest_file = max(events_files, key=lambda f: f.stat().st_mtime)
        return latest_file

    def check_job_completion(self, events_file: Path) -> bool:
        """Check if job is already completed by looking for job_complete event."""
        if not events_file.exists():
            return False

        try:
            with open(events_file, "r") as f:
                for line in f:
                    if line.strip():
                        event = json.loads(line.strip())
                        if event.get("event_type") == "job_complete":
                            return True
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Error reading events file {events_file}: {e}")

        return False

    def _format_event_for_logging(self, event: Event) -> str:
        """Format event for logging with enhanced details."""
        parts = [f"EVENT[{event.event_type.value}]", f"TIME[{datetime.fromtimestamp(event.timestamp).isoformat()}]"]

        if event.partition_id is not None:
            parts.append(f"PARTITION[{event.partition_id}]")

        if event.operation_name:
            parts.append(f"OP[{event.operation_name}]")
            if event.operation_idx is not None:
                parts.append(f"OP_IDX[{event.operation_idx}]")

        if event.duration is not None:
            # Handle case where duration might be a string (due to parameter order issues)
            try:
                if isinstance(event.duration, (int, float)):
                    parts.append(f"DURATION[{event.duration:.3f}s]")
                else:
                    parts.append(f"DURATION[{event.duration}]")
            except (ValueError, TypeError):
                parts.append(f"DURATION[{event.duration}]")

        parts.append(f"MSG[{event.message}]")

        if event.error_message:
            parts.append(f"ERROR[{event.error_message}]")

        if event.checkpoint_path:
            parts.append(f"CHECKPOINT[{os.path.basename(event.checkpoint_path)}]")

        if event.output_path:
            parts.append(f"OUTPUT[{os.path.basename(event.output_path)}]")

        if event.metadata:
            # Include key metadata in the log message
            key_metadata = {}
            for key in ["status", "retry_count", "error_type", "operation_class"]:
                if key in event.metadata:
                    key_metadata[key] = event.metadata[key]
            if key_metadata:
                parts.append(f"META[{json.dumps(key_metadata)}]")

        return " | ".join(parts)

    def get_events(
        self,
        event_type: Optional[EventType] = None,
        partition_id: Optional[int] = None,
        operation_name: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        limit: Optional[int] = None,
    ) -> List[Event]:
        """Get events with optional filtering."""
        with self.event_lock:
            filtered_events = []

            for event in self.events:
                # Apply filters
                if event_type and event.event_type != event_type:
                    continue
                if partition_id is not None and event.partition_id != partition_id:
                    continue
                if operation_name and event.operation_name != operation_name:
                    continue
                if start_time and event.timestamp < start_time:
                    continue
                if end_time and event.timestamp > end_time:
                    continue

                filtered_events.append(event)

            # Apply limit
            if limit:
                filtered_events = filtered_events[-limit:]

            return filtered_events

    def generate_status_report(self) -> str:
        """Generate a comprehensive status report."""
        with self.event_lock:
            total_events = len(self.events)
            if total_events == 0:
                return "No events logged yet."

            # Count event types
            event_counts = defaultdict(int)
            error_count = 0
            warning_count = 0

            for event in self.events:
                event_counts[event.event_type.value] += 1

            # Generate report
            report_lines = [
                "=== EVENT LOGGING STATUS REPORT ===",
                f"Total Events: {total_events}",
                f"Errors: {error_count}",
                f"Warnings: {warning_count}",
                "",
                "Event Type Distribution:",
            ]

            for event_type, count in sorted(event_counts.items()):
                percentage = (count / total_events) * 100
                report_lines.append(f"  {event_type}: {count} ({percentage:.1f}%)")

            return "\n".join(report_lines)

    def monitor_events(self, event_type: Optional[EventType] = None) -> Generator[Event, None, None]:
        """Monitor events in real-time."""
        last_event_count = len(self.events)

        while True:
            with self.event_lock:
                current_events = list(self.events)

            # Yield new events
            for event in current_events[last_event_count:]:
                if event_type is None or event.event_type == event_type:
                    yield event

            last_event_count = len(current_events)
            time.sleep(0.1)  # Check every 100ms

    @classmethod
    def list_available_jobs(cls, work_dir: str) -> List[Dict[str, Any]]:
        """List available jobs for resumption from a work directory."""
        available_jobs = []

        if not os.path.exists(work_dir):
            return available_jobs

        # Look for job directories (each job has its own directory)
        for item in os.listdir(work_dir):
            job_work_dir = os.path.join(work_dir, item)
            if os.path.isdir(job_work_dir):
                summary_file = os.path.join(job_work_dir, "job_summary.json")
                if os.path.exists(summary_file):
                    try:
                        with open(summary_file, "r") as f:
                            job_summary = json.load(f)
                        job_summary["work_dir"] = job_work_dir
                        available_jobs.append(job_summary)
                    except Exception as e:
                        logger.warning(f"Failed to load job summary from {summary_file}: {e}")

        return available_jobs


class EventLoggingMixin:
    """Mixin to add event logging capabilities to any executor."""

    def __init__(self, *args, **kwargs):
        """Initialize the mixin."""
        # Initialize event logging if not already done
        if not hasattr(self, "event_logger"):
            self._setup_event_logging()

    def _setup_event_logging(self):
        """Setup event logging for the executor."""
        # Get event logging configuration
        event_config = getattr(self.cfg, "event_logging", {})
        enabled = event_config.get("enabled", True)

        if not enabled:
            self.event_logger = None
            return

        # job_id and work_dir should already be resolved by resolve_job_directories() in config.py
        job_id = getattr(self.cfg, "job_id", None)
        if not job_id:
            raise ValueError(
                "job_id must be set before setting up event logging. "
                "This should have been done by resolve_job_id() in config.py"
            )

        # work_dir already includes job_id after resolve_job_directories
        # Create work directory and subdirectories
        os.makedirs(self.work_dir, exist_ok=True)

        # Use logs directory instead of event_logs
        logs_dir = os.path.join(self.work_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)

        self.event_logger = EventLogger(logs_dir, job_id=job_id, work_dir=self.work_dir)

        logger.info(f"Event logging initialized for {self.executor_type} executor")

    def _update_job_summary(self, status: str, end_time: Optional[float] = None, error_message: Optional[str] = None):
        """Update job summary with completion status."""
        # work_dir already includes job_id after resolve_job_directories
        summary_file = os.path.join(self.work_dir, "job_summary.json")

        if not os.path.exists(summary_file):
            return

        with open(summary_file, "r") as f:
            job_summary = json.load(f)

        job_summary.update(
            {
                "status": status,
                "end_time": end_time or time.time(),
                "duration": (end_time or time.time()) - job_summary.get("start_time", time.time()),
                "error_message": error_message,
            }
        )

        with open(summary_file, "w") as f:
            json.dump(job_summary, f, indent=2, default=str)

        # Display completion info
        if status == "completed":
            logger.info("=" * 60)
            logger.info("DataJuicer Job Completed Successfully")
            logger.info(f"Duration: {job_summary['duration']:.2f} seconds")
            logger.info("=" * 60)
        elif status == "failed":
            logger.error("=" * 60)
            logger.error("DataJuicer Job Failed")
            logger.error(f"Error: {error_message}")
            logger.error(f"Duration: {job_summary['duration']:.2f} seconds")
            logger.error("=" * 60)
            logger.error("To resume this job, use:")
            logger.error(f"  {job_summary['resumption_command']}")
            logger.error("=" * 60)

    def _load_job_summary(self) -> Optional[Dict[str, Any]]:
        """Load job summary if it exists."""
        # work_dir already includes job_id after resolve_job_directories
        summary_file = os.path.join(self.work_dir, "job_summary.json")

        if os.path.exists(summary_file):
            with open(summary_file, "r") as f:
                return json.load(f)
        return None

    def _get_config_name(self) -> str:
        """Extract a meaningful name from config file or project name."""
        # Try to get config file name first
        config_file = getattr(self.cfg, "config", None)
        if config_file:
            # Extract filename without extension and path
            config_name = os.path.splitext(os.path.basename(config_file))[0]
            # Clean up the name (remove special chars, limit length)
            config_name = re.sub(r"[^a-zA-Z0-9_-]", "_", config_name)
            config_name = config_name[:20]  # Limit length
            if config_name:
                return config_name

        # Fall back to project name
        project_name = getattr(self.cfg, "project_name", "dj")
        # Clean up project name
        project_name = re.sub(r"[^a-zA-Z0-9_-]", "_", project_name)
        project_name = project_name[:15]  # Limit length

        return project_name

    def _add_dag_context_to_metadata(
        self, metadata: Dict[str, Any], operation_name: str, operation_idx: int, partition_id: int
    ):
        """Add DAG context to metadata if DAGExecutionMixin is available."""
        # Check if DAGExecutionMixin is available and has the method to get DAG node
        if hasattr(self, "_get_dag_node_for_operation"):
            try:
                node_id = self._get_dag_node_for_operation(operation_name, operation_idx, partition_id=partition_id)
                if node_id:
                    metadata["dag_node_id"] = node_id
                else:
                    logger.debug(f"DAG node not found for operation {operation_name} (idx {operation_idx})")
            except Exception as e:
                logger.debug(f"Error getting DAG node for operation {operation_name}: {e}")

    def _log_event(self, event_type: EventType, message: str, **kwargs):
        """Log an event if event logging is enabled."""
        if self.event_logger is None:
            logger.warning(f"Event logger is None, cannot log event: {event_type.value}")
            return

        # Automatically capture process and thread IDs
        process_id = os.getpid()
        thread_id = threading.get_ident()

        # Generate event ID if not provided
        event_id = kwargs.pop("event_id", None)
        if event_id is None:
            timestamp = int(time.time())
            event_id = f"{event_type.value}_{timestamp}_{uuid4().hex[:8]}"

        logger.debug(f"Creating event: {event_type.value} - {message}")
        event = Event(
            event_type=event_type,
            timestamp=time.time(),
            message=message,
            event_id=event_id,
            process_id=process_id,
            thread_id=thread_id,
            **kwargs,
        )
        logger.debug(f"Logging event to event logger: {event_type.value}")
        self.event_logger.log_event(event)
        logger.debug(f"Successfully logged event: {event_type.value}")

    # Add new logging methods for job, partition, and op events
    def log_job_start(self, config, total_partitions):
        """Log job start with detailed configuration."""
        # Handle both dataset_path (string) and dataset (dict) configurations
        dataset_info = {}
        if "dataset_path" in config:
            dataset_info["dataset_path"] = config.get("dataset_path")
        if "dataset" in config:
            dataset_info["dataset"] = config.get("dataset")

        metadata = {
            "total_partitions": total_partitions,
            "config_summary": {
                **dataset_info,
                "executor_type": config.get("executor_type"),
                "partition_size": config.get("partition_size"),
                "checkpoint_strategy": config.get("checkpoint_strategy"),
                "storage_format": config.get("storage_format"),
                "compression": config.get("compression"),
            },
        }
        event_id = f"job_start_{int(time.time())}"
        self._log_event(
            EventType.JOB_START,
            "Job started",
            event_id=event_id,
            config=config,
            metadata=metadata,
            total_partitions=total_partitions,
        )

    def log_job_complete(self, duration, output_path=None):
        """Log job completion with performance metrics."""
        metadata = {"status": "completed", "duration_seconds": duration, "completion_time": time.time()}
        if output_path:
            metadata["output_path"] = output_path

        event_id = f"job_complete_{int(time.time())}"
        self._log_event(
            EventType.JOB_COMPLETE,
            f"Job completed successfully in {duration:.2f}s",
            event_id=event_id,
            status="completed",
            duration=duration,
            metadata=metadata,
        )
        self._update_job_summary("completed", error_message=None)

    def log_job_failed(self, error_message, duration):
        """Log job failure with error details."""
        metadata = {
            "status": "failed",
            "duration_seconds": duration,
            "failure_time": time.time(),
            "error_type": type(error_message).__name__ if error_message else "Unknown",
        }
        event_id = f"job_failed_{int(time.time())}"
        self._log_event(
            EventType.JOB_FAILED,
            f"Job failed: {error_message}",
            event_id=event_id,
            status="failed",
            error_message=error_message,
            duration=duration,
            metadata=metadata,
        )
        self._update_job_summary("failed", error_message=error_message)

    def log_partition_start(self, partition_id, partition_meta):
        """Log partition start with detailed metadata."""
        metadata = {
            "partition_path": partition_meta.get("partition_path"),
            "start_time": partition_meta.get("start_time"),
            "partition_size_bytes": partition_meta.get("file_size_bytes"),
            "sample_count": partition_meta.get("sample_count"),
        }
        event_id = f"partition_start_{partition_id}_{int(time.time())}"
        self._log_event(
            EventType.PARTITION_START,
            f"Partition {partition_id} started processing",
            event_id=event_id,
            partition_id=partition_id,
            partition_meta=partition_meta,
            metadata=metadata,
        )

    def log_partition_complete(self, partition_id, duration, output_path, success=True, error=None):
        """Log partition completion with performance metrics."""
        metadata = {
            "output_path": output_path,
            "duration_seconds": duration,
            "completion_time": time.time(),
            "success": success,
            "throughput_samples_per_second": None,  # Will be calculated if sample_count is available
        }

        if not success and error:
            metadata["error"] = error
            message = f"Partition {partition_id} completed with failure after {duration:.2f}s: {error}"
        else:
            message = f"Partition {partition_id} completed successfully after {duration:.2f}s"

        # Add debug logging to help diagnose issues
        logger.debug(f"Creating partition_complete event for partition {partition_id}")
        logger.debug(f"  Duration: {duration:.2f}s")
        logger.debug(f"  Success: {success}")
        logger.debug(f"  Output path: {output_path}")
        if error:
            logger.debug(f"  Error: {error}")

        # Use the _log_event method to ensure proper logging
        event_id = f"partition_complete_{partition_id}_{int(time.time())}"
        self._log_event(
            EventType.PARTITION_COMPLETE, message, event_id=event_id, partition_id=partition_id, metadata=metadata
        )

    def log_partition_failed(self, partition_id, error_message, retry_count):
        """Log partition failure with retry information."""
        metadata = {
            "retry_count": retry_count,
            "failure_time": time.time(),
            "error_type": type(error_message).__name__ if error_message else "Unknown",
        }
        event_id = f"partition_failed_{partition_id}_{int(time.time())}"
        self._log_event(
            EventType.PARTITION_FAILED,
            f"Partition {partition_id} failed after {retry_count} retries: {error_message}",
            event_id=event_id,
            partition_id=partition_id,
            error_message=error_message,
            retry_count=retry_count,
            status="failed",
            metadata=metadata,
        )

    def log_op_start(self, partition_id, operation_name, operation_idx, op_args, **kwargs):
        """Log operation start with detailed arguments."""
        metadata = {
            "operation_idx": operation_idx,
            "operation_args": op_args,
            "start_time": time.time(),
            "operation_class": operation_name,
        }
        # Merge any additional metadata from kwargs
        if "metadata" in kwargs:
            metadata.update(kwargs["metadata"])

        # Automatically add DAG context if DAGExecutionMixin is available
        self._add_dag_context_to_metadata(metadata, operation_name, operation_idx, partition_id)

        event_id = f"op_start_{partition_id}_{operation_idx}_{int(time.time())}"
        self._log_event(
            EventType.OP_START,
            f"Operation {operation_name} (idx {operation_idx}) started on partition {partition_id}",
            event_id=event_id,
            partition_id=partition_id,
            operation_name=operation_name,
            operation_idx=operation_idx,
            op_args=op_args,
            metadata=metadata,
        )

    def log_op_complete(
        self, partition_id, operation_name, operation_idx, duration, checkpoint_path, input_rows, output_rows, **kwargs
    ):
        """Log operation completion with detailed performance metrics."""
        # Calculate performance metrics
        throughput = input_rows / duration if duration > 0 and input_rows else 0
        reduction_ratio = (input_rows - output_rows) / input_rows if input_rows > 0 else 0

        metadata = {
            "duration_seconds": duration,
            "input_rows": input_rows,
            "output_rows": output_rows,
            "throughput_rows_per_second": throughput,
            "reduction_ratio": reduction_ratio,
            "checkpoint_path": checkpoint_path,
            "completion_time": time.time(),
            "operation_class": operation_name,
        }
        # Merge any additional metadata from kwargs
        if "metadata" in kwargs:
            metadata.update(kwargs["metadata"])

        # Automatically add DAG context if DAGExecutionMixin is available
        self._add_dag_context_to_metadata(metadata, operation_name, operation_idx, partition_id)

        event_id = f"op_complete_{partition_id}_{operation_idx}_{int(time.time())}"
        self._log_event(
            EventType.OP_COMPLETE,
            f"Operation {operation_name} (idx {operation_idx}) completed on partition {partition_id} - {input_rows}→{output_rows} rows in {duration:.3f}s",
            event_id=event_id,
            partition_id=partition_id,
            operation_name=operation_name,
            operation_idx=operation_idx,
            duration=duration,
            checkpoint_path=checkpoint_path,
            input_rows=input_rows,
            output_rows=output_rows,
            status="success",
            metadata=metadata,
        )

    def log_op_failed(self, partition_id, operation_name, operation_idx, error_message, retry_count, **kwargs):
        """Log operation failure with error details."""
        metadata = {
            "retry_count": retry_count,
            "failure_time": time.time(),
            "error_type": type(error_message).__name__ if error_message else "Unknown",
            "operation_class": operation_name,
        }
        # Merge any additional metadata from kwargs
        if "metadata" in kwargs:
            metadata.update(kwargs["metadata"])

        # Automatically add DAG context if DAGExecutionMixin is available
        self._add_dag_context_to_metadata(metadata, operation_name, operation_idx, partition_id)

        event_id = f"op_failed_{partition_id}_{operation_idx}_{int(time.time())}"
        self._log_event(
            EventType.OP_FAILED,
            f"Operation {operation_name} (idx {operation_idx}) failed on partition {partition_id}: {error_message}",
            event_id=event_id,
            partition_id=partition_id,
            operation_name=operation_name,
            operation_idx=operation_idx,
            error_message=error_message,
            retry_count=retry_count,
            status="failed",
            metadata=metadata,
        )

    def log_checkpoint_save(self, partition_id, operation_name, operation_idx, checkpoint_path):
        """Log checkpoint save with file information."""
        metadata = {
            "checkpoint_path": checkpoint_path,
            "operation_idx": operation_idx,
            "operation_class": operation_name,
            "save_time": time.time(),
        }
        event_id = f"checkpoint_save_{partition_id}_{operation_idx}_{int(time.time())}"
        self._log_event(
            EventType.CHECKPOINT_SAVE,
            f"Checkpoint saved for operation {operation_name} (idx {operation_idx}) on partition {partition_id}",
            event_id=event_id,
            partition_id=partition_id,
            operation_name=operation_name,
            operation_idx=operation_idx,
            checkpoint_path=checkpoint_path,
            metadata=metadata,
        )

    def log_checkpoint_load(self, partition_id, operation_name, operation_idx, checkpoint_path):
        """Log checkpoint load with file information."""
        metadata = {
            "checkpoint_path": checkpoint_path,
            "operation_idx": operation_idx,
            "operation_class": operation_name,
            "load_time": time.time(),
        }
        event_id = f"checkpoint_load_{partition_id}_{operation_idx}_{int(time.time())}"
        self._log_event(
            EventType.CHECKPOINT_LOAD,
            f"Checkpoint loaded for operation {operation_name} (idx {operation_idx}) on partition {partition_id}",
            event_id=event_id,
            partition_id=partition_id,
            operation_name=operation_name,
            operation_idx=operation_idx,
            checkpoint_path=checkpoint_path,
            metadata=metadata,
        )

    # DAG-specific event logging methods
    def log_dag_build_start(self, ast_info: Dict[str, Any]):
        """Log DAG build start with AST information."""
        metadata = {
            "ast_node_count": ast_info.get("node_count", 0),
            "ast_depth": ast_info.get("depth", 0),
            "ast_operation_types": ast_info.get("operation_types", []),
            "build_start_time": time.time(),
        }
        event_id = f"dag_build_start_{int(time.time())}"
        self._log_event(
            EventType.DAG_BUILD_START,
            "DAG build started from pipeline AST",
            event_id=event_id,
            metadata=metadata,
        )

    def log_dag_build_complete(self, dag_info: Dict[str, Any]):
        """Log DAG build completion with execution plan information."""
        metadata = {
            "dag_node_count": dag_info.get("node_count", 0),
            "dag_edge_count": dag_info.get("edge_count", 0),
            "parallel_groups_count": dag_info.get("parallel_groups_count", 0),
            "execution_plan_length": dag_info.get("execution_plan_length", 0),
            "build_duration": dag_info.get("build_duration", 0),
            "build_complete_time": time.time(),
        }
        event_id = f"dag_build_complete_{int(time.time())}"
        self._log_event(
            EventType.DAG_BUILD_COMPLETE,
            f"DAG build completed: {dag_info.get('node_count', 0)} nodes, {dag_info.get('edge_count', 0)} edges",
            event_id=event_id,
            metadata=metadata,
        )

    def log_dag_node_ready(self, node_id: str, node_info: Dict[str, Any]):
        """Log when a DAG node becomes ready for execution."""
        metadata = {
            "node_id": node_id,
            "op_name": node_info.get("op_name"),
            "op_type": node_info.get("op_type"),
            "dependencies_count": node_info.get("dependencies_count", 0),
            "dependents_count": node_info.get("dependents_count", 0),
            "execution_order": node_info.get("execution_order", -1),
            "ready_time": time.time(),
        }
        event_id = f"dag_node_ready_{node_id}_{int(time.time())}"
        self._log_event(
            EventType.DAG_NODE_READY,
            f"DAG node {node_id} ({node_info.get('op_name')}) ready for execution",
            event_id=event_id,
            metadata=metadata,
        )

    def log_dag_node_start(self, node_id: str, node_info: Dict[str, Any]):
        """Log when a DAG node starts execution."""
        metadata = {
            "node_id": node_id,
            "op_name": node_info.get("op_name"),
            "op_type": node_info.get("op_type"),
            "execution_order": node_info.get("execution_order", -1),
            "start_time": time.time(),
        }
        event_id = f"dag_node_start_{node_id}_{int(time.time())}"
        self._log_event(
            EventType.DAG_NODE_START,
            f"DAG node {node_id} ({node_info.get('op_name')}) started execution",
            event_id=event_id,
            metadata=metadata,
        )

    def log_dag_node_complete(self, node_id: str, node_info: Dict[str, Any], duration: float):
        """Log when a DAG node completes execution."""
        metadata = {
            "node_id": node_id,
            "op_name": node_info.get("op_name"),
            "op_type": node_info.get("op_type"),
            "execution_order": node_info.get("execution_order", -1),
            "duration_seconds": duration,
            "completion_time": time.time(),
        }
        event_id = f"dag_node_complete_{node_id}_{int(time.time())}"
        self._log_event(
            EventType.DAG_NODE_COMPLETE,
            f"DAG node {node_id} ({node_info.get('op_name')}) completed in {duration:.3f}s",
            event_id=event_id,
            metadata=metadata,
        )

    def log_dag_node_failed(self, node_id: str, node_info: Dict[str, Any], error_message: str, duration: float = 0):
        """Log when a DAG node fails execution."""
        metadata = {
            "node_id": node_id,
            "op_name": node_info.get("op_name"),
            "op_type": node_info.get("op_type"),
            "execution_order": node_info.get("execution_order", -1),
            "duration_seconds": duration,
            "error_message": error_message,
            "failure_time": time.time(),
        }
        event_id = f"dag_node_failed_{node_id}_{int(time.time())}"
        self._log_event(
            EventType.DAG_NODE_FAILED,
            f"DAG node {node_id} ({node_info.get('op_name')}) failed: {error_message}",
            event_id=event_id,
            metadata=metadata,
        )

    def log_dag_parallel_group_start(self, group_id: str, group_info: Dict[str, Any]):
        """Log when a parallel group starts execution."""
        metadata = {
            "group_id": group_id,
            "node_count": group_info.get("node_count", 0),
            "node_ids": group_info.get("node_ids", []),
            "op_types": group_info.get("op_types", []),
            "start_time": time.time(),
        }
        event_id = f"dag_parallel_group_start_{group_id}_{int(time.time())}"
        self._log_event(
            EventType.DAG_PARALLEL_GROUP_START,
            f"Parallel group {group_id} started with {group_info.get('node_count', 0)} nodes",
            event_id=event_id,
            metadata=metadata,
        )

    def log_dag_parallel_group_complete(self, group_id: str, group_info: Dict[str, Any], duration: float):
        """Log when a parallel group completes execution."""
        metadata = {
            "group_id": group_id,
            "node_count": group_info.get("node_count", 0),
            "completed_nodes": group_info.get("completed_nodes", 0),
            "failed_nodes": group_info.get("failed_nodes", 0),
            "duration_seconds": duration,
            "completion_time": time.time(),
        }
        event_id = f"dag_parallel_group_complete_{group_id}_{int(time.time())}"
        self._log_event(
            EventType.DAG_PARALLEL_GROUP_COMPLETE,
            f"Parallel group {group_id} completed: {group_info.get('completed_nodes', 0)}/{group_info.get('node_count', 0)} nodes in {duration:.3f}s",
            event_id=event_id,
            metadata=metadata,
        )

    def log_dag_execution_plan_saved(self, plan_path: str, plan_info: Dict[str, Any]):
        """Log when DAG execution plan is saved."""
        metadata = {
            "plan_path": plan_path,
            "node_count": plan_info.get("node_count", 0),
            "edge_count": plan_info.get("edge_count", 0),
            "parallel_groups_count": plan_info.get("parallel_groups_count", 0),
            "save_time": time.time(),
        }
        event_id = f"dag_execution_plan_saved_{int(time.time())}"
        self._log_event(
            EventType.DAG_EXECUTION_PLAN_SAVED,
            f"DAG execution plan saved to {plan_path}",
            event_id=event_id,
            metadata=metadata,
        )

    def log_dag_execution_plan_loaded(self, plan_path: str, plan_info: Dict[str, Any]):
        """Log when DAG execution plan is loaded."""
        metadata = {
            "plan_path": plan_path,
            "node_count": plan_info.get("node_count", 0),
            "edge_count": plan_info.get("edge_count", 0),
            "parallel_groups_count": plan_info.get("parallel_groups_count", 0),
            "load_time": time.time(),
        }
        event_id = f"dag_execution_plan_loaded_{int(time.time())}"
        self._log_event(
            EventType.DAG_EXECUTION_PLAN_LOADED,
            f"DAG execution plan loaded from {plan_path}",
            event_id=event_id,
            metadata=metadata,
        )

    def log_job_restart(
        self,
        restart_reason: str,
        original_start_time: float,
        resume_partitions: List[int],
        resume_from_operation: int,
        checkpoint_paths: List[str],
    ):
        """Log when a job is restarted after interruption."""
        metadata = {
            "restart_reason": restart_reason,
            "original_start_time": original_start_time,
            "restart_time": time.time(),
            "resume_partitions": resume_partitions,
            "resume_from_operation": resume_from_operation,
            "checkpoint_paths": checkpoint_paths,
        }
        event_id = f"job_restart_{int(time.time())}"
        self._log_event(
            EventType.JOB_RESTART,
            f"Job restarted after {restart_reason} interruption",
            event_id=event_id,
            metadata=metadata,
        )

    def log_partition_resume(self, partition_id: int, resume_operation: int, checkpoint_path: str, resume_reason: str):
        """Log when a partition is resumed from a checkpoint."""
        metadata = {
            "resume_operation": resume_operation,
            "checkpoint_path": checkpoint_path,
            "resume_reason": resume_reason,
        }
        event_id = f"partition_resume_{partition_id}_{int(time.time())}"
        self._log_event(
            EventType.PARTITION_RESUME,
            f"Partition {partition_id} resumed from operation {resume_operation} checkpoint",
            event_id=event_id,
            partition_id=partition_id,
            metadata=metadata,
        )

    def get_events(self, **kwargs) -> List[Event]:
        """Get events with optional filtering."""
        if self.event_logger is None:
            return []
        return self.event_logger.get_events(**kwargs)

    def generate_status_report(self) -> str:
        """Generate status report."""
        if self.event_logger is None:
            return "Event logging is disabled."
        return self.event_logger.generate_status_report()

    def monitor_events(self, event_type: Optional[EventType] = None) -> Generator[Event, None, None]:
        """Monitor events in real-time."""
        if self.event_logger is None:
            return
        yield from self.event_logger.monitor_events(event_type)

    def analyze_resumption_state(self, job_id: str) -> Dict[str, Any]:
        """
        Analyze event history to determine resumption state and generate resumption plan.

        Args:
            job_id: The job ID to analyze

        Returns:
            Dictionary containing resumption analysis and plan
        """
        if not self.event_logger:
            return {"error": "Event logger not available"}

        events_file = self.event_logger.jsonl_file
        if not os.path.exists(events_file):
            return {"error": f"Events file not found: {events_file}"}

        # Parse all events
        events = []
        with open(events_file, "r") as f:
            for line in f:
                try:
                    event = json.loads(line.strip())
                    events.append(event)
                except json.JSONDecodeError:
                    continue

        # Analyze events by type
        partition_starts = [e for e in events if e.get("event_type") == "partition_start"]
        partition_completes = [e for e in events if e.get("event_type") == "partition_complete"]
        partition_failures = [e for e in events if e.get("event_type") == "partition_failed"]
        op_starts = [e for e in events if e.get("event_type") == "op_start"]
        op_completes = [e for e in events if e.get("event_type") == "op_complete"]
        checkpoints = [e for e in events if e.get("event_type") == "checkpoint_saved"]

        # Determine job status
        job_status = self._determine_job_status(events, partition_completes, partition_failures)

        # Analyze partition states
        partition_states = self._analyze_partition_states(
            partition_starts, partition_completes, partition_failures, op_starts, op_completes
        )

        # Generate resumption plan
        resumption_plan = self._generate_resumption_plan(partition_states, checkpoints, job_status)

        # Calculate progress metrics
        progress_metrics = self._calculate_progress_metrics(partition_states, events)

        return {
            "job_id": job_id,
            "job_status": job_status,
            "total_events": len(events),
            "partition_states": partition_states,
            "resumption_plan": resumption_plan,
            "progress_metrics": progress_metrics,
            "analysis_timestamp": time.time(),
            "can_resume": resumption_plan["can_resume"],
            "resume_from_checkpoint": resumption_plan.get("resume_from_checkpoint"),
            "partitions_to_retry": resumption_plan.get("partitions_to_retry", []),
            "partitions_to_skip": resumption_plan.get("partitions_to_skip", []),
        }

    def _determine_job_status(
        self, events: List[Dict], partition_completes: List[Dict], partition_failures: List[Dict]
    ) -> str:
        """Determine the current job status based on events."""
        # Check if job has any completion events
        job_completes = [e for e in events if e.get("event_type") == "job_complete"]
        job_failures = [e for e in events if e.get("event_type") == "job_failed"]

        if job_completes:
            return "completed"
        elif job_failures:
            return "failed"
        elif partition_completes:
            # Check if all partitions are completed (success or failure)
            all_partitions_completed = all(
                pc.get("metadata", {}).get("success", False) or pc.get("metadata", {}).get("error") is not None
                for pc in partition_completes
            )
            if all_partitions_completed:
                return "completed_with_failures"
            else:
                return "running"
        else:
            return "not_started"

    def _analyze_partition_states(
        self,
        partition_starts: List[Dict],
        partition_completes: List[Dict],
        partition_failures: List[Dict],
        op_starts: List[Dict],
        op_completes: List[Dict],
    ) -> Dict[int, Dict]:
        """Analyze the state of each partition based on events."""
        partition_states = {}

        # Group events by partition ID
        for start_event in partition_starts:
            partition_id = start_event.get("partition_id")
            if partition_id is None:
                continue

            # Find the latest start event for this partition
            partition_starts_for_id = [e for e in partition_starts if e.get("partition_id") == partition_id]
            latest_start = max(partition_starts_for_id, key=lambda x: x.get("timestamp", 0))

            # Find completion events for this partition
            partition_completes_for_id = [e for e in partition_completes if e.get("partition_id") == partition_id]
            partition_failures_for_id = [e for e in partition_failures if e.get("partition_id") == partition_id]

            # Find operation events for this partition
            ops_for_partition = [e for e in op_starts if e.get("partition_id") == partition_id]
            op_completes_for_partition = [e for e in op_completes if e.get("partition_id") == partition_id]

            # Determine partition state
            state = self._determine_partition_state(
                partition_id,
                latest_start,
                partition_completes_for_id,
                partition_failures_for_id,
                ops_for_partition,
                op_completes_for_partition,
            )

            partition_states[partition_id] = state

        return partition_states

    def _determine_partition_state(
        self,
        partition_id: int,
        start_event: Dict,
        completes: List[Dict],
        failures: List[Dict],
        op_starts: List[Dict],
        op_completes: List[Dict],
    ) -> Dict:
        """Determine the detailed state of a specific partition."""
        # Find the latest completion event
        latest_complete = max(completes, key=lambda x: x.get("timestamp", 0)) if completes else None

        # Determine if partition is completed successfully
        is_completed = latest_complete and latest_complete.get("metadata", {}).get("success", False)
        is_failed = latest_complete and not latest_complete.get("metadata", {}).get("success", False)

        # Find the last operation that was started
        last_op_start = max(op_starts, key=lambda x: x.get("timestamp", 0)) if op_starts else None
        last_op_complete = max(op_completes, key=lambda x: x.get("timestamp", 0)) if op_completes else None

        # Determine current operation
        current_operation = None
        if last_op_start:
            current_operation = {
                "name": last_op_start.get("operation_name"),
                "idx": last_op_start.get("operation_idx"),
                "started_at": last_op_start.get("timestamp"),
                "completed": last_op_complete is not None
                and last_op_complete.get("timestamp", 0) > last_op_start.get("timestamp", 0),
            }

        return {
            "partition_id": partition_id,
            "status": "completed" if is_completed else "failed" if is_failed else "running",
            "start_time": start_event.get("timestamp"),
            "completion_time": latest_complete.get("timestamp") if latest_complete else None,
            "duration": latest_complete.get("metadata", {}).get("duration_seconds") if latest_complete else None,
            "success": is_completed,
            "error": latest_complete.get("metadata", {}).get("error") if latest_complete and not is_completed else None,
            "current_operation": current_operation,
            "retry_count": len([f for f in failures if f.get("partition_id") == partition_id]),
            "output_path": latest_complete.get("metadata", {}).get("output_path") if latest_complete else None,
        }

    def _generate_resumption_plan(
        self, partition_states: Dict[int, Dict], checkpoints: List[Dict], job_status: str
    ) -> Dict:
        """Generate a resumption plan based on partition states and checkpoints."""
        # Find partitions that need to be retried
        partitions_to_retry = []
        partitions_to_skip = []

        for partition_id, state in partition_states.items():
            if state["status"] == "failed":
                partitions_to_retry.append(partition_id)
            elif state["status"] == "completed":
                partitions_to_skip.append(partition_id)

        # Find the latest checkpoint
        latest_checkpoint = max(checkpoints, key=lambda x: x.get("timestamp", 0)) if checkpoints else None

        # Determine if we can resume based on job status and partition states
        if job_status == "completed":
            can_resume = False
            reason = "Job already completed successfully"
        elif job_status == "failed":
            can_resume = True
            reason = "Job failed, can resume from checkpoint or retry failed partitions"
        elif len(partitions_to_retry) > 0:
            can_resume = True
            reason = f"Found {len(partitions_to_retry)} failed partitions to retry"
        elif latest_checkpoint is not None:
            can_resume = True
            reason = "Found checkpoint to resume from"
        else:
            can_resume = False
            reason = "No failed partitions or checkpoints found"

        return {
            "can_resume": can_resume,
            "reason": reason,
            "resume_from_checkpoint": (
                latest_checkpoint.get("metadata", {}).get("checkpoint_path") if latest_checkpoint else None
            ),
            "partitions_to_retry": partitions_to_retry,
            "partitions_to_skip": partitions_to_skip,
            "total_partitions_to_process": len(partitions_to_retry),
            "estimated_remaining_work": len(partitions_to_retry) / len(partition_states) if partition_states else 0,
        }

    def _calculate_progress_metrics(self, partition_states: Dict[int, Dict], events: List[Dict]) -> Dict:
        """Calculate progress metrics based on partition states."""
        total_partitions = len(partition_states)
        completed_partitions = len([s for s in partition_states.values() if s["status"] == "completed"])
        failed_partitions = len([s for s in partition_states.values() if s["status"] == "failed"])
        running_partitions = len([s for s in partition_states.values() if s["status"] == "running"])

        # Calculate overall progress
        if total_partitions == 0:
            progress_percentage = 0
        else:
            progress_percentage = (completed_partitions / total_partitions) * 100

        # Calculate timing metrics
        job_start_events = [e for e in events if e.get("event_type") == "job_start"]
        start_time = job_start_events[0].get("timestamp") if job_start_events else None
        current_time = time.time()
        elapsed_time = current_time - start_time if start_time else 0

        return {
            "total_partitions": total_partitions,
            "completed_partitions": completed_partitions,
            "failed_partitions": failed_partitions,
            "running_partitions": running_partitions,
            "progress_percentage": progress_percentage,
            "elapsed_time_seconds": elapsed_time,
            "start_time": start_time,
            "current_time": current_time,
        }

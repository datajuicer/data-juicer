"""
Processing Snapshot Utility for DataJuicer

This module analyzes the current state of processing based on events.jsonl and DAG structure
to provide a comprehensive snapshot of what's done, what's not, and checkpointing status.
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from loguru import logger


class ProcessingStatus(Enum):
    """Processing status enumeration."""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CHECKPOINTED = "checkpointed"


@dataclass
class OperationStatus:
    """Status of a single operation."""

    operation_name: str
    operation_idx: int
    status: ProcessingStatus
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration: Optional[float] = None
    input_rows: Optional[int] = None
    output_rows: Optional[int] = None
    checkpoint_time: Optional[float] = None
    error_message: Optional[str] = None


@dataclass
class PartitionStatus:
    """Status of a single partition."""

    partition_id: int
    status: ProcessingStatus
    sample_count: Optional[int] = None
    creation_start_time: Optional[float] = None
    creation_end_time: Optional[float] = None
    processing_start_time: Optional[float] = None
    processing_end_time: Optional[float] = None
    current_operation: Optional[str] = None
    completed_operations: List[str] = None
    failed_operations: List[str] = None
    checkpointed_operations: List[str] = None
    error_message: Optional[str] = None

    def __post_init__(self):
        """Initialize mutable fields after dataclass creation."""
        if self.completed_operations is None:
            self.completed_operations = []
        if self.failed_operations is None:
            self.failed_operations = []
        if self.checkpointed_operations is None:
            self.checkpointed_operations = []


@dataclass
class JobSnapshot:
    """Complete snapshot of job processing status."""

    job_id: str
    job_start_time: Optional[float] = None
    job_end_time: Optional[float] = None
    total_duration: Optional[float] = None
    total_partitions: int = 0
    completed_partitions: int = 0
    failed_partitions: int = 0
    in_progress_partitions: int = 0
    total_operations: int = 0
    completed_operations: int = 0
    failed_operations: int = 0
    checkpointed_operations: int = 0
    partition_statuses: Dict[int, PartitionStatus] = None
    operation_statuses: Dict[str, OperationStatus] = None
    dag_structure: Dict = None
    checkpoint_strategy: Optional[str] = None
    checkpoint_frequency: Optional[str] = None
    last_checkpoint_time: Optional[float] = None
    resumable: bool = False
    overall_status: ProcessingStatus = ProcessingStatus.NOT_STARTED


class ProcessingSnapshotAnalyzer:
    """Analyzer for processing snapshots."""

    def __init__(self, work_dir: str):
        """Initialize the analyzer with work directory."""
        self.work_dir = Path(work_dir)
        self.events_file = self.work_dir / "events.jsonl"
        self.dag_file = self.work_dir / "dag_execution_plan.json"
        self.job_summary_file = self.work_dir / "job_summary.json"

    def load_events(self) -> List[Dict]:
        """Load events from events.jsonl file."""
        events = []
        if self.events_file.exists():
            try:
                with open(self.events_file, "r") as f:
                    for line in f:
                        events.append(json.loads(line.strip()))
                logger.info(f"Loaded {len(events)} events from {self.events_file}")
            except Exception as e:
                logger.error(f"Failed to load events: {e}")
        else:
            logger.warning(f"Events file not found: {self.events_file}")
        return events

    def load_dag_plan(self) -> Dict:
        """Load DAG execution plan."""
        dag_plan = {}
        if self.dag_file.exists():
            try:
                with open(self.dag_file, "r") as f:
                    dag_plan = json.load(f)
                logger.info(f"Loaded DAG plan from {self.dag_file}")
            except Exception as e:
                logger.error(f"Failed to load DAG plan: {e}")
        else:
            logger.warning(f"DAG file not found: {self.dag_file}")
        return dag_plan

    def load_job_summary(self) -> Dict:
        """Load job summary if available."""
        summary = {}
        if self.job_summary_file.exists():
            try:
                with open(self.job_summary_file, "r") as f:
                    summary = json.load(f)
                logger.info(f"Loaded job summary from {self.job_summary_file}")
            except Exception as e:
                logger.error(f"Failed to load job summary: {e}")
        return summary

    def extract_operation_pipeline(self, dag_plan: Dict) -> List[Dict]:
        """Extract operation pipeline from DAG plan."""
        operations = []
        try:
            if "process" in dag_plan:
                operations = dag_plan["process"]
            elif "operations" in dag_plan:
                operations = dag_plan["operations"]
            else:
                # Try to find operations in nested structure
                for key, value in dag_plan.items():
                    if isinstance(value, list) and value:
                        # Check if this looks like an operation list
                        if isinstance(value[0], dict) and any("name" in op or "type" in op for op in value):
                            operations = value
                            break
        except Exception as e:
            logger.error(f"Failed to extract operation pipeline: {e}")

        return operations

    def analyze_events(self, events: List[Dict]) -> Tuple[Dict[int, PartitionStatus], Dict[str, OperationStatus]]:
        """Analyze events to determine processing status."""
        partition_statuses = {}
        operation_statuses = {}

        # Track job-level events
        for event in events:
            event_type = event.get("event_type")
            timestamp = event.get("timestamp")

            if event_type == "job_start":
                # Extract checkpoint strategy from metadata
                metadata = event.get("metadata", {})
                # Note: checkpoint_strategy is extracted but not used in this method
                # It's used in generate_snapshot method
                pass

            elif event_type == "job_complete":
                # Note: job_end_time is extracted but not used in this method
                # It's used in generate_snapshot method
                pass

            elif event_type == "partition_creation_start":
                partition_id = event.get("partition_id")
                if partition_id not in partition_statuses:
                    partition_statuses[partition_id] = PartitionStatus(
                        partition_id=partition_id, status=ProcessingStatus.NOT_STARTED
                    )
                partition_statuses[partition_id].creation_start_time = timestamp

            elif event_type == "partition_creation_complete":
                partition_id = event.get("partition_id")
                if partition_id in partition_statuses:
                    partition_statuses[partition_id].creation_end_time = timestamp
                    metadata = event.get("metadata", {})
                    partition_statuses[partition_id].sample_count = metadata.get("sample_count")

            elif event_type == "partition_start":
                partition_id = event.get("partition_id")
                if partition_id in partition_statuses:
                    partition_statuses[partition_id].processing_start_time = timestamp
                    partition_statuses[partition_id].status = ProcessingStatus.IN_PROGRESS

            elif event_type == "partition_complete":
                partition_id = event.get("partition_id")
                if partition_id in partition_statuses:
                    partition_statuses[partition_id].processing_end_time = timestamp
                    partition_statuses[partition_id].status = ProcessingStatus.COMPLETED

            elif event_type == "partition_failed":
                partition_id = event.get("partition_id")
                if partition_id in partition_statuses:
                    partition_statuses[partition_id].status = ProcessingStatus.FAILED
                    partition_statuses[partition_id].error_message = event.get("error_message")

            elif event_type == "op_start":
                partition_id = event.get("partition_id")
                op_idx = event.get("operation_idx")
                op_name = event.get("operation_name")
                key = f"p{partition_id}_op{op_idx}_{op_name}"

                operation_statuses[key] = OperationStatus(
                    operation_name=op_name,
                    operation_idx=op_idx,
                    status=ProcessingStatus.IN_PROGRESS,
                    start_time=timestamp,
                )

                # Update partition status
                if partition_id in partition_statuses:
                    partition_statuses[partition_id].current_operation = op_name

            elif event_type == "op_complete":
                partition_id = event.get("partition_id")
                op_idx = event.get("operation_idx")
                op_name = event.get("operation_name")
                key = f"p{partition_id}_op{op_idx}_{op_name}"

                if key in operation_statuses:
                    operation_statuses[key].end_time = timestamp
                    operation_statuses[key].status = ProcessingStatus.COMPLETED
                    if operation_statuses[key].start_time:
                        operation_statuses[key].duration = timestamp - operation_statuses[key].start_time

                    metadata = event.get("metadata", {})
                    operation_statuses[key].input_rows = metadata.get("input_rows")
                    operation_statuses[key].output_rows = metadata.get("output_rows")

                    # Update partition status
                    if partition_id in partition_statuses:
                        partition_statuses[partition_id].completed_operations.append(op_name)

            elif event_type == "op_failed":
                partition_id = event.get("partition_id")
                op_idx = event.get("operation_idx")
                op_name = event.get("operation_name")
                key = f"p{partition_id}_op{op_idx}_{op_name}"

                if key in operation_statuses:
                    operation_statuses[key].status = ProcessingStatus.FAILED
                    operation_statuses[key].error_message = event.get("error_message")

                    # Update partition status
                    if partition_id in partition_statuses:
                        partition_statuses[partition_id].failed_operations.append(op_name)
                        partition_statuses[partition_id].status = ProcessingStatus.FAILED

            elif event_type == "checkpoint_save":
                partition_id = event.get("partition_id")
                op_idx = event.get("operation_idx")
                op_name = event.get("operation_name")
                key = f"p{partition_id}_op{op_idx}_{op_name}"

                if key in operation_statuses:
                    operation_statuses[key].checkpoint_time = timestamp
                    operation_statuses[key].status = ProcessingStatus.CHECKPOINTED

                    # Update partition status
                    if partition_id in partition_statuses:
                        partition_statuses[partition_id].checkpointed_operations.append(op_name)

        return partition_statuses, operation_statuses

    def determine_overall_status(
        self, partition_statuses: Dict[int, PartitionStatus], operation_statuses: Dict[str, OperationStatus]
    ) -> ProcessingStatus:
        """Determine overall job status."""
        if not partition_statuses:
            return ProcessingStatus.NOT_STARTED

        completed = sum(1 for p in partition_statuses.values() if p.status == ProcessingStatus.COMPLETED)
        failed = sum(1 for p in partition_statuses.values() if p.status == ProcessingStatus.FAILED)
        in_progress = sum(1 for p in partition_statuses.values() if p.status == ProcessingStatus.IN_PROGRESS)

        if failed > 0 and completed == 0:
            return ProcessingStatus.FAILED
        elif completed == len(partition_statuses):
            return ProcessingStatus.COMPLETED
        elif in_progress > 0 or completed > 0:
            return ProcessingStatus.IN_PROGRESS
        else:
            return ProcessingStatus.NOT_STARTED

    def calculate_statistics(
        self, partition_statuses: Dict[int, PartitionStatus], operation_statuses: Dict[str, OperationStatus]
    ) -> Dict:
        """Calculate processing statistics."""
        total_partitions = len(partition_statuses)
        completed_partitions = sum(1 for p in partition_statuses.values() if p.status == ProcessingStatus.COMPLETED)
        failed_partitions = sum(1 for p in partition_statuses.values() if p.status == ProcessingStatus.FAILED)
        in_progress_partitions = sum(1 for p in partition_statuses.values() if p.status == ProcessingStatus.IN_PROGRESS)

        total_operations = len(operation_statuses)
        completed_operations = sum(1 for op in operation_statuses.values() if op.status == ProcessingStatus.COMPLETED)
        failed_operations = sum(1 for op in operation_statuses.values() if op.status == ProcessingStatus.FAILED)
        checkpointed_operations = sum(
            1 for op in operation_statuses.values() if op.status == ProcessingStatus.CHECKPOINTED
        )

        return {
            "total_partitions": total_partitions,
            "completed_partitions": completed_partitions,
            "failed_partitions": failed_partitions,
            "in_progress_partitions": in_progress_partitions,
            "total_operations": total_operations,
            "completed_operations": completed_operations,
            "failed_operations": failed_operations,
            "checkpointed_operations": checkpointed_operations,
        }

    def generate_snapshot(self) -> JobSnapshot:
        """Generate a complete processing snapshot."""
        logger.info(f"Generating processing snapshot for work directory: {self.work_dir}")

        # Load data
        events = self.load_events()
        dag_plan = self.load_dag_plan()
        job_summary = self.load_job_summary()

        # Extract job ID from directory name
        job_id = self.work_dir.name

        # Analyze events
        partition_statuses, operation_statuses = self.analyze_events(events)

        # Calculate statistics
        stats = self.calculate_statistics(partition_statuses, operation_statuses)

        # Determine overall status
        overall_status = self.determine_overall_status(partition_statuses, operation_statuses)

        # Extract timing information from job summary first, then fall back to events
        job_start_time = None
        job_end_time = None
        total_duration = None

        if job_summary:
            # Use job summary timing if available (more accurate)
            job_start_time = job_summary.get("start_time")
            job_end_time = job_summary.get("end_time")
            total_duration = job_summary.get("duration")
        else:
            # Fall back to event-based timing
            for event in events:
                if event.get("event_type") == "job_start":
                    job_start_time = event.get("timestamp")
                elif event.get("event_type") == "job_complete":
                    job_end_time = event.get("timestamp")

            if job_start_time and job_end_time:
                total_duration = job_end_time - job_start_time

        # Determine resumability
        resumable = any(op.status == ProcessingStatus.CHECKPOINTED for op in operation_statuses.values())

        # Extract checkpoint information
        checkpoint_strategy = None
        last_checkpoint_time = None
        for event in events:
            if event.get("event_type") == "job_start":
                metadata = event.get("metadata", {})
                checkpoint_strategy = metadata.get("checkpoint_strategy")
            elif event.get("event_type") == "checkpoint_save":
                last_checkpoint_time = event.get("timestamp")

        return JobSnapshot(
            job_id=job_id,
            job_start_time=job_start_time,
            job_end_time=job_end_time,
            total_duration=total_duration,
            partition_statuses=partition_statuses,
            operation_statuses=operation_statuses,
            dag_structure=dag_plan,
            checkpoint_strategy=checkpoint_strategy,
            last_checkpoint_time=last_checkpoint_time,
            resumable=resumable,
            overall_status=overall_status,
            **stats,
        )

    def to_json_dict(self, snapshot: JobSnapshot) -> Dict:
        """Convert snapshot to JSON-serializable dictionary with comprehensive progress tracking."""
        # Load job summary for additional metadata
        job_summary = self.load_job_summary()

        # Convert partition statuses to JSON format
        partition_progress = {}
        for partition_id, partition in snapshot.partition_statuses.items():
            partition_progress[str(partition_id)] = {
                "status": partition.status.value,
                "sample_count": partition.sample_count,
                "creation_start_time": partition.creation_start_time,
                "creation_end_time": partition.creation_end_time,
                "processing_start_time": partition.processing_start_time,
                "processing_end_time": partition.processing_end_time,
                "current_operation": partition.current_operation,
                "completed_operations": partition.completed_operations,
                "failed_operations": partition.failed_operations,
                "checkpointed_operations": partition.checkpointed_operations,
                "error_message": partition.error_message,
                "progress_percentage": self._calculate_partition_progress(partition),
            }

        # Convert operation statuses to JSON format
        operation_progress = {}
        for op_key, operation in snapshot.operation_statuses.items():
            operation_progress[op_key] = {
                "operation_name": operation.operation_name,
                "operation_idx": operation.operation_idx,
                "status": operation.status.value,
                "start_time": operation.start_time,
                "end_time": operation.end_time,
                "duration": operation.duration,
                "input_rows": operation.input_rows,
                "output_rows": operation.output_rows,
                "checkpoint_time": operation.checkpoint_time,
                "error_message": operation.error_message,
                "progress_percentage": self._calculate_operation_progress(operation),
            }

        # Extract DAG structure information
        dag_info = {}
        if snapshot.dag_structure:
            dag_info = {
                "total_nodes": len(snapshot.dag_structure.get("nodes", [])),
                "total_edges": len(snapshot.dag_structure.get("edges", [])),
                "parallel_groups": len(snapshot.dag_structure.get("parallel_groups", [])),
                "execution_plan": snapshot.dag_structure.get("execution_plan", []),
                "metadata": snapshot.dag_structure.get("metadata", {}),
            }

        # Calculate overall progress percentages
        overall_progress = self._calculate_overall_progress(snapshot)

        # Build job information from job summary
        job_info = {
            "job_id": snapshot.job_id,
            "executor_type": job_summary.get("executor_type") if job_summary else None,
            "status": job_summary.get("status") if job_summary else snapshot.overall_status.value,
            "config_file": job_summary.get("config_file") if job_summary else None,
            "work_dir": job_summary.get("work_dir") if job_summary else None,
            "resumption_command": job_summary.get("resumption_command") if job_summary else None,
            "error_message": job_summary.get("error_message") if job_summary else None,
        }

        return {
            "job_info": job_info,
            "overall_status": snapshot.overall_status.value,
            "overall_progress": overall_progress,
            "job_start_time": snapshot.job_start_time,
            "job_end_time": snapshot.job_end_time,
            "total_duration": snapshot.total_duration,
            "timing": {
                "start_time": snapshot.job_start_time,
                "end_time": snapshot.job_end_time,
                "duration_seconds": snapshot.total_duration,
                "duration_formatted": (
                    self._format_duration(snapshot.total_duration) if snapshot.total_duration else None
                ),
                "job_summary_duration": job_summary.get("duration") if job_summary else None,
                "timing_source": "job_summary" if job_summary else "events",
            },
            "progress_summary": {
                "total_partitions": snapshot.total_partitions,
                "completed_partitions": snapshot.completed_partitions,
                "failed_partitions": snapshot.failed_partitions,
                "in_progress_partitions": snapshot.in_progress_partitions,
                "partition_progress_percentage": self._calculate_partition_progress_percentage(snapshot),
                "total_operations": snapshot.total_operations,
                "completed_operations": snapshot.completed_operations,
                "failed_operations": snapshot.failed_operations,
                "checkpointed_operations": snapshot.checkpointed_operations,
                "operation_progress_percentage": self._calculate_operation_progress_percentage(snapshot),
            },
            "checkpointing": {
                "strategy": snapshot.checkpoint_strategy,
                "last_checkpoint_time": snapshot.last_checkpoint_time,
                "checkpointed_operations_count": snapshot.checkpointed_operations,
                "resumable": snapshot.resumable,
                "checkpoint_progress": self._calculate_checkpoint_progress(snapshot),
                "checkpoint_dir": job_summary.get("checkpoint_dir") if job_summary else None,
            },
            "partition_progress": partition_progress,
            "operation_progress": operation_progress,
            "dag_structure": dag_info,
            "file_paths": {
                "event_log_file": job_summary.get("event_log_file") if job_summary else None,
                "event_log_dir": job_summary.get("event_log_dir") if job_summary else None,
                "checkpoint_dir": job_summary.get("checkpoint_dir") if job_summary else None,
                "metadata_dir": job_summary.get("metadata_dir") if job_summary else None,
                "backed_up_config_path": job_summary.get("backed_up_config_path") if job_summary else None,
            },
            "metadata": {
                "snapshot_generated_at": datetime.now().isoformat(),
                "events_analyzed": len(self.load_events()),
                "dag_plan_loaded": bool(snapshot.dag_structure),
                "job_summary_loaded": bool(job_summary),
                "job_summary_used": bool(job_summary),
            },
        }

    def _calculate_partition_progress(self, partition: PartitionStatus) -> float:
        """Calculate progress percentage for a partition."""
        if partition.status == ProcessingStatus.COMPLETED:
            return 100.0
        elif partition.status == ProcessingStatus.FAILED:
            return 0.0
        elif partition.status == ProcessingStatus.IN_PROGRESS:
            # Estimate progress based on completed operations
            total_ops = (
                len(partition.completed_operations)
                + len(partition.failed_operations)
                + len(partition.checkpointed_operations)
            )
            if total_ops > 0:
                return min(90.0, (total_ops / 8) * 100)  # Assume 8 operations per partition
            else:
                return 10.0  # Just started
        else:
            return 0.0

    def _calculate_operation_progress(self, operation: OperationStatus) -> float:
        """Calculate progress percentage for an operation."""
        if operation.status == ProcessingStatus.COMPLETED:
            return 100.0
        elif operation.status == ProcessingStatus.FAILED:
            return 0.0
        elif operation.status == ProcessingStatus.CHECKPOINTED:
            return 100.0  # Checkpointed operations are considered complete
        elif operation.status == ProcessingStatus.IN_PROGRESS:
            if operation.start_time:
                # Estimate progress based on time elapsed
                current_time = datetime.now().timestamp()
                elapsed = current_time - operation.start_time
                # Assume average operation takes 1 second
                estimated_duration = 1.0
                progress = min(90.0, (elapsed / estimated_duration) * 100)
                return max(10.0, progress)
            else:
                return 10.0
        else:
            return 0.0

    def _calculate_overall_progress(self, snapshot: JobSnapshot) -> Dict[str, float]:
        """Calculate overall progress percentages."""
        total_partitions = snapshot.total_partitions or 1
        total_operations = snapshot.total_operations or 1

        partition_progress = (snapshot.completed_partitions / total_partitions) * 100
        operation_progress = (snapshot.completed_operations / total_operations) * 100

        # Weighted overall progress (partitions and operations equally weighted)
        overall_progress = (partition_progress + operation_progress) / 2

        return {
            "overall_percentage": overall_progress,
            "partition_percentage": partition_progress,
            "operation_percentage": operation_progress,
        }

    def _calculate_partition_progress_percentage(self, snapshot: JobSnapshot) -> float:
        """Calculate partition progress percentage."""
        if snapshot.total_partitions == 0:
            return 100.0
        return (snapshot.completed_partitions / snapshot.total_partitions) * 100

    def _calculate_operation_progress_percentage(self, snapshot: JobSnapshot) -> float:
        """Calculate operation progress percentage."""
        if snapshot.total_operations == 0:
            return 100.0
        return (snapshot.completed_operations / snapshot.total_operations) * 100

    def _calculate_checkpoint_progress(self, snapshot: JobSnapshot) -> Dict[str, any]:
        """Calculate checkpoint progress information."""
        if snapshot.total_operations == 0:
            return {"percentage": 0.0, "checkpointed_operations": [], "checkpoint_coverage": 0.0}

        checkpoint_percentage = (snapshot.checkpointed_operations / snapshot.total_operations) * 100

        # Get list of checkpointed operations
        checkpointed_ops = []
        for op_key, operation in snapshot.operation_statuses.items():
            if operation.status == ProcessingStatus.CHECKPOINTED:
                checkpointed_ops.append(
                    {
                        "operation_key": op_key,
                        "operation_name": operation.operation_name,
                        "checkpoint_time": operation.checkpoint_time,
                    }
                )

        return {
            "percentage": checkpoint_percentage,
            "checkpointed_operations": checkpointed_ops,
            "checkpoint_coverage": checkpoint_percentage / 100.0,
        }

    def _format_duration(self, duration_seconds: float) -> str:
        """Format duration in human-readable format."""
        if duration_seconds is None:
            return None

        hours = int(duration_seconds // 3600)
        minutes = int((duration_seconds % 3600) // 60)
        seconds = int(duration_seconds % 60)

        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"


def create_snapshot(work_dir: str, detailed: bool = False) -> JobSnapshot:
    """Create and display a processing snapshot for a work directory."""
    analyzer = ProcessingSnapshotAnalyzer(work_dir)
    snapshot = analyzer.generate_snapshot()
    return snapshot


def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate DataJuicer processing snapshot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m data_juicer.utils.job.snapshot outputs/partition-checkpoint-eventlog/20250808_230030_501c9d
  python -m data_juicer.utils.job.snapshot /path/to/job/directory --human-readable
        """,
    )
    parser.add_argument("work_dir", help="Path to the DataJuicer work directory")
    parser.add_argument("--human-readable", action="store_true", help="Output in human-readable format instead of JSON")

    args = parser.parse_args()

    if not os.path.exists(args.work_dir):
        print(f"Error: Work directory '{args.work_dir}' does not exist")
        return 1

    try:
        snapshot = create_snapshot(args.work_dir)
        analyzer = ProcessingSnapshotAnalyzer(args.work_dir)

        if args.human_readable:
            # Human-readable output
            print("\n" + "=" * 80)
            print(f"DataJuicer Processing Snapshot - Job: {snapshot.job_id}")
            print("=" * 80)

            # Overall status
            status_emoji = {
                ProcessingStatus.NOT_STARTED: "⏳",
                ProcessingStatus.IN_PROGRESS: "🔄",
                ProcessingStatus.COMPLETED: "✅",
                ProcessingStatus.FAILED: "❌",
                ProcessingStatus.CHECKPOINTED: "💾",
            }

            print(
                f"\n📊 Overall Status: {status_emoji[snapshot.overall_status]} {snapshot.overall_status.value.upper()}"
            )

            # Timing information
            if snapshot.job_start_time:
                start_time = datetime.fromtimestamp(snapshot.job_start_time)
                print(f"🕐 Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

            if snapshot.total_duration:
                print(f"⏱️  Duration: {snapshot.total_duration:.2f} seconds")

            # Progress summary
            print(f"\n📈 Progress Summary:")
            print(f"   Partitions: {snapshot.completed_partitions}/{snapshot.total_partitions} completed")
            print(f"   Operations: {snapshot.completed_operations}/{snapshot.total_operations} completed")

            if snapshot.failed_partitions > 0:
                print(f"   ❌ Failed partitions: {snapshot.failed_partitions}")
            if snapshot.failed_operations > 0:
                print(f"   ❌ Failed operations: {snapshot.failed_operations}")
            if snapshot.checkpointed_operations > 0:
                print(f"   💾 Checkpointed operations: {snapshot.checkpointed_operations}")

            # Checkpointing information
            if snapshot.checkpoint_strategy:
                print(f"\n💾 Checkpointing:")
                print(f"   Strategy: {snapshot.checkpoint_strategy}")
                if snapshot.last_checkpoint_time:
                    checkpoint_time = datetime.fromtimestamp(snapshot.last_checkpoint_time)
                    print(f"   Last checkpoint: {checkpoint_time.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"   Resumable: {'Yes' if snapshot.resumable else 'No'}")

            print("\n" + "=" * 80)
        else:
            # JSON output (default)
            json_dict = analyzer.to_json_dict(snapshot)
            print(json.dumps(json_dict, indent=2))

        return 0

    except Exception as e:
        print(f"Error generating snapshot: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())

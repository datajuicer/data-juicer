#!/usr/bin/env python3
"""
DataJuicer Job Utilities - Common Functions

Shared utilities for job stopping and monitoring operations.
"""

import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import psutil
from loguru import logger


class JobUtils:
    """Common utilities for DataJuicer job operations."""

    def __init__(self, job_id: str, work_dir: str = None, base_dir: str = None):
        """
        Initialize job utilities.

        Args:
            job_id: The job ID to work with
            work_dir: Work directory that already includes job_id (preferred)
            base_dir: Base directory containing job outputs (deprecated, use work_dir instead)
        """
        self.job_id = job_id
        if work_dir:
            # work_dir already includes job_id
            self.work_dir = Path(work_dir)
        elif base_dir:
            # Legacy: construct work_dir from base_dir + job_id
            self.work_dir = Path(base_dir) / job_id
        else:
            # Default fallback
            self.work_dir = Path("outputs/partition-checkpoint-eventlog") / job_id

        # Set up logging
        logger.remove()
        logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {name}:{function}:{line} - {message}")

        if not self.work_dir.exists():
            raise FileNotFoundError(f"Job directory not found: {self.work_dir}")

    def load_job_summary(self) -> Optional[Dict[str, Any]]:
        """Load job summary from the work directory."""
        job_summary_file = self.work_dir / "job_summary.json"
        if not job_summary_file.exists():
            logger.error(f"Job summary not found: {job_summary_file}")
            return None

        try:
            with open(job_summary_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load job summary: {e}")
            return None

    def load_dataset_mapping(self) -> Dict[str, Any]:
        """Load dataset mapping information."""
        mapping_file = self.work_dir / "metadata" / "dataset_mapping.json"
        if mapping_file.exists():
            try:
                with open(mapping_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load dataset mapping: {e}")
        return {}

    def load_event_logs(self) -> List[Dict[str, Any]]:
        """Load and parse event logs."""
        events_file = self.work_dir / "events.jsonl"
        events = []

        if events_file.exists():
            try:
                with open(events_file, "r") as f:
                    for line in f:
                        try:
                            events.append(json.loads(line.strip()))
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                logger.error(f"Failed to read events file: {e}")
        else:
            logger.warning(f"Events file not found: {events_file}")

        return events

    def extract_process_thread_ids(self) -> Dict[str, Set[int]]:
        """
        Extract process and thread IDs from event logs.
        Returns a dict with 'process_ids' and 'thread_ids' sets.
        """
        events = self.load_event_logs()
        process_ids = set()
        thread_ids = set()

        for event in events:
            # Extract process ID
            if "process_id" in event and event["process_id"] is not None:
                process_ids.add(event["process_id"])

            # Extract thread ID
            if "thread_id" in event and event["thread_id"] is not None:
                thread_ids.add(event["thread_id"])

        logger.info(f"Found {len(process_ids)} unique process IDs and {len(thread_ids)} unique thread IDs")
        return {"process_ids": process_ids, "thread_ids": thread_ids}

    def find_processes_by_ids(self, process_ids: Set[int]) -> List[psutil.Process]:
        """Find running processes by their PIDs."""
        processes = []
        current_pid = os.getpid()

        for pid in process_ids:
            if pid == current_pid:
                logger.debug(f"Skipping current process PID {pid}")
                continue

            try:
                proc = psutil.Process(pid)
                if proc.is_running():
                    processes.append(proc)
                    logger.debug(f"Found running process PID {pid}")
                else:
                    logger.debug(f"Process PID {pid} is not running")
            except psutil.NoSuchProcess:
                logger.debug(f"Process PID {pid} no longer exists")
            except psutil.AccessDenied:
                logger.warning(f"Access denied to process PID {pid}")
            except Exception as e:
                logger.warning(f"Error checking process PID {pid}: {e}")

        return processes

    def find_threads_by_ids(self, thread_ids: Set[int]) -> List[threading.Thread]:
        """Find running threads by their IDs (if possible)."""
        # Note: Python doesn't provide a direct way to enumerate all threads
        # This is more of a placeholder for future implementation
        logger.info(f"Thread termination not implemented yet. Found {len(thread_ids)} thread IDs")
        return []

    def get_partition_status(self) -> Dict[int, Dict[str, Any]]:
        """Get current status of all partitions."""
        dataset_mapping = self.load_dataset_mapping()
        events = self.load_event_logs()

        partition_status = {}

        # Initialize from dataset mapping
        if "partitions" in dataset_mapping:
            for partition_info in dataset_mapping["partitions"]:
                partition_id = partition_info["partition_id"]
                partition_status[partition_id] = {
                    "status": partition_info.get("processing_status", "unknown"),
                    "sample_count": partition_info.get("sample_count", 0),
                    "start_time": partition_info.get("processing_start_time"),
                    "end_time": partition_info.get("processing_end_time"),
                    "error_message": partition_info.get("error_message"),
                    "current_op": None,
                    "completed_ops": [],
                    "checkpoints": [],
                }

        # Update from event logs
        for event in events:
            if "partition_id" in event:
                partition_id = event["partition_id"]
                if partition_id not in partition_status:
                    partition_status[partition_id] = {
                        "status": "unknown",
                        "sample_count": 0,
                        "start_time": None,
                        "end_time": None,
                        "error_message": None,
                        "current_op": None,
                        "completed_ops": [],
                        "checkpoints": [],
                    }

                # Track partition start/complete
                if event["event_type"] == "partition_start":
                    partition_status[partition_id]["start_time"] = event["timestamp"]
                    partition_status[partition_id]["status"] = "processing"

                elif event["event_type"] == "partition_complete":
                    partition_status[partition_id]["end_time"] = event["timestamp"]
                    partition_status[partition_id]["status"] = "completed"

                # Track operations
                elif event["event_type"] == "op_start":
                    partition_status[partition_id]["current_op"] = {
                        "name": event.get("operation_name", "Unknown"),
                        "idx": event.get("operation_idx", 0),
                        "start_time": event["timestamp"],
                    }

                elif event["event_type"] == "op_complete":
                    op_info = {
                        "name": event.get("operation_name", "Unknown"),
                        "idx": event.get("operation_idx", 0),
                        "duration": event.get("duration", 0),
                        "input_rows": event.get("input_rows", 0),
                        "output_rows": event.get("output_rows", 0),
                        "throughput": event.get("performance_metrics", {}).get("throughput", 0),
                        "reduction_ratio": event.get("performance_metrics", {}).get("reduction_ratio", 0),
                    }
                    partition_status[partition_id]["completed_ops"].append(op_info)
                    partition_status[partition_id]["current_op"] = None

                # Track checkpoints
                elif event["event_type"] == "checkpoint_save":
                    checkpoint_info = {
                        "operation_name": event.get("operation_name", "Unknown"),
                        "operation_idx": event.get("operation_idx", 0),
                        "checkpoint_path": event.get("checkpoint_path", ""),
                        "timestamp": event["timestamp"],
                    }
                    partition_status[partition_id]["checkpoints"].append(checkpoint_info)

        return partition_status

    def calculate_overall_progress(self) -> Dict[str, Any]:
        """Calculate overall job progress."""
        partition_status = self.get_partition_status()
        job_summary = self.load_job_summary()

        total_partitions = len(partition_status)
        completed_partitions = sum(1 for p in partition_status.values() if p["status"] == "completed")
        processing_partitions = sum(1 for p in partition_status.values() if p["status"] == "processing")
        failed_partitions = sum(1 for p in partition_status.values() if p["status"] == "failed")

        # Calculate total samples
        total_samples = sum(p.get("sample_count", 0) for p in partition_status.values())
        processed_samples = sum(
            p.get("sample_count", 0) for p in partition_status.values() if p["status"] == "completed"
        )

        # Calculate progress percentage
        progress_percentage = (completed_partitions / total_partitions * 100) if total_partitions > 0 else 0

        # Calculate estimated time remaining
        estimated_remaining = None
        if job_summary and "start_time" in job_summary and completed_partitions > 0:
            elapsed_time = time.time() - job_summary["start_time"]
            if completed_partitions > 0:
                avg_time_per_partition = elapsed_time / completed_partitions
                remaining_partitions = total_partitions - completed_partitions
                estimated_remaining = avg_time_per_partition * remaining_partitions

        return {
            "total_partitions": total_partitions,
            "completed_partitions": completed_partitions,
            "processing_partitions": processing_partitions,
            "failed_partitions": failed_partitions,
            "progress_percentage": progress_percentage,
            "total_samples": total_samples,
            "processed_samples": processed_samples,
            "estimated_remaining_seconds": estimated_remaining,
            "job_status": job_summary.get("status", "unknown") if job_summary else "unknown",
        }

    def get_operation_pipeline(self) -> List[Dict[str, Any]]:
        """Get the operation pipeline from config."""
        config_file = self.work_dir / "partition-checkpoint-eventlog.yaml"
        if not config_file.exists():
            return []

        # Try to find process section in config
        try:
            with open(config_file, "r") as f:
                content = f.read()

            # Simple parsing for process section
            operations = []
            lines = content.split("\n")
            in_process = False

            for line in lines:
                if line.strip().startswith("process:"):
                    in_process = True
                    continue
                elif in_process and line.strip().startswith("-"):
                    # Extract operation name
                    op_line = line.strip()
                    if ":" in op_line:
                        op_name = op_line.split(":")[0].replace("- ", "").strip()
                        operations.append({"name": op_name, "config": {}})

            return operations
        except Exception as e:
            logger.warning(f"Failed to parse operation pipeline: {e}")
            return []


def list_running_jobs(base_dir: str = "outputs/partition-checkpoint-eventlog") -> List[Dict[str, Any]]:
    """List all DataJuicer jobs and their status."""
    base_path = Path(base_dir)
    if not base_path.exists():
        return []

    jobs = []
    for job_dir in base_path.iterdir():
        if job_dir.is_dir():
            job_summary_file = job_dir / "job_summary.json"
            if job_summary_file.exists():
                try:
                    with open(job_summary_file, "r") as f:
                        job_summary = json.load(f)

                    # Check if processes are still running
                    events_file = job_dir / "events.jsonl"
                    process_ids = set()
                    if events_file.exists():
                        try:
                            with open(events_file, "r") as f:
                                for line in f:
                                    try:
                                        event_data = json.loads(line.strip())
                                        if "process_id" in event_data and event_data["process_id"] is not None:
                                            process_ids.add(event_data["process_id"])
                                    except json.JSONDecodeError:
                                        continue
                        except Exception:
                            pass

                    # Count running processes
                    running_processes = 0
                    for pid in process_ids:
                        try:
                            if psutil.Process(pid).is_running():
                                running_processes += 1
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass

                    jobs.append(
                        {
                            "job_id": job_dir.name,
                            "status": job_summary.get("status", "unknown"),
                            "start_time": job_summary.get("start_time"),
                            "processes": running_processes,
                            "work_dir": str(job_dir),
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to read job summary for {job_dir.name}: {e}")

    return sorted(jobs, key=lambda x: x.get("start_time", 0) or 0, reverse=True)

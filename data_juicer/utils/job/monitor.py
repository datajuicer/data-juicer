#!/usr/bin/env python3
"""
DataJuicer Job Progress Monitor

A utility to monitor and display progress information for DataJuicer jobs.
Shows partition status, operation progress, checkpoints, and overall job metrics.
"""

import os
import sys
import time
from datetime import datetime
from typing import Any, Dict

from data_juicer.utils.job.common import JobUtils


class JobProgressMonitor:
    """Monitor and display progress for DataJuicer jobs."""

    def __init__(self, job_id: str, base_dir: str = "outputs/partition-checkpoint-eventlog"):
        """
        Initialize the job progress monitor.

        Args:
            job_id: The job ID to monitor
            base_dir: Base directory containing job outputs
        """
        self.job_utils = JobUtils(job_id, base_dir=base_dir)
        self.job_id = job_id
        self.work_dir = self.job_utils.work_dir

    def display_progress(self, detailed: bool = False):
        """Display job progress information."""
        print(f"\n{'='*80}")
        print(f"DataJuicer Job Progress Monitor")
        print(f"Job ID: {self.job_id}")
        print(f"{'='*80}")

        # Load data
        job_summary = self.job_utils.load_job_summary()
        dataset_mapping = self.job_utils.load_dataset_mapping()
        partition_status = self.job_utils.get_partition_status()
        overall_progress = self.job_utils.calculate_overall_progress()

        # Job overview
        print(f"\n📊 JOB OVERVIEW")
        print(f"   Status: {overall_progress['job_status'].upper()}")
        print(f"   Dataset: {dataset_mapping.get('original_dataset_path', 'Unknown')}")
        print(f"   Total Samples: {dataset_mapping.get('original_dataset_size', 0):,}")
        print(f"   Partition Size: {dataset_mapping.get('partition_size', 0):,} samples")

        if job_summary and job_summary.get("start_time"):
            start_time = datetime.fromtimestamp(job_summary["start_time"])
            print(f"   Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        if job_summary and job_summary.get("duration"):
            print(f"   Duration: {job_summary['duration']:.1f} seconds")

        # Overall progress
        print(f"\n🎯 OVERALL PROGRESS")
        print(
            f"   Progress: {overall_progress['progress_percentage']:.1f}% "
            f"({overall_progress['completed_partitions']}/{overall_progress['total_partitions']} partitions)"
        )
        print(
            f"   Status: {overall_progress['completed_partitions']} completed, "
            f"{overall_progress['processing_partitions']} processing, "
            f"{overall_progress['failed_partitions']} failed"
        )
        print(f"   Samples: {overall_progress['processed_samples']:,}/{overall_progress['total_samples']:,}")

        if overall_progress["estimated_remaining_seconds"]:
            remaining_minutes = overall_progress["estimated_remaining_seconds"] / 60
            print(f"   Estimated Time Remaining: {remaining_minutes:.1f} minutes")

        # Partition status
        print(f"\n📦 PARTITION STATUS")
        for partition_id in sorted(partition_status.keys()):
            partition = partition_status[partition_id]
            status_icon = {"completed": "✅", "processing": "🔄", "failed": "❌", "unknown": "❓"}.get(
                partition["status"], "❓"
            )

            print(f"   Partition {partition_id:2d}: {status_icon} {partition['status'].upper()}")
            print(f"              Samples: {partition['sample_count']:,}")

            if partition["current_op"]:
                print(f"              Current: {partition['current_op']['name']} (op {partition['current_op']['idx']})")

            if partition["completed_ops"]:
                print(f"              Completed: {len(partition['completed_ops'])} operations")

            if partition["checkpoints"]:
                print(f"              Checkpoints: {len(partition['checkpoints'])} saved")

        if detailed:
            # Detailed operation information
            print(f"\n🔧 OPERATION DETAILS")
            for partition_id in sorted(partition_status.keys()):
                partition = partition_status[partition_id]
                if partition["completed_ops"]:
                    print(f"\n   Partition {partition_id}:")
                    for op in partition["completed_ops"]:
                        reduction = op.get("reduction_ratio", 0) * 100
                        print(
                            f"     {op['name']:25s} | "
                            f"Duration: {op['duration']:6.1f}s | "
                            f"Throughput: {op['throughput']:6.0f} rows/s | "
                            f"Reduction: {reduction:5.2f}%"
                        )

        # Checkpoint information
        print(f"\n💾 CHECKPOINT SUMMARY")
        total_checkpoints = sum(len(p["checkpoints"]) for p in partition_status.values())
        print(f"   Total Checkpoints: {total_checkpoints}")

        if detailed:
            for partition_id in sorted(partition_status.keys()):
                partition = partition_status[partition_id]
                if partition["checkpoints"]:
                    print(f"\n   Partition {partition_id} checkpoints:")
                    for checkpoint in partition["checkpoints"]:
                        checkpoint_time = datetime.fromtimestamp(checkpoint["timestamp"])
                        print(
                            f"     {checkpoint['operation_name']} (op {checkpoint['operation_idx']}) - "
                            f"{checkpoint_time.strftime('%H:%M:%S')}"
                        )

        # Add helpful hint for stopping the job
        print(f"\n💡 To stop this job: from data_juicer.utils.job_stopper import stop_job; stop_job('{self.job_id}')")
        print(f"{'='*80}")

    def get_progress_data(self) -> Dict[str, Any]:
        """Get progress data as a dictionary for programmatic use."""
        job_summary = self.job_utils.load_job_summary()
        dataset_mapping = self.job_utils.load_dataset_mapping()
        partition_status = self.job_utils.get_partition_status()
        overall_progress = self.job_utils.calculate_overall_progress()

        return {
            "job_id": self.job_id,
            "job_summary": job_summary,
            "dataset_mapping": dataset_mapping,
            "partition_status": partition_status,
            "overall_progress": overall_progress,
        }


def show_job_progress(
    job_id: str, base_dir: str = "outputs/partition-checkpoint-eventlog", detailed: bool = False
) -> Dict[str, Any]:
    """
    Utility function to show job progress.

    Args:
        job_id: The job ID to monitor
        base_dir: Base directory containing job outputs
        detailed: Whether to show detailed operation information

    Returns:
        Dictionary containing all progress data

    Example:
        >>> show_job_progress("20250728_233517_510abf")
        >>> show_job_progress("20250728_233517_510abf", detailed=True)
    """
    monitor = JobProgressMonitor(job_id, base_dir)
    monitor.display_progress(detailed)
    return monitor.get_progress_data()


def main():
    """Main entry point for the job progress monitor."""
    import argparse

    parser = argparse.ArgumentParser(description="Monitor DataJuicer job progress")
    parser.add_argument("job_id", help="Job ID to monitor")
    parser.add_argument(
        "--base-dir", default="outputs/partition-checkpoint-eventlog", help="Base directory containing job outputs"
    )
    parser.add_argument("--detailed", action="store_true", help="Show detailed operation information")
    parser.add_argument("--watch", action="store_true", help="Watch mode - continuously update progress")
    parser.add_argument("--interval", type=int, default=10, help="Update interval in seconds for watch mode")

    args = parser.parse_args()

    try:
        monitor = JobProgressMonitor(args.job_id, args.base_dir)

        if args.watch:
            print(f"Watching job {args.job_id} (press Ctrl+C to stop)...")
            try:
                while True:
                    os.system("clear" if os.name == "posix" else "cls")
                    monitor.display_progress(args.detailed)
                    time.sleep(args.interval)
            except KeyboardInterrupt:
                print("\nStopped watching.")
        else:
            monitor.display_progress(args.detailed)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

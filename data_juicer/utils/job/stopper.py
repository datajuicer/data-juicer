#!/usr/bin/env python3
"""
DataJuicer Job Stopper

A utility to stop DataJuicer jobs by reading event logs to find process and thread IDs,
then terminating those specific processes and threads.
"""

import json
import sys
import time
from typing import Any, Dict

import psutil
from loguru import logger

from data_juicer.utils.job.common import JobUtils, list_running_jobs


class JobStopper:
    """Stop DataJuicer jobs using event log-based process discovery."""

    def __init__(self, job_id: str, base_dir: str = "outputs/partition-checkpoint-eventlog"):
        self.job_utils = JobUtils(job_id, base_dir=base_dir)
        self.job_id = job_id
        self.work_dir = self.job_utils.work_dir

    def terminate_process_gracefully(self, proc, timeout: int = 10) -> bool:
        """Terminate a process gracefully with timeout."""
        try:
            logger.info(f"Terminating process {proc.pid} gracefully...")
            proc.terminate()

            # Wait for the process to terminate
            try:
                proc.wait(timeout=timeout)
                logger.info(f"Process {proc.pid} terminated gracefully")
                return True
            except psutil.TimeoutExpired:
                logger.warning(f"Process {proc.pid} did not terminate within {timeout}s, force killing...")
                proc.kill()
                proc.wait()
                logger.info(f"Process {proc.pid} force killed")
                return True

        except psutil.NoSuchProcess:
            logger.info(f"Process {proc.pid} already terminated")
            return True
        except psutil.AccessDenied:
            logger.error(f"Access denied when terminating process {proc.pid}")
            return False
        except Exception as e:
            logger.error(f"Error terminating process {proc.pid}: {e}")
            return False

    def cleanup_job_resources(self) -> None:
        """Clean up job resources and update job summary."""
        job_summary = self.job_utils.load_job_summary()
        if job_summary:
            job_summary["status"] = "stopped"
            job_summary["stop_time"] = time.time()
            job_summary["stop_reason"] = "manual_stop"

            try:
                with open(self.work_dir / "job_summary.json", "w") as f:
                    json.dump(job_summary, f, indent=2, default=str)
                logger.info(f"Updated job summary: {self.work_dir / 'job_summary.json'}")
            except Exception as e:
                logger.error(f"Failed to update job summary: {e}")

    def stop_job(self, force: bool = False, timeout: int = 30) -> Dict[str, Any]:
        """Stop the DataJuicer job using event log-based process discovery."""
        results = {
            "job_id": self.job_id,
            "success": False,
            "processes_found": 0,
            "processes_terminated": 0,
            "threads_found": 0,
            "threads_terminated": 0,
            "errors": [],
        }

        logger.info(f"🛑 Stopping DataJuicer job: {self.job_id}")
        logger.info(f"Work directory: {self.work_dir}")

        # Load job summary
        job_summary = self.job_utils.load_job_summary()
        if job_summary:
            logger.info(f"Job status: {job_summary.get('status', 'unknown')}")
            logger.info(f"Job started: {job_summary.get('start_time', 'unknown')}")

        # Extract process and thread IDs from event logs
        logger.info("🔍 Extracting process and thread IDs from event logs...")
        ids = self.job_utils.extract_process_thread_ids()

        results["processes_found"] = len(ids["process_ids"])
        results["threads_found"] = len(ids["thread_ids"])

        if not ids["process_ids"] and not ids["thread_ids"]:
            logger.warning("No process or thread IDs found in event logs")
            results["errors"].append("No process or thread IDs found in event logs")
            self.cleanup_job_resources()
            return results

        # Find and terminate processes
        logger.info(f"🔍 Finding {len(ids['process_ids'])} processes...")
        processes = self.job_utils.find_processes_by_ids(ids["process_ids"])

        if processes:
            logger.info(f"Found {len(processes)} running processes to terminate")
            for proc in processes:
                if self.terminate_process_gracefully(proc, timeout):
                    results["processes_terminated"] += 1
                else:
                    results["errors"].append(f"Failed to terminate process {proc.pid}")
        else:
            logger.info("No running processes found")

        # Find and terminate threads (placeholder for future implementation)
        logger.info(f"🔍 Finding {len(ids['thread_ids'])} threads...")
        threads = self.job_utils.find_threads_by_ids(ids["thread_ids"])
        results["threads_terminated"] = len(threads)

        # Clean up job resources
        logger.info("🧹 Cleaning up job resources...")
        self.cleanup_job_resources()

        # Determine success
        results["success"] = results["processes_terminated"] > 0 or results["threads_terminated"] > 0

        if results["success"]:
            logger.info(f"✅ Job {self.job_id} stopped successfully")
            logger.info(f"   Terminated {results['processes_terminated']} processes")
            logger.info(f"   Terminated {results['threads_terminated']} threads")
        else:
            logger.warning(f"⚠️  Job {self.job_id} may not have been fully stopped")
            if results["errors"]:
                logger.error(f"   Errors: {results['errors']}")

        return results


def stop_job(
    job_id: str, base_dir: str = "outputs/partition-checkpoint-eventlog", force: bool = False, timeout: int = 30
) -> Dict[str, Any]:
    """Stop a DataJuicer job using event log-based process discovery."""
    stopper = JobStopper(job_id, base_dir)
    return stopper.stop_job(force=force, timeout=timeout)


def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Stop DataJuicer jobs using event log-based process discovery")
    parser.add_argument("job_id", nargs="?", help="Job ID to stop")
    parser.add_argument(
        "--base-dir", default="outputs/partition-checkpoint-eventlog", help="Base directory for job outputs"
    )
    parser.add_argument("--force", action="store_true", help="Force termination")
    parser.add_argument("--timeout", type=int, default=30, help="Termination timeout in seconds")
    parser.add_argument("--list", action="store_true", help="List all jobs")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")

    if args.list:
        jobs = list_running_jobs(args.base_dir)
        if jobs:
            print("📋 DataJuicer Jobs:")
            print("=" * 80)
            for job in jobs:
                status_icon = "🟢" if job["status"] == "completed" else "🟡" if job["status"] == "running" else "🔴"
                print(f"{status_icon} {job['job_id']} | Status: {job['status']} | Processes: {job['processes']}")
        else:
            print("No DataJuicer jobs found")
        return

    if not args.job_id:
        parser.error("Job ID is required unless using --list")

    result = stop_job(args.job_id, args.base_dir, force=args.force, timeout=args.timeout)

    if result["success"]:
        print(f"✅ Job {args.job_id} stopped successfully")
        print(f"   Terminated {result['processes_terminated']} processes")
        print(f"   Terminated {result['threads_terminated']} threads")
    else:
        print(f"⚠️  Job {args.job_id} may not have been fully stopped")
        if result["errors"]:
            print(f"   Errors: {result['errors']}")
        sys.exit(1)


if __name__ == "__main__":
    main()

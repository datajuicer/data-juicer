#!/usr/bin/env python3
"""
Comprehensive Demo for DataJuicer Job Management & Monitoring

This script demonstrates all the implemented job management features:
1. Processing Snapshot Utility - Comprehensive job status analysis with JSON output
2. Job Management Tools - Monitor and manage DataJuicer processing jobs
3. Resource-Aware Partitioning - Automatic resource optimization for distributed processing
4. Job-specific directory isolation
5. Flexible storage paths for event logs and checkpoints
6. Configurable checkpointing strategies
7. Event logging with JSONL format
8. Job resumption capabilities
9. Comprehensive job management

Usage:
    # IMPORTANT: This script must be run from the Data-Juicer root directory
    cd /path/to/data-juicer
    python demos/partition_and_checkpoint/run_demo.py
"""

import os
import subprocess
import time
import json
from pathlib import Path
import re


def run_data_juicer_command(config_file, job_id=None, extra_args=None):
    """Run a DataJuicer command and return the result."""
    cmd = ["dj-process", "--config", config_file]
    if job_id:
        cmd.extend(["--job_id", job_id])
    if extra_args:
        cmd.extend(extra_args)
    
    print(f"Running: {' '.join(cmd)}")
    print("-" * 80)
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end_time = time.time()
    
    print(f"Exit code: {result.returncode}")
    print(f"Duration: {end_time - start_time:.2f} seconds")
    print("-" * 80)
    
    if result.stdout:
        print("STDOUT:")
        print(result.stdout)
    
    if result.stderr:
        print("STDERR:")
        print(result.stderr)
    
    return result


def run_snapshot_analysis(job_id, work_dir="./outputs/partition-checkpoint-eventlog"):
    """Run the processing snapshot utility to analyze job status."""
    print(f"\n📊 Processing Snapshot Analysis for {job_id}:")
    print("=" * 60)
    
    # Run the snapshot utility
    job_dir = os.path.join(work_dir, job_id)
    cmd = ["python", "-m", "data_juicer.utils.job.snapshot", job_dir]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            snapshot_data = json.loads(result.stdout)
            print("✅ Snapshot Analysis Results:")
            print(f"   Job Status: {snapshot_data.get('overall_status', 'unknown')}")
            print(f"   Progress: {snapshot_data.get('overall_progress', {}).get('overall_percentage', 0):.1f}%")
            print(f"   Duration: {snapshot_data.get('timing', {}).get('duration_formatted', 'unknown')}")
            print(f"   Partitions: {snapshot_data.get('progress_summary', {}).get('completed_partitions', 0)}/{snapshot_data.get('progress_summary', {}).get('total_partitions', 0)}")
            print(f"   Operations: {snapshot_data.get('progress_summary', {}).get('completed_operations', 0)}/{snapshot_data.get('progress_summary', {}).get('total_operations', 0)}")
            print(f"   Resumable: {snapshot_data.get('checkpointing', {}).get('resumable', False)}")
        else:
            print(f"❌ Snapshot analysis failed: {result.stderr}")
    except Exception as e:
        print(f"❌ Error running snapshot analysis: {e}")
    
    print("=" * 60)


def check_directory_structure(job_id, work_dir="./outputs/partition-checkpoint-eventlog"):
    """Check and display the job-specific directory structure."""
    job_dir = os.path.join(work_dir, job_id)
    
    print(f"\n📁 Job Directory Structure for {job_id}:")
    print("=" * 60)
    
    if os.path.exists(job_dir):
        for root, dirs, files in os.walk(job_dir):
            level = root.replace(job_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                print(f"{subindent}{file}")
    else:
        print(f"Job directory {job_dir} does not exist")
    
    print("=" * 60)


def check_flexible_storage(job_id):
    """Check job storage directories."""
    print(f"\n💾 Job Storage for {job_id}:")
    print("=" * 60)
    
    # Check event logs in job directory
    event_log_file = f"./outputs/partition-checkpoint-eventlog/{job_id}/events.jsonl"
    if os.path.exists(event_log_file):
        size = os.path.getsize(event_log_file)
        print(f"✅ Event Logs: {event_log_file} ({size} bytes)")
    else:
        print(f"❌ Event Logs: {event_log_file} not found")
    
    # Check logs directory
    logs_dir = f"./outputs/partition-checkpoint-eventlog/{job_id}/logs"
    if os.path.exists(logs_dir):
        print(f"✅ Logs Directory: {logs_dir}")
        for file in os.listdir(logs_dir):
            file_path = os.path.join(logs_dir, file)
            size = os.path.getsize(file_path)
            print(f"   📄 {file} ({size} bytes)")
    else:
        print(f"❌ Logs Directory: {logs_dir} not found")
    
    # Check checkpoints in job directory
    checkpoint_dir = f"./outputs/partition-checkpoint-eventlog/{job_id}/checkpoints"
    if os.path.exists(checkpoint_dir):
        print(f"✅ Checkpoints: {checkpoint_dir}")
        for file in os.listdir(checkpoint_dir):
            file_path = os.path.join(checkpoint_dir, file)
            if os.path.isfile(file_path):
                size = os.path.getsize(file_path)
                print(f"   💾 {file} ({size} bytes)")
            else:
                print(f"   📁 {file}/")
    else:
        print(f"❌ Checkpoints: {checkpoint_dir} not found")
    
    print("=" * 60)


def check_job_summary(job_id, work_dir="./outputs/partition-checkpoint-eventlog"):
    """Check and display job summary."""
    job_dir = os.path.join(work_dir, job_id)
    summary_file = os.path.join(job_dir, "job_summary.json")
    
    print(f"\n📋 Job Summary for {job_id}:")
    print("=" * 60)
    
    if os.path.exists(summary_file):
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        
        print(f"Job ID: {summary.get('job_id')}")
        print(f"Status: {summary.get('status')}")
        print(f"Start Time: {summary.get('start_time')}")
        print(f"Job Directory: {summary.get('job_dir')}")
        print(f"Event Log File: {summary.get('event_log_file')}")
        print(f"Checkpoint Directory: {summary.get('checkpoint_dir')}")
        print(f"Resumption Command: {summary.get('resumption_command')}")
    else:
        print(f"Job summary file {summary_file} not found")
    
    print("=" * 60)


def check_resource_optimization():
    """Check resource-aware partitioning configuration."""
    print(f"\n⚙️ Resource-Aware Partitioning Check:")
    print("=" * 60)
    
    # Check if resource optimization is enabled in config
    config_file = "configs/demo/partition-checkpoint-eventlog.yaml"
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config_content = f.read()
        
        if "resource_optimization:" in config_content and "auto_configure: true" in config_content:
            print("✅ Resource optimization is enabled")
            print("   - Automatic partition size optimization")
            print("   - Worker count optimization")
            print("   - 64MB partition targeting")
        else:
            print("ℹ️ Resource optimization not enabled (using manual configuration)")
    else:
        print(f"❌ Config file {config_file} not found")
    
    print("=" * 60)


def get_latest_job_id(work_dir):
    """Get the most recently created job_id directory in work_dir."""
    if not os.path.exists(work_dir):
        return None
    job_dirs = [d for d in os.listdir(work_dir) if os.path.isdir(os.path.join(work_dir, d))]
    if not job_dirs:
        return None
    # Sort by creation time (descending)
    job_dirs = sorted(job_dirs, key=lambda d: os.path.getctime(os.path.join(work_dir, d)), reverse=True)
    return job_dirs[0]


def main():
    """Run the comprehensive demo."""
    print("🚀 DataJuicer Job Management & Monitoring Demo")
    print("=" * 80)
    
    # IMPORTANT: This script must be run from the Data-Juicer root directory
    # Check if we're in the root directory by looking for key files/directories
    if not os.path.exists("configs") or not os.path.exists("data_juicer"):
        print("❌ Error: This script must be run from the Data-Juicer root directory!")
        print("   Current directory:", os.getcwd())
        print("   Expected to find: configs/ and data_juicer/ directories")
        print("\n   Please run:")
        print("   cd /path/to/data-juicer")
        print("   python demos/partition_and_checkpoint/run_demo.py")
        return
    
    config_file = "configs/demo/partition-checkpoint-eventlog.yaml"
    work_dir = "./outputs/partition-checkpoint-eventlog"
    
    # Ensure the config file exists
    if not os.path.exists(config_file):
        print(f"❌ Config file {config_file} not found!")
        print("Please run this script from the DataJuicer root directory.")
        return
    
    # Check resource optimization configuration
    check_resource_optimization()
    
    # Demo 1: First run with new job (auto-generated job_id)
    print("\n🎯 Demo 1: First Run (New Job, Auto-generated job_id)")
    print("=" * 80)
    result1 = run_data_juicer_command(config_file)
    job_id_1 = get_latest_job_id(work_dir)
    if result1.returncode == 0 and job_id_1:
        print(f"✅ First run completed successfully! (job_id: {job_id_1})")
        check_directory_structure(job_id_1, work_dir)
        check_flexible_storage(job_id_1)
        check_job_summary(job_id_1, work_dir)
        run_snapshot_analysis(job_id_1, work_dir)
    else:
        print("❌ First run failed!")
        return
    
    # Demo 2: Resume the same job
    print("\n🎯 Demo 2: Resume Job")
    print("=" * 80)
    result2 = run_data_juicer_command(config_file, job_id_1)
    if result2.returncode == 0:
        print("✅ Job resumption completed successfully!")
        print("Note: This should be much faster than the first run due to checkpoint resumption.")
        check_job_summary(job_id_1, work_dir)
        run_snapshot_analysis(job_id_1, work_dir)
    else:
        print("❌ Job resumption failed!")
    
    # Demo 3: New job with different checkpoint strategy (auto-generated job_id)
    print("\n🎯 Demo 3: Different Checkpoint Strategy")
    print("=" * 80)
    extra_args = ["--checkpoint.strategy", "every_partition"]
    result3 = run_data_juicer_command(config_file, None, extra_args)
    job_id_2 = get_latest_job_id(work_dir)
    if result3.returncode == 0 and job_id_2:
        print(f"✅ Different checkpoint strategy completed successfully! (job_id: {job_id_2})")
        check_directory_structure(job_id_2, work_dir)
        check_flexible_storage(job_id_2)
        check_job_summary(job_id_2, work_dir)
        run_snapshot_analysis(job_id_2, work_dir)
    else:
        print("❌ Different checkpoint strategy failed!")
    
    # Demo 4: List available jobs
    print("\n🎯 Demo 4: List Available Jobs")
    print("=" * 80)
    if os.path.exists(work_dir):
        print("Available job directories:")
        for item in os.listdir(work_dir):
            item_path = os.path.join(work_dir, item)
            if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, "job_summary.json")):
                print(f"  📁 {item}")
    else:
        print(f"Work directory {work_dir} not found")
    
    print("\n🎉 Demo completed!")
    print("=" * 80)
    print("Key Features Demonstrated:")
    print("✅ Processing Snapshot Utility - Comprehensive job status analysis with JSON output")
    print("✅ Job Management Tools - Monitor and manage DataJuicer processing jobs")
    print("✅ Resource-Aware Partitioning - Automatic resource optimization for distributed processing")
    print("✅ Job-specific directory isolation")
    print("✅ Event logging with JSONL format")
    print("✅ Human-readable logs with multiple levels")
    print("✅ Configurable checkpointing strategies")
    print("✅ Job resumption capabilities")
    print("✅ Comprehensive job management with job_summary.json")
    print("✅ Fast resumption from checkpoints")


if __name__ == "__main__":
    main() 
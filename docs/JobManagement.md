# Job Management & Monitoring

Data-Juicer provides comprehensive job management and monitoring capabilities to help you track, analyze, and optimize your data processing workflows.

## Overview

The job management system includes:

- **Processing Snapshot Utility**: Detailed analysis of job status and progress
- **Resource-Aware Partitioning**: Automatic optimization of distributed processing
- **Enhanced Logging**: Centralized logging with rotation and retention
- **Job Monitoring Tools**: Real-time tracking of processing jobs

## Processing Snapshot Utility

The Processing Snapshot Utility provides comprehensive analysis of Data-Juicer job processing status based on `events.jsonl` and DAG structure.

### Features

- **JSON Output**: Machine-readable format for automation and integration
- **Progress Tracking**: Detailed partition and operation progress
- **Checkpointing Analysis**: Checkpoint status and resumability information
- **Timing Analysis**: Precise timing from job summary or events
- **Resource Utilization**: Partition and operation-level statistics

### Usage

#### Basic Snapshot
```bash
python -m data_juicer.utils.job.snapshot outputs/partition-checkpoint-eventlog/20250809_040053_a001de
```

#### Human-Readable Output
```bash
python -m data_juicer.utils.job.snapshot outputs/partition-checkpoint-eventlog/20250809_040053_a001de --human-readable
```

### JSON Output Structure

```json
{
  "job_info": {
    "job_id": "20250809_040053_a001de",
    "executor_type": "ray_partitioned",
    "status": "completed",
    "config_file": ["configs/demo/partition-checkpoint-eventlog.yaml"],
    "work_dir": "./outputs/partition-checkpoint-eventlog/20250809_040053_a001de",
    "resumption_command": "dj-process --config [Path_fr(...)] --job_id 20250809_040053_a001de",
    "error_message": null
  },
  "overall_status": "completed",
  "overall_progress": {
    "overall_percentage": 100.0,
    "partition_percentage": 100.0,
    "operation_percentage": 100.0
  },
  "timing": {
    "start_time": 1754712053.496651,
    "end_time": 1754712325.323669,
    "duration_seconds": 271.82701802253723,
    "duration_formatted": "4m 31s",
    "job_summary_duration": 271.82701802253723,
    "timing_source": "job_summary"
  },
  "progress_summary": {
    "total_partitions": 18,
    "completed_partitions": 18,
    "failed_partitions": 0,
    "in_progress_partitions": 0,
    "partition_progress_percentage": 100.0,
    "total_operations": 144,
    "completed_operations": 144,
    "failed_operations": 0,
    "checkpointed_operations": 0,
    "operation_progress_percentage": 100.0
  },
  "checkpointing": {
    "strategy": "every_op",
    "last_checkpoint_time": 1754712320.123456,
    "checkpointed_operations_count": 72,
    "resumable": true,
    "checkpoint_progress": {
      "percentage": 50.0,
      "checkpointed_operations": [...],
      "checkpoint_coverage": 0.5
    },
    "checkpoint_dir": "./outputs/partition-checkpoint-eventlog/20250809_040053_a001de/checkpoints"
  },
  "partition_progress": {
    "0": {
      "status": "completed",
      "sample_count": 20000,
      "creation_start_time": 1754712074.356004,
      "creation_end_time": 1754712074.366004,
      "processing_start_time": 1754712074.366004,
      "processing_end_time": 1754712074.456004,
      "current_operation": null,
      "completed_operations": ["clean_links_mapper", "clean_email_mapper", ...],
      "failed_operations": [],
      "checkpointed_operations": [],
      "error_message": null,
      "progress_percentage": 100.0
    }
  },
  "operation_progress": {
    "p0_op0_clean_links_mapper": {
      "operation_name": "clean_links_mapper",
      "operation_idx": 0,
      "status": "completed",
      "start_time": 1754712074.356004,
      "end_time": 1754712074.366004,
      "duration": 0.01,
      "input_rows": 20000,
      "output_rows": 19363,
      "checkpoint_time": null,
      "error_message": null,
      "progress_percentage": 100.0
    }
  },
  "file_paths": {
    "event_log_file": "./outputs/partition-checkpoint-eventlog/20250809_040053_a001de/events.jsonl",
    "event_log_dir": "./outputs/partition-checkpoint-eventlog/20250809_040053_a001de/logs",
    "checkpoint_dir": "./outputs/partition-checkpoint-eventlog/20250809_040053_a001de/checkpoints",
    "metadata_dir": "./outputs/partition-checkpoint-eventlog/20250809_040053_a001de/metadata",
    "backed_up_config_path": "./outputs/partition-checkpoint-eventlog/20250809_040053_a001de/partition-checkpoint-eventlog.yaml"
  },
  "metadata": {
    "snapshot_generated_at": "2025-08-09T13:33:54.770298",
    "events_analyzed": 367,
    "dag_plan_loaded": true,
    "job_summary_loaded": true,
    "job_summary_used": true
  }
}
```

## Resource-Aware Partitioning

The Resource-Aware Partitioning system automatically optimizes partition sizes and worker counts based on available cluster resources and data characteristics.

### Features

- **Automatic Resource Detection**: Analyzes local and cluster resources
- **Data-Driven Optimization**: Samples data to determine optimal partition sizes
- **Modality-Aware**: Different optimization strategies for text, image, audio, video, and multimodal data
- **64MB Target**: Optimizes partitions to target 64MB per partition
- **Worker Count Optimization**: Automatically determines optimal number of Ray workers

### Configuration

Enable resource optimization in your config:

```yaml
# Resource optimization configuration
resource_optimization:
  auto_configure: true  # Enable automatic optimization

# Manual configuration (used when auto_configure: false)
# partition:
#   size: 10000  # Number of samples per partition
#   max_size_mb: 128  # Maximum partition size in MB
# np: 2  # Number of Ray workers
```

### Optimization Process

1. **Resource Detection**: Analyzes CPU, memory, GPU, and cluster resources
2. **Data Sampling**: Samples dataset to understand data characteristics
3. **Modality Analysis**: Determines data modality and applies appropriate optimizations
4. **Partition Calculation**: Calculates optimal partition size targeting 64MB
5. **Worker Optimization**: Determines optimal number of Ray workers
6. **Application**: Applies optimizations to the processing pipeline

## Enhanced Logging System

The enhanced logging system provides centralized logging with rotation and retention policies.

### Features

- **Centralized Logging**: All logs managed through `logger_utils.py`
- **Log Rotation**: Automatic rotation based on file size
- **Retention Policies**: Configurable retention and cleanup
- **Compression**: Automatic compression of rotated logs
- **Multiple Levels**: Separate log files for different log levels

### Configuration

```python
from data_juicer.utils.logger_utils import setup_logger

# Setup logger with rotation and retention
setup_logger(
    save_dir="./outputs",
    filename="log.txt",
    max_log_size_mb=100,  # Rotate at 100MB
    backup_count=5        # Keep 5 backup files
)
```

### Log Structure

```
outputs/
├── job_20250809_040053_a001de/
│   ├── events.jsonl          # Event log (JSONL format)
│   ├── logs/                 # Log directory
│   │   ├── events.log        # Event log (human-readable)
│   │   ├── log.txt           # Main log file
│   │   ├── log_DEBUG.txt     # Debug level logs
│   │   ├── log_ERROR.txt     # Error level logs
│   │   └── log_WARNING.txt   # Warning level logs
│   ├── checkpoints/          # Checkpoint directory
│   ├── partitions/           # Partition directory
│   └── job_summary.json      # Job summary
```

## Job Management Tools

### Job Utilities

```python
from data_juicer.utils.job import JobUtils, create_snapshot

# Create job utilities
job_utils = JobUtils("./outputs")

# List running jobs
running_jobs = job_utils.list_running_jobs()

# Load event logs
events = job_utils.load_event_logs()

# Create processing snapshot
snapshot = create_snapshot("./outputs/job_20250809_040053_a001de")
```

### Event Analysis

The system tracks various event types:

- **Job Events**: `job_start`, `job_complete`
- **Partition Events**: `partition_creation_start`, `partition_creation_complete`, `partition_start`, `partition_complete`, `partition_failed`
- **Operation Events**: `op_start`, `op_complete`, `op_failed`
- **Checkpoint Events**: `checkpoint_save`
- **DAG Events**: `dag_build_start`, `dag_build_complete`, `dag_execution_plan_saved`

## Best Practices

### 1. Enable Resource Optimization

Always enable resource optimization for production workloads:

```yaml
resource_optimization:
  auto_configure: true
```

### 2. Monitor Job Progress

Use the snapshot utility to monitor long-running jobs:

```bash
# Check job status
python -m data_juicer.utils.job.snapshot /path/to/job/directory

# Get detailed analysis
python -m data_juicer.utils.job.snapshot /path/to/job/directory --human-readable
```

### 3. Configure Logging

Set appropriate log rotation and retention:

```python
setup_logger(
    save_dir="./outputs",
    max_log_size_mb=100,
    backup_count=5
)
```

### 4. Use Checkpointing

Enable checkpointing for long-running jobs:

```yaml
checkpoint:
  enabled: true
  strategy: "every_op"  # or "every_partition", "every_n_ops"
```

### 5. Monitor Resource Usage

The snapshot utility provides detailed resource utilization information:

- Partition-level progress and timing
- Operation-level performance metrics
- Checkpoint coverage and resumability
- Overall job efficiency statistics

## Integration Examples

### Automation Script

```python
import json
import subprocess
from pathlib import Path

def monitor_job(job_dir: str):
    """Monitor a Data-Juicer job and return status."""
    result = subprocess.run([
        "python", "-m", "data_juicer.utils.job.snapshot", job_dir
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        snapshot = json.loads(result.stdout)
        return {
            "status": snapshot["overall_status"],
            "progress": snapshot["overall_progress"]["overall_percentage"],
            "duration": snapshot["timing"]["duration_formatted"],
            "resumable": snapshot["checkpointing"]["resumable"]
        }
    else:
        return {"error": result.stderr}

# Usage
status = monitor_job("./outputs/job_20250809_040053_a001de")
print(f"Job Status: {status['status']}, Progress: {status['progress']:.1f}%")
```

### Dashboard Integration

The JSON output format makes it easy to integrate with monitoring dashboards:

```python
def get_job_metrics(job_dir: str):
    """Extract key metrics for dashboard display."""
    snapshot = create_snapshot(job_dir)
    
    return {
        "job_id": snapshot.job_id,
        "status": snapshot.overall_status.value,
        "progress": {
            "partitions": f"{snapshot.completed_partitions}/{snapshot.total_partitions}",
            "operations": f"{snapshot.completed_operations}/{snapshot.total_operations}"
        },
        "timing": {
            "duration": snapshot.total_duration,
            "start_time": snapshot.job_start_time
        },
        "checkpointing": {
            "resumable": snapshot.resumable,
            "strategy": snapshot.checkpoint_strategy
        }
    }
```

## Troubleshooting

### Common Issues

1. **Job Not Starting**: Check resource availability and configuration
2. **Slow Performance**: Enable resource optimization and check partition sizes
3. **Memory Issues**: Reduce partition size or enable checkpointing
4. **Log File Growth**: Configure log rotation and retention policies

### Debug Commands

```bash
# Check job status
python -m data_juicer.utils.job.snapshot /path/to/job

# Analyze events
python -c "import json; events = [json.loads(line) for line in open('/path/to/job/events.jsonl')]; print(f'Total events: {len(events)}')"

# Check resource usage
python -c "from data_juicer.core.executor.partition_size_optimizer import ResourceDetector; print(ResourceDetector.detect_local_resources())"
```

## API Reference

### ProcessingSnapshotAnalyzer

```python
from data_juicer.utils.job.snapshot import ProcessingSnapshotAnalyzer

analyzer = ProcessingSnapshotAnalyzer(job_dir)
snapshot = analyzer.generate_snapshot()
json_data = analyzer.to_json_dict(snapshot)
```

### ResourceDetector

```python
from data_juicer.core.executor.partition_size_optimizer import ResourceDetector

# Detect local resources
local_resources = ResourceDetector.detect_local_resources()

# Detect Ray cluster
cluster_resources = ResourceDetector.detect_ray_cluster()

# Calculate optimal worker count
optimal_workers = ResourceDetector.calculate_optimal_worker_count()
```

### PartitionSizeOptimizer

```python
from data_juicer.core.executor.partition_size_optimizer import PartitionSizeOptimizer

optimizer = PartitionSizeOptimizer()
recommendations = optimizer.get_partition_recommendations(dataset, modality)
```

This comprehensive job management system provides the tools you need to monitor, optimize, and troubleshoot Data-Juicer processing jobs effectively.

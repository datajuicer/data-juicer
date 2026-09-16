# Job Management

DataJuicer provides utilities for monitoring and managing processing jobs.

## Processing Snapshot

Analyze job status from event logs and DAG structure.

```bash
# JSON output
python -m data_juicer.utils.job.snapshot /path/to/job_dir

# Human-readable output
python -m data_juicer.utils.job.snapshot /path/to/job_dir --human-readable
```

Output includes:
- Job status and progress percentage
- Partition completion counts
- Operation metrics
- Checkpoint coverage
- Timing information

## Resource-Aware Partitioning

The system automatically optimizes partition sizes based on cluster resources and data characteristics.

```yaml
partition:
  mode: "auto"
  target_size_mb: 256  # Target partition size (configurable)
```

The optimizer:
1. Detects CPU, memory, and GPU resources
2. Samples data to determine modality and memory usage
3. Calculates partition size targeting the configured size (default 256MB)
4. Determines optimal worker count

## Logging

Logs are organized per job with rotation and retention:

```
{job_dir}/
├── events_{timestamp}.jsonl   # Machine-readable events
├── logs/
│   ├── log.txt                # Main log
│   ├── log_DEBUG.txt          # Debug logs
│   ├── log_ERROR.txt          # Error logs
│   └── log_WARNING.txt        # Warning logs
└── job_summary.json           # Summary (on completion)
```

Configure logging:
```python
from data_juicer.utils.logger_utils import setup_logger

setup_logger(
    save_dir="./outputs",
    filename="log.txt",
    max_log_size_mb=100,
    backup_count=5
)
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

local = ResourceDetector.detect_local_resources()
cluster = ResourceDetector.detect_ray_cluster()
workers = ResourceDetector.calculate_optimal_worker_count()
```

### PartitionSizeOptimizer

```python
from data_juicer.core.executor.partition_size_optimizer import PartitionSizeOptimizer

optimizer = PartitionSizeOptimizer(cfg)
recommendations = optimizer.get_partition_recommendations(dataset, pipeline)
```

## Troubleshooting

Check job status:
```bash
python -m data_juicer.utils.job.snapshot /path/to/job
```

Analyze events:
```bash
cat /path/to/job/events_*.jsonl | head -20
```

Check resources:
```python
from data_juicer.core.executor.partition_size_optimizer import ResourceDetector
print(ResourceDetector.detect_local_resources())
```

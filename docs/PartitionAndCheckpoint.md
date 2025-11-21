# DataJuicer Fault-Tolerant Processing with Checkpointing and Event Logging

This directory contains the implementation of fault-tolerant, resumable DataJuicer processing with comprehensive checkpointing, partitioning, and event logging capabilities.

## 🚀 Features Implemented

### ✅ Core Features
- **Job-Specific Directory Isolation**: Each job gets its own dedicated directory structure
- **Configurable Checkpointing Strategies**: Multiple checkpointing frequencies and strategies
- **Spark-Style Event Logging**: Comprehensive event tracking in JSONL format for resumability
- **Job Resumption Capabilities**: Resume failed or interrupted jobs from the last checkpoint
- **Comprehensive Job Management**: Job summaries, metadata tracking, and resumption commands

### ✅ Checkpointing Strategies
- `EVERY_OP`: Checkpoint after every operation (most resilient, slower)
- `EVERY_N_OPS`: Checkpoint after every N operations (configurable)
- `MANUAL`: Checkpoint only after specified operations
- `DISABLED`: Disable checkpointing entirely

### ✅ Event Logging
- **Human-readable logs**: Loguru-based logging for debugging and monitoring
- **Machine-readable logs**: JSONL format for programmatic analysis and resumption
- **Comprehensive event types**: Job start/complete/failed, partition events, operation events, checkpoint events
- **Real-time monitoring**: Live event streaming and status reporting

### ✅ Job Management
- **Meaningful Job IDs**: Format: `{YYYYMMDD}_{HHMMSS}_{config_name}_{unique_suffix}`
- **Job Summary Files**: Comprehensive metadata for each job run
- **Resumption Commands**: Automatic generation of exact commands to resume jobs
- **Job Validation**: Validation of job resumption parameters and existing state

## 📁 Directory Structure

```
{work_dir}/
├── {job_id}/                    # Job-specific directory
│   ├── job_summary.json         # Job metadata and resumption info
│   ├── events.jsonl             # Machine-readable events (JSONL format)
│   ├── dag_execution_plan.json  # DAG execution plan
│   ├── partition-checkpoint-eventlog.yaml  # Backed up config file
│   ├── metadata/                # Job metadata files
│   │   ├── dataset_mapping.json
│   │   └── final_mapping_report.json
│   ├── logs/                    # Human-readable logs
│   │   ├── export_processed.jsonl_time_*.txt           # Main log file
│   │   ├── export_processed.jsonl_time_*_DEBUG.txt     # Debug level logs
│   │   ├── export_processed.jsonl_time_*_WARNING.txt   # Warning level logs
│   │   └── export_processed.jsonl_time_*_ERROR.txt     # Error level logs
│   ├── checkpoints/             # Checkpoint data
│   │   ├── checkpoint_*.json    # Checkpoint metadata
│   │   └── partition_*/         # Partition checkpoint data
│   ├── partitions/              # Input data partitions
│   ├── processed.jsonl/         # Intermediate processing results
│   └── results/                 # Final processing results
```

## 🛠️ Configuration

### Configuration Structure

The configuration uses a **logical nested structure** that groups related settings by concern:

#### New Logical Structure (Recommended)
```yaml
# Partitioning configuration
partition:
  size: 1000  # Number of samples per partition
  max_size_mb: 64  # Maximum partition size in MB



# Intermediate storage configuration for partition and checkpoint data (format, compression, and lifecycle management)
intermediate_storage:
  # File format and compression
  format: "parquet"  # parquet, arrow, jsonl
  compression: "snappy"  # snappy, gzip, none
  use_arrow_batches: true
  arrow_batch_size: 500
  arrow_memory_mapping: false
  
  # File lifecycle management
  preserve_intermediate_data: true  # Keep temporary files for debugging/resumption
  cleanup_temp_files: true
  cleanup_on_success: false
  retention_policy: "keep_all"  # keep_all, keep_failed_only, cleanup_all
  max_retention_days: 7
```

#### Legacy Flat Structure (Still Supported)
```yaml
# Legacy flat configuration (still works)
partition_size: 1000
max_partition_size_mb: 64
preserve_intermediate_data: true
storage_format: "parquet"
use_arrow_batches: true
arrow_batch_size: 500
arrow_memory_mapping: false
```

**Note**: The system reads from the new nested sections first, then falls back to the legacy flat configuration if not found.

### Configuration Sections Explained

#### `partition` - Partitioning and Resilience
Controls how the dataset is split and how failures are handled:

**Two Partition Modes:**

1. **Auto Mode** (Recommended - `mode: "auto"`):
   - Automatically analyzes your data characteristics and system resources
   - Calculates optimal partition size targeting ~64MB per partition
   - Determines optimal number of partitions based on dataset size
   - Configures optimal worker count based on available CPU cores
   - No manual tuning required - adapts to your hardware and data
   - Configuration:
     - `mode`: `"auto"`
     - `size`: Fallback partition size (samples) - used if auto-analysis fails
     - `max_size_mb`: Fallback max partition size (MB) - used if auto-analysis fails

2. **Manual Mode** (`mode: "manual"`):
   - You specify the exact number of partitions to create
   - Useful when you know your optimal partitioning strategy
   - Configuration:
     - `mode`: `"manual"`
     - `num_of_partitions`: Exact number of partitions to create
     - `size` and `max_size_mb` are ignored in manual mode


#### `intermediate_storage` - Intermediate Data Management
Controls file formats, compression, and lifecycle management for intermediate data:
- **File Format & Compression**:
  - `format`: Storage format (`parquet`, `arrow`, `jsonl`)
  - `compression`: Compression algorithm (`snappy`, `gzip`, `none`)
  - `use_arrow_batches`: Use Arrow batch processing
  - `arrow_batch_size`: Arrow batch size
  - `arrow_memory_mapping`: Enable memory mapping
- **File Lifecycle Management**:
  - `preserve_intermediate_data`: Keep temporary files for debugging
  - `cleanup_temp_files`: Enable automatic cleanup
  - `cleanup_on_success`: Clean up even on successful completion
  - `retention_policy`: File retention strategy (`keep_all`, `keep_failed_only`, `cleanup_all`)
  - `max_retention_days`: Auto-cleanup after X days

### Basic Configuration
```yaml
# Enable fault-tolerant processing
executor_type: ray_partitioned

# Job management
job_id: my_experiment_001  # Optional: auto-generated if not provided

# Checkpointing configuration
checkpoint:
  enabled: true
  strategy: every_op  # every_op, every_n_ops, manual, disabled
  n_ops: 2            # For every_n_ops strategy
  op_names:           # For manual strategy
    - clean_links_mapper
    - whitespace_normalization_mapper

# Event logging configuration
event_logging:
  enabled: true
  max_log_size_mb: 100
  backup_count: 5

# Partitioning configuration
partition:
  mode: "auto"          # Auto mode - optimal partitioning based on data analysis
  size: 5000            # Fallback partition size (samples) - used if auto-analysis fails
  max_size_mb: 64       # Fallback max partition size (MB) - used if auto-analysis fails
  # Note: num_of_partitions is calculated automatically in auto mode

# Alternative: Manual partition mode
# partition:
#   mode: "manual"        # Manual mode - specify exact number of partitions
#   num_of_partitions: 8  # Split dataset into exactly 8 partitions
#   # Note: size and max_size_mb are ignored in manual mode
  


# Intermediate storage configuration for partition and checkpoint data (format, compression, and lifecycle management)
intermediate_storage:
  # File format and compression
  format: "parquet"  # parquet, arrow, jsonl
  compression: "snappy"  # snappy, gzip, none
  use_arrow_batches: true
  arrow_batch_size: 500
  arrow_memory_mapping: false
  
  # File lifecycle management
  preserve_intermediate_data: true  # Keep temporary files for debugging/resumption
  cleanup_temp_files: true
  cleanup_on_success: false
  retention_policy: "keep_all"  # keep_all, keep_failed_only, cleanup_all
  max_retention_days: 7
```

## 📊 Partition Modes Explained

### Auto Mode (Recommended)
**When to use:** Most use cases, especially when you want optimal performance without manual tuning.

**Benefits:**
- ✅ Automatically adapts to your data characteristics (text length, modality, etc.)
- ✅ Optimizes for your system resources (CPU, memory, GPU)
- ✅ Targets ~64MB per partition for optimal memory usage
- ✅ Calculates optimal number of partitions based on dataset size
- ✅ No manual tuning required

**Example output:**
```
🔧 Auto-configuring partition settings based on data characteristics...
📊 Dataset analysis complete:
  Total samples: 10000
  Recommended partition size: 5000 samples
  Calculated partitions: 2
  Recommended max size: 64 MB
  Recommended workers: 4
```

### Manual Mode
**When to use:** When you have specific requirements or know your optimal partitioning strategy.

**Benefits:**
- ✅ Full control over partition count
- ✅ Predictable resource usage
- ✅ Useful for debugging or specific workflows
- ✅ Can be more efficient for known dataset patterns

**Example:**
```yaml
partition:
  mode: "manual"
  num_of_partitions: 8  # Always creates exactly 8 partitions
```

## 🚀 Quick Start

### 1. Basic Usage

#### Auto Partition Mode (Recommended)
```bash
# Run with auto-generated job ID and auto partition optimization
dj-process --config configs/demo/partition-checkpoint-eventlog.yaml --partition.mode auto

# Run with custom job ID
dj-process --config configs/demo/partition-checkpoint-eventlog.yaml --partition.mode auto --job_id my_experiment_001
```

#### Manual Partition Mode
```bash
# Run with manual partition configuration (4 partitions)
dj-process --config configs/demo/partition-checkpoint-eventlog.yaml --partition.mode manual --partition.num_of_partitions 4

# Run with custom job ID
dj-process --config configs/demo/partition-checkpoint-eventlog.yaml --partition.mode manual --partition.num_of_partitions 4 --job_id my_experiment_001
```

### 2. Resume a Job
```bash
# Resume using the job ID
dj-process --config configs/demo/partition-checkpoint-eventlog.yaml --job_id my_experiment_001
```

### 3. Different Checkpoint Strategies
```bash
# Checkpoint every operation (most resilient)
dj-process --config configs/demo/partition-checkpoint-eventlog.yaml --job_id every_op_test --checkpoint.strategy every_op

# Checkpoint every 3 operations
dj-process --config configs/demo/partition-checkpoint-eventlog.yaml --job_id n_ops_test --checkpoint.strategy every_n_ops --checkpoint.n_ops 3

# Manual checkpointing
dj-process --config configs/demo/partition-checkpoint-eventlog.yaml --job_id manual_test --checkpoint.strategy manual --checkpoint.op_names clean_links_mapper,whitespace_normalization_mapper
```

### 4. Run Comprehensive Demo
```bash
# Run the full demo showcasing all features
python demos/partition_and_checkpoint/run_demo.py
```

## 📊 Monitoring and Debugging

### View Job Information
```bash
# Check job summary
cat ./outputs/partition-checkpoint-eventlog/{job_id}/job_summary.json

# View event logs
cat ./outputs/partition-checkpoint-eventlog/{job_id}/events.jsonl

# View human-readable logs
cat ./outputs/partition-checkpoint-eventlog/{job_id}/logs/export_processed.jsonl_time_*.txt

# View DAG execution plan
cat ./outputs/partition-checkpoint-eventlog/{job_id}/dag_execution_plan.json
```

### List Available Jobs
```bash
# List all job directories
ls -la ./outputs/partition-checkpoint-eventlog/
```

### Check Job Structure
```bash
# Check job directory structure
ls -la ./outputs/partition-checkpoint-eventlog/{job_id}/

# Check logs directory
ls -la ./outputs/partition-checkpoint-eventlog/{job_id}/logs/

# Check checkpoints directory
ls -la ./outputs/partition-checkpoint-eventlog/{job_id}/checkpoints/
```

## 📈 Job Management Utilities

DataJuicer provides comprehensive job management utilities for monitoring progress and stopping running jobs. These utilities are located in `data_juicer/utils/job/` and provide both command-line and programmatic interfaces.

### 📊 Job Progress Monitor

A comprehensive utility to monitor and display progress information for DataJuicer jobs. Shows partition status, operation progress, checkpoints, and overall job metrics.

#### Features

- **Real-time Progress Tracking**: Monitor job progress with partition-level details
- **Operation Performance**: View detailed operation metrics including throughput and data reduction
- **Checkpoint Monitoring**: Track checkpoint saves and recovery points
- **Watch Mode**: Continuously monitor jobs with automatic updates
- **Programmatic Access**: Use as a Python function for integration into other tools

#### Command Line Usage

##### Basic Usage
```bash
# Show basic progress for a job
python -m data_juicer.utils.job.monitor 20250728_233517_510abf

# Show detailed progress with operation metrics
python -m data_juicer.utils.job.monitor 20250728_233517_510abf --detailed

# Watch mode - continuously update progress every 10 seconds
python -m data_juicer.utils.job.monitor 20250728_233517_510abf --watch

# Watch mode with custom update interval (30 seconds)
python -m data_juicer.utils.job.monitor 20250728_233517_510abf --watch --interval 30

# Use custom base directory
python -m data_juicer.utils.job.monitor 20250728_233517_510abf --base-dir /custom/path
```

##### Command Line Options
- `job_id`: The job ID to monitor (required)
- `--base-dir`: Base directory containing job outputs (default: `outputs/partition-checkpoint-eventlog`)
- `--detailed`: Show detailed operation information
- `--watch`: Watch mode - continuously update progress
- `--interval`: Update interval in seconds for watch mode (default: 10)

#### Python API

##### Basic Function Usage
```python
from data_juicer.utils.job.monitor import show_job_progress

# Show progress and get data
data = show_job_progress("20250728_233517_510abf")

# Show detailed progress
data = show_job_progress("20250728_233517_510abf", detailed=True)

# Use custom base directory
data = show_job_progress("20250728_233517_510abf", base_dir="/custom/path")
```

##### Class-based Usage
```python
from data_juicer.utils.job.monitor import JobProgressMonitor

# Create monitor instance
monitor = JobProgressMonitor("20250728_233517_510abf")

# Display progress
monitor.display_progress(detailed=True)

# Get progress data as dictionary
data = monitor.get_progress_data()

# Access specific information
job_status = data['overall_progress']['job_status']
progress_percentage = data['overall_progress']['progress_percentage']
partition_status = data['partition_status']
```

### 🛑 Job Stopper

A utility to stop running DataJuicer jobs by reading event logs to find process and thread IDs, then terminating those specific processes and threads.

#### Features

- **Precise Process Termination**: Uses event logs to identify exact processes and threads to terminate
- **Graceful Shutdown**: Sends SIGTERM first for graceful shutdown, then SIGKILL if needed
- **Safety Checks**: Validates job existence and running status before stopping
- **Comprehensive Logging**: Detailed logging of termination process
- **Programmatic Access**: Can be used as a Python function or command-line tool

#### Command Line Usage

##### Basic Usage
```bash
# Stop a job gracefully (SIGTERM)
python -m data_juicer.utils.job.stopper 20250728_233517_510abf

# Force stop a job (SIGKILL)
python -m data_juicer.utils.job.stopper 20250728_233517_510abf --force

# Stop with custom timeout (60 seconds)
python -m data_juicer.utils.job.stopper 20250728_233517_510abf --timeout 60

# Use custom base directory
python -m data_juicer.utils.job.stopper 20250728_233517_510abf --base-dir /custom/path

# List all running jobs
python -m data_juicer.utils.job.stopper --list
```

##### Command Line Options
- `job_id`: The job ID to stop (required, unless using --list)
- `--base-dir`: Base directory containing job outputs (default: `outputs/partition-checkpoint-eventlog`)
- `--force`: Force kill with SIGKILL instead of graceful SIGTERM
- `--timeout`: Timeout in seconds for graceful shutdown (default: 30)
- `--list`: List all running jobs instead of stopping one

#### Python API

##### Basic Function Usage
```python
from data_juicer.utils.job.stopper import stop_job

# Stop a job gracefully
result = stop_job("20250728_233517_510abf")

# Force stop a job
result = stop_job("20250728_233517_510abf", force=True)

# Stop with custom timeout
result = stop_job("20250728_233517_510abf", timeout=60)

# Use custom base directory
result = stop_job("20250728_233517_510abf", base_dir="/custom/path")
```

##### Class-based Usage
```python
from data_juicer.utils.job.stopper import JobStopper

# Create stopper instance
stopper = JobStopper("20250728_233517_510abf")

# Stop the job
result = stopper.stop_job(force=False, timeout=30)

# Check if job is running
is_running = stopper.is_job_running()

# Get job summary
summary = stopper.get_job_summary()
```

### 🔧 Common Utilities

Both the monitor and stopper utilities share common functionality through `data_juicer.utils.job.common`:

```python
from data_juicer.utils.job.common import JobUtils, list_running_jobs

# List all running jobs
running_jobs = list_running_jobs()

# Create job utilities instance
job_utils = JobUtils("20250728_233517_510abf")

# Load job summary
summary = job_utils.load_job_summary()

# Load event logs
events = job_utils.load_event_logs()

# Get partition status
partition_status = job_utils.get_partition_status()
```

### Output Information

#### Job Overview
- Job status (completed, processing, failed, etc.)
- Dataset path and size
- Partition configuration
- Start time and duration

#### Overall Progress
- Progress percentage
- Partition completion status
- Sample processing counts
- Estimated time remaining (for running jobs)

#### Partition Status
- Individual partition status with visual indicators
- Sample counts per partition
- Current operation (if processing)
- Number of completed operations
- Number of saved checkpoints

#### Operation Details (with --detailed flag)
- Per-partition operation performance
- Duration, throughput, and data reduction metrics
- Operation completion order

#### Checkpoint Summary
- Total number of checkpoints saved
- Checkpoint details by partition and operation
- Timestamp information

### Example Output

```
================================================================================
DataJuicer Job Progress Monitor
Job ID: 20250728_233517_510abf
================================================================================

📊 JOB OVERVIEW
   Status: COMPLETED
   Dataset: /Users/yilei.z/Downloads/c4-train.00000-of-01024.jsonl
   Total Samples: 356,317
   Partition Size: 50,000 samples
   Start Time: 2025-07-28 16:35:18
   Duration: 441.1 seconds

🎯 OVERALL PROGRESS
   Progress: 100.0% (8/8 partitions)
   Status: 8 completed, 0 processing, 0 failed
   Samples: 356,317/356,317

📦 PARTITION STATUS
   Partition  0: ✅ COMPLETED
              Samples: 44,539
              Completed: 8 operations
              Checkpoints: 2 saved
   Partition  1: ✅ COMPLETED
              Samples: 44,540
              Completed: 8 operations
              Checkpoints: 2 saved
   ...

💾 CHECKPOINT SUMMARY
   Total Checkpoints: 16
```

### Integration Examples

#### Monitoring Multiple Jobs
```python
from data_juicer.utils.job.monitor import show_job_progress

job_ids = ["job1", "job2", "job3"]
for job_id in job_ids:
    try:
        data = show_job_progress(job_id)
        print(f"Job {job_id}: {data['overall_progress']['progress_percentage']:.1f}%")
    except FileNotFoundError:
        print(f"Job {job_id}: Not found")
```

#### Custom Monitoring Script
```python
from data_juicer.utils.job.monitor import JobProgressMonitor
import time

def monitor_job_until_completion(job_id, check_interval=30):
    monitor = JobProgressMonitor(job_id)
    
    while True:
        data = monitor.get_progress_data()
        status = data['overall_progress']['job_status']
        
        if status == 'completed':
            print(f"Job {job_id} completed!")
            break
        elif status == 'failed':
            print(f"Job {job_id} failed!")
            break
        
        print(f"Job {job_id} still running... {data['overall_progress']['progress_percentage']:.1f}%")
        time.sleep(check_interval)
```

#### Job Management Workflow
```python
from data_juicer.utils.job.monitor import show_job_progress
from data_juicer.utils.job.stopper import stop_job
from data_juicer.utils.job.common import list_running_jobs

# List all running jobs
running_jobs = list_running_jobs()
print(f"Found {len(running_jobs)} running jobs")

# Monitor and potentially stop jobs
for job_info in running_jobs:
    job_id = job_info['job_id']
    
    # Check progress
    try:
        data = show_job_progress(job_id)
        progress = data['overall_progress']['progress_percentage']
        
        # Stop jobs that are stuck (less than 10% progress after 1 hour)
        if progress < 10 and data['overall_progress']['elapsed_time_seconds'] > 3600:
            print(f"Stopping stuck job {job_id} (progress: {progress:.1f}%)")
            stop_job(job_id, force=True)
        else:
            print(f"Job {job_id}: {progress:.1f}% complete")
            
    except Exception as e:
        print(f"Error monitoring job {job_id}: {e}")
```

## 🤖 Auto-Configuration System

### **Smart Partition Sizing by Modality**

DataJuicer now includes an intelligent auto-configuration system that automatically determines optimal partition sizes based on your data characteristics:

#### **How It Works**

1. **Modality Detection**: Analyzes your dataset to detect the primary modality (text, image, audio, video, multimodal)
2. **Dataset Analysis**: Examines sample characteristics (text length, media counts, file sizes)
3. **Pipeline Complexity**: Considers the complexity of your processing operations
4. **Resource Optimization**: Adjusts partition sizes for optimal memory usage and fault tolerance

#### **Modality-Specific Optimizations**

| Modality | Default Size | Max Size | Memory Multiplier | Use Case |
|----------|--------------|----------|-------------------|----------|
| **Text** | 5000 samples | 20000 | 1.0x | Efficient processing, low memory, target 64MB partitions |
| **Image** | 1000 samples | 5000 | 5.0x | Moderate memory, image processing, target 64MB partitions |
| **Audio** | 500 samples | 2000 | 8.0x | High memory, audio processing, target 64MB partitions |
| **Video** | 200 samples | 1000 | 20.0x | Very high memory, complex processing, target 64MB partitions |
| **Multimodal** | 800 samples | 3000 | 10.0x | Multiple modalities, moderate complexity, target 64MB partitions |

#### **Enable Auto-Configuration**

```yaml
partition:
  mode: "auto"  # Enable automatic optimization
  # Fallback values used if auto-analysis fails
  size: 5000
  max_size_mb: 64
```

#### **Manual Override**

```yaml
partition:
  mode: "manual"  # Use manual partition configuration
  num_of_partitions: 8  # Specify exact number of partitions
  # size and max_size_mb are ignored in manual mode
```

## 📊 Partition Sizing Guidelines

### **Why Smaller Partitions Are Better**

**Fault Tolerance**: Smaller partitions mean smaller units of failure. If a partition fails, you lose less work.

**Recovery Speed**: Failed partitions can be retried faster, reducing overall job time.

**Progress Visibility**: More granular progress tracking and faster feedback.

**Memory Efficiency**: Lower memory usage per partition, better for resource-constrained environments.

**Debugging**: Easier to isolate and debug issues in smaller chunks.

### **Partition Size Recommendations**

| Use Case | Partition Size | When to Use |
|----------|---------------|-------------|
| **Debugging** | 50-100 samples | Quick iterations, testing, small datasets |
| **Production** ⭐ | 100-300 samples | Most use cases, good balance |
| **Large Datasets** | 300-500 samples | Stable processing, large datasets |
| **Very Large** | 500+ samples | Only when failure risk is minimal |

### **Factors to Consider**

- **Dataset Size**: Larger datasets can use larger partitions
- **Processing Complexity**: Complex operations benefit from smaller partitions
- **Failure Rate**: Higher failure rates need smaller partitions
- **Memory Constraints**: Limited memory requires smaller partitions
- **Time Sensitivity**: Faster feedback needs smaller partitions

## 🔧 Implementation Details

### Core Components

1. **`EventLoggingMixin`** (`data_juicer/core/executor/event_logging_mixin.py`)
   - Provides event logging capabilities to executors
   - Manages job-specific directories and flexible storage
   - Handles job summary creation and validation
   - Implements Spark-style event logging schema

2. **`PartitionedRayExecutor`** (`data_juicer/core/executor/ray_executor_partitioned.py`)
   - Extends Ray executor with partitioning and fault tolerance
   - Implements configurable checkpointing strategies
   - Integrates with EventLoggingMixin for comprehensive logging
   - Handles job resumption from checkpoints

3. **Configuration Integration** (`data_juicer/config/config.py`)
   - Added command-line arguments for job management
   - Added checkpointing configuration options
   - Added flexible storage path configuration

### Event Types
- `JOB_START`, `JOB_COMPLETE`, `JOB_FAILED`
- `PARTITION_START`, `PARTITION_COMPLETE`, `PARTITION_FAILED`
- `OP_START`, `OP_COMPLETE`, `OP_FAILED`
- `CHECKPOINT_SAVE`, `CHECKPOINT_LOAD`
- `PROCESSING_START`, `PROCESSING_COMPLETE`, `PROCESSING_ERROR`
- `RESOURCE_USAGE`, `PERFORMANCE_METRIC`
- `WARNING`, `INFO`, `DEBUG`

## 🎯 Use Cases

### 1. Large Dataset Processing
- Process datasets that are too large for memory
- Automatic partitioning with fault tolerance
- Resume processing after failures

### 2. Experimental Workflows
- Track different experiments with meaningful job IDs
- Compare results across different configurations
- Maintain experiment history and reproducibility

### 3. Production Pipelines
- Robust error handling and recovery
- Comprehensive monitoring and logging
- Flexible storage for different performance requirements

### 4. Research and Development
- Iterative development with checkpoint resumption
- Detailed event logging for analysis
- Configurable checkpointing for different scenarios

## 🔍 Troubleshooting

### Common Issues

1. **Job resumption fails**
   - Check if job summary exists: `ls -la ./outputs/{work_dir}/{job_id}/job_summary.json`
   - Verify checkpoint files exist: `ls -la /tmp/large_checkpoints/{job_id}/`

2. **Event logs not found**
   - Check flexible storage paths: `ls -la /tmp/fast_event_logs/{job_id}/`
   - Verify event logging is enabled in config

3. **Checkpointing not working**
   - Verify checkpoint strategy in config
   - Check if checkpoint directory is writable
   - Ensure checkpoint.enabled is true

4. **Performance issues**
   - Adjust partition size based on available memory
   - Consider different checkpoint strategies
   - Use appropriate storage formats (parquet for large datasets)

### Debug Commands
```bash
# Check Ray cluster status
ray status

# View Ray dashboard
open http://localhost:8265

# Check DataJuicer logs
tail -f /tmp/fast_event_logs/{job_id}/event_logs/events.log
```

## 📊 Understanding Intermediate Data

### What is Intermediate Data?

Intermediate data refers to temporary results generated during the processing pipeline that exist between operations and before the final output. In DataJuicer's partitioned processing, this includes:

1. **Partition-level intermediate data**: Results after each operation within a partition
2. **Operation-level intermediate data**: Data that exists between operations (e.g., after `clean_links_mapper` but before `whitespace_normalization_mapper`)
3. **Checkpoint intermediate data**: Temporary files created during checkpointing

### When to Preserve Intermediate Data

**Enable `preserve_intermediate_data: true` when you need:**
- **Debugging**: Inspect what the data looks like after each operation
- **Resumption**: If a job fails, see exactly where it failed and what the data looked like
- **Analysis**: Understand how each operation transforms the data
- **Development**: Iterate on processing pipelines with detailed inspection

**Disable `preserve_intermediate_data: false` when you want:**
- **Performance**: Faster processing with less disk I/O
- **Storage efficiency**: Reduced disk space usage
- **Production**: Clean processing without temporary file accumulation

### Example Directory Structure with Intermediate Data

```
{job_dir}/intermediate/
├── partition_000000/
│   ├── op_000_clean_links_mapper.parquet      # After clean_links_mapper
│   ├── op_001_clean_email_mapper.parquet      # After clean_email_mapper
│   ├── op_002_whitespace_normalization_mapper.parquet
│   └── op_003_fix_unicode_mapper.parquet      # After fix_unicode_mapper
└── partition_000001/
    ├── op_000_clean_links_mapper.parquet
    └── ...
```

## 📈 Performance Considerations

### Checkpointing Overhead
- `EVERY_OP`: Highest overhead, maximum resilience
- `EVERY_N_OPS`: Configurable overhead (balance between resilience and performance)
- `MANUAL`: Minimal overhead, requires careful planning
- `DISABLED`: No overhead, no resilience

### Storage Recommendations
- **Event logs**: Use fast storage (SSD) for real-time monitoring
- **Checkpoints**: Use large capacity storage (HDD/network storage) for cost efficiency
- **Partitions**: Use local storage for processing speed

### Memory Management
- Adjust `partition_size` based on available memory
- Use `max_partition_size_mb` to limit partition size
- Consider `preserve_intermediate_data` for debugging vs. performance

## 🎉 Success Metrics

The implementation successfully demonstrates:
- ✅ **Fault Tolerance**: Jobs can resume after failures
- ✅ **Scalability**: Handles large datasets through partitioning
- ✅ **Observability**: Comprehensive logging and monitoring
- ✅ **Flexibility**: Configurable checkpointing strategies
- ✅ **Usability**: Simple command-line interface with meaningful job IDs
- ✅ **Performance**: Fast resumption from checkpoints
- ✅ **Reliability**: Robust error handling and validation

## 🔮 Future Enhancements

Potential areas for future development:
- **Distributed checkpointing**: Multi-node checkpoint coordination
- **Incremental checkpointing**: Only save changed data
- **Checkpoint compression**: Reduce storage requirements
- **Advanced monitoring**: Web-based dashboard for job monitoring
- **Checkpoint versioning**: Support for multiple checkpoint versions
- **Integration with external systems**: Cloud storage, monitoring systems 
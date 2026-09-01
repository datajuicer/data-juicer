# Partitioned Processing with Checkpointing

This document describes DataJuicer's fault-tolerant processing system with partitioning, checkpointing, and event logging.

## Overview

The `ray_partitioned` executor splits datasets into partitions and processes them with configurable checkpointing. Failed jobs can resume from the last checkpoint.

**Checkpointing strategies:**
- `every_n_ops` - checkpoint every N operations (default, balanced)
- `every_op` - checkpoint after every operation (max protection, impacts performance)
- `manual` - checkpoint only after specified operations (best for known expensive ops)
- `disabled` - no checkpointing (best performance)

## Directory Structure

```
{work_dir}/{job_id}/
├── job_summary.json              # Job metadata (created on completion)
├── gpu_probe_results.json         # Auto-probed GPU operator resources
├── events_{timestamp}.jsonl      # Machine-readable event log
├── dag_execution_plan.json       # DAG execution plan
├── checkpoints/                  # Checkpoint data
│   ├── partitioning_info.json    # Saved row boundaries and partition hashes
│   └── checkpoint_op_*.parquet/  # Per-operation partition checkpoints
├── partitions/                   # Input partitions
├── logs/                         # Human-readable logs
└── metadata/                     # Job metadata
```

## Configuration

### Partition Modes

**Auto mode** (recommended) - analyzes data and resources to determine optimal partitioning:

```yaml
executor_type: ray_partitioned

partition:
  mode: "auto"
  max_concurrent_partitions: "auto"  # Resource-aware driver concurrency
  max_gpu_workers_per_device: 5       # Conservative model-replica cap per GPU
  max_concurrent_gpu_probes: "auto"   # Fill available GPU/CPU slots; set an integer to cap parallel probes
  gpu_preflight_enabled: true          # false skips preflight and uses explicit resources/actor counts
  gpu_probe_timeout_seconds: null     # Optional per-probe timeout; null disables termination
  gpu_probe_warmup_batches: 1         # Warmup batches before steady-state timing
  gpu_probe_steady_batches: 3         # Batches used for steady-state throughput
  execution_group_size: "auto"       # Logical partitions sharing one GPU actor lifecycle
  max_initialization_overhead_ratio: 0.1  # Allowed model-init share per execution group
  target_size_mb: 256    # Target partition size (128, 256, 512, or 1024)
  size: 5000             # Fallback if auto-analysis fails
  max_size_mb: 256       # Fallback max size
```

**Manual mode** - specify exact partition count:

```yaml
partition:
  mode: "manual"
  num_of_partitions: 8
  max_concurrent_partitions: "auto"
```

Logical partitions and GPU execution concurrency are independent. Logical
partitions define checkpoint/recovery granularity and the per-partition data
bound. Preflight throughput and cluster resources determine GPU actor counts.
An execution group sends several logical partitions through one Ray actor-pool
lifecycle while still writing a separate checkpoint for every logical
partition. `execution_group_size: "auto"` uses measured initialization time,
throughput, data volume, and `max_initialization_overhead_ratio`; a positive
integer overrides it. Resume safely falls back to serial per-partition
execution because partitions may be at different checkpoint boundaries.

The automatic actor planner starts with one actor per profiled GPU stage and
adds actors to the current pipeline bottleneck. Every addition must fit the
cluster CPU budget, Ray GPU scheduling fractions, measured per-device GPU
memory, the per-device actor cap, and useful-batch limits. If the cluster
cannot host the one-actor-per-stage minimum, execution fails before the formal
job instead of increasing partitions or oversubscribing GPU memory.

#### GPU memory preflight

For a fixed-resource control run, set `gpu_preflight_enabled: false`. The
executor will not sample input rows, create disposable probe actors, or write
`gpu_probe_results.json`. Configure CUDA `num_gpus`, `memory`, and fixed
`num_proc` values explicitly, and normally disable `auto_op_parallelism` too.

In `ray_partitioned` mode (including manual partitioning), ordinary single-GPU
CUDA Mapper/Filter operators are profiled before the formal experiment. An op
without `memory`/`num_gpus` also receives resource estimates. Explicit values
still win, but throughput is measured for automatic `num_proc` planning:

1. The executor takes a fixed prefix of the input. Its size is the largest
   `batch_size` among operators that still need probing.
2. Operators may declare `input_columns` and `output_columns`, including
   nested paths such as `__dj__meta__.quality_score`. The executor builds a
   conservative data-dependency graph from these contracts.
3. Targets proven independent, with only compatible CPU Mapper/Filter
   ancestors, can run concurrently. Each disposable Ray worker receives the
   lightweight source rows, replays its required CPU ancestors locally, and
   reserves one full GPU for the target. Large intermediate NumPy values are
   therefore not round-tripped through the driver. The default
   `max_concurrent_gpu_probes: "auto"` fills the dependency-safe GPU and CPU
   slots. Set a positive integer to cap concurrent model loading when checkpoint
   storage or host-memory bandwidth is constrained.
   Workers log dependency replay and measured-target timing. The driver logs
   each completion immediately and emits a progress heartbeat every 30 seconds.
   `gpu_probe_timeout_seconds` can fail a stalled target with its operator name;
   the default `null` keeps timeout termination disabled.
4. Missing contracts, GPU-to-GPU dependencies, runtime-environment conflicts,
   and dataset-level operations conservatively fall back to the original
   ordered recipe replay. If an earlier filter leaves too few rows, surviving
   rows are cycled to fill one target batch.
5. The disposable worker constructs the target only once and separately
   records model initialization, warmup, and multiple steady-state batches.
   It reports steady input throughput and output ratio. Defaults are one
   warmup batch and three measured batches.
6. A probe does not poll CUDA from a background thread during model
   initialization, avoiding `cudaMemGetInfo` contention with large
   `model.to(cuda)` transfers on the same CUDA context. The disposable worker
   lets the operator initialize CUDA as a formal actor would, then combines
   PyTorch's allocator peak with final persistent device usage (which also
   covers non-PyTorch runtimes such as Paddle). The measured value receives
   10% headroom and becomes `memory_fraction`. Ray's scheduling `num_gpus` is `max(memory_fraction,
   1 / max_gpu_workers_per_device)`; the default allows at most five
   auto-probed model actors per physical GPU.
7. Resource values, phased timing, throughput, output ratio, probe mode, and
   replayed dependencies are saved in `{work_dir}/gpu_probe_results.json`. A resume
   reuses entries when the operator configuration, GPU model/capacity, and
   timing-batch settings, and per-device worker cap match, and re-probes
   otherwise. The source YAML is not overwritten.

For example, independent image taggers sharing a CPU resize can opt into the
parallel path without changing their Python classes:

```yaml
process:
  - bucket_resize_mapper:
      input_columns: [images]
      output_columns: [_bucket_img]
  - image_quality_mapper:
      input_columns: [images, _bucket_img]
      output_columns: [__dj__meta__.quality_score]
  - image_rotation_mapper:
      input_columns: [images, _bucket_img]
      output_columns: [__dj__meta__.rotation_*]
```

`None`/omitted metadata means unknown, not empty. Existing recipes therefore
retain ordered probing until their operators declare a complete contract.

An explicit `memory` or `num_gpus` always wins and is not overwritten by the
measurement. Preflight currently supports ordinary Mapper/Filter operators
that fit on one GPU.
GPU `Pipeline` operators, a `Pipeline` before a pending target, and multi-GPU
operators require explicit resources. Empty input, probe errors, OOM, and an
invalid zero peak fail before formal partition workers start.

### Checkpointing

```yaml
checkpoint:
  enabled: true
  strategy: every_n_ops  # every_n_ops (default), every_op, manual, disabled
  n_ops: 5               # Default: checkpoint every 5 operations
  op_names:              # For manual strategy - checkpoint after expensive ops
    - document_deduplicator
    - embedding_mapper
```

When checkpointing is enabled, the initial run saves
`checkpoints/partitioning_info.json`. For every logical partition, this file
records:

- `start_row` (inclusive) and `end_row` (exclusive) in the ordered input;
- the partition row count;
- a stable hash of the complete partition contents;
- the partition hash algorithm used by the writer.

These values allow an explicit resume to recreate the original logical
partitions even when Ray produces a different physical block layout in the new
process. Complete partition hashes are sensitive to row order, independent of
Ray batch boundaries, and validated before any checkpoint is reused.

### Intermediate Storage

```yaml
intermediate_storage:
  format: "parquet"              # parquet, arrow, jsonl
  compression: "snappy"          # snappy, gzip, none
  preserve_intermediate_data: true
  retention_policy: "keep_all"   # keep_all, keep_failed_only, cleanup_all
```

## Usage

### Running Jobs

```bash
# Auto partition mode
dj-process --config config.yaml --partition.mode auto

# Manual partition mode
dj-process --config config.yaml --partition.mode manual --partition.num_of_partitions 4

# Optional: start a new job with a custom job ID
dj-process --config config.yaml --job_id my_experiment_001
```

When checkpointing is enabled, a new `ray_partitioned` job prints a resume
token:

```text
Resume token: 20260805_115141_81270d. Rerun the original command with
--resume 20260805_115141_81270d to resume this job.
```

### Resuming Jobs

```bash
# Use the token printed by the initial run. Keep the input and recipe unchanged.
dj-process --config config.yaml --resume 20260805_115141_81270d

# A custom ID from the initial run can be used in the same way.
dj-process --config config.yaml --resume my_experiment_001
```

`--resume` is supported only by the `ray_partitioned` executor. It performs a
strict resume in the following order:

1. locate the original work and checkpoint directories;
2. verify that the current configuration matches the original run;
3. load the saved partition count, row boundaries, and content hashes;
4. recreate the original partitions with the saved row boundaries;
5. validate every complete partition hash;
6. load completed checkpoints and process only the unfinished work.

If metadata is missing, row boundaries are invalid, the input has changed, or
a content hash does not match, explicit resume stops with an error and leaves
the existing checkpoints unchanged.

`--job_id` remains available for custom job naming and backward compatibility.
For fault-tolerant continuation, prefer `--resume`: the legacy `--job_id`
resumption path keeps its previous behavior and may clear mismatched
checkpoints before starting fresh. Metadata created by an older Data-Juicer
version can still be read, but it does not contain the row boundaries and full
content hashes required by explicit resume; `--resume` therefore rejects it
without deleting its checkpoints.

If both arguments are supplied, their values must be identical:

```bash
dj-process --config config.yaml \
  --job_id my_experiment_001 \
  --resume my_experiment_001
```

### Checkpoint Strategies

```bash
# Every operation
dj-process --config config.yaml --checkpoint.strategy every_op

# Every N operations
dj-process --config config.yaml --checkpoint.strategy every_n_ops --checkpoint.n_ops 3

# Manual
dj-process --config config.yaml --checkpoint.strategy manual --checkpoint.op_names op1,op2
```

## Auto-Configuration

In auto mode, the optimizer:
1. Samples the dataset to detect modality (text, image, audio, video, multimodal)
2. Measures memory usage per sample
3. Analyzes pipeline complexity
4. Calculates partition size targeting the configured `target_size_mb`

Default partition sizes by modality:

| Modality | Default Size | Max Size | Memory Multiplier |
|----------|--------------|----------|-------------------|
| Text | 10000 | 50000 | 1.0x |
| Image | 2000 | 10000 | 5.0x |
| Audio | 1000 | 4000 | 8.0x |
| Video | 400 | 2000 | 20.0x |
| Multimodal | 1600 | 6000 | 10.0x |

## Job Management Utilities

### Monitor

```bash
# Show progress
python -m data_juicer.utils.job.monitor {job_id}

# Detailed view
python -m data_juicer.utils.job.monitor {job_id} --detailed

# Watch mode
python -m data_juicer.utils.job.monitor {job_id} --watch --interval 10
```

```python
from data_juicer.utils.job.monitor import show_job_progress

data = show_job_progress("job_id", detailed=True)
```

### Stopper

```bash
# Graceful stop
python -m data_juicer.utils.job.stopper {job_id}

# Force stop
python -m data_juicer.utils.job.stopper {job_id} --force

# List running jobs
python -m data_juicer.utils.job.stopper --list
```

```python
from data_juicer.utils.job.stopper import stop_job

stop_job("job_id", force=True, timeout=60)
```

### Common Utilities

```python
from data_juicer.utils.job.common import JobUtils, list_running_jobs

running_jobs = list_running_jobs()

job_utils = JobUtils("job_id")
summary = job_utils.load_job_summary()
events = job_utils.load_event_logs()
```

## Event Types

- `job_start`, `job_complete`, `job_failed`
- `partition_start`, `partition_complete`, `partition_failed`
- `op_start`, `op_complete`, `op_failed`
- `checkpoint_save`, `checkpoint_load`

## Performance Considerations

### Checkpoint vs Ray Optimization Trade-off

**Key insight: Checkpointing interferes with Ray's automatic optimization.**

Ray optimizes execution by fusing operations together and pipelining data. Each checkpoint forces materialization, which breaks the optimization window:

```
Without checkpoints:     op1 → op2 → op3 → op4 → op5
                         |___________________________|
                              Ray optimizes entire window

With every_op:           op1 | op2 | op3 | op4 | op5
                         materialize at each | (5 barriers)

With every_n_ops(5):     op1 → op2 → op3 → op4 → op5 |
                         |_____________________________|
                              Ray optimizes all 5 ops
```

### Checkpoint Cost Analysis

| Cost Type | Typical Value |
|-----------|---------------|
| Checkpoint write | ~2-5 seconds |
| Cheap op execution | ~1-2 seconds |
| Expensive op execution | minutes to hours |

**For cheap operations, checkpointing costs MORE than re-running on failure.**

Example pipeline analysis:
```
filter(1s) → mapper(2s) → deduplicator(300s) → filter(1s)

Strategy         | Overhead  | Protection Value
-----------------|-----------|------------------
every_op         | ~20s      | Save 1-304s on failure
after dedup only | ~5s       | Save 300s on failure
disabled         | 0s        | Re-run everything
```

### Strategy Recommendations

| Job Duration | Recommended Strategy | Rationale |
|--------------|---------------------|-----------|
| < 10 min | `disabled` | Re-running is cheap |
| 10-60 min | `every_n_ops` (n=5) | Balanced protection |
| > 60 min with expensive ops | `manual` | Checkpoint after expensive ops only |
| Unstable infrastructure | `every_n_ops` (n=2-3) | Accept overhead for reliability |

### Operation Categories

**Expensive operations (checkpoint after these):**
- `*_deduplicator` - Global state, expensive computation
- `*_embedding_*` - Model inference
- `*_model_*` - Model inference
- `*_vision_*` - Image/video processing
- `*_audio_*` - Audio processing

**Cheap operations (skip checkpointing):**
- `*_filter` - Simple filtering
- `clean_*` - Text cleaning
- `remove_*` - Field removal

### Storage Recommendations

- Event logs: fast storage (SSD)
- Checkpoints: large capacity storage
- Partitions: local storage

### Partition Sizing Trade-offs

- Smaller partitions: better fault tolerance, more scheduling overhead
- Larger partitions: less overhead, coarser recovery granularity

## Troubleshooting

**Job resumption fails:**
```bash
ls -la ./outputs/{work_dir}/{job_id}/job_summary.json
ls -la ./outputs/{work_dir}/{job_id}/checkpoints/
cat ./outputs/{work_dir}/{job_id}/checkpoints/partitioning_info.json
```

Use the resume token printed by the original run and rerun the same recipe with
`--resume`. Check the error log for configuration mismatch, missing partition
metadata, invalid row boundaries, or partition content hash mismatch. Explicit
resume does not delete checkpoints when validation fails.

**Check Ray status:**
```bash
ray status
```

**View logs:**
```bash
cat ./outputs/{work_dir}/{job_id}/events_*.jsonl
tail -f ./outputs/{work_dir}/{job_id}/logs/*.txt
```

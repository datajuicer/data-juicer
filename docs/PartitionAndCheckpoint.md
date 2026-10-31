# Partitioned Processing with Checkpointing

This document describes DataJuicer's fault-tolerant processing system with partitioning, checkpointing, and event logging.

## Overview

The `ray_partitioned` executor processes a dataset on Ray Data and can checkpoint every operator. Failed jobs can resume from the committed work.

**Checkpointing strategies:**
- `every_op` - commit the output of every Mapper/Filter batch (default)
- `disabled` - no checkpointing (best performance)

## Directory Structure

```
{work_dir}/{job_id}/
├── job_summary.json              # Job metadata (created on completion)
├── gpu_probe_results.json         # Auto-probed GPU operator resources
├── events_{timestamp}.jsonl      # Machine-readable event log
├── dag_execution_plan.json       # DAG execution plan
├── checkpoints/                  # checkpoint_dir
│   └── partition_recovery/
│       ├── run.json              # Input fingerprint and recipe identity
│       ├── commits/              # One immutable manifest per committed batch
│       ├── exports/              # One record per export attempt
│       └── data/                 # Parquet files referenced by the manifests
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
  max_concurrent_gpu_probes: "auto"   # Probe one target at a time; set an integer to opt into parallel probes
  gpu_preflight_enabled: true          # false skips preflight and uses explicit resources/actor counts
  gpu_probe_timeout_seconds: null     # Optional per-probe timeout; null disables termination
  gpu_probe_warmup_batches: 1         # Warmup batches before steady-state timing
  gpu_probe_steady_batches: 3         # Batches used for steady-state throughput
  gpu_probe_sample_offset: 0          # Leading rows skipped when sampling for preflight
  gpu_probe_sample_shuffle: false     # true randomizes block order before sampling
  gpu_probe_sample_seed: 42           # Seed for the shuffled sample; null re-samples, and never reuses the report
  target_size_mb: 256    # Target partition size (128, 256, 512, or 1024)
  size: 5000             # Fallback if auto-analysis fails; max input rows per commit when checkpointing
  max_size_mb: 256       # Fallback max size
```

**Manual mode** - specify exact partition count:

```yaml
partition:
  mode: "manual"
  num_of_partitions: 8
  max_concurrent_partitions: "auto"
```

Partitions are cut at exact row boundaries, so every partition holds the same
row count give or take one row. This is deliberately not Ray's
`Dataset.split(n)`, which distributes whole blocks: a dataset with fewer blocks
than `num_of_partitions` would otherwise produce empty trailing partitions (ten
rows in one block split four ways gives `[10, 0, 0, 0]`), and unevenly sized
blocks would skew partitions even when there are enough of them. A partition
count larger than the row count fails the job before any actor starts.

Partitions only apply when no operator is checkpointed. A checkpointed run streams the
whole dataset through one Ray Data plan instead and forms its recovery commits
as batches finish (see [Checkpointing](#checkpointing)), so `partition.mode` and
`num_of_partitions` do not affect it.

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
   `batch_size` among operators that still need probing. A dataset ordered by
   length, resolution or source makes that prefix unrepresentative and biases
   both measured memory and throughput. `gpu_probe_sample_shuffle: true`
   randomizes block order first, so the sample comes from a random block instead
   of the first one; only block references are reordered, keeping the preflight
   cost proportional to the sample rather than to the dataset. Alternatively,
   `gpu_probe_sample_offset` skips a known-unrepresentative prefix. The two are
   alternatives rather than a combination: a successful shuffle has already left
   the head, so the offset is reported as ignored. Changing either option
   invalidates `gpu_probe_results.json`, because measurements from a different
   part of the dataset are not comparable. A shuffle without
   `gpu_probe_sample_seed` samples different rows on every run, so its report is
   written for observability but never reused as a cache.
2. Operators may declare `input_columns` and `output_columns`, including
   nested paths such as `__dj__meta__.quality_score`. The executor builds a
   conservative data-dependency graph from these contracts.
3. Targets proven independent, with only compatible CPU Mapper/Filter
   ancestors, can run concurrently. Each disposable Ray worker receives the
   lightweight source rows, replays its required CPU ancestors locally, and
   reserves one full GPU for the target. Large intermediate NumPy values are
   therefore not round-tripped through the driver. The default
   `max_concurrent_gpu_probes: "auto"` probes one target at a time: concurrent
   probes initialize several models at once, and neither checkpoint storage nor
   host-memory bandwidth is a Ray resource, so nothing automatic can tell whether
   the node can sustain them. Set a positive integer to opt into parallel probing
   when checkpoints are local or the I/O headroom is known; the value is still
   bounded by the dependency-safe GPU and CPU slots.
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
   warmup batch and three measured batches. Replay and measurement use the
   recipe's `skip_op_error`, so a few invalid rows are skipped exactly as in the
   formal run instead of aborting the job at preflight. Because one measured
   batch is repeated, a batch that returns nothing is not a usable measurement:
   a Mapper that returns no rows always fails preflight with its operator name
   (with `skip_op_error` the whole batch took the error path, without it the
   operator returned empty where it should have raised), while a
   Filter that keeps nothing only warns that its throughput is worst-case.
   Preflight no longer sets the `DATA_JUICER_GPU_PREFLIGHT_OP` marker, because
   operators that reacted to it with extra CUDA synchronizations could turn a
   normal model transfer into a multi-minute stall; custom operators must not
   depend on observing that variable.
6. A probe does not poll CUDA from a background thread during model
   initialization, avoiding `cudaMemGetInfo` contention with large
   `model.to(cuda)` transfers on the same CUDA context. The disposable worker
   lets the operator initialize CUDA as a formal actor would, then combines
   PyTorch's allocator peak with final persistent device usage (which also
   covers non-PyTorch runtimes such as Paddle). The measured value receives
   10% headroom and becomes `memory_fraction`. Ray's scheduling `num_gpus` is `max(memory_fraction,
   1 / max_gpu_workers_per_device)`; the default allows at most five
   auto-probed model actors per physical GPU.
7. Resource values, phased timing, throughput, output ratio, probe mode,
   replayed dependencies, and where the sample actually came from are saved in
   `{work_dir}/gpu_probe_results.json`. A resume
   reuses entries when the operator configuration, GPU model/capacity, and
   timing-batch settings, per-device worker cap, and sampling policy match, and
   re-probes otherwise. The source YAML is not overwritten.

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

An explicit `memory` or `num_gpus` wins over the measurement, except that a
`num_gpus` below the measured memory fraction is raised to it with a warning:
Ray packs actors by `num_gpus` alone, so a smaller request would let it place
more actors on a device than its memory allows. Preflight currently supports
ordinary Mapper/Filter operators that fit on one GPU.
GPU `Pipeline` operators, a `Pipeline` before a pending target, and multi-GPU
operators require explicit resources. Empty input, probe errors, OOM, and an
invalid zero peak fail before formal partition workers start.

### Checkpointing

```yaml
checkpoint:
  enabled: true
  strategy: every_op     # every_op (default) or disabled
partition:
  size: 5000             # Maximum input rows per commit
checkpoint_dir: s3://bucket/jobs/my_job   # Optional; defaults to <work_dir>/checkpoints
```

Every leading Mapper/Filter of the recipe is a save point. Data stays on Ray
Data: operators read and write blocks through the object store as in an
uncheckpointed run, and each worker writes a copy of its output batch to
`checkpoint_dir` as a side effect before passing the batch on. A commit is
formed at that moment rather than when the input is read: it records which input
rows the batch consumed and which Parquet files it produced, so a Mapper that
expands rows or a Filter that drops all of them is still recorded exactly.
Checkpointed stages pass `batch_size=None` to Ray, so small blocks are not
combined to reach `partition.size`. Each Ray block is split into commits
of at most `partition.size` input rows, without changing the operator's internal
inference `batch_size`. Without `partition.size`, the limit is derived from
preflight throughput (about 30 seconds of work, between 64 and 2048 rows) and
defaults to 512.

The first Deduplicator, `Pipeline` or other operator that needs all rows at
once ends the checkpointed prefix. It and every later operator run unchanged on
the merged output without checkpoints, and a warning names the operator.

Commits are coordinated by a metadata-only Ray actor pinned to the driver node.
It hands out commit sequence numbers and rejects a batch whose input rows
overlap a different committed batch. Before running the operator, a retry looks
up a durable commit by the stage and consumed lineage; if found, it reads the
original output without running the operator. Concurrent attempts use the
first committed result; later attempts reuse its output. Each commit is published
as a manifest that is created only if absent, so a commit is never overwritten.

`checkpoint_dir` must be visible to every Ray worker. Supported locations are a
shared POSIX path and `s3://` (AWS S3 or an S3-compatible service such as
MinIO, through `boto3` with conditional `If-None-Match` writes). Other schemes
are rejected at startup.

### Checkpoints and Temporary Files

Checkpoints are kept after the job finishes so that it can be resumed. The
executor cleans up its temporary working files when the run exits.

Checkpoint cleanup removes only files owned by an attempt that are proven not
to be committed. A process kill or ambiguous remote write can still leave
orphan files; there is no automatic sweep of all unreferenced files, because
another worker could still be writing or committing them. See
[RecoveryPartitionProtocol.md](RecoveryPartitionProtocol.md#storage) for the
cleanup boundaries. Empty checkpoint exports remain directories for resumption.

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
3. read and fingerprint the input again and compare it, its row count, schema
   and the checkpointed operator configuration with `run.json`;
4. load commit manifests with at most 32 concurrent storage `get` calls,
   register commits in stage order, and determine each operator's unconsumed rows;
5. run each operator only on those rows plus the committed rows of the previous
   operator that it has not consumed yet, reading committed rows from storage on
   the workers;
6. verify after the export that every input row was consumed and delivered
   exactly once;
7. only then remove the export files that the interrupted runs left behind.

`partition.size` is not part of the identity and may change on resume. If the
recorded input fingerprint, row count, schema or checkpointed operator
configuration does not match, explicit resume stops with an error and leaves
the existing checkpoints unchanged. Starting a new run with the ID of an
existing checkpoint is rejected rather than mixed with it.

The caller must preserve source-row identity and read order: the same source
offset must still refer to the same logical record. The fingerprint is an
additional diagnostic guard, not a guarantee that every payload byte is
unchanged. Non-JSON values use their display representation, which can omit
tensor elements; contents of referenced media or model files are not hashed.
Keep payload consistency under your own dataset/version management and start
a new job when the logical input changes.

The export is written while the operators run, so an interrupted job usually
leaves a partial export behind. A resumed job writes the complete export under
new file names and removes the earlier files only after its own export is
complete and verified, so a failed resume never loses exported output. Files in
the export directory that the job did not write are kept and reported in a
warning. A partial Lance export must be removed before resuming.

Use `--job_id` to name a job. To resume it, add `--resume` with the original job ID to the original command.

If both arguments are supplied, their values must be identical:

```bash
dj-process --config config.yaml \
  --job_id my_experiment_001 \
  --resume my_experiment_001
```

### Checkpoint Strategies

```bash
# Every Mapper/Filter (default when checkpointing is enabled)
dj-process --config config.yaml --checkpoint.strategy every_op

# No checkpoints
dj-process --config config.yaml --checkpoint.strategy disabled
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

### Checkpoint Cost

Checkpointing does not add a materialization barrier between operators: Ray
Data keeps fusing, pipelining and scheduling operators as in an uncheckpointed
run. The added cost is the side write of every operator's output batch, one
small manifest per commit, a fingerprint pass over the input at startup, and
removing the internal lineage columns before the export. For cheap text
operators the write can cost more than re-running them; disable checkpointing
for short jobs.

Two optional consumers require retained final results: setting
`export_shard_size` materializes the export input for exact sizing, and asking
the Python API for a returned dataset materializes any operators after the
checkpointed prefix so export and later reads share one execution. The CLI
discards the return value and uses `skip_return=True`. Explicit Filter
statistics exports outside the checkpointed prefix also retain their computed
statistics for both writing and filtering.

### Storage Recommendations

- Event logs: fast storage (SSD)
- Checkpoints: storage shared by every worker with enough capacity for every
  operator's output

### Commit Size Trade-offs

- Smaller `partition.size`: less work lost per failure, more manifests and files
- Larger `partition.size`: fewer files, more rows recomputed after a failure

## Troubleshooting

**Job resumption fails:**
```bash
ls -la ./outputs/{work_dir}/{job_id}/job_summary.json
ls -la ./outputs/{work_dir}/{job_id}/checkpoints/partition_recovery/commits/
cat ./outputs/{work_dir}/{job_id}/checkpoints/partition_recovery/run.json
```

Use the resume token printed by the original run and rerun the same recipe with
`--resume`. Check the error log for configuration mismatch, a changed input,
read order or operator configuration, or commits without a `run.json`.
Explicit resume does not delete checkpoints when validation fails.

**Check Ray status:**
```bash
ray status
```

**View logs:**
```bash
cat ./outputs/{work_dir}/{job_id}/events_*.jsonl
tail -f ./outputs/{work_dir}/{job_id}/logs/*.txt
```

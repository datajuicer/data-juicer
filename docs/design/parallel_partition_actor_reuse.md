# Design Doc: Concurrent Partition Processing with GPU Scoping

**Author:** Data-Juicer Team
**Created:** 2026-03-09
**Updated:** 2026-03-17
**Status:** Implemented
**Branch:** `feat/cyrusz/parallel-partition-actor-reuse`

---

## 1. Problem Statement

### Current Behavior (Before This Change)

The `PartitionedRayExecutor` processes partitions **sequentially**, creating new GPU actors for each partition:

```
Partition 1 → [Create Actors] → [Load Models] → [Process] → [Actors GC'd]
Partition 2 → [Create Actors] → [Load Models] → [Process] → [Actors GC'd]
Partition 3 → [Create Actors] → [Load Models] → [Process] → [Actors GC'd]
```

### Problems

1. **Repeated Model Loading**: Heavy GPU models (e.g., VideoBLIP ~20GB) are loaded N times for N partitions
2. **GPU Idle Time**: GPUs sit idle between partitions during actor teardown/creation
3. **Poor Scalability**: Processing time scales linearly with partition count due to model loading overhead

### Impact

For a typical video processing pipeline with 3 GPU operators and 10 partitions:
- Model loading time: ~60s per operator × 3 operators × 10 partitions = **30 minutes of pure overhead**
- This overhead can exceed actual processing time for smaller datasets

---

## 2. Implemented Solution: Concurrent Partition Processing

### Overview

Instead of sequential processing with shared actor pools (originally proposed), we implemented **concurrent partition processing** where all partitions run in parallel as independent Ray remote tasks, each with its own scoped GPU actors:

```
┌──────────────────────────────────────────────────────────────────┐
│                  Concurrent Partition Processing                   │
│                                                                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐       ┌──────────┐    │
│  │ Task P0  │  │ Task P1  │  │ Task P2  │  ...  │ Task P7  │    │
│  │ 1 GPU    │  │ 1 GPU    │  │ 1 GPU    │       │ 1 GPU    │    │
│  │ Actor    │  │ Actor    │  │ Actor    │       │ Actor    │    │
│  └──────────┘  └──────────┘  └──────────┘       └──────────┘    │
│       ↕              ↕              ↕                  ↕          │
│    GPU 0          GPU 1          GPU 2             GPU 7          │
└──────────────────────────────────────────────────────────────────┘

All partitions processed concurrently, each with its own scoped actor
```

### Why Concurrent Instead of Sequential + Actor Reuse

The original design proposed sequential processing with detached shared actor pools. During implementation, we chose concurrent processing because:

1. **Simpler architecture**: No need for detached actor lifecycle management, pool coordination, or cross-partition actor sharing
2. **Better GPU utilization**: All GPUs are busy simultaneously instead of sequentially
3. **Natural Ray fit**: Each partition is a self-contained Ray remote task — no complex orchestration
4. **Same model loading cost**: Each GPU loads the model once per partition, but all load concurrently (~60s wall time vs. N × 60s sequential)
5. **Maintained benefits**: Checkpointing, resume, and memory control per partition are all preserved

### Key Design Principles

1. **Concurrent partition processing**: All partitions run in parallel (up to `max_concurrent_partitions`)
2. **Concurrency scoping**: Each partition's GPU ops get `num_proc = total_gpus // max_concurrent_partitions` actors
3. **Forced actor mode**: GPU ops are set to `ray_execution_mode = "actor"` inside the remote task (where CUDA is not visible)
4. **Per-partition checkpointing**: Each remote task manages its own checkpoint state
5. **Resume support**: Skip completed partitions on restart

---

## 3. Detailed Design

### 3.1 Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    PartitionedRayExecutor                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                 _process_partitions_concurrent()                  │   │
│  │                                                                   │   │
│  │  1. Extract serializable config values                           │   │
│  │  2. Submit Ray remote tasks (one per partition)                  │   │
│  │  3. Collect results, union partitions                            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│              ┌───────────────┼───────────────┐                          │
│              ▼               ▼               ▼                          │
│  ┌─────────────────┐ ┌─────────────┐ ┌─────────────┐                  │
│  │ Remote Task P0  │ │ Remote Task │ │ Remote Task │ ...              │
│  │                 │ │ P1          │ │ P2          │                  │
│  │ - load_ops()    │ │             │ │             │                  │
│  │ - force actor   │ │  (same)     │ │  (same)     │                  │
│  │   mode for GPU  │ │             │ │             │                  │
│  │ - scope conc.   │ │             │ │             │                  │
│  │ - process data  │ │             │ │             │                  │
│  │ - checkpoint    │ │             │ │             │                  │
│  └─────────────────┘ └─────────────┘ └─────────────┘                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Execution Flow

```
Phase 1: Dataset Splitting
──────────────────────────

Job Start
    │
    ▼
┌──────────────────────────┐
│ Repartition to N blocks  │  Ensure enough blocks for N partitions
└───────────┬──────────────┘
            │
            ▼
┌──────────────────────────┐
│ Split into N partitions  │  Each partition gets ~equal rows
└───────────┬──────────────┘


Phase 2: Concurrent Processing
──────────────────────────────
            │
    ┌───────┼───────┬───────┬───── ... ─────┐
    ▼       ▼       ▼       ▼               ▼
┌──────┐┌──────┐┌──────┐┌──────┐       ┌──────┐
│ P0   ││ P1   ││ P2   ││ P3   │       │ P7   │
│1 GPU ││1 GPU ││1 GPU ││1 GPU │       │1 GPU │
└──┬───┘└──┬───┘└──┬───┘└──┬───┘       └──┬───┘
   │       │       │       │               │
   ▼       ▼       ▼       ▼               ▼
 [Load]  [Load]  [Load]  [Load]  ...    [Load]   ← Models load concurrently
   │       │       │       │               │
   ▼       ▼       ▼       ▼               ▼
[Process][Process][Process][Process]    [Process] ← All GPUs busy
   │       │       │       │               │
   ▼       ▼       ▼       ▼               ▼
 [Ckpt]  [Ckpt]  [Ckpt]  [Ckpt]  ...  [Ckpt]    ← Per-partition checkpoint


Phase 3: Merge Results
──────────────────────
    └───────┴───────┴───────┴───── ... ─────┘
                        │
                        ▼
              ┌──────────────────┐
              │  Union partitions │
              └──────────────────┘
                        │
                        ▼
                    Job End
```

### 3.3 Concurrency Scoping

The critical mechanism that prevents GPU over-allocation:

```python
# Inside each remote task:
for op in task_ops:
    # Step 1: Force actor mode (MUST be before scope_op_concurrency)
    if getattr(op, "num_gpus", 0) and op.num_gpus > 0:
        op.ray_execution_mode = "actor"

    # Step 2: Scope concurrency — divides num_proc by max_concurrent_partitions
    op.num_proc = scope_op_concurrency(op, max_concurrent_partitions)
```

**Why order matters:**
- The remote task runs on a CPU-only node (no GPU assigned to the task itself)
- `torch.cuda.is_available()` returns `False` in the remote task
- Without explicitly setting `ray_execution_mode = "actor"`, `use_ray_actor()` returns `False`
- `scope_op_concurrency()` only divides `num_proc` for actor-mode ops
- If actor mode is not set first, `num_proc` stays at the full value (e.g., 8), causing each partition to request all 8 GPUs → deadlock

**Example with 8 GPUs, 8 partitions:**
- `num_proc` original = 8 (wants 8 GPU actors)
- `scope_op_concurrency(op, 8)` → `8 // 8 = 1` (1 GPU actor per partition)
- 8 partitions × 1 GPU = 8 GPUs total → fits exactly

### 3.4 Remote Task Design

Each partition is processed by an independent `@ray.remote(num_cpus=0)` task that:

1. **Re-creates ops from config** — avoids serialization issues with GPU operator state
2. **Forces actor mode** — sets `ray_execution_mode = "actor"` for GPU ops
3. **Scopes concurrency** — divides `num_proc` by `max_concurrent_partitions`
4. **Manages its own checkpoints** — creates a local `RayCheckpointManager`
5. **Handles resume** — checks for existing checkpoints before processing

The task requests `num_cpus=0` because the actual compute is done by Ray Data actors/tasks spawned within.

### 3.5 Dataset Splitting

```python
# Repartition to ensure enough blocks, then split
dataset.data = dataset.data.repartition(self.num_partitions)
partitions = dataset.data.split(self.num_partitions)
```

- `repartition(N)` ensures at least N blocks exist (lazy, adds a shuffle stage)
- `split(N)` distributes blocks across N independent `Dataset` objects
- Without repartition, split may produce empty partitions if there are fewer blocks than partitions

---

## 4. Configuration

```yaml
partition:
  mode: 'auto'                           # 'auto' | 'manual'
  num_of_partitions: 8                   # Number of partitions
  max_concurrent_partitions: 8           # Max partitions running in parallel

checkpoint:
  enabled: true
  dir: './checkpoints'
  strategy: 'per_op'                     # Checkpoint after each operator
```

The `max_concurrent_partitions` parameter controls how many partitions run simultaneously and how GPU resources are divided. Setting it equal to the number of GPUs (one partition per GPU) is typical for GPU-bound workloads.

---

## 5. Performance Comparison

### Timeline: Sequential vs Concurrent

**Before (Sequential, no actor reuse):**
```
Time ────────────────────────────────────────────────────────────────────▶

P0: [Load 60s][Process 120s][GC]
P1:                              [Load 60s][Process 120s][GC]
P2:                                                           [Load 60s][Process 120s]

Total: 3 × (60 + 120) = 540s
GPU idle: ~67% of total time
```

**After (Concurrent, 8 partitions on 8 GPUs):**
```
Time ────────────────────────────────────────────────────────────────────▶

P0: [Load 60s][Process 120s]
P1: [Load 60s][Process 120s]     ← All load concurrently
P2: [Load 60s][Process 120s]
...
P7: [Load 60s][Process 120s]

Total: 60 + 120 = 180s (wall time)
GPU idle: ~0% during processing
```

### Observed Results

**Setup:** 8× A100 80GB, 6000 video samples, VideoAestheticsFilter

| Mode | Time | GPU Utilization |
|------|------|-----------------|
| Pure GPU (no partitioning) | ~1100s | 100% on all 8 GPUs |
| Concurrent partitions (8) | ~1100-1300s | 100% on all 8 GPUs |
| Sequential (old, deadlocked) | ∞ (deadlock) | 8/8 GPU allocated, 14+ pending |

The concurrent approach matches pure GPU mode performance while adding partition-level checkpointing and resume capability.

---

## 6. Checkpointing and Resume

### Checkpoint Structure

Each remote task manages its own checkpoints:

```
checkpoints/
├── partitioning_info.json        # Partition metadata for validation
├── partition_0/
│   ├── op_0_video_aesthetics_filter/
│   │   ├── data.parquet
│   │   └── _SUCCESS
│   └── ...
├── partition_1/
│   └── ...
└── ...
```

### Resume Flow

```
Resume from Crash (Partition 2 was in progress)
──────────────────────────────────────────────

1. Load partitioning_info.json
2. Validate current partitions match saved metadata
3. Submit all partition tasks concurrently
4. Each task independently:
   - Checks its own checkpoint state
   - Skips completed ops (loads from checkpoint)
   - Resumes from last incomplete op
5. Collect results and union
```

---

## 7. Error Handling

### Partition Task Failure

If a remote task fails:
- Other partitions continue processing independently
- Failed partition's actors are cleaned up by Ray
- On retry/resume, the failed partition restarts from its last checkpoint

### GPU Resource Deadlock Prevention

The concurrency scoping mechanism prevents deadlock by ensuring:
- Total GPU requests across all concurrent partitions ≤ available GPUs
- `num_proc` is divided by `max_concurrent_partitions` for actor-mode ops
- Actor mode is set before scoping (critical ordering requirement)

---

## 8. Known Limitations and Future Work

1. **No actor reuse across partitions**: Each partition loads models independently. For workloads dominated by model loading time, a shared actor pool approach (the original design) could reduce overhead.

2. **Repartition cost**: `repartition()` adds a shuffle stage. For large datasets this is cheap relative to processing, but for small datasets it adds overhead.

3. **Single block per partition**: After split, each partition typically has one block, which means the entire partition is processed as a single batch by the actor. This prevents streaming output — no progress is visible until the whole partition completes.

4. **`max_concurrent_partitions` tuning**: Must be ≤ available GPUs for GPU-bound workloads. Auto-detection sets it to the GPU count, but mixed CPU/GPU pipelines may benefit from different values.

---

## 9. Design Decision Log

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Sequential vs concurrent | Concurrent | Better GPU utilization, simpler architecture |
| Shared actors vs per-partition | Per-partition | Avoids detached actor lifecycle complexity |
| Repartition before split | Always repartition | Avoids materializing dataset to check num_blocks |
| Actor mode + scoping order | Actor mode first | Required for scope_op_concurrency to work correctly |
| Remote task num_cpus | 0 | Task is just an orchestrator; actual compute uses Ray Data actors |

---

## References

- [Ray Actors Documentation](https://docs.ray.io/en/latest/ray-core/actors.html)
- [Ray Data User Guide](https://docs.ray.io/en/latest/data/data.html)
- Source: `data_juicer/core/executor/ray_executor_partitioned.py`
- Source: `data_juicer/core/executor/concurrency_scoping.py`

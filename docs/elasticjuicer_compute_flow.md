# Queue-aware compute flow for resident DAG pools

Compute flow is an opt-in scheduler for the [resident DAG executor](elasticjuicer_resident_dag.md).
It assigns idle workers from a fixed Ray pool to execution groups using queued
rows, in-flight work and measured service cost. Each worker retains its models
and each operator retains its own adaptive microbatch state.

The executor is an explicit Python API. Existing `ray` and `ray_partitioned`
entry points and their configuration remain unchanged.

## Two control layers

The driver estimates remaining worker-seconds from queued/in-flight rows and
an exponentially smoothed seconds-per-row measurement. It periodically updates
integer concurrency targets, with hysteresis to reduce small allocation changes.
The first nonempty sample for each actor/group is excluded as warmup. Empty
filtered batches do not consume warmup or contribute service-rate samples.

Targets are soft: running tasks finish normally. An idle worker can borrow work
when the target allocation would leave it unused. Queue-age priority prevents
starvation when there are more groups than workers. A completed group releases
its workers for remaining work at task boundaries.

Within each actor, each operator uses the existing `AdaptiveBatchController`
and `OOMSafeAdaptiveMapper`. Global scheduling never changes batch limits or
overwrites local success/OOM observations. Worker feedback measures the complete
group duration, including parent reconstruction and local retries, but excludes
driver queueing and downstream export.

## Usage

Initialize Ray and construct an explicit `ResidentDAGPlan` as shown in the
resident DAG guide, then enable compute flow:

```python
from data_juicer.core.elasticjuicer.compute_flow import ComputeFlowConfig
from data_juicer.core.executor.resident_dag_executor import ResidentDAGExecutor

flow = ComputeFlowConfig(
    pools=("gpu",),
    interval_seconds=0.5,
    starvation_seconds=5.0,
)
with ResidentDAGExecutor(
    plan, adaptive=True, compute_flow=flow, artifact_dir="new-run-directory"
) as executor:
    for result in executor.iter_batches(input_batches):
        write_idempotently(result.columns(), result.rejected_ids)
```

| Option | Default | Meaning |
| --- | --- | --- |
| `mode` | `"dynamic"` | Queue/rate-driven targets; `"static"` pins actor slots to groups. |
| `pools` | `()` | Select all GPU pools; explicitly name CPU pools to include them. |
| `interval_seconds` | `0.5` | Minimum interval between target updates per pool. |
| `ewma_alpha` | `0.25` | Weight of a new seconds-per-row observation. |
| `starvation_seconds` | `5.0` | Queue age at which waiting work gets priority at the next idle opportunity. |
| `hysteresis_workers` | `1` | Minimum target change; activation/deactivation bypasses hysteresis. |
| `initial_slots` | `()` | Optional `(group, count)` pairs covering each specified pool exactly. |

For an eight-worker pool, `initial_slots=(("quality", 2), ("embedding", 6))`
sets a starting allocation. In dynamic mode, subsequent plans adapt this split.
Static mode requires at least one worker per selected group and does not lend
idle slots. Dynamic mode also supports fewer workers than groups.

`DAGNode.batch_size` is the local microbatch cap. `initial_batch_size` optionally
starts adaptive execution below that cap. Fixed batching (`adaptive=False`)
always uses the cap and propagates OOMs. `compute_flow=None` retains the resident
executor's downstream-first policy and FIFO idle-worker reuse.

## Diagnostics and limits

With `artifact_dir` enabled, each run records target plans and actual actor
leases in `compute_flow.jsonl`. Lease events identify pool, slot, group, reason,
queue wait, reassignment and the corresponding pool-plan epoch. `events.jsonl`
records node execution, instance identity and local batching; `summary.json`
contains run totals and controller lifetime counters across successive passes.

Compute flow changes which task uses an idle actor. Ray CPU/GPU reservations and
pool size remain fixed. There is no preemption, model eviction, migration or
automatic placement search. Every model assigned to a GPU pool must fit on each
actor's GPU, together with the peak working memory of its execution group.
The DAG payload budget does not bound physical VRAM or temporary model memory.

Exact joins, ID conservation, admission budgets, pause/drain and failure cleanup
come from the resident DAG executor. Durable checkpoints, actor replacement,
exactly-once external commits and automatic cross-pool reallocation are outside
this API. A scheduling change is not evidence of an overall speedup.

## Reproducible real-image comparison

The benchmark decodes and verifies source images on CPU, runs QualiCLIP and
DINOv2 branches on GPU, joins by ID/hash, and exports complete outputs. It
compares shared-pool, static-allocation, dynamic-flow and colocated parallel-group
controls with fixed/adaptive local batching. It requires local trusted model
weights and definitions plus PyTorch, torchvision and transformers.

```bash
python tools/elasticjuicer/benchmark_compute_flow.py \
  --input /path/to/images.jsonl --qualiclip /path/to/QualiCLIP \
  --dinov2 /path/to/dino-v2 --output /path/to/new-run \
  --gpus 8 --rows 512 --passes 2 --repeats 1
```

Each JSONL record needs a unique `id`, `image_path` and `image_sha256`. Before
creating pools, the script estimates decoded RGB payload sizes from actual image
dimensions. Adjust `--object-budget-gib` (default 64) and `--inflight` (default
16) to the available host memory. An insufficient budget fails before pool
creation. Run this script in its own process; it creates and closes a local Ray
cluster.

`provenance.json` records code/environment/input identity; `results.json` records
timings and output comparisons. Cold and warm timings exclude Ray startup and
post-run verification. Compare identical inputs, models, outputs and budgets
over repeated paired runs before drawing performance conclusions. The short
historical eight-H20 experiment on the original development branch established
mechanism correctness, but did not outperform the shared-pool or colocated-group
controls. It has not been repeated on this main-based port.

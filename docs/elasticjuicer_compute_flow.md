# Compute flow inside `ray_partitioned`

Compute flow is an optional execution policy of `PartitionedRayExecutor`.
Continue using `executor_type: ray_partitioned`, the existing `process` recipe,
partition settings, checkpoint manager and resume command. There is no additional
executor type or user-authored DAG.

For eligible checkpoint intervals, a fixed pool of GPU workers keeps the
interval's operator instances resident. Different input batches can be at
different stages of the original operator sequence. The driver assigns idle
workers to stages according to ready/in-flight rows, observed service cost and
queue age. Each operator uses the existing `RayAdaptiveMapperActor`, including
its independent local batch controller, OOM retries and recovery behavior.

## Enable

The feature is disabled when `partition.compute_flow` is null (the default) or
its `enabled` option is false. Example settings to add to an existing recipe:

```yaml
executor_type: ray_partitioned
op_fusion: false
elastic_juicer_adaptive_batching: true
elastic_juicer_profile_seed: false
partition:
  # Keep the existing partition mode, size and execution_group_size settings.
  gpu_preflight_enabled: true
  compute_flow:
    enabled: true
    workers: 2
    batch_size: 32
    max_inflight_batches: 8
    max_batch_bytes: 134217728
    memory_fraction_limit: 0.8
    interval_seconds: 0.5
    starvation_seconds: 5.0
```

Each selected operator must be a batched CUDA Mapper supporting Ray actors and
explicitly opted into adaptive batching (`adaptive_batching: true` or its class
capability). This asserts row/order preservation, slice independence and retry
safety. Set `skip_op_error: false`. Operators must preserve the executor's
internal routing columns and support multiple operator instances in one process.
External side effects need operator-owned idempotency.

The current implementation selects a whole checkpoint interval only when **all**
its operators qualify. Filters, CPU operators, dataset-level operations,
per-operator runtime environments and error-skipping use the existing Ray Data
path. Selection happens before resident execution. A failure after resident work
starts fails the interval; it never silently replays the interval through another
execution path.

## Partitions, checkpoint barriers and resume

Logical partition IDs remain attached to every row until the existing checkpoint
manager saves each partition. Internal sequence IDs additionally check row order
and row conservation after every resident stage, then disappear from the result.
Completed batches are returned in input order as a materialized Ray Dataset.
The existing exporter and checkpoint manager consume that Dataset normally.

The scheduler never crosses a checkpoint boundary. With `every_op`, each interval
contains one operator: no cross-stage compute flow is possible. With an existing
multi-operator checkpoint interval (for example `every_n_ops`), batches can flow
through those operators concurrently on different workers. Enabling compute flow
does not change checkpoint frequency or merge existing boundaries.

A pool is reused across consecutive logical execution groups when their operator
interval, stage identities and resource requirements match. A different interval
replaces the pool. The pool is released before a legacy/global-operation path,
final export, and on all normal or exceptional exits from `run()`.

Checkpoint resume uses the existing per-partition path, including heterogeneous
resume positions. Completed checkpoint rows are not sent through resident
workers again. This first version does not use compute flow during recovery;
checkpoint identity and recovery decisions remain owned by `PartitionedRayExecutor`.

## Resource admission

Each resident worker reserves one whole GPU and the maximum CPU requirement of
its operators. `workers` controls the **shared** worker count for a selected
interval, replacing the separate per-operator actor counts there. Existing actor
counts still apply to fallback intervals. The executor never allocates both kinds
of GPU pool for the same interval. At least one CPU remains available for Ray Data
metadata, input preparation and checkpoint/export tasks.

GPU preflight must provide a measured memory fraction for every selected
operator. The sum of those complete per-operator peak estimates must be at most
`memory_fraction_limit`. This conservative admission rule includes all models
and sums activation estimates even though each actor runs only one stage at a
time. Missing estimates or an excessive sum select the legacy path. Scheduling
fractions alone are not treated as proof that models fit together.

The first version supports a single GPU node with hardware matching the preflight
measurements. Multiple GPU nodes use the existing path. Preflight estimates do
not guarantee future inputs will fit: local adaptive retries remain necessary,
and model-construction failures abort the interval. There is no live pool resize,
model migration, preemption or eviction. Shared stage-profile seeding is rejected
when compute flow is enabled because co-resident resource envelopes need separate
compatibility validation.

`max_inflight_batches` bounds admitted batches, while `max_batch_bytes` bounds
each Arrow input/intermediate batch. Completed blocks retained by the materialized
Dataset and operator temporary allocations are outside this in-flight budget.
There is no claim of bounded total dataset storage or physical VRAM from these
Arrow limits.

## Scheduling and diagnostics

`mode: dynamic` is the default. Ready and in-flight row counts are weighted by
smoothed seconds per input row. Each actor/stage's first nonempty observation is
excluded as warmup. Targets update no more often than `interval_seconds`, with
`hysteresis_workers` (default 1) reducing small changes. `ewma_alpha` defaults to
0.25. Running work finishes normally; idle workers may borrow tasks, and queue-age
priority prevents starvation. Targets never overwrite local batch/OOM state.

`mode: static` assigns fixed worker slots to stages as an experimental control.
It requires at least one worker per stage; otherwise the interval uses the
existing execution path. Static mode uses identical data transport and local
adaptive batching. `startup_timeout_seconds` defaults to 120 and
`task_timeout_seconds` to 600.

Each run has a unique directory under `work_dir/compute_flow/`, preserving failed
run evidence across resume. `events.jsonl` records actual pool instances,
plan epochs, actor leases, completed stages, local OOM counts and fallback reasons.
`summary.json` records executed intervals, pool reuse and controller totals;
its directory is also available in `cfg._resolved_compute_flow_summary["artifact_dir"]`.
The failure field describes resident execution
failures; job state remains in the executor's existing event log.

## Verification

```bash
PYTHONPATH=.:tests python -m pytest -q \
  tests/core/elasticjuicer/test_compute_flow.py \
  tests/core/elasticjuicer/test_partition_compute_flow.py
```

The transport tests run the public `PartitionedRayExecutor.run()` with real local
Ray workers, logical GPU reservations, injected preflight estimates and injected
OOMs. They check output/order equality with the disabled path, model-instance
reuse, reassignment, local retry isolation, checkpoint barriers, completed and
partial-checkpoint resume, fallback decisions and resource cleanup.

These CPU-arithmetic tests do not establish physical GPU compatibility or
throughput improvements. Earlier standalone resident-DAG measurements do not
validate this integrated path. Real-model GPU and repeated end-to-end performance
comparisons remain required before recommending it as a default.

# Adaptive microbatching and OOM recovery for Ray GPU Mappers

This opt-in runtime builds on the GPU actor planner and execution groups
merged in Data-Juicer PR1054 (`1e9720d01610ece3cb0d3e3026c0bf88fbbf42c8`).
The portable controller,
microbatch wrapper and stage identity code are adapted from ElasticJuicer
`9e8b2552fa3a8fa6733c3afd94e9abef5176525f`.

## What is included

| Component | Behavior |
| --- | --- |
| Stable stage identities | Stamp the complete prepared recipe before probing and checkpoint slicing. Distinguish repeated operators, preserve stamps through partition copies and exclude the per-run work directory. |
| Adaptive microbatch controller | Keep bounded success/OOM state inside each actor. Shrink after OOM and cautiously grow below known failing sizes. |
| Lossless OOM retry wrapper | Retry the same input offset, isolate failed input mutations, validate output row counts and merge successful slices in order. |
| Ray adapter | Select the adapter only for explicitly opted-in CUDA Mappers. Keep PR1054's outer batch size, ActorPool strategy, CPU/GPU reservations, execution groups and logical checkpoints. |

The runtime uses no second global planner or coordinating service. Resident
workers, dynamic actor replanning, ControlService/Captain, distributed memory
quotas, profile persistence/seeding and actor-startup economics policies are
outside this first batch. Controller state lasts for one actor incarnation;
a new actor starts at the configured batch size.

## Enable explicitly

The global switch defaults to **false**, so existing recipes retain the original
execution path. A recipe must also opt its eligible operator in:

```yaml
executor_type: ray_partitioned
elastic_juicer_adaptive_batching: true
op_fusion: false
process:
  - your_retry_safe_gpu_mapper:
      adaptive_batching: true
      batch_size: 8
      # Keep the usual PR1054 resource settings and preflight configuration.
```

`your_retry_safe_gpu_mapper` is a placeholder for a registered custom Mapper,
not a newly bundled operator. An operator author can instead declare
`_supports_adaptive_batching = True` on the Mapper class after validating the
contract below. A recipe's `adaptive_batching: false` overrides that class
declaration. No built-in model operator is enabled by default in this patch.

The ordinary `ray` executor can use the adapter too. Partition grouping and
checkpoint guarantees belong to `ray_partitioned`.

## Operator contract and error handling

An opted-in operator must:

1. Be a batched CUDA `Mapper` using a Ray actor, with at most one GPU reserved
   per actor and a positive integer `batch_size`.
2. Preserve row count and row order, and return compatible columns across
   slices. Its result must be independent of how the outer batch is sliced.
3. Preserve PR1054's internal logical-partition tag exactly. The adapter checks
   this tag before accepting each slice.
4. Be safe to retry. Failed calls must not leave external writes or model state
   that changes the result of the retry. Copying inputs protects nested input
   mutations; it cannot undo file/database writes or arbitrary model state.
5. Let allocation OOM exceptions escape `process_batched`. Operator-internal
   broad exception handlers that hide OOMs must be fixed before opting in.

CPU operators, Filters, reducers/deduplicators, multi-GPU actors, operator fusion
and sample tracing are outside the adapter's supported scope. Explicitly
opting in an unsupported configuration raises an error.

The adapter calls `process_batched` below the ordinary Mapper error-skipping
wrapper. Classified allocation OOMs shrink the microbatch and retry the same
input offset. Failed call frames are cleared before CUDA cache cleanup. At the
minimum size it waits and retries up to three times; retries are bounded.
An unrecoverable OOM propagates even if `skip_op_error: true`. Row-count, schema
and partition-tag contract errors also propagate.

Ordinary operator errors retain the existing batch-level skip policy:
`skip_op_error: false` raises; `true` drops the entire Ray outer batch, including
any successful microbatches accumulated within that call. Thus the lossless
claim applies to recovered OOMs, not to deliberately skipped ordinary errors.

The microbatch maximum is the existing configured/probed `batch_size`. The
adapter never increases a GPU reservation or an actor count. Initial model
construction and PR1054 preflight remain outside runtime microbatch recovery;
the configured probe batch must fit. A sample-based probe is not a guarantee
that every later input fits, which is the runtime recovery case addressed here.

## Stage identities and checkpoints

Stage IDs contain the full-recipe index, repeated-configuration occurrence,
operator fingerprint and readable name. Constructor arguments and ordinary
JSON-compatible recipe options contribute to the fingerprint. Arbitrary opaque
Python objects with process-specific representations are not a supported
restart-stability contract.

The partitioned executor assigns identities after fusion and before resource
probing. Runtime injection of resource fields does not restamp an operator.
The manifest is available as `cfg._resolved_stage_identities`; GPU probe records
and throughput actor plans include `stage_id`, and adaptive retry logs use it.
Existing checkpoint keys and persisted dataset layouts are unchanged. This
patch does not add a separate runtime-plan writer or profile store.

## Validation

Focused tests cover controller bounds, retry budgets, mutable inputs, Arrow
batches, row/tag preservation, ordinary errors, terminal OOM propagation,
configuration defaults, actor dispatch, repeated-stage identity, resource
planning and the PR's partition/resume behavior.

Validation on the merged PR1054 base used Python 3.12 and Ray 2.55.1 across
2026-09-15/16. The selection contains 483 configuration, controller, executor,
base-operator, fingerprint and RayDataset regression tests, plus one real-Ray
end-to-end test. Of the regression tests, 455 passed in the restricted sandbox;
the remaining 28 required process access and reported passing on the host.
That host command reached its 180-second timeout during teardown, after the
passing pytest summary. The separate end-to-end command completed successfully
with one passing test in 55.41 seconds. Changed Python files also passed Black,
isort, flake8 and `git diff --check`. This is targeted validation, not the entire
repository's model-dependent test suite or a throughput benchmark.

```bash
python -m pytest -q tests/core/elasticjuicer \
  tests/core/executor/test_gpu_planning_regressions.py \
  tests/core/executor/test_gpu_memory_probe.py \
  tests/core/executor/test_auto_partition_cluster.py \
  tests/core/executor/test_ray_executor_partitioned_parallel.py \
  tests/core/executor/test_ray_partition_resume.py \
  tests/core/executor/test_partition_size_optimizer.py \
  tests/config/test_config_functions.py \
  tests/ops/test_base_op.py \
  tests/utils/test_fingerprint_utils.py \
  tests/core/data/test_ray_dataset.py
```

The end-to-end test starts an isolated real Ray cluster with a logical GPU
resource and a CPU-safe threshold Mapper that injects OOMs. It exercises public
partitioned execution, two repeated Mappers, grouped logical checkpoints,
checkpoint reuse and exported row IDs. It does not measure physical GPU memory
limits or throughput.

`tools/benchmark/elasticjuicer_clarity_smoke.py` is an optional GPU correctness
check for the external Vgen `CustomClarityMapper`. Supply `--vgen-root`,
`--model-path`, `--input` (JSONL) and `--output` (report JSON). It requires trained
`new_query` and `new_query_head` weights and refuses an ordinary Qwen checkpoint
without that head. It compares real Clarity inference on fixed batches with
adaptive inference under **injected** OOMs. Model weights, images and the external
Vgen source are not bundled with this repository.

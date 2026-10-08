# Stage batch profiles for Ray adaptive Mappers

StageProfile builds on PR1070's actor-local adaptive microbatch controller and
PR1054's execution groups. It shares successful microbatch sizes and exclusive
OOM upper bounds between actors in **one `PartitionedRayExecutor.run()`**.
Later execution groups and replacement actors can start with compatible prior
experience instead of repeating the same initial OOM probes.

The switch defaults to **false**. This feature does not require PR1072's probe
resource sampler. It changes neither actor allocation nor dataset checkpoints.
It does not establish a throughput or wall-time improvement for real models.

## Enable and declare compatibility

```yaml
executor_type: ray_partitioned
elastic_juicer_adaptive_batching: true
elastic_juicer_profile_seed: true
op_fusion: false
process:
  - your_retry_safe_gpu_mapper:
      adaptive_batching: true
      batch_size: 8
```

The placeholder Mapper must satisfy the retry, slicing, row/order and partition
tag contracts in [ElasticJuicerFirstBatch.md](ElasticJuicerFirstBatch.md).
Profile reuse additionally requires two declarations on its class:

```python
elastic_juicer_cost_columns = ("image_height", "image_width", "token_budget")

def elastic_juicer_profile_resource_context(self):
    # Use the actual assigned device after model initialization. The budget
    # comes from this operator's resource contract, not a free-memory sample.
    import torch
    properties = torch.cuda.get_device_properties(self.device)
    return {
        "resource_class": properties.name,
        "memory_budget_bytes": self.memory_budget_bytes,
        "device_total_memory_bytes": properties.total_memory,
        "budget_epoch": self.budget_epoch,
    }
```

These are an **operator-author contract**, not generic recipe options added to
every built-in Mapper. `self.device`, `self.memory_budget_bytes` and
`self.budget_epoch` are example operator-owned fields. The author must include
all relevant input cost and resource distinctions: transformed image canvas,
frame count, token limits, device/MIG class, model precision or mutable budget
epochs as appropriate. Include immutable model/precision options in constructor
arguments so they also contribute to the operator fingerprint. Fractional Ray
GPU reservations alone do not enforce a physical memory budget.

Only outer batches with homogeneous, non-null declared cost values are eligible.
Missing columns, mixed costs, opaque/non-finite JSON values, missing resource
declarations and invalid budgets bypass both profile reads and publication.
The resource declaration must return finite JSON with a nonempty string
`resource_class` and positive integer `memory_budget_bytes`. Exact matching is
intentional; there is no similarity model or generic distribution fingerprint.
Incomplete cost/resource declarations cannot guarantee compatibility.

## Safety and lifetime

Each key contains the run UUID, full pipeline and operator fingerprints, stable
stage identity, input-cost digest, resource-context digest, resolved CPU/GPU,
memory and actor-concurrency envelope, runtime environment and static batch
limits. Fingerprints come from the complete prepared recipe before resource
injection. Repeated operators remain distinct stages. Raw cost and resource
values are hashed before they enter the artifact.

The actor reads at most once, before its first local success/OOM observation.
It validates schema, identity, bounds and TTL again before calling
`AdaptiveBatchController.seed_bounds()`. A seed stays advisory: it cannot exceed
static bounds, replace fresh local evidence or clear the actor's OOM limit.
An input/resource change in a running actor does not reset its controller or
trigger a new seed. The actor may therefore stay conservative after a heavy
batch; bounded capacity recovery is a separate controller policy.

Successful sizes are actual validated observations, never an inferred success
at `oom_upper_bound - 1`. Profile merging takes the smallest live exclusive OOM
bound and the largest live observed success below it. Each piece of evidence
expires independently after one hour from receipt by the service. Fresh
successes cannot extend the life of an older OOM observation. OOM at the minimum
size does not produce a reusable prior; local bounded retries still apply.

Observations are buffered inside the actor and published only after the entire
outer batch passes row/tag checks and output merging. Terminal OOM, ordinary
skips, schema/row/tag failures and resource context changes during the call
discard the buffer. Failed input mutations retain PR1070's copy-and-retry
semantics. Profile publication is advisory telemetry, not a data commit or an
exactly-once checkpoint mechanism.

One unnamed, driver-owned Ray actor with zero reserved CPUs holds at most 256
contexts, with bounded evidence and source/seeded incarnation lists per context.
It survives data-actor replacement while the driver and profile service remain
alive. Service loss, invalid priors and RPC failure fall back to the local
controller. A data actor disables its profile client after its first RPC error;
the error adds at most one two-second timeout for that incarnation. Service
startup has a separate ten-second timeout.

Eligible outer batches publish once after delivery, with a bounded RPC wait, so
the next execution group sees completed evidence. There is no RPC per
microbatch. Cost validation scans the declared columns before and after each
outer batch. This opt-in overhead should be measured on target workloads.

## Audit artifact and recovery scope

At run exit, including failure, the driver attempts to write
`work_dir/elastic_juicer_stage_profiles.json` using a temporary file, `fsync` and
atomic replacement. It always releases its owned profile actor. Artifact or
service failures cannot change the data-processing result.

The JSON includes the run/pipeline identity, `scope: driver_run`, profile bounds,
source and successfully seeded actor incarnation IDs, and read/hit/miss,
publication, expiry, eviction and rejected-request counters. Consumer-side
context skips, seed rejection and service errors appear in actor logs and
`profile_diagnostics`. `cfg._resolved_stage_profile_summary` holds the final
snapshot when saving succeeds.

The file is an audit snapshot; this PR does **not** load it on a subsequent run.
A new run UUID prevents inheritance across drivers, repeated executor runs or
work-directory reuse. Dataset checkpoint resume keeps its existing behavior and
may learn a new profile from the remaining data. Cross-driver profile restore
requires a separate persistence compatibility contract and validation matrix.

## Validation

```bash
PYTHONPATH=.:tests python -m pytest -q tests/core/elasticjuicer/test_stage_profile.py
PYTHONPATH=.:tests CUDA_VISIBLE_DEVICES=0 python -m pytest -q \
  tests/core/elasticjuicer/test_stage_profile_e2e.py \
  tests/core/elasticjuicer/test_partitioned_adaptive_e2e.py
```

The real-Ray tests use a CPU-safe Mapper with injected OOMs and logical GPU
reservations. They kill and replace an actor, verify actual seeding and run
seed-off/on public execution across four execution groups with two repeated
stages. They compare output rows, values, nested mutation counts, exported IDs,
OOM attempts and completed checkpoint reuse. These tests do not allocate CUDA
memory or benchmark real-model throughput.

Validation on 2026-10-08 with Python 3.10.19 passed 540 distinct tests: 54
StageProfile unit tests, 70 existing adaptive/controller/identity tests, 413
configuration/executor/probe/base-operator/RayDataset regressions, and three
real-Ray end-to-end tests. In the deterministic four-group/two-stage fixture,
seed-off/on reduced injected OOM attempts from 24 to 6 with identical output
and completed-checkpoint reuse. This is mechanism validation, not a performance
claim. Changed Python files passed Black/isort; flake8 found no new issues
(the base configuration module retains two existing F824 warnings).

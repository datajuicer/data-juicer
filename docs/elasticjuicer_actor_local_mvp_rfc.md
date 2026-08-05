# ElasticJuicer Actor-local MVP Control Contract

- Status: Ready for maintainer review; approval pending
- Scope: EJ-0; normative input to EJ-9b and EJ-10
- Date: 2026-07-20
- Evidence baseline: R3 corrective validation (`20260718T131224Z_r3_corrective`)

## 1. Decision summary

For the MVP, each Ray mapper actor owns the only component allowed to choose its next executable micro-batch size: `AdaptiveBatchController`.

The driver/coordinator may send a versioned hard upper bound. It may make an actor more conservative, but it cannot select the actor's next batch size, overwrite success/OOM bounds, or force growth. Capacity recovery is represented by a non-binding hint. The actor decides whether and when local evidence is sufficient to reopen a bounded probe.

Runtime metrics are lossy observability data, not an audit log. Missing, stale, low-confidence, or gapped metrics mean unknown; they never mean zero utilization, zero memory pressure, successful work, or lost business rows.

For a strict mapper invocation, the MVP guarantees that every successfully processed input row contributes exactly once to the returned output under its own OOM retry loop, with row-count/value correctness and input order within that batch. It does not guarantee exactly-once external side effects, extend the guarantee across Ray task reconstruction, or add a global output-order guarantee across parallel Ray actors. A caller that requires global deterministic order must enable and validate Ray's existing `preserve_order` execution mode independently.

These rules are intentionally narrow. The MVP does not change actor count, Ray resource reservations, placement, stage parallelism, or pipeline topology.

## 2. Motivation and current-state conflict

The R3 implementation establishes a working actor-local controller, lossless retry loop, actor-owned RSS/CUDA sampler, bounded asynchronous metrics transport, and a versioned quota object. Validation demonstrated bounded OOM probing, shrink/recovery behavior, bounded metrics pressure, and value/count correctness.

The current code nevertheless exposes two remote ways to erase local OOM knowledge:

- `BatchSizeQuota.reset_oom_bound=True` calls `controller.reset_oom_bound()` while applying a driver quota.
- `RayAdaptiveMapperActor.reset_oom_bound()` exposes the same operation directly.

Those paths conflict with the controller's stated sole-authority contract. They are treated as transitional EJ-9 behavior, not as the EJ-9b contract. EJ-9b must remove them from the remote control surface after compatibility impact is checked. Raising a quota alone is not evidence that an actor's usable memory has recovered.

## 3. Goals and priorities

The control objective is ordered, not blended:

1. Preserve row/value correctness and never skip or duplicate a slice because of adaptive retry.
2. Contain OOMs and fail explicitly when the configured minimum batch size cannot run.
3. Maximize sustained goodput within locally proven safety bounds and the current remote cap.
4. Use utilization and profiling estimates only as diagnostic or advisory evidence.

When objectives conflict, the earlier item wins. In particular, throughput or utilization does not justify deleting an OOM bound.

Normative terms `MUST`, `MUST NOT`, `SHOULD`, and `MAY` have their usual RFC meanings.

## 4. Authority and actuator model

### 4.1 Actor-local authority

Each actor incarnation owns one controller and its associated state:

- current batch size;
- configured minimum and static maximum;
- coordinator hard limit;
- successful lower bound and exclusive OOM upper bound;
- cooldown, probe, retry, and recovery counters.

Only that controller may turn observations and constraints into the next micro-batch size. The executable size is always bounded by:

```text
min(remaining_rows, static_max_batch_size, hard_limit,
    local_oom_upper_bound - 1 when present)
```

The actor MUST apply local success/OOM observations synchronously with execution. Remote services MUST NOT mutate learned local bounds, counters, or current batch size directly.

### 4.2 Allowed remote actuators

The MVP remote control surface contains exactly two fields:

| Actuator | Strength | Meaning |
| --- | --- | --- |
| `max_batch_size` | Binding upper cap | The actor MUST NOT execute a larger batch. It cannot force the actor to increase. |
| `capacity_recovery_hint` | Non-binding hint | Conditions outside the actor may have improved. The actor MAY schedule a bounded local re-probe after its own safety checks. |

There is no remote `target_batch_size`, `set_current_batch_size`, or `reset_oom_bound` actuator.

The coordinator may lower `hard_limit` and may later relax that cap up to the actor's static maximum. Relaxing a cap only removes a remote constraint: it does not assert local capacity, clear learned bounds, select a larger batch, or bypass actor-local probing. Thus the coordinator is not restricted to monotonically lowering the cap, but it is restricted to changing this upper-bound constraint.

The local controller may also reopen probing using its configured success-based recovery policy. A hint can make that policy eligible; it cannot bypass cooldown, local probe budgets, the hard limit, or static bounds.

### 4.3 Recovery-hint acceptance

On a newer valid hint, the actor records the hint but retains its OOM bound. It may reopen one probe only when all of the following are true:

- the hint is fresh and addressed to the current job and actor incarnation;
- at least `oom_reprobe_successes` full-size successes have occurred since the last OOM;
- cooldown is complete;
- the per-bound `max_oom_reprobes` budget is not exhausted;
- the current hard limit permits a larger batch.

The probe remains subject to the controller's ordinary proportional growth and midpoint logic. A failed recovery probe restores/tightens the OOM bound and applies exponential backoff already represented by the controller. A successful execution at or above the former failing size is the only proof of recovered capacity.

### 4.4 Size hierarchy and authority under `ray_partitioned`

When the adaptive path runs inside `PartitionedRayExecutor`, three distinct sizes coexist. They MUST keep distinct names and distinct owners; none of them may be referred to as plain `batch_size` in configs, metrics, or logs:

| Size | Owner (single authority) | Meaning |
| --- | --- | --- |
| `partition_rows` | `ray_partitioned` executor | Checkpoint/fault-recovery granularity. Fixed at split time; never mutated mid-run. |
| outer map batch (`op.batch_size`) | Ray Data operator configuration | The input batch Ray Data hands to one actor call. Static per operator. |
| executable micro-batch | actor-local controller | The slice the controller actually allows the operator to execute. Runtime-adaptive. |

Invariants:

- executable micro-batch <= outer map batch <= `partition_rows`;
- the partitioned executor decides partition boundaries and checkpoints, and MUST NOT influence micro-batch selection;
- the actor-local controller (and, when enabled, the Captain cap per section 4.2) MUST NOT modify partition count, partition boundaries, or actor parallelism;
- the same operator instance keeps exactly one stable `stage-NNNNNN:op_name` identity for the whole job, across partitions and checkpoint groups (PR-RP-EJ-3); distinct operator instances keep distinct identities;
- the partitioned executor owns exactly one job-scoped Captain lifecycle: it is started once before the partition loop and closed in `finally`, mirroring `RayExecutor.run` (PR-RP-EJ-2);
- each partition rebuilds its ActorPool; new incarnations MAY inherit learned bounds only through the advisory stage-profile seed defined in section 4.5 (PR-RP-EJ-4).

The joint path is validated by `tests/core/elasticjuicer/test_ray_partitioned_adaptive_e2e.py` (lossless per-partition OOM retry, shared job-scoped services, checkpoint resume and partial-failure recompute, a Captain lifecycle spanning partitions, and cross-partition profile inheritance).

### 4.5 Cross-partition profile seeding (PR-RP-EJ-4)

Under `ray_partitioned`, each partition rebuilds its ActorPool, so without seeding every partition re-probes the OOM boundary from scratch. Seeding transfers that learning as an advisory prior while preserving the single-authority rule of section 4.1.

Message. A `StageProfile` carries `job_id`, the stable `stage_id` (section 4.4), `op_name`, `safe_batch_size` (best proven success size), `oom_upper_bound` (tightest observed failure size), and `observed_at_ms`. A profile MUST carry at least one learned bound and MUST be job-scoped; the `ControlService` rejects any cross-job profile.

Reporting. An actor publishes its learned bounds fire-and-forget at outer-batch boundaries, and only when they changed since its last successful report. Report failures are logged and retried at the next boundary; they never block or fail data processing.

Merge. The `ControlService` keeps one profile per `stage_id` and merges conservatively: the tightest OOM bound (minimum), the best proven safe size (maximum), and the safe size is then capped below the merged OOM bound. Merging never widens previously recorded evidence.

Seeding. A new incarnation performs one bounded, best-effort `get_stage_profile` fetch at construction, before its first slice. `AdaptiveBatchController.seed_bounds` adopts the profile only as an initial prior: it MUST run before the first observation, it can never relax static bounds, and an advisory OOM bound at or below the static minimum is ignored rather than bricking the incarnation. After seeding, the actor-local controller remains the sole authority over the next micro-batch; a seed is a prior, never a command.

Failure semantics. Any fetch failure, timeout, or missing profile degrades to an unseeded start; any report failure degrades to a no-op. Seeding is a latency optimization only and MUST NOT affect losslessness.

Configuration. The product path enables seeding with `elastic_juicer_profile_seed` (default `true` when `elastic_juicer_adaptive_batching` is enabled). The actor constructor flag `profile_seed_enabled` defaults to `false` so unit tests and embedders opt in explicitly.

## 5. EJ-9b message contract

Quota delivery and metrics delivery are separate channels and have separate failure semantics. An actor first registers this identity and its immutable bounds:

```text
ActorRegistrationV1 {
  schema_version: 1
  job_id: string
  stage_id: string
  op_name: string
  actor_id: string
  actor_incarnation_id: string
  static_min_batch_size: positive integer
  static_max_batch_size: positive integer
}
```

`actor_incarnation_id` is a fresh UUID for every actor process construction, including reconstruction of the same logical actor. The proposed quota envelope is:

```text
QuotaEnvelopeV1 {
  schema_version: 1
  job_id: string
  actor_id: string
  actor_incarnation_id: string
  revision: positive integer
  issued_at_ms: integer
  expires_at_ms: integer
  max_batch_size: positive integer
  capacity_recovery_hint: boolean = false
  reason: optional string
}
```

Requirements:

- Revisions are strictly monotonic per `(job_id, actor_id, actor_incarnation_id)`.
- Duplicate or lower revisions MUST be ignored and reported diagnostically.
- Identity or schema mismatch MUST be rejected; it MUST NOT partially apply.
- `expires_at_ms` MUST be later than `issued_at_ms`.
- A message already expired on receipt MUST be ignored.
- `max_batch_size` below the actor minimum MUST be rejected; a value above the static maximum is clamped to that maximum and reported as such.
- A newly accepted lower cap takes effect before the next slice.
- Expiry never means capacity is free. Once accepted, the last hard cap remains fail-safe until a newer valid envelope changes it or the actor incarnation ends.
- A newer envelope may raise the cap, but raising it does not clear an OOM bound or immediately grow the batch.

An actor restart creates a new `actor_incarnation_id` and resets local controller history. Revisions may restart for the new identity. Messages for a prior incarnation MUST be rejected. `actor_id` alone is insufficient for restart safety.

If quota delivery is unavailable, an existing actor continues under its last accepted cap and local evidence. A new actor starts with its configured static cap and local conservative initial batch size; it does not infer a cap from missing metrics or stale messages.

### 5.1 Delivery architecture

The job owns one explicit `ControlService`, separate from the metrics sink. An actor submits its registration when constructed and maintains at most one registration request and one quota-poll request in flight.

Before every micro-slice decision, including an OOM retry, the actor advances this
local state machine with `ray.wait(timeout=0)`. It resolves only references
already reported ready, applies any cached quota before calling
`next_batch_size()`, and otherwise continues immediately. Poll submission is
rate-limited by `control_poll_interval_sec`. There is no blocking `ray.get`,
service RPC wait, or Ray Data private ActorPool access on the data path.

This boundary-driven design intentionally uses no actor-lifetime polling thread. It avoids a Ray Core teardown race when the control actor or worker terminates while preserving asynchronous delivery and failure isolation.

## 6. Memory metric definitions

Memory metrics are not interchangeable. Every observation MUST retain source, scope, timestamp, unit, and confidence.

| Metric | Definition and scope | Permitted MVP use |
| --- | --- | --- |
| `rss_start_mb` / `rss_end_mb` | Host RSS of the actor process at batch boundaries | Actor-process trend and leak diagnostics |
| `rss_peak_mb` | Maximum sampled actor-process RSS during the measured batch | Actor-process peak diagnostic; sampling may undercount short spikes |
| `rss_delta_mb` | End RSS minus start RSS for the actor process | Directional diagnostic, not total batch memory |
| `cuda.allocated_mb` | Live bytes owned by tensors in the actor's current PyTorch CUDA allocator/device at batch end | Local allocator pressure diagnostic |
| `cuda.reserved_mb` | Bytes held by that allocator at batch end | Local allocator footprint; not device-wide use |
| `cuda.peak_allocated_mb` | Per-batch peak allocated bytes after resetting allocator peak stats at batch entry | Strongest current local GPU batch-peak signal, but PyTorch-allocator scoped |
| Adapter/system probe memory | Host/device observation outside the actor ownership boundary | Offline profiling and diagnosis only |

No one metric is a complete device-capacity truth. RSS excludes other processes and can miss short-lived peaks; CUDA allocator metrics exclude non-PyTorch allocation and other actors; system probes lack actor attribution. Consequently:

- The hard safety boundary is the actor's observed success/OOM history plus configured bounds.
- Metrics MAY advise caps or recovery hints when fresh and sufficiently confident.
- Metrics MUST NOT directly select the next batch or erase OOM evidence.
- `None`/unavailable CUDA data is unknown, not zero.
- Negative RSS delta is valid and does not prove spare capacity.

## 7. Metrics transport and schema semantics

Metrics are best-effort and bounded on both producer and sink. Producer saturation may drop a new event; sink saturation may evict an old event. This behavior protects the data path and is not a processing failure.

The current `ActorMetricsEvent` identity and sequence fields are retained, and EJ-9b/EJ-10 must evolve them to an explicit envelope. Scope and confidence attach to each signal group because one snapshot contains both process- and device-scoped values:

```text
ActorMetricsEventV1 {
  schema_version: 1
  job_id: string
  actor_id: string
  actor_incarnation_id: string
  op_name: string
  sequence: positive integer
  observed_at_ms: integer
  emitted_at_ms: integer
  source: string
  snapshot: {
    process: {
      scope: process
      confidence: number in [0, 1]
      rss_start_mb, rss_end_mb, rss_peak_mb, rss_delta_mb
      rss_peak_confidence: number in [0, 1]
    }
    cuda: optional {
      scope: device
      confidence: number in [0, 1]
      device_index, allocated_mb, reserved_mb, peak_allocated_mb
    }
    batch_size, latency_ms, throughput, succeeded, error_type
  }
}
```

Semantics:

- Sequence is monotonic per actor incarnation and includes attempted emissions. A gap indicates telemetry loss or omission only.
- Sequence MUST NOT be used to infer input offsets, exactly-once delivery, or row loss.
- `observed_at_ms` is the measurement time; `emitted_at_ms` is transport time.
- A coordinator MUST compare metric age against a configured `metrics_ttl_ms` before using it.
- A stale, missing, unsupported-schema, identity-mismatched, or low-confidence observation is `unknown`.
- `unknown` data may hold or lower a cap conservatively; it MUST NOT trigger expansion or a recovery hint.
- Every resource signal MUST carry a scope and confidence; a mixed process/device event MUST NOT collapse them into one ambiguous value.
- Aggregation MUST preserve scope. Process RSS cannot be added to device allocator memory as if they described the same resource.
- Dropped-event counters on producer and sink are diagnostics and SHOULD be exported with snapshots.
- Resource measurement completion and event emission are separate steps. The
  actor MUST first apply `observe_oom()` or `observe_success()` and then attach
  the resulting controller state to that same attempt's event. A later event
  MUST NOT be required to observe the transition.

For V1, sampler-derived boundary values have confidence `1.0` as measurements of their stated scope. `rss_peak_confidence` is computed from polling interval and observed coverage and is capped below `1.0`; a batch with no interior RSS sample reports `0.25`. Captain uses the minimum relevant confidence. Confidence expresses measurement fitness for its declared scope, not probability that a batch will succeed. System-level Adapter observations use a separate system-scoped schema and are never inserted into actor process/device fields.

### 7.1 Stage coordination and lifecycle

Every pipeline occurrence receives a job-local topology identity of the form
`stage-<index>:<op-name>`; repeated instances of the same operator therefore do
not share a stage. Active membership comes from the control service's latest
registration for each logical actor ID, so metrics from a prior incarnation do
not participate after reconstruction.

One Captain decision cycle selects the latest observation for every active
actor in a `(job_id, stage_id)` snapshot. Safety shrink remains per actor and
may proceed for a reliably observed hot actor. Recovery or expansion requires
one fresh, sufficiently confident, low-pressure observation from every active
stage member in the same cycle. Missing, stale, lossy, restarted-without-new-
metrics, or high-pressure members hold recovery for the entire stage. This is
the MVP fairness rule for heterogeneous actors; it does not require equal
absolute caps.

The product Ray executor owns an explicitly configured Captain lifecycle. It
starts before lazy RayDataset execution is materialized and stops in a
`finally` block. Each metrics/control RPC has a finite timeout; failures retain
pending delivery, enter bounded backoff, and do not affect actor-local data
processing. Captain is disabled by default and requires an explicit complete
process-RSS or CUDA high/low watermark pair.

## 8. Data correctness and ordering

Within a single `OOMSafeAdaptiveMapper` invocation:

- the same input slice is retried after a classified OOM;
- the input offset advances only after success;
- successful micro-batch outputs are merged in input-slice order;
- row-count validation is required by default;
- a non-OOM exception is propagated, except for the operator's existing explicit `skip_op_error` policy;
- an OOM at minimum batch size is propagated;
- adaptive retry itself MUST NOT skip or duplicate rows.

Across parallel Ray actors, this RFC guarantees output count and values, not global row order. Global deterministic order is an executor concern and is opt-in through the existing Ray `preserve_order=True` path. EJ-9b and EJ-10 MUST NOT silently enable it because it has independent scheduling/performance implications. Tests that compare unordered parallel output MUST use stable row identities or multisets; tests claiming deterministic order MUST explicitly enable and validate that mode.

The metrics sequence is unrelated to data ordering.

The exactly-once statement is limited to returned output within one mapper invocation. Operators that perform external side effects before raising OOM require their own idempotency mechanism, and Ray task/actor reconstruction requires a separate end-to-end delivery contract. The existing explicit `skip_op_error` policy may intentionally return an empty batch after a non-OOM operator failure and is therefore outside strict count/value correctness.

## 9. Failure behavior

- Metrics sink/reporter failure is isolated: processing continues and the failure is observable through warnings/counters where possible.
- Quota delivery failure does not block actor-local control or the data path.
- Malformed, stale, expired, or misaddressed control messages fail closed: no state is partially changed.
- Coordinator restart does not reset actor controller state. It must reconstruct revisions and actor incarnation identities before issuing control.
- Actor restart discards that actor's learned state; it must not inherit messages for the prior incarnation.
- When evidence is unknown, no control path may infer resource availability or force growth.

## 10. Explicit non-goals

The actor-local MVP does not implement or control:

- dynamic actor count or ActorPool resizing;
- Ray task/actor CPU or GPU reservation changes;
- placement groups, node placement, migration, or preemption;
- stage parallelism, pipeline topology, fusion, or operator reordering;
- cross-stage backlog control or work stealing;
- global device-memory arbitration between actors;
- Predictor/PBT/Tower optimization or learned online policies;
- exactly-once metrics delivery or a durable audit log;
- a new executor, sampler, metrics sink, controller, or MicroScheduler;
- global output ordering unless the existing executor option is explicitly enabled.

## 11. Implementation mapping

The current EJ-9b/EJ-10 implementation maps the contract as follows:

1. Add actor-incarnation identity, schema version, issue/expiry times, and strict revision checks to quota delivery.
2. Replace remote `reset_oom_bound` with `capacity_recovery_hint` and actor-owned acceptance logic.
3. Remove or make private `RayAdaptiveMapperActor.reset_oom_bound()` so the driver has no authority bypass.
4. Preserve `set_hard_limit()` as the single binding remote actuator.
5. Add metric envelope identity/version/freshness/confidence without coupling metrics and quota transports.
6. Reuse the existing RayDataset ActorPool, controller, mapper, sampler, and bounded metrics sink.

The legacy remote reset field and public actor reset method are absent from the
MVP control surface. Reintroducing either requires an RFC revision and explicit
maintainer approval.

## 12. Required validation for EJ-9b/EJ-10

Contract tests MUST demonstrate:

- lower cap clamps the next slice without changing learned success/OOM bounds;
- a lower cap arriving during one outer batch clamps the immediately following
  micro-slice without a synchronous control wait;
- higher cap never clears an OOM bound or forces immediate growth;
- a hint alone cannot clear a bound and only enables a bounded actor-local probe after required evidence;
- stale, duplicate, expired, wrong-job, wrong-actor, and prior-incarnation messages do not change state;
- quota delivery failure leaves local processing functional;
- metric gaps/staleness/unavailability never become zero and never trigger expansion;
- producer and sink queues remain bounded under pressure;
- OOM retry preserves count and values with no adaptive duplicate/skip;
- unordered parallel tests do not claim global ordering;
- actor restart and coordinator restart follow the identity/revision rules above.
- OOM and success events themselves contain post-transition controller state;
- stage skew, partial/stale membership, heterogeneous bounds, repeated
  operators, and actor reconstruction obey section 7.1;
- Captain RPC timeout/backoff and product start/stop lifecycle are bounded and
  failure-isolated.

EJ-10 Captain-lite may consume only the two actuators defined in section 4.2. Any additional actuator or authority exception requires a revision of this RFC before implementation.

## 13. Approval record

The implementation follows these proposed decisions, but no independently
verifiable maintainer approval is recorded in this repository. A task
continuation is not treated as architecture approval. Link the approving
issue, PR review, or maintainer comment here before changing this RFC to
`Accepted`.

- [ ] Approve actor-local controller as the sole next-batch authority.
- [ ] Approve removal of remote direct OOM-bound reset in favor of a non-binding recovery hint.
- [ ] Approve memory, metrics loss/freshness/confidence, and restart semantics.
- [ ] Approve invocation-scoped output exactly-once/value correctness without external-side-effect or default global-order guarantees.
- [ ] Approve the explicit non-goals and the two-field actuator surface.

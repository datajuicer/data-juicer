# Multi-node and conditional compute flow: PR #1116 supplement

This supplement extends the resident execution already exposed by
`PartitionedRayExecutor`. The public entry point remains
`executor_type: ray_partitioned`; `partition.compute_flow: null` still disables
the feature. Existing linear, one-worker-per-GPU configurations retain their
defaults. No separate executor or user-authored DAG is required.

## Configure the shared pool

Merge the following fragment into an existing recipe whose operators meet the
contracts below. Model paths and the `process` list remain recipe-specific.

```yaml
executor_type: ray_partitioned
op_fusion: false
elastic_juicer_adaptive_batching: true
elastic_juicer_profile_seed: false
partition:
  gpu_preflight_enabled: true
  compute_flow:
    enabled: true
    execution: linear
    mode: dynamic
    allow_multi_node: true
    workers: 80
    actors_per_gpu: 5
    max_resident_models: 1
    batch_size: auto
    max_inflight_batches: 128
    memory_fraction_limit: 0.9
    startup_timeout_seconds: 600
    task_timeout_seconds: 1200
```

This is an example budget for two nodes with eight GPUs each. Four equivalent
nodes use 160 workers at the same density; choose concurrency and memory settings
from measured preflight data. `workers` is the total pool size across the cluster,
not a per-node count. Each worker requests one fifth of a GPU and the maximum CPU
requirement of its operators. A free CPU is also required for Ray Data and
checkpoint/export work. This executor connects to the caller's Ray cluster; it
does not start DLC services, allocate nodes, or configure network ports.

The existing checkpoint interval must contain multiple GPU stages for capacity
to move between them. `every_op` keeps its single-stage barriers. Conditional
fork/join requires its Topic and selected leaf operators in the same interval;
enabling this policy never merges checkpoint boundaries automatically.

Multi-node use is explicit. Nodes need compatible GPU hardware, package versions,
operator code and accessible data/model/output paths. Fractional requests control
Ray reservations, not physical GPU memory isolation. Admission estimates must
cover every actor replica. The pool and its GPU identities are recorded in
`compute_flow/<run-id>/events.jsonl` and `summary.json`.

`mode: static` is a control using the same transport and actor-local batching.
An optional `stage_slots: [N1, N2, ...]` gives positive initial counts in GPU
operator order and must sum to `workers`. Static mode retains those assignments;
dynamic mode can reassign idle workers as demand changes. The list must cover
exactly the selected interval. Omit it for automatic initial allocation.

## Explicit single-model lifecycle

The default `max_resident_models: 0` keeps every interval operator resident and
admits `sum(operator_peak_fractions) * actors_per_gpu`. Single-model mode uses
`max(operator_peak_fractions) * actors_per_gpu`, subject to the same configured
memory fraction limit. These are estimates, not guarantees for future inputs.

Every selected mapper must provide all three attributes:

```python
elastic_juicer_lazy_model = True

def elastic_juicer_load_model(self):
    # Load or move this mapper's model onto its assigned GPU.
    ...

def elastic_juicer_release_model(self):
    # Release this mapper's GPU model before another mapper is loaded.
    ...
```

The mapper constructor must actually be lazy. Hooks own backend resources and
should tolerate repeated loads/releases. A Torch implementation may move weights
to CPU and reload them; a backend without offload support may destroy and recreate
its predictor. Both costs remain inside the measured execution interval.
Advertising these attributes does not make an eagerly constructed model lazy.
Built-in operators without this contract use the existing execution path.

Switching stages does not recreate the mapper's adaptive batch controller or
reset its OOM history. An optional callable `elastic_juicer_oom_cleanup()` augments
the existing CUDA cleanup for backends with their own allocation caches. Operator
errors after work starts propagate without replay through a fallback executor.

## Conditional fan-out and join

`execution: linear` remains the default. Conditional routing contracts are honored
by the existing adaptive actor in the linear path as well. To schedule independent
branches concurrently, explicitly choose `execution: conditional` and provide:

- One contiguous GPU interval whose first mapper declares
  `conditional_topic = True` and preserves all source fields while producing
  `required_ops`, a list of stage names for every row.
- Independent leaves declaring `conditional_stage = "ocr"`, for example, and
  `conditional_outputs = {"ocr": "string"}`. Supported declared types are
  `float32`, `string`, `list_float32`, and `list_string`.
- Row-preserving, retry-safe, actor-capable batched CUDA mappers, with
  `skip_op_error: false` and the ordinary adaptive-batching opt-in contract.

Only selected rows reach a leaf. Empty branches do not invoke the model and
produce typed null columns. Each leaf reads the Topic snapshot, cannot consume
siblings' outputs, and must preserve identities, routing metadata and other input
columns. Unknown or duplicate route names, overlapping writers, missing selected
results and reordered identities fail the interval. The join restores source
order and retains all source rows, including rows selecting no optional branch.

CPU operators can precede/follow the contiguous GPU region. The conditional path
uses a bounded source producer and CPU join/suffix tasks, allowing overlap with
GPU work. It does not cross existing checkpoint boundaries. Explicitly replayed
output columns are replaced rather than appended twice. Generic recipes retain
their existing fallback behavior, and resume retains the executor's original
per-partition checkpoint recovery path.

## Validation and evidence boundary

Run the focused tests from the repository root:

```bash
PYTHONPATH=.:tests python -m pytest -q tests/core/elasticjuicer tests/config
```

Tests cover row routing and typed nulls, selected-only execution, joins and order,
local OOM cleanup, pool reuse, fractional resource admission, lazy lifecycle hooks,
controller preservation, checkpoint barriers and fallback. A two-Ray-node test
uses logical GPU resources and CPU arithmetic to verify explicit multi-node
opt-in and a shared fractional pool. It does not simulate physical GPU speed.

The runtime originates from the locally tested `8ac9c6f` experimental branch on
top of PR #1116 head `4628c5b`. Separate H20 experiments used 20,000 source images
repeated into 200,000 fresh inference tasks, five real models and five actors per
GPU. Completed framework comparisons covered 16 and 32 GPUs. A later feature
ablation snapshot had 26/45 completed two-node cells and one completed four-node
cell; a DLC heartbeat-file read failure interrupted the next four-node cell.
That supervisor and the benchmark's Parquet reader belong to the external
benchmark harness and are not part of this product patch.

In the completed two-node fixed-batch static/dynamic pairs, dynamic execution was
faster in three repetitions (paired ratios 1.174, 1.402, 1.361). The measurements
include model switching and export, but describe a controlled route-drift image
workload. They do not establish universal speedups, complete five-repeat
four-node scaling, or adaptive OOM-recovery gains on this workload. The feature
therefore remains opt-in. Distribution packaging keeps historical measurement
provenance separate from the new commit's local test results.

# Optional GPU probe resource sampling

`ray_partitioned` can record worker process resources during GPU preflight:

```yaml
executor_type: ray_partitioned
partition:
  gpu_preflight_enabled: true
  gpu_probe_resource_sampling: true
  gpu_probe_warmup_batches: 1
  gpu_probe_steady_batches: 3
  max_concurrent_gpu_probes: 2
```

Sampling defaults to `false`. Existing probe concurrency, memory headroom,
actor planning and execution groups retain their existing policies. Sampling
adds overhead to the measured probe durations; enable it for diagnostics and
compare against a run with sampling disabled before drawing performance
conclusions. It does not introduce runtime actor sampling or a control service.

## Report

Each operator in `work_dir/gpu_probe_results.json` receives a
`resource_samples` object when sampling is enabled. Both dependency-safe
parallel probes and ordered replay are supported. The phases are operator
construction, each warmup batch, and each steady batch. CPU dependencies
replayed before a target are outside these target observations.

The object contains:

- `worker`: PID, hostname, Ray node ID, Ray GPU IDs and
  `CUDA_VISIBLE_DEVICES`. GPU IDs must be interpreted together with the node
  ID. CUDA device index `0` is the worker-local index, not the physical card ID.
- `samples`: host wall time, process CPU seconds, RSS at phase boundaries,
  sampled RSS peak and poll count, and CUDA allocator counters at each
  boundary. CPU seconds sum user and system CPU time across process threads;
  they may exceed wall time. Child processes are excluded.
- `sampling_errors`: counters for unavailable process, CUDA or polling
  observations. Unavailable values are `null`, not zero.
- `dropped_samples`: phases omitted after the 256-record bound. Work continues
  after the bound, without further sampling threads.

RSS is polled every 10 ms during each phase. Its peak is a sampled lower bound
and can miss short allocations. Only RSS is polled in the background. CUDA
counters are read on the operator thread at phase boundaries; sampling never
initializes CUDA, resets its peak counters, or adds device synchronizations.

The CUDA counters describe the **current worker's PyTorch allocator**, not
all memory on the card. `peak_allocated_bytes` and `peak_reserved_bytes` are
**cumulative since worker startup**, not isolated per-phase peaks. Boundary
`allocated_bytes` after warmup can show retained model state, but the difference
between cumulative peaks and residency is not a fitted per-row memory model.
Other runtimes' transient allocations are not captured by these counters.

Phase wall time measures host execution; asynchronous GPU work may finish
later unless the operator synchronizes itself. It is not CUDA event timing.
The existing final synchronization and conservative memory-planning formula
remain in place.

Changing the sampling flag invalidates the probe cache, so enabling it obtains
fresh diagnostics. With sampling disabled, reports written before this option
remain reusable under the existing cache rules. Reused resource samples retain
their original worker identities and timestamps; they are historical evidence,
not new measurements of the current worker or device placement.

## Validation

The optional script runs real Ray GPU tasks with deterministic CUDA arithmetic
and synthetic allocations. It compares sampling off/on, checks distinct GPU
assignment and overlapping targets, verifies cache reuse and ordered outputs,
and checks propagation of actual oversized-allocation OOMs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  PYTHONPATH="$PWD" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python tools/validate_gpu_probe_resources.py \
    --gpus 4 --rounds 2 --output /path/to/new-validation-directory
```

Use idle devices and a new output directory. Each task reserves a full GPU.
The script adds a 0.1-second allocation hold for RSS sampling, so its times are
diagnostic and do not establish real-model throughput improvements. The OOM
checks preserve preflight failure semantics; they do not implement recovery.

On 2026-09-16, validation passed on PPU-ZW810 devices with 96 GiB per device,
Python 3.12, PyTorch 2.11.0 (CUDA API 13.0), and Ray 2.55.1:

- Two GPUs, one off/on pair: two distinct assigned devices and two overlapping
  targets; cache reuse, ordered output equality, and real allocation OOM checks
  passed.
- Eight GPUs, two counterbalanced off/on pairs: both sampled runs used eight
  distinct assigned devices with eight overlapping targets. No samples were
  dropped and no sampling errors were recorded.
- Allocator reserved peaks and GPU reservations matched between off/on pairs.
  Probe wall-time medians were 7.251 s off and 7.312 s on for the eight-GPU
  runs. Two pairs and the synthetic allocation hold are insufficient to infer
  general overhead or real-model throughput gains.
- 290 focused regression tests covering sampling, GPU probing, configuration,
  resource planning, partition concurrency, resume and partition sizing passed.

These are CUDA-compatible PPU results, not NVIDIA compatibility results or a
real-model benchmark. The GPU checks exercise the preflight and ordered worker
paths, not an entire GPU pipeline through checkpoint recovery.

## Selective port

This change adapts ElasticJuicer commit
`4274a9d` (`feat(elasticjuicer): sample actor-owned resources`) into PR1054's
existing probe. It relocates the sampler under `core/executor`, removes the
experimental metrics-schema dependency, adds CPU and Ray assignment evidence,
and avoids resetting allocator peaks used by the main repository's planner.
It can be merged independently of adaptive microbatch/OOM runtime PR #1070.

Batch experience persistence, cost-aware batch adjustment and execution-group
actor replanning remain separate follow-up work.

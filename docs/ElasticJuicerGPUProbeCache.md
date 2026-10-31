# Reusing GPU preflight reports across job work directories

GPU preflight reports contain measured memory requirements and throughput used
by the partitioned Ray executor's startup planner. Config initialization adds
the current job's `work_dir` to every operator. Hashing that output directory
made an otherwise compatible report miss when imported into a new job.

The probe configuration hash now excludes only the top-level `work_dir` inside
each operator's configuration. It does not modify the live configuration.
Batch size, model options, precision, resource settings and other constructor
arguments continue to participate in the hash. Existing operator identity,
sampling-policy, report-schema and GPU hardware checks still apply.

## Import and compatibility

The executor reads `gpu_probe_results.json` from its current job work directory.
Report placement remains explicit: a caller can copy an appropriate report
into that directory before preflight. This change adds no cross-job cache
discovery, and a copied report is still subject to the existing validation.
Use reports measured for the intended input workload; this change does not
introduce dataset-content or model-file-content hashing.

Reports generated before this fix with a `work_dir` in their configuration hash
are re-probed once because their hashes differ from the normalized identity.
The report format remains unchanged.

This startup measurement cache is separate from the job-scoped adaptive batch
profiles in [ElasticJuicerStageProfile.md](ElasticJuicerStageProfile.md).

## Focused regression coverage

```bash
PYTHONPATH=.:tests python -m pytest -q tests/core/executor/test_gpu_memory_probe.py
```

The added tests import a report between two job directories and assert that no
sample rows are requested or model stage is run. They also assert that changing
batch size, model path, precision or memory budget triggers a fresh probe.
All 54 tests in this file passed on the port to upstream main
`ae85f6916ab9bd132a187a15b8c1cb5269e20921` with Python 3.10.19.
The test host disabled its unavailable optional cuDF import; the probes use
test doubles and do not allocate physical GPUs.

## Completed real-model evidence

The original fix was tested at product commit
`fa5eee0d34430f2333f392845783958dab64886c`, using frozen benchmark harness
`943cddf777aa56f6c53dc053483cea2fc87fbaa1`. These are historical real-machine
runs, not a new GPU rerun of the port to current main.

The pipeline used CPU image metadata, custom batched QualiCLIP RN50 quality
scoring and DINOv2-large embedding Mappers with trained local weights and real
qwen-image inputs, on eight NVIDIA H20 GPUs. QualiCLIP used 224-pixel inputs;
DINOv2 used 896 pixels for the heavy cohorts and 448 for W0. Reports were
explicitly imported by the harness. Adaptive StageProfile seeding was disabled.

| Cohort | Cache runs | Cache/static paired median wall ratio | Bootstrap 95% interval |
| --- | ---: | ---: | --- |
| Heavy, 2,048 images | 10 | 1.0325 | [0.9927, 1.0766] |
| Heavy, 10,000 images | 10 | 1.0243 | [0.9907, 1.0461] |
| Heavy, 20,000 images | 10 | 1.0196 | [1.0132, 1.0490] |
| W0, 2,048 images | 10 | 1.0382 | [1.0006, 1.0571] |

Each of the 40 cache runs reused both model records, executed zero model
preflight attempts and recorded successful work on all eight GPU UUIDs. Output
IDs and full quality/embedding values were checked against the corresponding
fixed-plan references; recorded maximum absolute differences were zero.

Ratios below one indicate less wall time. The heavy static baseline was chosen
from a batch-size grid at a fixed actor allocation, not a global layout optimum.
The observed cache/static medians satisfy the predeclared 1.05 gate, but the
confidence intervals do not all lie below 1.05. These results support cache reuse
and near-baseline startup overhead; they do not establish an overall speedup,
runtime actor replanning or completion of the broader system matrix.

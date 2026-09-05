# Elastic Multi-Node Sharding on Shared Storage

## Overview

This demo adds a "shard across nodes, use Ray inside each node" execution mode
for large JSONL datasets:

1. Pre-split the complete input into a fixed, deterministic set of JSONL
   shards.
2. Let independent Workers dynamically claim shards through a shared
   POSIX/NAS/CPFS directory.
3. Process each claimed shard with Data-Juicer's `ray` executor on the claiming
   node.
4. Publish validated completion metadata and let an idle Worker claim another
   shard.
5. Validate and merge all successful shard outputs in their original order.

This design does not require a cross-node Ray cluster. The shared filesystem
coordinates nodes, while every node has its own independent Ray runtime.

The main package now exposes this capability as `dj-process-sharded`. Run it
once on a single machine, or have a scheduler run the same command with the
same `job-dir`, `run-id`, and arguments on every node:

```bash
dj-process-sharded run \
  --config /mnt/shared/recipes/process.yaml \
  --job-dir /mnt/shared/data-juicer-jobs/job-001 \
  --num-shards 32 \
  --run-id submission-001
```

The command checks the recipe first. Mapper/Filter-only recipes with local
JSONL input and output use elastic sharding. Recipes containing whole-dataset
operators such as Deduplicator, Selector, Grouper, Aggregator, or Pipeline—or
unsupported input/output settings—print the reason and run one original
`dj-process` invocation on the elected coordinator. Runtime failures after
sharding starts never fall back, preventing duplicate full-dataset work.

Inspect a sharded job with `dj-process-sharded status --job-dir <job-dir>`.
Requeue terminal failures with
`dj-process-sharded retry --job-dir <job-dir> --all-failed`, then invoke `run`
with a new `run-id`. Arguments for the same `job-dir/run-id` must not change.

> **DLC launch topology matters.** `dlc_job.py dlc` requires a job type that
> broadcasts the configured startup command to every Worker. A PAI-DLC MPIJob
> does not do that: its command runs on the Launcher, which must use `mpirun`
> and DLC's `/etc/mpi/hostfile` to start one process on every GPU Worker. In
> both cases each GPU Worker still uses its own node-local Ray runtime.

```text
                 Worker-broadcast DLC job submission
                                  |
             +--------------------+--------------------+
             |                    |                    |
       DLC Worker 0         DLC Worker 1         DLC Worker N
       claim shard A        claim shard B        claim shard C
       node-local Ray       node-local Ray       node-local Ray
             |                    |                    |
             +--------------------+--------------------+
                                  |
                        shared NAS/CPFS job-dir
                manifest / shards / locks / done / attempts
                                  |
                       validate and ordered merge
```

## Key advantages

- **One submission for multiple nodes**: use either Worker broadcast or an
  MPIJob Launcher with `mpirun`; neither requires logging in to every machine.
- **Dynamic load balancing**: a Worker claims another shard after finishing its
  current one, so faster nodes naturally process more work.
- **Ray remains available inside every node**: every shard is processed by the
  Data-Juicer Ray executor using that node's CPU/GPU resources.
- **No rank dependency**: coordination does not require `RANK`, `WORLD_SIZE`,
  or a static hostname-to-file mapping.
- **Auditable and reproducible**: the manifest records input fingerprints,
  recipe hash, Data-Juicer commit, Ray configuration, and shard order.
- **Exclusive claims**: POSIX `O_CREAT|O_EXCL` prevents two active Workers from
  owning the same shard. A successful or terminally failed claim remains at
  the same path as a durable fence, so a short NAS/CPFS metadata-visibility
  delay cannot reopen an already finished shard.
- **Failure handling**: failed attempts can retry, and expired claims can be
  reclaimed by another Worker.
- **Integrity validation**: row counts, byte counts, and SHA256 values are
  checked before a result is accepted and merged.
- **Deterministic order**: directory inputs are sorted, shards are contiguous,
  and merge follows manifest order.
- **Low integration risk**: the state machine lives in
  `tools/elastic_sharding.py` and does not modify existing executors, recipe
  schemas, or operators; this directory retains scheduler examples and a
  compatibility wrapper.

## Comparison with existing approaches

| Approach | Splitting | Dynamic claims | Recovery | Ray inside node | Cross-node Ray |
| --- | --- | --- | --- | --- | --- |
| `tools/data_resplit.py` | Pre-split | No | No | User-managed | No |
| `ray_partitioned` | Runtime | Ray scheduling | Ray job | Yes | Yes |
| `dj-process-sharded` | Pre-split | Shared filesystem | Timeout/retry | Yes | No |

This demo is useful when:

- the input is large and fixed, inspectable shards are desirable;
- DLC Workers share NAS/CPFS but should not form one Ray cluster;
- nodes have different speeds or may be restarted;
- per-shard logs, attempts, ownership, and verifiable outputs are required.

## Files

```text
demos/elastic_sharding/
├── shard_job.py             # Generic prepare/worker/status/retry/merge CLI
├── dlc_job.py               # Launcher for DLC job types that broadcast to Workers
├── two_node_test.py         # Backward-compatible strict two-node wrapper
├── configs/
│   ├── demo.yaml            # CPU-only Mapper/Filter recipe
│   ├── gpu_demo.yaml        # One-GPU-per-node CPU + GPU smoke test
│   └── gpu_demo_4gpu.yaml   # Single-node, four-GPU smoke test
├── data/
│   └── gpu-demo-dataset.jsonl
├── README.md
└── README_ZH.md
```

`shard_job.py` is a compatibility wrapper for the main
`dj-process-sharded` command. For
Worker-broadcast job types, `dlc_job.py` coordinates one-time preparation and
finalization around any number of DLC Workers. `two_node_test.py` keeps the
original strict two-node defaults for backward compatibility. An MPIJob
Launcher should instead call the lower-level `prepare`, `worker`, and `merge`
commands around an `mpirun -np N -npernode 1` worker launch.

## Important path concepts

Do not confuse these three paths:

- `--config`: the Data-Juicer YAML recipe.
- `--dataset-path`: the JSONL file or directory to pre-split. When specified,
  it overrides `dataset_path` in the recipe.
- `--job-dir`: the shared working directory containing shards, locks, attempts,
  logs, states, and results. It is not the original dataset directory.

Example:

```bash
python demos/elastic_sharding/dlc_job.py dlc \
  --config /mnt/shared/recipes/my_process.yaml \
  --dataset-path /mnt/shared/input/my_dataset.jsonl \
  --job-dir /mnt/shared/data-juicer-jobs/my-job-001 \
  --nodes 4 \
  --num-shards 16
```

## Requirements and current scope

- All Workers must run the same Data-Juicer version and dependencies.
- The job directory must be on a shared POSIX filesystem supporting atomic
  create, atomic rename, hard links, and `fcntl` advisory locks, such as a
  normally configured NAS/NFS/CPFS.
- Every Worker must see the job directory, input JSONL, and local media at
  identical paths.
- Inputs are local `.jsonl` files or local directories recursively containing
  `.jsonl` files.
- Every line must be a UTF-8 JSON object. Blank lines, JSON arrays as rows, and
  malformed JSON are rejected.
- Only shard-independent Mapper and Filter operators are currently accepted.
- Deduplicators, Selectors, Groupers, Aggregators, Pipelines, and other
  whole-dataset operations are rejected during `prepare`.
- Claims use a static timeout without a heartbeat. The timeout must be longer
  than the longest expected shard runtime.
- The job directory stores normalized shards and attempt results, so reserve
  enough capacity. Media files are referenced by path and are not copied.

## PAI-DLC Worker-broadcast quick start

This section is **not** the MPIJob launch procedure. Use it only after
confirming that the selected DLC job type runs the startup command on every
Worker. For MPIJob, configure the command once on the Launcher and have that
command run preparation, `mpirun` one process per Worker, and merge.

### 1. Configure the DLC job

Create one DLC job with:

- Framework: a `PyTorch`-style job that starts the user command on every
  Worker, without `torchrun`.
- Worker count: any positive number, for example `4`, with one script process
  per Worker.
- Do not select DLC's `Ray` framework for this demo. DLC Ray creates a
  cross-node Ray cluster, while this design intentionally uses independent
  node-local Ray runtimes.
- Mount the same Data-Juicer code at the same path on all Workers, for example
  `/mnt/data/data-juicer`.
- Mount the same read-write NAS/CPFS at the same path, for example
  `/mnt/shared`.
- Use the same image with Data-Juicer's Ray dependencies on all Workers.

Only one startup command and a Worker count are configured on the PAI-DLC job
page; the selected job type must dispatch that command to every Worker. See the official
[Create a training job](https://help.aliyun.com/zh/pai/create-a-training-task)
guide.

### 2. Enter one startup command

Enter this command once in the DLC job configuration:

```bash
cd /mnt/data/data-juicer && \
python demos/elastic_sharding/dlc_job.py dlc \
  --job-dir /mnt/shared/data-juicer-jobs/multi-node-job-001 \
  --nodes 4 \
  --num-shards 16 \
  --ray-address local
```

The defaults are:

- recipe: `demos/elastic_sharding/configs/demo.yaml`;
- input: `demos/data/demo-dataset.jsonl` from that recipe;
- shard count: 4 unless explicitly set;
- expected DLC Workers: not enforced in the default elastic mode;
- Ray: independent `local` mode on every node;
- merged result: `<job-dir>/merged.jsonl`.

### 3. Automatic workflow

The selected Worker-broadcast job type runs the same `dlc` entry point on
every Worker:

1. Atomically elect one prepare coordinator through shared storage.
2. Validate the input and recipe, then pre-split exactly once.
3. Let every live Worker claim shards without a per-Worker cap.
4. Process each shard through `tools/process_data.py` with
   `--executor_type ray --ray_address local`.
5. Wait for all shards to complete or reach terminal failure.
6. Atomically elect one finalize coordinator.
7. Report participating hostnames, validate outputs, and merge.
8. Let every other instance read the same final result and exit with the same
   code.

Useful successful log messages include:

```text
elected as DLC prepare coordinator
Starting DLC worker on hostname=...
elected as DLC finalize coordinator
PASS: 16 shards were completed by 4 node(s)
```

Inspect status from any environment that mounts the job directory:

```bash
python demos/elastic_sharding/dlc_job.py status \
  --job-dir /mnt/shared/data-juicer-jobs/multi-node-job-001
```

The merged output is:

```text
/mnt/shared/data-juicer-jobs/multi-node-job-001/merged.jsonl
```

### Elastic mode and strict participation mode

The default is **elastic mode**:

- `--nodes` is optional and informational;
- Workers have no claim cap;
- if fewer Workers start than requested, the live Workers can still claim all
  remaining shards;
- finalization requires completed shards, not a specific Worker count.

This is the recommended production behavior:

```bash
python demos/elastic_sharding/dlc_job.py dlc \
  --job-dir /mnt/shared/data-juicer-jobs/elastic-job-001 \
  --num-shards 32 \
  --ray-address local
```

Use **strict participation mode** only when testing that every configured DLC
Worker actually processed at least one shard:

```bash
python demos/elastic_sharding/dlc_job.py dlc \
  --job-dir /mnt/shared/data-juicer-jobs/strict-job-001 \
  --nodes 4 \
  --num-shards 16 \
  --require-all-nodes \
  --ray-address local
```

Strict mode sets a per-Worker claim cap and verifies at least `--nodes`
distinct completion hostnames. A missing Worker therefore causes the strict
job to wait and eventually fail. Use a shard count that is a multiple of the
node count.

## GPU smoke test: one recipe with CPU and GPU operators

`configs/gpu_demo.yaml` verifies that a claimed shard can move through both
CPU and GPU operators inside the node-local Ray executor:

| Order | Operator | Resource requested from Ray |
| --- | --- | --- |
| 1 | `whitespace_normalization_mapper` | 1 CPU |
| 2 | `query_sentiment_detection_mapper` | 1 CPU + 1 GPU |
| 3 | `query_topic_detection_mapper` | 1 CPU + 1 GPU |
| 4 | `text_pair_similarity_filter` | 1 CPU + 1 GPU |
| 5 | `text_length_filter` | 1 CPU |

The two GPU Mappers write sentiment and topic labels into `meta`. The GPU
Filter uses `openai/clip-vit-base-patch32` and writes
`text_pair_similarity` into `__dj__stats__`. All three GPU operators set
`num_gpus: 1`, so Ray must schedule their tasks on a real GPU; this is not a
CPU-fallback test.

Before submitting the job:

- give every DLC Worker at least one visible NVIDIA GPU;
- install the CUDA-enabled PyTorch, Ray, Transformers, and Data-Juicer
  dependencies in the image;
- make these three models available to every Worker:
  `mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis`,
  `dstefa/roberta-base_topic_classification_nyt_news`, and
  `openai/clip-vit-base-patch32`;
- the first run downloads the models from Hugging Face unless the corresponding
  recipe fields are changed to pre-downloaded paths;
- prefer models baked into the image or a pre-populated cache for a
  multi-node test, rather than making all Workers download them concurrently.

Check one Worker image first:

```bash
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
```

### Single-node, four-GPU command

Use `gpu_demo_4gpu.yaml` when one machine has four visible GPUs:

```bash
cd /mnt/data/data-juicer && \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python demos/elastic_sharding/dlc_job.py dlc \
  --config demos/elastic_sharding/configs/gpu_demo_4gpu.yaml \
  --job-dir /mnt/shared/data-juicer-jobs/gpu-4card-smoke-001 \
  --nodes 1 \
  --num-shards 1 \
  --ray-address local \
  --output /mnt/shared/data-juicer-jobs/gpu-4card-smoke-001/merged.jsonl
```

The four-GPU recipe uses `override_num_blocks: 4`, `batch_size: 1`,
`num_proc: 4`, and `num_gpus: 1` for each GPU operator. One node-local Ray
runtime can therefore schedule four one-GPU tasks concurrently.

Keep `--num-shards 1` for this four-row smoke test. A single DLC Worker claims
shards sequentially, so `--num-shards 4` would create four one-row Ray jobs and
would normally exercise only one GPU at a time. For a larger input, keep each
shard large enough to contain at least four Ray blocks.

You can observe placement in another terminal:

```bash
watch -n 1 nvidia-smi
```

Because the bundled input is tiny, GPU utilization may be brief. A larger
`--dataset-path` is better for sustained utilization and throughput
measurements.

### Multi-node, one-GPU-per-node command

For a strict two-GPU-node DLC smoke test, configure two Workers and enter this
single startup command once:

```bash
cd /mnt/data/data-juicer && \
python demos/elastic_sharding/dlc_job.py dlc \
  --config demos/elastic_sharding/configs/gpu_demo.yaml \
  --job-dir /mnt/shared/data-juicer-jobs/gpu-smoke-001 \
  --nodes 2 \
  --num-shards 4 \
  --require-all-nodes \
  --ray-address local \
  --output /mnt/shared/data-juicer-jobs/gpu-smoke-001/merged.jsonl
```

The bundled dataset has four rows, so it supports at most four non-empty
shards. To test more nodes or realistic throughput, pass a larger JSONL with
both `text` and `target_text` fields:

```bash
python demos/elastic_sharding/dlc_job.py dlc \
  --config demos/elastic_sharding/configs/gpu_demo.yaml \
  --dataset-path /mnt/shared/input/text-pairs.jsonl \
  --job-dir /mnt/shared/data-juicer-jobs/gpu-large-smoke-001 \
  --nodes 8 \
  --num-shards 32 \
  --require-all-nodes \
  --ray-address local
```

After completion, verify ownership and the GPU-generated metadata/statistics:

```bash
python demos/elastic_sharding/dlc_job.py status \
  --job-dir /mnt/shared/data-juicer-jobs/gpu-smoke-001

rg -n 'query_(sentiment|topic)_label|text_pair_similarity' \
  /mnt/shared/data-juicer-jobs/gpu-smoke-001/merged.jsonl
```

Each Worker handles one shard attempt at a time. The bundled recipe deliberately
sets `ray_execution_mode: task`, `num_proc: 1`, and `num_gpus: 1` for every GPU
operator. Their Ray tasks can therefore reuse a single GPU on each node instead
of requiring three GPUs concurrently. For a production recipe, tune operator
concurrency, GPU fractions, batch size, model size, and shard size for the
hardware.

## Important: use an existing Mapper/Filter recipe and your own JSONL

The demo is not limited to the bundled recipe. The intended real-world usage is
to reuse an existing Data-Juicer Mapper/Filter recipe and point
`--dataset-path` at a large user-owned JSONL dataset.

### Recipe compatibility rules

1. `executor_type` may be `default` or `ray`; Workers always override it with
   `ray`.
2. The recipe must contain an explicit `process` list.
3. Every operator must resolve to a Mapper or Filter and must not be a global
   operation.
4. Operators must not set `stats_export_path`, because different shards would
   collide on the same statistics file.
5. A fixed `save_dir` is allowed with a warning; users must ensure generated
   filenames cannot collide across shards.
6. `custom_operator_paths` is supported. Relative paths resolve from the
   Data-Juicer repository root, and custom operators must still inherit Mapper
   or Filter.
7. When an operator declares `index_key`, preparation fills missing values
   with the global input row index. Existing values remain unchanged.

Generally suitable operations include:

- per-sample text cleanup Mappers;
- per-sample text, image, audio, or video property Mappers;
- Filters based only on the current sample and its computed statistics;
- custom Mappers/Filters with no cross-sample state or shared fixed outputs.

Operations that are not directly shard-safe include:

- global deduplication;
- Selectors requiring global ordering or sampling;
- Groupers, Aggregators, and Pipelines;
- operators with cross-sample state or one global output.

Example recipe:

```yaml
project_name: my-elastic-job
dataset_path: /mnt/shared/input/default.jsonl
export_path: /mnt/shared/output/ignored-by-shard-worker.jsonl
executor_type: ray
ray_address: local

text_key: text
image_key: images
audio_key: audios
video_key: videos

process:
  - whitespace_normalization_mapper:
      text_key: text
  - text_length_filter:
      text_key: text
      min_len: 10
      max_len: 10000
```

For every claim, the Worker overrides the recipe's `dataset_path` with the
current shard and overrides `export_path` with the isolated attempt output.
The same recipe can therefore be safely reused for all shards.

### Requirements for your JSONL

Single-file input:

```text
/mnt/shared/input/my_dataset.jsonl
```

Directory input:

```text
/mnt/shared/input/my_dataset/
├── 000.jsonl
├── 001.jsonl
└── nested/
    └── 002.jsonl
```

Directories are scanned recursively and sorted by relative path. Each line
must be an object:

```json
{"id": 1, "text": "example", "images": ["media/1.jpg"]}
```

Media path behavior:

- absolute paths are preserved;
- `http://`, `https://`, `s3://`, `gs://`, and `hdfs://` values are preserved;
- relative paths become absolute:
  - relative to the JSONL's parent for single-file input;
  - relative to the dataset root for directory input;
- `images`, `audios`, and `videos` must be string lists or `null`. Change their
  names through `image_key`, `audio_key`, and `video_key` in the recipe.

### Run your recipe and input on any number of broadcast-started DLC Workers

The Worker-broadcast DLC job still contains only one configured startup
command:

```bash
cd /mnt/data/data-juicer && \
python demos/elastic_sharding/dlc_job.py dlc \
  --config /mnt/shared/recipes/my_process.yaml \
  --dataset-path /mnt/shared/input/my_dataset.jsonl \
  --job-dir /mnt/shared/data-juicer-jobs/my-dataset-001 \
  --nodes 4 \
  --num-shards 4 \
  --ray-address local \
  --output /mnt/shared/output/my_dataset.processed.jsonl
```

In this command:

- `--config` reuses the existing Mapper/Filter recipe;
- `--dataset-path` overrides the recipe's original input;
- `--job-dir` stores all state and intermediate results for this run;
- `--output` selects the final merged JSONL;
- four live Workers dynamically claim four shards and use Ray inside each
  node.

Start with a small sample and one shard per node, then scale the input after
the smoke test succeeds.

### Choosing the shard count

- `1 <= num_shards <= total JSONL rows` must hold.
- For best throughput, keep the shard count no greater than the Worker node
  count, ideally equal to it: one shard per node.
- Fewer shards leave nodes idle. More shards make a Worker process multiple
  shards sequentially and start another Data-Juicer process for each shard.
- Use more shards than nodes only for significant workload skew or finer retry
  granularity, and make each shard heavy enough to amortize process and
  node-local Ray startup.
- A shard should complete well before `lock_timeout_secs`.
- Too many shards also increase Ray startup, metadata, and small-file overhead.
- Default elastic mode has no per-Worker cap, so the shard count does not need
  to be a multiple of the Worker count.
- With `--require-all-nodes`, set `--num-shards` equal to `--nodes`.

### When the recipe or input changes

Preparation records the recipe SHA256, input file SHA256 values, sizes, mtimes,
row counts, normalized-content SHA256, and Data-Juicer commit.

- Repeating an identical request is an idempotent no-op.
- Do not reuse an old job directory after changing the recipe, input, shard
  count, or Ray address.
- Use a new directory for each independent run, such as `my-dataset-001` and
  `my-dataset-002`.

## `dlc_job.py` parameter reference

### `dlc`

Runs the complete Worker-broadcast DLC workflow.

| Option | Required | Default | Meaning |
| --- | --- | --- | --- |
| `--job-dir` | Yes | None | Shared job directory used by every Worker |
| `--config` | No | `configs/demo.yaml` | Existing shard-safe Data-Juicer recipe |
| `--dataset-path` | No | Recipe value | Override with a JSONL file or directory |
| `--nodes` | No | Not enforced | Informational in elastic mode; required in strict mode |
| `--num-shards` | No | `4` | Exact number of non-empty shards |
| `--require-all-nodes` | No | false | Cap claims and require all `--nodes` hostnames |
| `--ray-address` | No | `local` | Ray address used independently in each node |
| `--output` | No | `<job-dir>/merged.jsonl` | Final merged JSONL |
| `--run-id` | No | DLC Job ID | Identifier shared by Workers in one submission |
| `--wait-timeout-secs` | No | `126000` (35h) | Maximum prepare/completion/finalize wait |
| `--poll-interval-secs` | No | `2` | DLC coordination polling interval |

`--wait-timeout-secs` controls cross-instance DLC coordination. It is different
from the per-shard claim timeout.

The launcher reads the submission identity from `PAI_JOB_ID`, `DLC_JOB_ID`, or
`JOB_ID`. Outside DLC, pass the same `--run-id` to every Worker and use a new
value for each new submission. This keeps terminal coordination state from one
submission out of later submissions that reuse the same `job-dir`.

### Other wrapper subcommands

| Command | Option | Default and meaning |
| --- | --- | --- |
| `prepare` | `--job-dir` | Required shared directory |
|  | `--config` | Bundled demo recipe |
|  | `--dataset-path` | Optional recipe input override |
|  | `--num-shards` | `4` |
|  | `--ray-address` | `local` |
| `worker` | `--job-dir` | Required |
|  | `--max-shards` | Unlimited unless explicitly set |
|  | `--ray-address` | Optional manifest override |
| `status` | `--job-dir` | Required; prints all shards |
| `verify` | `--job-dir` | Required |
|  | `--output` | Defaults inside the job directory |
|  | `--expect-nodes` | `1`; minimum distinct completion hostnames |

## Complete `shard_job.py` parameter reference

### `prepare`

Validate the recipe, scan the input, and atomically publish a prepared job.

| Option | Required | Default | Meaning |
| --- | --- | --- | --- |
| `--config` | Yes | None | Data-Juicer YAML recipe |
| `--dataset-path` | No | Recipe value | Override input JSONL file/directory |
| `--job-dir` | Yes | None | New shared POSIX job directory |
| `--num-shards` | Yes | None | Exact number of non-empty shards |
| `--lock-timeout-secs` | No | `126000` | Per-shard claim timeout stored in manifest |
| `--max-retries` | No | `3` | Retries after the first failure; four total failures |
| `--poll-interval-secs` | No | `20` | Worker wait when no claim is available |
| `--ray-address` | No | `local` | Ray address stored for Workers |

### `worker`

Continuously claim and process shards.

| Option | Required | Default | Meaning |
| --- | --- | --- | --- |
| `--job-dir` | Yes | None | Shared directory created by prepare |
| `--max-shards` | No | Unlimited | Maximum claims; failed attempts also count |
| `--lock-timeout-secs` | No | Manifest value | Override claim timeout for this Worker |
| `--max-retries` | No | Manifest value | Override failure retries for this Worker |
| `--poll-interval-secs` | No | Manifest value | Override no-claim polling interval |
| `--ray-address` | No | Manifest value | Override Ray address for this Worker |
| `--allow-version-mismatch` | No | false | Allow a Worker commit mismatch intentionally |

Without `--max-shards`, a Worker keeps claiming until all shards complete or
the job reaches terminal failure.

### `status`

Read job state without changing it.

| Option | Required | Default | Meaning |
| --- | --- | --- | --- |
| `--job-dir` | Yes | None | Job directory |
| `--lock-timeout-secs` | No | Manifest value | Only affects whether locks display as stale |
| `--json` | No | false | Print machine-readable JSON |
| `--all` | No | false | Print every shard and owner in text mode |

### `retry`

Archive terminal failure state and requeue shards. Select exactly one mode:
`--all-failed` or one or more `--shard-id` options.

| Option | Required | Default | Meaning |
| --- | --- | --- | --- |
| `--job-dir` | Yes | None | Job directory |
| `--all-failed` | Conditional | false | Requeue every failed shard |
| `--shard-id` | Conditional | None | Requeue this ID; may be repeated |

Old failure metadata, attempts, and the terminal claim fence move into
`state/history`; retry does not overwrite history. Removing that fence is the
final atomic requeue step.

### `merge`

Revalidate and merge all completed results in manifest order.

| Option | Required | Default | Meaning |
| --- | --- | --- | --- |
| `--job-dir` | Yes | None | Fully completed job |
| `--output` | Yes | None | Final JSONL path |
| `--lock-timeout-secs` | No | Manifest value | Used while calculating pre-merge status |
| `--overwrite` | No | false | Replace an existing output |

## Generic manual or scheduler workflow

Use the lower-level CLI when a scheduler other than DLC launches the Workers.

### 1. Prepare

```bash
python demos/elastic_sharding/shard_job.py prepare \
  --config /mnt/shared/recipes/my_process.yaml \
  --dataset-path /mnt/shared/input/my_dataset.jsonl \
  --job-dir /mnt/shared/data-juicer-jobs/my-job-001 \
  --num-shards 16 \
  --lock-timeout-secs 126000 \
  --max-retries 3 \
  --poll-interval-secs 20 \
  --ray-address local
```

Preparation uses two streaming passes:

- pass one validates JSONL, fingerprints inputs, normalizes media paths, and
  assigns missing global `index_key` values;
- pass two writes contiguous, non-empty shards approximately balanced by
  normalized bytes;
- input changes between passes are detected;
- the complete stage directory is atomically renamed to `job-dir`.

### 2. Start a Worker on each node

Have the scheduler start this on every node:

```bash
python demos/elastic_sharding/shard_job.py worker \
  --job-dir /mnt/shared/data-juicer-jobs/my-job-001
```

### 3. Inspect status

```bash
python demos/elastic_sharding/shard_job.py status \
  --job-dir /mnt/shared/data-juicer-jobs/my-job-001 --all

python demos/elastic_sharding/shard_job.py status \
  --job-dir /mnt/shared/data-juicer-jobs/my-job-001 --json
```

States are:

- `pending`: not claimed;
- `running`: has a non-expired claim;
- `stale`: claim age exceeds the timeout and the next claimant may reclaim it;
- `committing`: the claim is terminal but the terminal marker is not yet
  visible to this filesystem client;
- `conflict`: marker/claim metadata disagree; Workers stop instead of risking
  duplicate processing;
- `done`: validated completion metadata is published;
- `failed`: terminal failure after the retry limit.

### 4. Requeue failures

After fixing the root cause:

```bash
python demos/elastic_sharding/shard_job.py retry \
  --job-dir /mnt/shared/data-juicer-jobs/my-job-001 \
  --all-failed
```

Or select shards:

```bash
python demos/elastic_sharding/shard_job.py retry \
  --job-dir /mnt/shared/data-juicer-jobs/my-job-001 \
  --shard-id part-00003-of-00016 \
  --shard-id part-00007-of-00016
```

Then start Workers again.

### 5. Merge

```bash
python demos/elastic_sharding/shard_job.py merge \
  --job-dir /mnt/shared/data-juicer-jobs/my-job-001 \
  --output /mnt/shared/output/my_dataset.processed.jsonl
```

Merge parses every JSONL row again and checks the row count and SHA256 from
completion metadata. It refuses an existing output unless `--overwrite` is
explicit.

## Ray execution modes

### Default: local Ray per attempt

```bash
--ray-address local
```

Every shard attempt starts an independent Ray instance, which is cleaned up
when the Data-Juicer process exits. This is simple and isolated, but adds a Ray
startup cost per attempt.

### Optional: persistent Ray head on each node

If the scheduler guarantees one Worker per node, run separately on each node:

```bash
ray start --head
python demos/elastic_sharding/shard_job.py worker \
  --job-dir /mnt/shared/data-juicer-jobs/my-job-001 \
  --ray-address auto
ray stop
```

Do not let every node's `auto` resolve to one shared Ray cluster; that changes
the intended topology.

## Job directory layout

```text
my-job-001/
├── manifest.json
├── recipe.yaml
├── shards/
│   └── part-xxxxx-of-xxxxx.jsonl
├── cache/
├── attempts/<shard-id>/<attempt-id>/
│   ├── attempt.json
│   ├── process.log
│   ├── logs/
│   ├── checkpoints/
│   ├── partitions/
│   ├── ray-output.jsonl/
│   └── processed.jsonl
├── state/
│   ├── locks/
│   ├── stale_locks/
│   ├── done/
│   ├── failed/
│   └── history/
│       ├── failed/
│       ├── attempts/
│       └── claims/
└── merge.json
```

The one-command DLC entry point also creates a sibling directory:

```text
.<job-dir-name>.dlc-coordination/
└── <submission-id-hash>/
    ├── prepare.lock
    ├── prepare-result.json
    ├── abort.lock
    ├── abort.json
    ├── finalize.lock
    └── finalize-result.json
```

If `XDG_CACHE_HOME` or `HF_HOME` is not explicitly set, Workers use
`<job-dir>/cache`.

## Failure semantics

- A shard has one visible claim, although stale takeover can briefly overlap
  an old attempt and a new attempt.
- Successful and terminally failed claims are rewritten in place to `done` or
  `failed` and retained as fences. Only explicit `retry` archives a failed
  terminal fence and makes that shard claimable again.
- Every attempt has an isolated directory and cannot overwrite another result.
- Only the first successful atomic done publication is accepted.
- `max_retries=3` means three retries after the initial failure, allowing four
  failed attempts before terminal state.
- A shard reaching the retry limit publishes `state/failed`, and the job exits
  with code 2.
- Run `retry` only after fixing the root cause; it rejects non-terminal claims.
- Input or recipe changes require a new job directory, not `retry`.
- If a DLC prepare/finalize coordinator is killed before publishing its phase
  result, other instances wait until `wait_timeout_secs` and then fail. A new
  submission uses a new coordination generation and can safely retry with the
  same unchanged `job-dir`.

## Exit codes

- `0`: success; for a Worker, it may also mean `--max-shards` was reached.
- `1`: `status` found an incomplete job without terminal failure.
- `2`: parameter, input, recipe, version, or runtime error, or terminal shard
  failure.

Any nonzero Worker exit should fail the DLC job.

## Troubleshooting

### Preparation fails

Check:

- JSONL is UTF-8, contains one object per line, and has no blank lines;
- `num_shards` does not exceed the row count;
- the recipe has no global operator or `stats_export_path`;
- recipe and data relative paths resolve from the Data-Juicer repository root;
- a mismatched request is not reusing an old job directory.

### Worker or Ray fails

Inspect:

```text
<job-dir>/attempts/<shard-id>/<attempt-id>/process.log
<job-dir>/attempts/<shard-id>/<attempt-id>/attempt.json
```

Also check:

- Ray dependencies in the image;
- node CPU, GPU, shared memory, and temporary storage;
- the intended `local` or node-local `auto` Ray address;
- whether `lock_timeout_secs` is shorter than real shard runtime;
- whether every Worker uses the same Data-Juicer commit.

### DLC keeps waiting

Check:

- this is a Worker-broadcast job type, not an MPIJob whose command ran only on
  the Launcher;
- in strict mode, DLC Worker count equals `--nodes`; in elastic mode,
  `--nodes` may be omitted;
- every Worker started the same command;
- `job-dir` is one shared mount, not separate local directories with the same
  path string;
- every Worker resolved the same `--run-id` or DLC Job ID;
- the current generation's `prepare-result.json`, `abort.json`, and
  `finalize-result.json` below the sibling coordination directory;
- with `--require-all-nodes`, failed attempts did not consume the strict
  mode's per-Worker claim cap.

## Tests

```bash
python -m pytest -q tests/demos/test_elastic_sharding.py
```

Coverage includes:

- deterministic splitting, input ordering, and media path normalization;
- recipe safety validation and default-to-Ray override;
- concurrent exclusive claims and stale-lock takeover;
- isolated attempts, Ray command construction, and multi-file output
  materialization;
- retry limits, selected retry, and ordered merge;
- distinct two-host ownership verification;
- DLC prepare/finalize election and failure propagation.

See [README_ZH.md](README_ZH.md) for the Chinese guide.

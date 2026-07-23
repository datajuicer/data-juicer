# Elastic Multi-Node Sharding on Shared Storage

This demo pre-splits a large JSONL dataset into a fixed number of shards and
lets independent machines dynamically claim them through a shared POSIX/NAS
directory. After claiming a shard, each machine processes it with the
Data-Juicer `ray` executor. The default `ray_address=local` creates independent
node-local Ray instances.

Unlike `tools/data_resplit.py`, this demo also provides claims, stale-lock
recovery, retries, status, and ordered merge. Unlike `ray_partitioned`, there
is no single Ray driver coordinating all machines: the shared filesystem
coordinates machines, while Ray schedules work within each machine.

## Requirements and scope

- The job directory must be on a POSIX filesystem with atomic create/rename
  semantics and `fcntl` advisory locks, such as a normally configured NAS/NFS.
- Inputs are local JSONL files or directories containing JSONL files.
- Only shard-independent Mapper and Filter operators are accepted. Global or
  dataset-level operators are rejected during preparation.
- Every machine must have Data-Juicer's Ray dependencies installed. With the
  default `local` mode, run one worker per machine to avoid resource contention
  between multiple node-local Ray instances.
- Claims use a static timeout without heartbeats. Set the timeout longer than
  the longest expected shard runtime.
- All workers must access the same data and job directory and run the same
  Data-Juicer commit.

## Low-level commands (manual, non-DLC)

Run all commands from the Data-Juicer repository root:

DLC users can skip this section and use the single startup command in the next
section.

```bash
python demos/elastic_sharding/shard_job.py prepare \
  --config demos/elastic_sharding/configs/demo.yaml \
  --job-dir /shared/data-juicer-jobs/demo \
  --num-shards 4 \
  --ray-address local

# Run this same command on every machine.
python demos/elastic_sharding/shard_job.py worker \
  --job-dir /shared/data-juicer-jobs/demo

python demos/elastic_sharding/shard_job.py status \
  --job-dir /shared/data-juicer-jobs/demo --all

python demos/elastic_sharding/shard_job.py retry \
  --job-dir /shared/data-juicer-jobs/demo --all-failed

python demos/elastic_sharding/shard_job.py merge \
  --job-dir /shared/data-juicer-jobs/demo \
  --output /shared/data-juicer-results/demo.jsonl
```

`prepare` performs two streaming passes and creates the exact requested number
of non-empty, contiguous shards, approximately balanced by normalized byte
size. Relative paths in configured image, audio, and video fields are rewritten
against the original dataset root. Fields requested through an operator's
`index_key` are populated with the global input index when absent; existing
values are preserved.

## One-command DLC two-node smoke test

`two_node_test.py` is a thin integration-test wrapper. It uses the included
recipe and Data-Juicer's existing `demos/data/demo-dataset.jsonl` by default,
creates four shards, and caps each worker at two shards so both machines must
participate.

Create one DLC job with the following settings:

- Select the `PyTorch` framework and configure two Worker nodes. `torchrun` is
  not used; this choice makes DLC execute the startup command on each Worker.
  Do not select DLC's `Ray` framework for this test: it would create a
  cross-node Ray cluster, while this demo intentionally uses shared-storage
  coordination between nodes and an independent Ray runtime inside each node.
- Start one script process per Worker.
- Mount the same Data-Juicer code path on both Workers, for example
  `/mnt/data/data-juicer`.
- Mount one read-write NAS/CPFS directory at the same path, such as
  `/mnt/shared`, on both Workers.
- Use the same image with Data-Juicer's Ray dependencies on both Workers.

PAI-DLC requires only one startup-command configuration and a Worker count;
see the official
[Create a training job](https://help.aliyun.com/zh/pai/create-a-training-task)
guide.

Enter this startup command once in the DLC job configuration:

```bash
cd /mnt/data/data-juicer && \
python demos/elastic_sharding/two_node_test.py dlc \
  --job-dir /mnt/shared/data-juicer-jobs/two-node-test-001 \
  --nodes 2 \
  --num-shards 4 \
  --ray-address local
```

DLC starts that entry point on both Workers. The instances elect exactly one
prepare coordinator through the shared directory, process at most two shards
each with independent node-local Ray executors, wait for all four shards, and
elect exactly one finalizer. The finalizer checks that two distinct hostnames
participated and merges the output. Rank environment variables are not needed.

Useful successful log messages include:

```text
elected as DLC prepare coordinator
Starting test worker on hostname=...
elected as DLC finalize coordinator
PASS: 4 shards were completed by 2 node(s)
```

The default merged result is
`/mnt/shared/data-juicer-jobs/two-node-test-001/two-node-merged.jsonl`.
Both Workers in one DLC job must use the exact same `--job-dir`. Re-running an
already successful job is idempotent; use a new job directory after changing
the input or recipe, or when starting an independent test. `--nodes` must equal
the DLC Worker count. The command rejects shard/node combinations that cannot
force every expected node to process at least one shard.

Use `dlc --config <recipe> --dataset-path <jsonl>` to test another existing
Mapper/Filter recipe and input. The explicit `prepare`, `worker`, and `verify`
subcommands remain available for manual use or other schedulers.

Before submitting the DLC job, verify that:

- The job directory, input JSONL, and referenced local media paths are visible
  at identical paths on both Workers.
- The shard count does not exceed the input row count. Four shards is the
  recommended two-node smoke-test setting.
- `--ray-address local` is present. It starts Ray inside the node processing
  the claimed shard and never connects that node to the other Worker's Ray.
- A nonzero exit from either Worker fails the DLC job. Inspect
  `<job-dir>/attempts/*/*/process.log` and the sibling hidden coordination
  directory's `abort.json`.
- The default coordination wait is 35 hours. If a coordinator is forcibly
  terminated before publishing its phase result, the remaining Worker
  eventually exits with code 2; submit a new job directory for the next run.

Workers claim one shard at a time with atomic `O_CREAT|O_EXCL`, process it in
an isolated attempt directory, and then claim another shard. The defaults are
a 126000-second (35-hour) lock timeout and three retries after the initial
attempt. Use `--max-shards` to cap one worker invocation. Explicit Hugging Face
or XDG cache environment variables are honored; otherwise workers reuse the
cache inside the shared job directory.

The worker always overrides the processing command to `executor_type=ray`,
regardless of whether the source recipe says `default` or `ray`. The default
Ray address, `local`, starts an isolated Ray instance for every shard attempt.
If each machine already runs its own persistent Ray head, use
`--ray-address auto` during preparation or on the worker to reuse it and avoid
per-shard startup. Do not point every machine at one shared Ray address if the
intended topology is filesystem coordination between machines and Ray only
within each machine.

For persistent mode, run the following separately on every machine:

```bash
ray start --head
python demos/elastic_sharding/shard_job.py worker \
  --job-dir /shared/data-juicer-jobs/demo \
  --ray-address auto
# After this machine no longer processes shards:
ray stop
```

Ray writes a directory of output files for each shard. The worker validates
these files and materializes one canonical `processed.jsonl` before publishing
the shard's done record. Manifest schema version 2 records the Ray execution
settings; jobs prepared with schema version 1 must be prepared again.

`merge` is allowed only after every shard succeeds. It validates each output's
JSONL content, row count, and SHA256 before atomically publishing the ordered
result. Existing output is preserved unless `--overwrite` is explicit.

See [README_ZH.md](README_ZH.md) for the complete Chinese guide, state layout,
and exit-code reference.

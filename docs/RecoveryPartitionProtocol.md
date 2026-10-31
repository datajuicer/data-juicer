# Checkpoint commit protocol

This document describes how `ray_partitioned` checkpoints the leading
Mapper/Filter operators of a recipe (`checkpoint.strategy: every_op`). For
configuration and usage see [PartitionAndCheckpoint.md](PartitionAndCheckpoint.md).

## Data path

All operators run in one Ray Data plan, and rows move between them through
the Ray object store exactly as without checkpoints. Checkpointed stages pass
`batch_size=None` to Ray, so small blocks are not combined to reach
`partition.size`. Within each Ray block, workers form commits of at most
`partition.size` input rows; the operator's internal inference `batch_size`
is unchanged. Checkpointing is a side path: a worker writes its output to
storage, commits it, and only then yields it downstream. A downstream commit
therefore never references a row whose producing batch is not committed.

`RayBatchPlan` records typed batch operations, statistics exports, and global
materialization barriers. The primary operator declares its scheduling
resources explicitly; helper operations do not determine stage placement.
Native execution hands the operations to Ray Data. A checkpoint worker keeps
callable instances across batches and uses Ray's batch UDF transform for
invocation, generator handling, and output validation. The remaining private
Ray batching and whole-block scheduling calls are isolated in
`ray_batch_adapter.py`; they remain a Ray-version compatibility boundary.

## Lineage keys

Every row carries two internal columns while it is inside the plan:

- `__dj_ckpt_seq__`: the commit that produced the row, or `-1` for an input row;
- `__dj_ckpt_idx__`: the row's position within that commit (the input offset for
  input rows).

Before an operator runs, the worker records the batch's keys as compact
ranges `(seq, start, stop)` and drops the columns, so operators never see them.
After the commit, the output is tagged with the new commit's sequence number and
local positions. The columns are removed before the export.

## Commits

A commit is a small JSON manifest at `commits/{stage:04d}-{seq:016d}.json`
recording:

- the operator stage and its sequence number;
- the consumed key ranges;
- the produced Parquet files under `data/`, each with row count and SHA-256;
- the Filter statistics files, if the stage exports statistics.

Commits are not planned ahead. Their boundaries follow Ray blocks, split into
at most `partition.size` input rows per commit. A Mapper may expand rows and a
Filter may drop all of them; an empty output still commits its consumed ranges.

Before running the operator, a worker looks up a durable commit by the stage
and consumed lineage. If one exists, it reads and yields that commit's output
without running the operator. Concurrent attempts may both compute, but the
first to commit wins; later attempts reuse its output.

A Ray actor pinned to the driver node coordinates commits. It handles metadata
only; data files are written by the workers. Under a lock it checks the consumed
ranges against the stage's committed ranges:

- no overlap: a new sequence number is assigned and the manifest is written
  with a create-if-absent operation;
- an identical consumed set: the existing commit is returned, and the worker
  yields that commit's files instead of its own. This covers Ray task retries
  and lineage reconstruction, because consumers may already hold the first
  result;
- any other overlap: the run fails, since the same row would be counted twice.

If a manifest write fails, the coordinator stops accepting commits and the run
must be resumed.

## Storage

`LocalStore` writes to a temporary file, fsyncs it, publishes with `os.link`
(create if absent) or `os.replace`, and fsyncs the directory. `S3Store` uses
`boto3` `put_object` with `IfNoneMatch='*'`, which AWS S3 and S3-compatible
services such as MinIO implement as a conditional create. Other URI schemes are
rejected because they do not provide that guarantee here.

Cleanup removes only files owned by the failed attempt: partial table writes,
statistics written before an operator fails, and files of a losing duplicate
attempt. If a synchronous POSIX manifest write fails and the coordinator
confirms that its manifest is absent, it also removes that attempt's data and
statistics. If the manifest exists, or a remote write's outcome is ambiguous,
the files are retained. Cleanup never scans and deletes all unreferenced files:
another worker could still publish a reference to one. Process termination
and ambiguous remote failures can therefore leave orphan files; automatic
reclamation of those files requires a separate ownership/fencing protocol.

## Run identity

`run.json` records the protocol version, an order-sensitive fingerprint of the
input computed on Ray workers, the row count, the Arrow schema, a hash of the
checkpointed operators' configuration, and the stage count. The input is not
copied. A new run fails if commits already exist; a resume fails if the recorded
fingerprint, row count, schema or operator configuration does not match, or if
commits exist without `run.json`. The commit size is not part of the identity.

The caller must keep source-row identity and read order stable: source offset
`i` must still identify the same logical record on resume. The input fingerprint
is an additional diagnostic guard, not a dataset version or a guarantee of
byte-for-byte content equality. Non-JSON values use their string representation
(which can omit tensor elements), and referenced media or model files are not
hashed. Payload consistency is the caller's responsibility; changing the logical
dataset requires a new job ID/checkpoint directory.

## Resume

Manifest loading uses at most 32 concurrent storage `get` calls and registers
commits in stage order, giving each stage's committed consumed ranges. The
resumed plan is built stage by stage:

- the input of stage 0 is the input rows that stage 0 has not consumed;
- the input of stage `k` is the stream from stage `k - 1` plus the committed
  stage `k - 1` rows that stage `k` has not consumed, read from storage on the
  workers;
- the output is the stream from the last stage plus all committed last-stage
  rows.

The final map records every delivered key. After the export, the coordinator
verifies that every stage consumed all rows produced by its predecessor and
that every final row was delivered. Statistics are published only after that
check.

## Export

The export streams with the operators, so an interrupted run usually leaves part
of it behind, and a resumed run writes the complete output again. Each run
registers an export attempt, `exports/{n:08d}.json`, and names its files
`dj{n:04d}-{id}_*`:

1. before writing, files of attempts older than the previous one are removed,
   so repeated failures keep at most one earlier partial export;
2. the attempt writes the complete output and the delivery check above runs;
3. only then are the files of every earlier attempt removed.

The executor calls these phases explicitly: `exporter.export()`,
`recovery.finish()` (delivery verification and statistics publication), then
`exporter.complete_attempt()`. The exporter does not trigger checkpoint
completion through a callback. If verification or statistics publication
fails, the latest earlier export remains available.

A failed resume therefore never leaves less exported output than it found.
Files in the export directory that no attempt of the job wrote are kept, with a
warning. Formats that Ray does not write as named files (Lance) cannot replace
a partial export; resume stops until it is removed.

An empty checkpoint export remains a directory, so a later attempt can write
to the same destination. Ordinary empty exports retain their existing empty
file behavior.

When `export_shard_size` is set, the exporter materializes its final input once
and reuses it for exact size/row counting and writing. When an API caller asks
for a returned dataset and the recipe has operators after the checkpointed
prefix, the executor materializes their final result once for both export and
subsequent API reads. It does not rerun those operators with already-used state.
The CLI uses `skip_return=True`; ordinary unsharded exports keep streaming.

Outside checkpointed stages, an explicit Filter statistics export materializes
the computed statistics once so its writer and downstream filtering share the
same computation. Within checkpointed stages, the commit already owns both the
statistics and filtered output.

## Scope

Only the leading Mapper/Filter operators are checkpointed. A Deduplicator,
`Pipeline`, shared-state Filter, or `converge_after` operator needs all of its
input at once; it and every later operator run without checkpoints on the
merged output. External side effects of operators are outside this contract.

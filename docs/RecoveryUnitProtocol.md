# Streaming recovery units

`partition.recovery_mode: streaming` in `ray_partitioned` uses this protocol
for Mapper/Filter chains when checkpointing is enabled. Its checkpoint
identity is a fixed range of **input rows**, rather than a row in an operator's
output or a Ray block. A Mapper may emit multiple rows and a Filter may emit
none without changing the meaning of a completed unit.

The executor streams the ordered input into durable, addressable Parquet unit
files once. The snapshot manifest records its content-and-order fingerprint,
Arrow schema, per-file hashes, row count, and unit size. On resume, the executor
validates the snapshot files and hashes the current source before reusing any
committed work. A changed source, changed operator configuration, or changed
manual unit size fails closed. An auto-selected size is stored and reused on
resume. The current auto heuristic targets about 30 seconds of measured
single-actor work, clamped to 64–2048 input rows; without a throughput probe
it uses 512. Set `partition.unit_size` to a positive integer to choose retry
granularity explicitly. If `unit_size` is `auto` and `partition.size` is set,
the existing manual size is used as the unit size.

Each `(unit, operator stage)` has at most one committed attempt. Workers write
immutable attempt-owned Parquet parts, then the coordinator atomically commits
the winner's file hashes, row counts, and Arrow schema in SQLite. Beginning a
replacement attempt fences late workers from older generations. A zero-row
result is a real commit with a schema. Uncommitted attempt files are unreachable
garbage. The next operator reads only committed files; the final Ray dataset
reads only committed final-stage files in input-unit order. No schema probe
executes an operator plan with checkpoint side effects.

The Ray runner uses bounded queues and keeps each operator actor alive across
many units. It currently finishes one operator stage before starting the next,
so actors are reused within a stage but stages do not overlap. The operator
adapter instantiates the operator once per actor and calls it in its configured
`batch_size` chunks. Its current implementation supports ordinary per-input
Mapper/Filter behavior, including batched expansion and all-filtered units.
Filter `stats_export_path` falls back to partition recovery because that export
needs a separate transactional sink. `stream_segments` and `stream_fsync` are
accepted for config compatibility but ignored: unit recovery commits every
operator stage and always fsyncs its files and SQLite transaction.

This is a single-host local-filesystem protocol. Every Ray worker must see the
same POSIX checkpoint directory and source snapshot. A multi-node deployment
needs a shared durable object store and a conditional-commit metadata service.
The input snapshot is currently written through the driver, which may be a
throughput bottleneck on very large datasets. The stage-major barriers and
per-stage Parquet I/O are also material costs to measure against the prior
tee-sink head. For a fair comparison, use separate job IDs/work directories
and the same input, operators, actor resources, and `partition.unit_size`.

Cross-unit state and external side effects are outside this retry contract.
Such operators need an explicit state snapshot, idempotent/transactional sink,
or a separate barrier. Mapper output schemas that cannot be inferred from a
nonempty unit require an explicit empty-output schema declaration; guessing
through a side-effecting lazy plan would recreate the original failure mode.

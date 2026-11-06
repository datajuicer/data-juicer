"""Side-path checkpointing for operator stages that stream through Ray Data.

Rows keep flowing between operators through the Ray object store. Each
checkpointed batch writes its complete output to storage and commits it
through a metadata-only coordinator before yielding it downstream, so a
downstream commit never references an uncommitted upstream row.
"""

import json
import threading
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import ray
from loguru import logger
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from data_juicer.core.data.ray_batch_plan import BatchPlanWorker
from data_juicer.core.executor.partition_protocol import (
    IDX_COLUMN,
    KEY_COLUMNS,
    SEQ_COLUMN,
    SOURCE_SEQ,
    CommitLog,
    OutputFile,
    ProtocolError,
    commit_key,
    concat_outputs,
    discard_failed_commit,
    discard_files,
    key_ranges,
    open_export_attempt,
    read_tables,
    select_ranges,
    with_keys,
    write_tables,
)


@ray.remote(num_cpus=0, max_concurrency=64)
class _CommitCoordinator:
    """Serializes lineage checks; storage writes of distinct commits overlap."""

    def __init__(self, store, stage_count, source_rows):
        self.store = store
        self.log = CommitLog(store, stage_count, source_rows)
        self.durable = set(self.log.commits)
        self.condition = threading.Condition()
        self.failure = None

    def _check(self):
        if self.failure is not None:
            raise ProtocolError("The checkpoint log stopped accepting commits") from self.failure

    def plan(self):
        with self.condition:
            return self.log.plan()

    def commit(self, stage, consumed, files, stats_files):
        with self.condition:
            self._check()
            commit, new = self.log.prepare(stage, consumed, files, stats_files)
            if not new:
                self.condition.wait_for(lambda: commit.seq in self.durable or self.failure is not None)
                self._check()
                return commit
        try:
            if not self.store.put_if_absent(commit_key(commit.stage, commit.seq), commit.to_bytes()):
                raise ProtocolError(f"Checkpoint commit {commit.seq} already exists; another driver shares this job")
        except BaseException as error:
            # The in-memory log now disagrees with storage; only a fresh resume can repair it.
            with self.condition:
                self.failure = error
                self.condition.notify_all()
            discard_failed_commit(self.store, commit)
            raise
        with self.condition:
            self.durable.add(commit.seq)
            self.condition.notify_all()
        return commit

    def lookup(self, stage, consumed):
        """Return the durable commit of each consumed set, or ``None`` if it is not committed."""
        with self.condition:
            self._check()
            commits = [self.log.find(stage, item) for item in consumed]
            self.condition.wait_for(
                lambda: self.failure is not None or all(c is None or c.seq in self.durable for c in commits)
            )
            self._check()
            return commits

    def summary(self):
        with self.condition:
            return self.log.summary()

    def deliver(self, consumed):
        with self.condition:
            self._check()
            self.log.deliver(consumed)

    def verify(self):
        with self.condition:
            self._check()
            return self.log.verify()


@dataclass(frozen=True)
class StageContext:
    store: object
    stage: int
    coordinator: object
    max_rows: int


def _commit_slice(context, worker, batch, consumed):
    stats_files = ()

    def write_stats(tables):
        nonlocal stats_files
        tables = list(tables)
        stats_files = write_tables(context.store, tables, kind="stats")
        return tables

    data = batch.drop_columns(list(KEY_COLUMNS))
    try:
        outputs = [table for table in worker.process_partition([data], write_stats=write_stats) if table.num_rows]
        files = write_tables(context.store, outputs)
    except BaseException:
        # The commit RPC has not started, so these stats cannot be referenced.
        discard_files(context.store, (item.key for item in stats_files))
        raise
    commit = ray.get(context.coordinator.commit.remote(context.stage, consumed, files, stats_files))
    if commit.files != files or commit.stats_files != stats_files:
        # A concurrent attempt won; consumers may already hold its output.
        winning_keys = {item.key for item in (*commit.files, *commit.stats_files)}
        discard_files(context.store, (item.key for item in (*files, *stats_files) if item.key not in winning_keys))
        outputs = list(read_tables(context.store, commit.files))
    return commit, outputs


def _checkpoint_batch(context, worker, batch):
    """Commit a Ray block in slices of at most ``max_rows`` rows.

    A retried block replays the slices it already committed instead of
    recomputing them, because consumers may already hold those outputs.
    """
    slices = [batch.slice(start, context.max_rows) for start in range(0, batch.num_rows, context.max_rows)]
    if not slices:
        return
    consumed = [key_ranges(item[SEQ_COLUMN].to_numpy(), item[IDX_COLUMN].to_numpy()) for item in slices]
    committed = ray.get(context.coordinator.lookup.remote(context.stage, consumed))
    for piece, ranges, commit in zip(slices, consumed, committed):
        if commit is None:
            commit, outputs = _commit_slice(context, worker, piece, ranges)
        else:
            outputs = list(read_tables(context.store, commit.files))
        if outputs:
            yield with_keys(concat_outputs(outputs), commit.seq)


class CheckpointStageActor:
    def __init__(self, context, plan):
        self.context = context
        self.worker = BatchPlanWorker(plan)

    def __call__(self, batch):
        yield from _checkpoint_batch(self.context, self.worker, batch)


def run_checkpoint_task(batch, context, plan):
    yield from _checkpoint_batch(context, BatchPlanWorker(plan), batch)


def _tag_source(batch, pending):
    batch = batch.append_column(SEQ_COLUMN, pa.array(np.full(batch.num_rows, SOURCE_SEQ, dtype=np.int64)))
    return batch if pending is None else select_ranges(batch, pending)


def _deliver_batch(batch, coordinator):
    if batch.num_rows:
        ray.get(coordinator.deliver.remote(key_ranges(batch[SEQ_COLUMN].to_numpy(), batch[IDX_COLUMN].to_numpy())))
    return batch.drop_columns(list(KEY_COLUMNS))


def _read_entries(batch, store):
    for item in batch.to_pylist():
        tables = list(read_tables(store, [OutputFile(**entry) for entry in json.loads(item["files"])]))
        if not tables:
            continue
        table = concat_outputs(tables)
        if item["keyed"]:
            table = with_keys(table, item["seq"])
            ranges = json.loads(item["ranges"])
            if ranges is not None:
                table = select_ranges(table, ranges)
        if table.num_rows:
            yield table


def _stored_dataset(store, entries):
    """Read committed files on Ray workers; ``entries`` are ``(files, seq, ranges)``."""
    if not entries:
        return None
    items = [
        {
            "files": json.dumps([item.__dict__ for item in files]),
            "keyed": seq is not None,
            "seq": SOURCE_SEQ if seq is None else seq,
            "ranges": json.dumps(ranges),
        }
        for files, seq, ranges in entries
    ]
    return ray.data.from_items(items, override_num_blocks=min(len(items), 1024)).map_batches(
        _read_entries, fn_kwargs={"store": store}, batch_size=1, batch_format="pyarrow"
    )


def _union(*datasets):
    present = [dataset for dataset in datasets if dataset is not None]
    if not present:
        return None
    return present[0].union(*present[1:]) if len(present) > 1 else present[0]


@dataclass(frozen=True)
class RecoveryStage:
    recovery: "PartitionRecovery"
    index: int

    @property
    def context(self):
        return StageContext(self.recovery.store, self.index, self.recovery.coordinator, self.recovery.partition_size)

    def register_stats(self, export):
        self.recovery.stats_exports[self.index] = export


class PartitionRecovery:
    """Build one streaming plan over unconsumed rows and committed outputs.

    The input of stage ``k`` is the stream from stage ``k - 1`` plus the
    committed stage ``k - 1`` rows that stage ``k`` never consumed.
    """

    def __init__(self, store, source, source_rows, stage_count, partition_size):
        if partition_size < 1:
            raise ValueError("partition.size must be positive")
        self.store = store
        self.source = source
        self.source_rows = source_rows
        self.stage_count = stage_count
        self.partition_size = partition_size
        self.coordinator = None
        self.plan = None
        self.stats_exports = {}

    def open(self):
        """Start the coordinator and return the unconsumed source rows, if any."""
        node_id = ray.get_runtime_context().get_node_id()
        self.coordinator = _CommitCoordinator.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id, soft=False)
        ).remote(self.store, self.stage_count, self.source_rows)
        self.plan = ray.get(self.coordinator.plan.remote())
        source, self.source = self.source, None
        pending = self.plan.source
        if not pending:
            return None
        offsets = ray.data.range(self.source_rows).rename_columns({"id": IDX_COLUMN})
        return source.zip(offsets).map_batches(
            _tag_source,
            fn_kwargs={"pending": None if pending == ((0, self.source_rows),) else pending},
            batch_format="pyarrow",
            batch_size=None,
        )

    def stage_input(self, index, upstream):
        pending = self.plan.stages.get(index, ())
        return _union(
            upstream,
            _stored_dataset(self.store, [(item.commit.files, item.commit.seq, item.ranges) for item in pending]),
        )

    def stage(self, index):
        return RecoveryStage(self, index)

    def output(self, upstream):
        """Strip lineage keys after recording that every final row went downstream."""
        final = _stored_dataset(self.store, [(commit.files, commit.seq, None) for commit in self.plan.final])
        data = _union(upstream, final)
        if data is None:
            return ray.data.from_items([])
        return data.map_batches(
            _deliver_batch,
            fn_kwargs={"coordinator": self.coordinator},
            batch_format="pyarrow",
            batch_size=None,
            zero_copy_batch=True,
        )

    def export_attempt(self):
        """Name this run's export files; the caller controls completion and cleanup."""
        from data_juicer.core.ray_exporter import ExportAttempt

        prefix, earlier = open_export_attempt(self.store)
        return ExportAttempt(prefix, earlier)

    def finish(self):
        """Verify complete delivery, then publish statistics of every commit."""
        stats = ray.get(self.coordinator.verify.remote())
        for stage, (commits, consumed, produced) in sorted(ray.get(self.coordinator.summary.remote()).items()):
            logger.info(
                f"Checkpoint stage {stage}: {commits} commit(s), {consumed} rows in "
                f"({consumed / commits:.1f} per commit), {produced} rows out"
            )
        for stage, export in sorted(self.stats_exports.items()):
            files = stats.get(stage, ())
            if files:
                entries = [((item,), None, None) for item in files]
                export.publish(_stored_dataset(self.store, entries), [item.key for item in files])

    def committed_output(self):
        """Read the complete committed output without the coordinator, for re-execution."""
        final = ray.get(self.coordinator.plan.remote()).final
        data = _stored_dataset(self.store, [(commit.files, None, None) for commit in final])
        return ray.data.from_items([]) if data is None else data

    def close(self):
        if self.coordinator is not None:
            ray.kill(self.coordinator, no_restart=True)
            self.coordinator = None

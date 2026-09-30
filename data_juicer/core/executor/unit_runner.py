"""Single-host reference runner for the recovery-unit protocol.

This deliberately keeps scheduling separate from committing. Production Ray
workers may execute units concurrently, but only the coordinator owns the
LocalUnitStore and conditionally accepts their immutable output files.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence, Tuple, Union

import pyarrow as pa
import pyarrow.parquet as pq

from data_juicer.core.executor.unit_protocol import Attempt, Commit, LocalUnitStore, PreparedOutput, ProtocolError, Unit

TableSource = Callable[[int, int], Iterable[pa.Table]]
TableOperator = Callable[[pa.Table], Iterable[pa.Table]]


@dataclass(frozen=True)
class UnitStage:
    operator: TableOperator
    empty_schema: Callable[[pa.Schema], pa.Schema]


def execute_unit_attempt(
    store: LocalUnitStore,
    attempt: Attempt,
    unit: Unit,
    source: TableSource,
    stage_spec: Union[TableOperator, UnitStage],
    upstream: Commit = None,
) -> PreparedOutput:
    """Produce immutable files for one attempt without deciding its winner."""
    operator = stage_spec.operator if isinstance(stage_spec, UnitStage) else stage_spec
    empty_schema = stage_spec.empty_schema if isinstance(stage_spec, UnitStage) else None
    if upstream is None:
        if attempt.stage != 0:
            raise ProtocolError("Upstream stage has not committed")
        input_tables = source(unit.start, unit.stop)
        expected_input_rows = unit.stop - unit.start
        upstream_schema = None
    else:
        if attempt.input_commit_id != upstream.attempt.attempt_id:
            raise ProtocolError("Attempt does not reference the upstream commit")
        input_tables = (pq.read_table(store.root / item.path) for item in upstream.files)
        expected_input_rows = upstream.rows
        upstream_schema = upstream.schema

    def output_tables():
        input_rows = 0
        input_schema = upstream_schema
        emitted = False
        for table in input_tables:
            if not isinstance(table, pa.Table):
                raise TypeError("Unit source must yield PyArrow tables")
            input_rows += table.num_rows
            if input_schema is None:
                input_schema = table.schema
            for output in operator(table):
                emitted = True
                yield output
        if input_rows != expected_input_rows:
            raise ProtocolError("Unit input row count differs from the recovery plan")
        if not emitted:
            if empty_schema is None or input_schema is None:
                raise ProtocolError("Empty output requires a stage schema contract")
            yield pa.Table.from_batches([], schema=empty_schema(input_schema))

    return store.write_output(attempt, output_tables())


def run_local_units(
    store: LocalUnitStore,
    units: Sequence[Unit],
    source: TableSource,
    stages: Sequence[Union[TableOperator, UnitStage]],
) -> Tuple[pa.Table, ...]:
    """Run/recover independent unit stages and return committed final blocks.

    A stage callback must emit its *complete* output for each input table and
    raise on failure. Its output may contain zero, one, or many tables/rows.
    It must not publish external side effects: retries can rerun an unfinished
    unit. The caller must plan the run and pin its source/config fingerprints.
    """
    if not stages:
        raise ValueError("At least one stage is required")

    for stage, stage_spec in enumerate(stages):
        ready = {unit.unit_id for unit in store.pending(stage)}
        for unit in units:
            if unit.unit_id not in ready:
                # Validate a prior winner before treating this stage as done.
                committed = store.get_commit(unit.unit_id, stage)
                if committed is None and stage == 0:
                    raise ProtocolError("Unit is absent from the recovery plan")
                continue

            upstream = store.get_commit(unit.unit_id, stage - 1) if stage else None
            attempt = store.begin(unit.unit_id, stage)
            output = execute_unit_attempt(store, attempt, unit, source, stage_spec, upstream)
            store.commit(attempt, output)

    return collect_final_units(store, units, len(stages) - 1)


def collect_final_units(store: LocalUnitStore, units: Sequence[Unit], stage: int) -> Tuple[pa.Table, ...]:
    """Read winners in input-unit order, retaining an all-empty schema."""
    final = []
    final_schema = None
    for unit in units:
        commit = store.get_commit(unit.unit_id, stage)
        if commit is None:
            raise ProtocolError("Final unit stage has not committed")
        if final_schema is None:
            final_schema = commit.schema
        elif not final_schema.equals(commit.schema, check_metadata=True):
            raise ProtocolError("Final unit schemas differ")
        final.extend(pq.read_table(Path(store.root / item.path)) for item in commit.files)
    if not final and final_schema is not None:
        final.append(pa.Table.from_batches([], schema=final_schema))
    return tuple(final)

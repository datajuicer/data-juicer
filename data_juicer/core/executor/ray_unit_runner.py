"""Local-filesystem Ray scheduler for recovery units.

This adapter runs on a single host where every Ray actor can access the same
POSIX checkpoint directory. It is not a remote/object-store backend.
"""

from collections import deque
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import pyarrow as pa
import ray

from data_juicer.core.executor.unit_protocol import LocalUnitStore, PreparedOutput, ProtocolError, Unit
from data_juicer.core.executor.unit_runner import (
    TableOperator,
    TableSource,
    UnitStage,
    collect_final_units,
    execute_unit_attempt,
)


@dataclass(frozen=True)
class RayStage:
    operator: Union[TableOperator, UnitStage]
    actors: int = 1
    num_cpus: float = 1
    num_gpus: float = 0
    runtime_env: Optional[dict] = None


@ray.remote
class _UnitStageActor:
    def __init__(self, directory: str, run_id: str, source: TableSource, stage_spec):
        self.store = LocalUnitStore(directory, run_id)
        self.source = source
        self.stage_spec = stage_spec

    def execute(self, attempt, unit, upstream):
        return execute_unit_attempt(self.store, attempt, unit, self.source, self.stage_spec, upstream)


def run_ray_units(
    store: LocalUnitStore,
    units: Sequence[Unit],
    source: TableSource,
    stages: Sequence[RayStage],
    *,
    materialize_output: bool = True,
) -> Optional[Tuple[pa.Table, ...]]:
    """Run/recover units with one durable coordinator and reused stage actors.

    The caller must initialize Ray and provide an immutable addressable source.
    Actors write attempt-owned files; only this coordinator commits winners.
    Actor failures leave pending units for a later invocation to replay.
    """
    if not ray.is_initialized():
        raise RuntimeError("Ray must be initialized before running recovery units")
    if not stages:
        raise ValueError("At least one stage is required")

    for stage, spec in enumerate(stages):
        if spec.actors < 1:
            raise ValueError("Each Ray stage needs at least one actor")
        ready = {unit.unit_id for unit in store.pending(stage)}
        queue = deque(unit for unit in units if unit.unit_id in ready)
        for unit in units:
            if unit.unit_id not in ready:
                committed = store.get_commit(unit.unit_id, stage)
                if committed is None:
                    raise ProtocolError("Unit is absent from the recovery plan")
        if not queue:
            continue

        actors = [
            _UnitStageActor.options(
                num_cpus=spec.num_cpus,
                num_gpus=spec.num_gpus,
                runtime_env=spec.runtime_env,
            ).remote(str(store.root), store.run_id, source, spec.operator)
            for _ in range(min(spec.actors, len(queue)))
        ]
        in_flight = {}
        empty_results = []
        canonical_schema = None

        def submit(actor, unit):
            upstream = store.get_commit(unit.unit_id, stage - 1) if stage else None
            attempt = store.begin(unit.unit_id, stage)
            ref = actor.execute.remote(attempt, unit, upstream)
            in_flight[ref] = (actor, attempt, unit.stop - unit.start if upstream is None else upstream.rows)

        try:
            for actor in actors:
                submit(actor, queue.popleft())
            while in_flight:
                ready_refs, _ = ray.wait(list(in_flight), num_returns=1)
                ref = ready_refs[0]
                actor, attempt, input_rows = in_flight.pop(ref)
                output = ray.get(ref)
                if output.files:
                    if canonical_schema is None:
                        canonical_schema = output.schema
                    elif not canonical_schema.equals(output.schema, check_metadata=True):
                        raise ProtocolError("Unit output schemas differ within a stage")
                    store.commit(attempt, output)
                else:
                    empty_results.append((attempt, output, input_rows))
                if queue:
                    submit(actor, queue.popleft())
            if canonical_schema is None:
                computed = [output.schema for _, output, input_rows in empty_results if input_rows]
                canonical_schema = computed[0] if computed else empty_results[0][1].schema
                if any(not canonical_schema.equals(schema, check_metadata=True) for schema in computed[1:]):
                    raise ProtocolError("Unit output schemas differ within a stage")
            for attempt, output, _ in empty_results:
                store.commit(attempt, PreparedOutput((), canonical_schema))
        finally:
            for actor in actors:
                try:
                    ray.kill(actor, no_restart=True)
                except ray.exceptions.RayError:
                    pass

    if materialize_output:
        return collect_final_units(store, units, len(stages) - 1)
    return None

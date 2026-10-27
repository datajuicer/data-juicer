"""Internal resident GPU segments owned by PartitionedRayExecutor.

The caller owns partitions, checkpoint barriers, recovery and final export.
This module never starts/stops Ray, checkpoints rows, or retries whole tasks.
"""

import copy
import json
import math
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4

import pyarrow as pa
import ray
from loguru import logger

from data_juicer.ops.base_op import TAGGING_OPS
from data_juicer.utils.constant import Fields

from .compute_flow import ComputeFlowController, ComputeFlowError, ReadyWork
from .ray_adaptive_mapper import RayAdaptiveMapperActor, adaptive_batching_enabled
from .stage_identity import operator_fingerprint, stamped_stage_identity

_ROW_COLUMN = "__data_juicer_compute_flow_row_id__"
_PARTITION_COLUMN = "__data_juicer_logical_partition_id__"


@dataclass(frozen=True)
class _Group:
    pool: str = "gpu"


@dataclass(frozen=True)
class _Pool:
    workers: int
    num_gpus: int = 1


def _compiled(stages, workers):
    return {
        "pools": {"gpu": _Pool(workers)},
        "groups": {stage: _Group() for stage in stages},
        "group_order": tuple(stages),
    }


def segment_rejection(ops, options):
    """Return a fallback reason before any resident operator has executed."""
    if not ops:
        return "empty operator interval"
    fractions = []
    for op in ops:
        if not adaptive_batching_enabled(op, True):
            return "interval contains a mapper without the adaptive/retry-safe contract"
        if "ray" not in op._supported_exec_modes or not op.use_ray_actor():
            return "interval contains a mapper without Ray actor support"
        if op.skip_op_error or op.runtime_env:
            return "error-skipping or per-operator runtime environments require Ray Data"
        if not 0 < float(op.num_gpus or 0) <= 1:
            return "GPU requests must be resolved before resident execution"
        fraction = getattr(op, "_gpu_memory_fraction", None)
        if type(fraction) not in (int, float) or not math.isfinite(fraction) or fraction <= 0:
            return "co-residency requires measured GPU preflight memory fractions"
        fractions.append(fraction)
    # Sum complete per-operator peak estimates, including activation budgets.
    # This deliberately overestimates co-resident usage: only one stage runs
    # at a time on each worker. It remains an estimate, not an OOM guarantee.
    if sum(fractions) > options.memory_fraction_limit:
        return "sum of measured operator memory fractions exceeds the co-residency limit"
    if options.mode == "static" and options.workers < len(ops):
        return "static allocation requires one worker per stage"
    return None


class _ResidentWorker:
    def __init__(self, specs, max_batch_bytes):
        self.max_batch_bytes = max_batch_bytes
        self.runners = {spec["stage_id"]: RayAdaptiveMapperActor(**spec) for spec in specs}

    def ready(self):
        return {
            "operators": {name: runner.actor_id for name, runner in self.runners.items()},
            "node_id": ray.get_runtime_context().get_node_id(),
            "gpu_ids": ray.get_gpu_ids(),
        }

    def process(self, stage, table, terminal):
        start = time.perf_counter()
        runner = self.runners[stage]
        # Match RayDataset._run_single_op's metadata setup for tagging Mappers.
        if runner.op._name in TAGGING_OPS.modules and Fields.meta not in table.column_names:
            table = table.append_column(Fields.meta, pa.array([{} for _ in range(table.num_rows)]))
        expected = table[_ROW_COLUMN]
        tags = table[_PARTITION_COLUMN] if _PARTITION_COLUMN in table.column_names else None
        retries = runner.mapper.oom_retries
        result = runner(table)
        output = result if isinstance(result, pa.Table) else pa.Table.from_pydict(result)
        if _ROW_COLUMN not in output.column_names or not output[_ROW_COLUMN].equals(expected):
            raise ComputeFlowError("resident mapper changed row identity/order")
        if tags is not None and (
            _PARTITION_COLUMN not in output.column_names or not output[_PARTITION_COLUMN].equals(tags)
        ):
            raise ComputeFlowError("resident mapper changed logical partition identity")
        if output.nbytes > self.max_batch_bytes:
            raise ComputeFlowError("resident output exceeded max_batch_bytes")
        if terminal:
            output = output.drop_columns([_ROW_COLUMN])
        return output, {
            "stage": stage,
            "rows": table.num_rows,
            "bytes": output.nbytes,
            "seconds": time.perf_counter() - start,
            "operator_instance": runner.actor_id,
            "oom_retries": runner.mapper.oom_retries - retries,
            "batch": asdict(runner.controller.state),
        }


class PartitionComputeFlowSession:
    """One run's optional execution service, called only by grouped partitions."""

    def __init__(self, options, work_dir):
        self.options = options
        self.root = Path(work_dir) / "compute_flow" / uuid4().hex
        self.root.mkdir(parents=True, exist_ok=True)
        self.events = (self.root / "events.jsonl").open("w")
        self.actors = {}
        self.signature = None
        self.flow = None
        self.pool_number = 0
        self.calls = 0
        self.executed = 0
        self.fallbacks = []
        self.pools = []
        self.failure = None

    def _event(self, **event):
        self.events.write(json.dumps({"call": self.calls, "pool_number": self.pool_number, **event}) + "\n")
        self.events.flush()

    def fallback(self, reason):
        self._close_pool()
        if reason not in self.fallbacks:
            self.fallbacks.append(reason)
            logger.info(f"Compute flow uses existing Ray Data execution: {reason}")
        self._event(kind="fallback", reason=reason)

    def _close_pool(self):
        actors, self.actors = self.actors, {}
        if self.flow is not None:
            self.pools[-1]["controller"] = self.flow.summary()
        self.flow, self.signature = None, None
        for actor in actors.values():
            try:
                ray.kill(actor, no_restart=True)
            except ray.exceptions.RayError:
                pass

    def _ensure_pool(self, ops):
        stages = tuple(stamped_stage_identity(op) for op in ops)
        signature = tuple(
            (stage, operator_fingerprint(op), op.batch_size, op.num_cpus, op.num_gpus, op._gpu_memory_fraction)
            for stage, op in zip(stages, ops)
        )
        if signature == self.signature:
            return stages
        self._close_pool()
        cpus = max(float(op.num_cpus or 1) for op in ops)
        required = {"CPU": cpus * self.options.workers, "GPU": self.options.workers}
        gpu_nodes = [node for node in ray.nodes() if node["Alive"] and node["Resources"].get("GPU", 0) > 0]
        if len(gpu_nodes) != 1:
            raise ComputeFlowError("GPU topology changed before resident pool creation")
        total = gpu_nodes[0]["Resources"]
        if required["GPU"] > total.get("GPU", 0) or required["CPU"] + 1 > total.get("CPU", 0):
            raise ComputeFlowError(f"compute-flow pool needs {required} plus one CPU for Ray Data/checkpoint tasks")
        # Only one pool exists at a time. Allow Ray's asynchronous actor teardown
        # to finish before replacing it; never stop or take over another job.
        deadline = time.monotonic() + min(5, self.options.startup_timeout_seconds)
        while any(required[k] > ray.available_resources().get(k, 0) for k in required):
            if time.monotonic() >= deadline:
                raise ComputeFlowError(f"insufficient free resources for compute-flow pool: {required}")
            time.sleep(0.05)
        specs = [
            dict(
                op_class=type(op),
                op_args=copy.deepcopy(op._init_args),
                op_kwargs=copy.deepcopy(op._init_kwargs),
                max_batch_size=op.batch_size,
                stage_id=stage,
            )
            for stage, op in zip(stages, ops)
        ]
        self.pool_number += 1
        self.flow = ComputeFlowController(_compiled(stages, self.options.workers), self.options.controller_config())
        self.pools.append({"number": self.pool_number, "stages": stages, "workers": self.options.workers})
        try:
            remote = ray.remote(max_restarts=0, max_task_retries=0)(_ResidentWorker)
            for slot in range(self.options.workers):
                self.actors[("gpu", slot)] = remote.options(num_cpus=cpus, num_gpus=1).remote(
                    specs, self.options.max_batch_bytes
                )
            identities = ray.get(
                [actor.ready.remote() for actor in self.actors.values()], timeout=self.options.startup_timeout_seconds
            )
            self.pools[-1]["instances"] = identities
            self._event(kind="pool", stages=stages, instances=identities, resources=required)
        except BaseException:
            self._close_pool()
            raise
        self.signature = signature
        return stages

    def execute(self, dataset, ops):
        """Return a materialized Dataset, or None for a pre-execution fallback."""
        self.calls += 1
        reason = segment_rejection(ops, self.options)
        if reason:
            self.fallback(reason)
            return None
        gpu_nodes = [node for node in ray.nodes() if node["Alive"] and node["Resources"].get("GPU", 0) > 0]
        if len(gpu_nodes) != 1:
            self.fallback("resident compute flow currently requires a single GPU node")
            return None
        schema = dataset.schema()
        if schema is not None and _ROW_COLUMN in schema.names:
            raise ComputeFlowError(f"input contains reserved column {_ROW_COLUMN}")
        # Finish the preceding Ray Data segment before reserving a resident
        # GPU pool. No independent Ray Data actor pool runs this same interval.
        dataset = dataset.materialize()
        source = iter(
            dataset.iter_batches(batch_size=self.options.batch_size, batch_format="pyarrow", prefetch_batches=0)
        )
        try:
            first = next(source)
        except StopIteration:
            return dataset
        try:
            stages = self._ensure_pool(ops)
            result = self._execute_batches(first, source, stages)
            self.executed += 1
            return result
        except BaseException as error:
            self.failure = {"type": type(error).__name__, "message": str(error)}
            self._close_pool()
            raise  # Never fall back after any resident work has started.
        finally:
            close = getattr(source, "close", None)
            if close:
                close()

    def _execute_batches(self, first, source, stages):
        idle = deque(self.actors)
        ready = {stage: deque() for stage in stages}
        payloads, pending, finished = {}, {}, {}
        next_stage = dict(zip(stages, stages[1:]))
        unit, row_offset, active = 0, 0, 0
        exhausted = False
        current = first
        while not exhausted or active:
            while not exhausted and active < self.options.max_inflight_batches:
                if current is None:
                    try:
                        current = next(source)
                    except StopIteration:
                        exhausted = True
                        break
                table, current = current, None
                table = table.append_column(_ROW_COLUMN, pa.array(range(row_offset, row_offset + table.num_rows)))
                if table.nbytes > self.options.max_batch_bytes:
                    raise ComputeFlowError("resident input exceeded max_batch_bytes")
                row_offset += table.num_rows
                payloads[unit] = ray.put(table)
                ready[stages[0]].append(ReadyWork(unit, table.num_rows, time.monotonic()))
                unit, active = unit + 1, active + 1
            if not active:
                break
            now = time.monotonic()
            running = {stage: [] for stage in stages}
            for task in pending.values():
                running[task["stage"]].append(task["rows"])
            for event in self.flow.replan(ready, running, now):
                self._event(**event)
            counts = {stage: len(rows) for stage, rows in running.items()}
            while idle:
                choice = self.flow.choose("gpu", idle, ready, counts, now)
                if choice is None:
                    break
                key, stage, reason = choice
                work = ready[stage].popleft()
                idle.remove(key)
                output, metadata = (
                    self.actors[key]
                    .process.options(num_returns=2)
                    .remote(stage, payloads[work.unit], stage == stages[-1])
                )
                pending[metadata] = dict(
                    unit=work.unit, stage=stage, key=key, output=output, rows=work.rows, submitted=time.monotonic()
                )
                counts[stage] += 1
                self._event(unit=work.unit, **self.flow.dispatched(key, stage, reason))
            if not pending:
                raise ComputeFlowError("compute-flow scheduler made no progress")
            done, _ = ray.wait(list(pending), num_returns=1, timeout=0.2)
            for ref in done:
                task = pending.pop(ref)
                metadata = ray.get(ref)
                if time.monotonic() - task["submitted"] > self.options.task_timeout_seconds:
                    raise TimeoutError("compute-flow task exceeded its timeout")
                self.flow.observe(task["stage"], task["key"], metadata["rows"], metadata["seconds"])
                self._event(kind="complete", unit=task["unit"], slot=task["key"][1], **metadata)
                idle.append(task["key"])
                if task["stage"] in next_stage:
                    stage = next_stage[task["stage"]]
                    payloads[task["unit"]] = task["output"]
                    ready[stage].append(ReadyWork(task["unit"], metadata["rows"], time.monotonic()))
                else:
                    finished[task["unit"]] = task["output"]
                    del payloads[task["unit"]]
                    active -= 1
            if any(
                time.monotonic() - task["submitted"] > self.options.task_timeout_seconds for task in pending.values()
            ):
                raise TimeoutError("compute-flow task timed out")
        # Reordering only references preserves the deterministic input order.
        # Final blocks belong to the materialized Dataset, outside the bounded
        # in-flight budget, just as in the executor's ordinary checkpoint path.
        return ray.data.from_arrow_refs([finished[index] for index in range(unit)])

    def release(self):
        """Drain has finished; release reservations before another GPU path."""
        self._close_pool()

    def finish(self):
        self.release()
        summary = {
            "artifact_dir": str(self.root),
            "config": asdict(self.options),
            "executed_intervals": self.executed,
            "fallbacks": self.fallbacks,
            "pools": self.pools,
            "failure": self.failure,
        }
        try:
            (self.root / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        finally:
            self.events.close()
        return summary

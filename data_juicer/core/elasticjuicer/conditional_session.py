"""Bounded conditional fork/join over PR1116's resident workers.

Tower owns leases. Existing per-operator Captain objects own microbatches.
The public partitioned executor retains checkpoint and final export authority.
"""

import copy
import queue
import threading
import time
from collections import deque

import pyarrow as pa
import ray

from data_juicer.core.data.ray_dataset import RayDataset

from .compute_flow import ComputeFlowError, ReadyWork
from .conditional_flow import ROUTES, ROW, ConditionalPlan, as_table, check_identity
from .partition_compute_flow import (
    PartitionComputeFlowSession,
    _PoolUnavailable,
    segment_rejection,
)


class _Source:
    """One bounded producer; source failures are delivered to the event loop."""

    def __init__(self, source):
        self.source = source
        self.queue = queue.Queue(maxsize=2)
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name="conditional-input", daemon=True)
        self.thread.start()

    def _run(self):
        try:
            for table in self.source:
                while not self.stop.is_set():
                    try:
                        self.queue.put(("table", table), timeout=0.1)
                        break
                    except queue.Full:
                        continue
                if self.stop.is_set():
                    return
            item = ("end", None)
        except BaseException as error:
            item = ("error", error)
        while not self.stop.is_set():
            try:
                self.queue.put(item, timeout=0.1)
                return
            except queue.Full:
                continue

    def close(self):
        self.stop.set()
        self.thread.join(timeout=10)


def _complete_unit(plan, suffix, base, stages, *deltas):
    start = time.monotonic()
    table = plan.join(base, dict(zip(stages, deltas)))
    identities = table[ROW]
    for cls, args, kwargs in suffix:
        op = cls(*args, **kwargs)
        preserved = table.drop_columns([key for key in getattr(op, "_output_columns", ()) if key in table.column_names])
        table = as_table(op.process_batched(table.to_pydict()))
        check_identity(preserved, table)
        for key in preserved.column_names:
            table = table.set_column(table.schema.get_field_index(key), key, preserved[key])
        if ROW not in table.column_names or not table[ROW].equals(identities):
            raise ComputeFlowError("CPU suffix changed row identity/order")
    return table.drop_columns([ROW]), dict(kind="join", rows=table.num_rows, seconds=time.monotonic() - start)


class ConditionalLinearSession(PartitionComputeFlowSession):
    """Keep PR1116's linear GPU policy for the new conditional op contract.

    Only the declared Topic workload gets CPU prefix/suffix segmentation;
    ordinary recipes retain the original admission/fallback behavior.
    """

    def execute(self, dataset, ops):
        gpu = [i for i, op in enumerate(ops) if getattr(op, "accelerator", None) == "cuda"]
        if not gpu or not getattr(ops[gpu[0]], "conditional_topic", False):
            return super().execute(dataset, ops)
        first, last = gpu[0], gpu[-1] + 1
        if gpu != list(range(first, last)):
            return super().execute(dataset, ops)
        if first:
            dataset = RayDataset(dataset, cfg=getattr(self, "cfg", None)).process(ops[:first]).data.materialize()
        result = super().execute(dataset, ops[first:last])
        if result is None:
            # CPU prefix is already materialized; continue from it rather than
            # replaying it through the executor's generic fallback.
            return RayDataset(dataset, cfg=getattr(self, "cfg", None)).process(ops[first:]).data.materialize()
        if last < len(ops):
            result = RayDataset(result, cfg=getattr(self, "cfg", None)).process(ops[last:]).data.materialize()
        return result


class ConditionalFlowSession(PartitionComputeFlowSession):
    def execute(self, dataset, ops):
        self.calls += 1
        gpu = [i for i, op in enumerate(ops) if getattr(op, "accelerator", None) == "cuda"]
        if not gpu or gpu != list(range(min(gpu), max(gpu) + 1)):
            self.fallback("conditional execution requires one contiguous GPU region")
            return None
        first, last = gpu[0], gpu[-1] + 1
        gpu_ops, prefix, suffix = ops[first:last], ops[:first], ops[last:]
        if not getattr(gpu_ops[0], "conditional_topic", False):
            self.fallback("checkpoint interval has no conditional Topic root")
            return None
        reason = segment_rejection(gpu_ops, self.options)
        if reason:
            self.fallback(reason)
            return None
        nodes = [n for n in ray.nodes() if n["Alive"] and n["Resources"].get("GPU", 0)]
        if not nodes or (len(nodes) != 1 and not self.options.allow_multi_node):
            self.fallback("conditional flow requires one GPU node")
            return None
        stages = tuple(getattr(op, "_elastic_juicer_stage_identity") for op in gpu_ops)
        plan = ConditionalPlan.compile(stages, gpu_ops)
        if dataset.count() == 0:
            return dataset
        source = None
        try:
            try:
                self._ensure_pool(gpu_ops)
            except _PoolUnavailable as error:
                self.fallback(str(error))
                return None
            if prefix:
                dataset = RayDataset(dataset, cfg=getattr(self, "cfg", None)).process(prefix).data
            size = (
                max(op.batch_size for op in gpu_ops) if self.options.batch_size == "auto" else self.options.batch_size
            )
            source = _Source(iter(dataset.iter_batches(batch_size=size, batch_format="pyarrow", prefetch_batches=0)))
            suffix_specs = [(type(op), copy.deepcopy(op._init_args), copy.deepcopy(op._init_kwargs)) for op in suffix]
            result = self._run_conditional(source, plan, stages, suffix_specs)
            self.executed += 1
            return result
        except BaseException as error:
            self.failure = dict(type=type(error).__name__, message=str(error))
            self._close_pool()
            raise
        finally:
            if source:
                source.close()

    def _run_conditional(self, source, plan, stages, suffix):
        ready = {s: deque() for s in stages}
        idle, pending, units, finished = deque(self.actors), {}, {}, {}
        unit, offset, exhausted = 0, 0, False
        join = ray.remote(num_cpus=1, max_retries=0)(_complete_unit)
        last_progress = time.monotonic()
        peak = 0
        while not exhausted or units or pending:
            while not exhausted and len(units) < self.options.max_inflight_batches:
                try:
                    kind, table = source.queue.get_nowait()
                except queue.Empty:
                    break
                if kind == "error":
                    raise table
                if kind == "end":
                    exhausted = True
                    break
                if ROW in table.column_names or table.nbytes > self.options.max_batch_bytes:
                    raise ComputeFlowError("invalid or oversized conditional input")
                if not table.num_rows:
                    continue
                table = table.append_column(ROW, pa.array(range(offset, offset + table.num_rows)))
                offset += table.num_rows
                units[unit] = dict(base=ray.put(table), outputs={}, waiting=None)
                ready[plan.topic].append(ReadyWork(unit, table.num_rows, time.monotonic()))
                unit += 1
                peak = max(peak, len(units))
                last_progress = time.monotonic()
            running = {s: [] for s in stages}
            for task in pending.values():
                if task["kind"] == "gpu":
                    running[task["stage"]].append(task["rows"])
            now = time.monotonic()
            for event in self.flow.replan(ready, running, now):
                self._event(**event)
            counts = {s: len(rows) for s, rows in running.items()}
            while idle:
                choice = self.flow.choose("gpu", idle, ready, counts, now)
                if choice is None:
                    break
                key, stage, reason = choice
                work = ready[stage].popleft()
                state = units[work.unit]
                indices = None if stage == plan.topic else state["routes"][stage]
                output, metadata = (
                    self.actors[key].process.options(num_returns=2).remote(stage, state["base"], False, indices)
                )
                pending[metadata] = dict(
                    kind="gpu",
                    stage=stage,
                    key=key,
                    unit=work.unit,
                    output=output,
                    rows=work.rows,
                    submitted=time.monotonic(),
                )
                idle.remove(key)
                counts[stage] += 1
                self._event(unit=work.unit, **self.flow.dispatched(key, stage, reason))
            if not pending:
                if exhausted and not units:
                    break
                if time.monotonic() - last_progress > self.options.task_timeout_seconds:
                    raise TimeoutError("conditional source made no progress")
                time.sleep(0.01)
                continue
            done, _ = ray.wait(list(pending), num_returns=1, timeout=0.1)
            for ref in done:
                task = pending.pop(ref)
                state = units[task["unit"]]
                last_progress = time.monotonic()
                if task["kind"] == "join":
                    metadata = ray.get(ref)
                    finished[task["unit"]] = task["output"]
                    del units[task["unit"]]
                    self._event(unit=task["unit"], **metadata)
                    continue
                metadata = ray.get(ref)
                stage = task["stage"]
                idle.append(task["key"])
                self.flow.observe(stage, task["key"], metadata["rows"], metadata["seconds"])
                self._event(kind="complete", unit=task["unit"], slot=task["key"][1], **metadata)
                if stage == plan.topic:
                    state["base"] = task["output"]
                    routes = plan.route(pa.table({ROUTES: metadata["required_ops"]}))
                    state["routes"] = routes
                    state["waiting"] = {s for s, indices in routes.items() if indices}
                    for leaf in state["waiting"]:
                        ready[leaf].append(ReadyWork(task["unit"], len(routes[leaf]), time.monotonic()))
                else:
                    if stage not in state["waiting"]:
                        raise ComputeFlowError("duplicate conditional completion")
                    state["outputs"][stage] = task["output"]
                    state["waiting"].remove(stage)
                if not state["waiting"]:
                    selected = tuple(state["outputs"])
                    output_ref, metadata_ref = join.options(num_returns=2).remote(
                        plan, suffix, state["base"], selected, *(state["outputs"][s] for s in selected)
                    )
                    # Return the data reference independently of the metadata;
                    # the driver never copies full feature tables to inspect it.
                    pending[metadata_ref] = dict(
                        kind="join", unit=task["unit"], output=output_ref, submitted=time.monotonic()
                    )
            if any(time.monotonic() - t["submitted"] > self.options.task_timeout_seconds for t in pending.values()):
                raise TimeoutError("conditional task timed out")
        self._event(
            kind="conditional_summary",
            input_batches=unit,
            input_rows=offset,
            peak_inflight_batches=peak,
            execution="conditional",
        )
        if not finished:
            return ray.data.from_arrow(pa.table({}))
        return ray.data.from_arrow_refs([finished[i] for i in range(unit)])

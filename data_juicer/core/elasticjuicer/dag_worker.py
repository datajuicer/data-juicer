"""Resident DJ operators shared across several physical DAG execution groups."""

import copy
import importlib
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from uuid import uuid4

import numpy as np

from .adaptive_mapper import OOMSafeAdaptiveMapper
from .batch_controller import AdaptiveBatchController
from .dag_plan import (
    DAGContractError,
    DAGPayload,
    merge_parents,
    payload_bytes,
    validate_columns,
)
from .fixed_batch_controller import FixedBatchController
from .oom import is_oom_error


def materialize_payload(payload, compiled, id_column, cache=None):
    """Resolve a lineage of small deltas inside the worker, never the driver."""
    import ray

    cache = {} if cache is None else cache
    if isinstance(payload, ray.ObjectRef):
        if payload not in cache:
            cache[payload] = materialize_payload(ray.get(payload), compiled, id_column, cache)
        return cache[payload]
    if not isinstance(payload, DAGPayload):
        raise DAGContractError("invalid DAG payload reference")
    if payload.origin is None:
        return payload
    node = compiled["nodes"][payload.origin]
    expected = node.parents or ("$input",)
    if set(payload.bases) != set(expected):
        raise DAGContractError("delta parent lineage does not match the logical DAG")
    if set(payload.columns) != {id_column, *node.writes}:
        raise DAGContractError("delta contains undeclared columns")
    if payload.writers != {key: node.name for key in node.writes}:
        raise DAGContractError("delta column ownership does not match its logical node")
    parents = [materialize_payload(payload.bases[name], compiled, id_column, cache) for name in expected]
    base = merge_parents(parents, node, compiled, id_column)
    ids = validate_columns(payload.columns, id_column)
    positions = {key: i for i, key in enumerate(base.columns[id_column])}
    if not set(ids) <= positions.keys() or (node.row_policy == "preserve" and set(ids) != positions.keys()):
        raise DAGContractError("delta changed row identity/cardinality")
    columns = {key: [values[positions[i]] for i in ids] for key, values in base.columns.items()}
    columns.update(payload.columns)
    return DAGPayload(columns, {**base.writers, **payload.writers})


def _operator_input(value):
    # Isolate mutable containers without copying large image buffers. Operators
    # explicitly opt into the immutable-input, replay-safe DAG contract.
    if isinstance(value, np.ndarray):
        view = value.view()
        view.flags.writeable = False
        return view
    if isinstance(value, dict):
        return {k: _operator_input(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_operator_input(v) for v in value)
    return value


class _NodeRunner:
    def __init__(self, node, pool, adaptive, id_column):
        from data_juicer.ops.base_op import Mapper
        from data_juicer.ops.load import load_ops

        self.node, self.id_column = node, id_column
        recipe = copy.deepcopy(node.operator)
        name = next(iter(recipe))
        kwargs = recipe[name] = dict(recipe[name] or {})
        if kwargs.get("skip_op_error", False):
            raise DAGContractError("DAG operators cannot skip errors")
        if kwargs.get("runtime_env"):
            raise DAGContractError("DAG v1 requires one compatible runtime environment per job")
        kwargs.update(
            batch_size=node.batch_size,
            num_proc=1,
            auto_op_parallelism=False,
            num_cpus=pool.num_cpus,
            num_gpus=pool.num_gpus,
            skip_op_error=False,
        )
        self.op = load_ops([recipe])[0]
        if not isinstance(self.op, Mapper) or not self.op.is_batched_op():
            raise DAGContractError("DAG nodes require batched DJ Mappers")
        if getattr(self.op, "elastic_juicer_dag_safe", False) is not True:
            raise DAGContractError("mapper must opt into elastic_juicer_dag_safe (immutable inputs, replay-safe)")
        controller_type = AdaptiveBatchController if adaptive else FixedBatchController
        initial = node.initial_batch_size if adaptive and node.initial_batch_size is not None else node.batch_size
        self.controller = controller_type(initial_batch_size=initial, max_batch_size=node.batch_size)
        self.mapper = OOMSafeAdaptiveMapper(
            self._call,
            self.controller,
            validate_output_rows=node.row_policy == "preserve",
            max_floor_retries=0,
            oom_cleanup=self._cleanup if pool.num_gpus else None,
            label=node.name,
        )
        self.instance_id = uuid4().hex
        self.total_calls = 0
        self.attempts = []

    @staticmethod
    def _cleanup():
        import torch

        torch.cuda.empty_cache()

    def _call(self, batch):
        expected = tuple(batch[self.id_column])
        started = time.perf_counter()
        attempt = {"rows": len(expected), "success": False, "oom": False}
        try:
            output = self.op.process_batched(_operator_input(batch))
            if not isinstance(output, dict):
                raise DAGContractError("DAG mapper must return a column dictionary")
            actual = validate_columns(output, self.id_column)
            if not set(actual) <= set(expected) or (
                self.node.row_policy == "preserve" and set(actual) != set(expected)
            ):
                raise DAGContractError(f"{self.node.name}: mapper changed row identity/cardinality")
            if not set(self.node.writes) <= output.keys():
                raise DAGContractError(f"{self.node.name}: missing declared output column")
            positions = {key: i for i, key in enumerate(actual)}
            ordered = [key for key in expected if key in positions]
            result = {self.id_column: ordered}
            result.update({key: [output[key][positions[i]] for i in ordered] for key in self.node.writes})
            attempt["success"] = True
            self.total_calls += 1
            return result
        except BaseException as error:
            attempt["oom"] = is_oom_error(error)
            raise
        finally:
            attempt["seconds"] = time.perf_counter() - started
            self.attempts.append(attempt)

    def run(self, payload):
        columns = payload.columns
        ids = validate_columns(columns, self.id_column)
        if not set(self.node.reads) <= columns.keys():
            raise DAGContractError(f"{self.node.name}: missing declared input column")
        self.attempts = []
        if ids:
            inputs = {key: columns[key] for key in set(self.node.reads) | {self.id_column}}
            delta = self.mapper(inputs)
        else:
            delta = {key: [] for key in (*self.node.writes, self.id_column)}
        positions = {key: i for i, key in enumerate(ids)}
        output_ids = delta[self.id_column]
        kept = [positions[key] for key in output_ids]
        result = {key: [values[i] for i in kept] for key, values in columns.items()}
        result.update(delta)
        writers = {**payload.writers, **{key: self.node.name for key in self.node.writes}}
        return DAGPayload(result, writers)

    def diagnostics(self):
        return {
            "operator_instance": self.instance_id,
            "model_calls": self.total_calls,
            "controller": asdict(self.controller.state),
            "oom_retries": self.mapper.oom_retries,
        }


class ResidentDAGWorker:
    """One Ray actor, one GPU at most, multiple persistent per-node operators.

    A worker runs one group at a time. Independent operators within a parallel
    group run in separate threads and retain independent local controllers.
    Per-model GPU peak sampling is intentionally absent: concurrent models share
    the same CUDA allocator, so resetting its peak per model would be misleading.
    """

    def __init__(self, plan, pool_name, slot, adaptive):
        import ray

        self.plan, self.compiled = plan, plan.compile()
        self.pool = self.compiled["pools"][pool_name]
        self.pool_name, self.slot = pool_name, slot
        self.instance_id = uuid4().hex
        self.closed = False
        self.runners = {}
        self.threads = None
        self.device = None
        if self.pool.num_gpus:
            import torch

            if len(ray.get_gpu_ids()) != 1 or torch.cuda.device_count() != 1:
                raise DAGContractError("GPU pool actor must own exactly one visible GPU")
            torch.cuda.set_device(0)
            props = torch.cuda.get_device_properties(0)
            self.device = {"uuid": str(props.uuid), "name": props.name, "memory_bytes": props.total_memory}
        try:
            for module in plan.operator_modules:
                importlib.import_module(module)
            groups = [g for g in plan.groups if g.pool == pool_name]
            names = {n for g in groups for n in g.nodes}
            for name in self.compiled["order"]:
                if name in names:
                    self.runners[name] = _NodeRunner(self.compiled["nodes"][name], self.pool, adaptive, plan.id_column)
            self.threads = ThreadPoolExecutor(max_workers=max(len(g.nodes) if g.parallel else 1 for g in groups))
        except BaseException:
            self.close()
            raise

    def ready(self):
        import ray

        memory = None
        if self.device:
            import torch

            memory = {"allocated_bytes": torch.cuda.memory_allocated(), "reserved_bytes": torch.cuda.memory_reserved()}
        return {
            "worker_id": self.instance_id,
            "pool": self.pool_name,
            "slot": self.slot,
            "pid": os.getpid(),
            "node_id": ray.get_runtime_context().get_node_id(),
            "device": self.device,
            "torch_memory_at_ready": memory,
            "operators": {n: r.diagnostics() for n, r in self.runners.items()},
        }

    def _run_node(self, name, parents):
        if self.device:
            import torch

            torch.cuda.set_device(0)  # CUDA device selection is thread-local.
        node = self.compiled["nodes"][name]
        payload = merge_parents(
            [parents[p] for p in node.parents or ("$input",)], node, self.compiled, self.plan.id_column
        )
        started = time.time()
        output = self.runners[name].run(payload)
        ended = time.time()
        if name == self.plan.output and self.plan.output_columns:
            keys = set(self.plan.output_columns) | {self.plan.id_column}
            if not keys <= output.columns.keys():
                raise DAGContractError("missing terminal output column")
            output = DAGPayload(
                {key: output.columns[key] for key in keys}, {key: output.writers.get(key, "$input") for key in keys}
            )
        size = payload_bytes(output.columns)
        if size > self.plan.max_batch_bytes:
            raise DAGContractError(f"{name}: output exceeded max_batch_bytes")
        actual = set(output.columns[self.plan.id_column])
        metadata = {
            "node": name,
            "started": started,
            "ended": ended,
            "pid": os.getpid(),
            "thread": threading.get_ident(),
            "worker_id": self.instance_id,
            "input_rows": len(payload.columns[self.plan.id_column]),
            "output_rows": len(actual),
            "output_ids": list(output.columns[self.plan.id_column]),
            "rejected_ids": [i for i in payload.columns[self.plan.id_column] if i not in actual],
            "payload_bytes": size,
            "attempts": self.runners[name].attempts,
            **self.runners[name].diagnostics(),
        }
        return output, metadata

    def process(self, group_name, parent_names, parent_refs, *payloads):
        group_started = time.perf_counter()
        if self.closed:
            raise RuntimeError("resident DAG worker is closed")
        group = self.compiled["groups"][group_name]
        if group.pool != self.pool_name:
            raise DAGContractError("group dispatched to the wrong pool")
        import ray

        if not (len(parent_names) == len(parent_refs) == len(payloads)) or not all(
            isinstance(ref, ray.ObjectRef) for ref in parent_refs
        ):
            raise DAGContractError("expected top-level payloads and nested parent ObjectRefs")
        cache = {}
        parents = {
            name: materialize_payload(value, self.compiled, self.plan.id_column, cache)
            for name, value in zip(parent_names, payloads)
        }
        references = dict(zip(parent_names, parent_refs))
        results, metadata = {}, []
        names = [n for n in self.compiled["order"] if n in group.nodes]
        if group.parallel:
            futures = {n: self.threads.submit(self._run_node, n, parents) for n in names}
            # Await every branch before propagating failure. No sibling may
            # continue using a model after this group has returned/failed.
            error = None
            for name, future in futures.items():
                try:
                    results[name], info = future.result()
                    metadata.append(info)
                except BaseException as failure:
                    error = error or failure
            if error is not None:
                raise error
        else:
            for name in names:
                results[name], info = self._run_node(name, {**parents, **results})
                metadata.append(info)
        emitted = {}
        for name in names:
            node, value = self.compiled["nodes"][name], results[name]
            if name == self.plan.output:
                emitted[name] = value  # Consumers need not resolve lineage.
            else:
                bases = {p: emitted[p] if p in emitted else references[p] for p in node.parents or ("$input",)}
                emitted[name] = DAGPayload(
                    {key: value.columns[key] for key in (self.plan.id_column, *node.writes)},
                    {key: name for key in node.writes},
                    name,
                    bases,
                )
            info = next(e for e in metadata if e["node"] == name)
            info["emitted_column_bytes"] = payload_bytes(emitted[name].columns)
            info["parent_reference_count"] = len(emitted[name].bases)
        # Worker-side monotonic time includes parent materialization, local OOM
        # retries and all nodes, without driver waits or consumer/export time.
        duration = time.perf_counter() - group_started
        for info in metadata:
            info["group_seconds"] = duration
            info["group_input_rows"] = metadata[0]["input_rows"]
        return (*[emitted[n] for n in group.nodes], metadata)

    def close(self):
        self.closed = True
        if self.threads is not None:
            self.threads.shutdown(wait=True)
        self.runners.clear()

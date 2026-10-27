"""Experimental, explicit logical DAG and resident-pool execution contract.

Logical dependencies never depend on placement. Group contraction must preserve
acyclicity; a parallel group may contain only independent, disjoint writers.
This is deliberately a fixed physical plan, not an automatic fusion optimizer.
"""

import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from typing import Dict, Tuple


class DAGContractError(ValueError):
    pass


@dataclass(frozen=True)
class DAGNode:
    name: str
    operator: dict
    parents: Tuple[str, ...] = ()
    reads: Tuple[str, ...] = ()
    writes: Tuple[str, ...] = ()
    batch_size: int = 32
    row_policy: str = "preserve"
    join_keys: Tuple[str, ...] = ()
    initial_batch_size: int = None


@dataclass(frozen=True)
class ExecutionGroup:
    name: str
    nodes: Tuple[str, ...]
    pool: str
    parallel: bool = False


@dataclass(frozen=True)
class WorkerPool:
    name: str
    workers: int
    num_cpus: float = 1
    num_gpus: int = 0


def _topological(dependencies):
    remaining = {key: set(value) for key, value in dependencies.items()}
    if any(not value <= remaining.keys() for value in remaining.values()):
        raise DAGContractError("unknown dependency")
    ordered = []
    while remaining:
        ready = [key for key, value in remaining.items() if not value]
        if not ready:
            raise DAGContractError("cycle in logical DAG or contracted execution groups")
        for key in ready:
            ordered.append(key)
            del remaining[key]
        for value in remaining.values():
            value.difference_update(ready)
    return tuple(ordered)


def _unique(items, kind):
    names = [item.name for item in items]
    if not names or any(not isinstance(n, str) or not n or n == "$input" for n in names):
        raise DAGContractError(f"{kind} must have nonempty names")
    if len(set(names)) != len(names):
        raise DAGContractError(f"duplicate {kind} name")
    return dict(zip(names, items))


@dataclass(frozen=True)
class ResidentDAGPlan:
    nodes: Tuple[DAGNode, ...]
    groups: Tuple[ExecutionGroup, ...]
    pools: Tuple[WorkerPool, ...]
    output: str
    operator_modules: Tuple[str, ...] = ()
    id_column: str = "id"
    output_columns: Tuple[str, ...] = ()
    max_inflight_units: int = 8
    max_batch_bytes: int = 128 * 1024**2
    object_budget_bytes: int = 8 * 1024**3
    task_timeout_seconds: float = 600
    startup_timeout_seconds: float = 600

    def compile(self):
        nodes, groups, pools = (_unique(self.nodes, "node"), _unique(self.groups, "group"), _unique(self.pools, "pool"))
        for value in (self.max_inflight_units, self.max_batch_bytes, self.object_budget_bytes):
            if type(value) is not int or value < 1:
                raise DAGContractError("inflight and byte budgets must be positive integers")
        if not self.id_column or any(
            not math.isfinite(v) or v <= 0 for v in (self.task_timeout_seconds, self.startup_timeout_seconds)
        ):
            raise DAGContractError("invalid ID column or timeout")
        for node in nodes.values():
            if not isinstance(node.operator, dict) or len(node.operator) != 1:
                raise DAGContractError("each node requires one DJ operator recipe")
            if type(node.batch_size) is not int or node.batch_size < 1:
                raise DAGContractError("node batch_size must be positive")
            if node.initial_batch_size is not None and (
                type(node.initial_batch_size) is not int or not 1 <= node.initial_batch_size <= node.batch_size
            ):
                raise DAGContractError("initial_batch_size must be within [1, batch_size]")
            if node.row_policy not in ("preserve", "subset"):
                raise DAGContractError("row_policy must be preserve or subset")
            if self.id_column in node.writes or len(set(node.writes)) != len(node.writes):
                raise DAGContractError("ID is immutable and output columns must be unique")
            if any(not isinstance(c, str) or not c for c in (*node.reads, *node.writes, *node.join_keys)):
                raise DAGContractError("column names must be nonempty strings")
            if len(set(node.parents)) != len(node.parents):
                raise DAGContractError("duplicate parent")
        order = _topological({n: node.parents for n, node in nodes.items()})
        ancestors = {"$input": set()}
        for name in order:
            ancestors[name] = {"$input"}
            for parent in nodes[name].parents:
                ancestors[name].update(ancestors[parent] | {parent})
        if self.output not in nodes or set(nodes) != ancestors[self.output] - {"$input"} | {self.output}:
            raise DAGContractError("every node must contribute to the single output node")
        owner = {}
        for group in groups.values():
            if group.pool not in pools or not group.nodes:
                raise DAGContractError("group requires a known pool and at least one node")
            written = set()
            for name in group.nodes:
                if name not in nodes or name in owner:
                    raise DAGContractError("each node must belong to exactly one group")
                owner[name] = group.name
                if group.parallel:
                    if ancestors[name] & set(group.nodes) or written & set(nodes[name].writes):
                        raise DAGContractError("parallel group contains dependent or conflicting writers")
                    written.update(nodes[name].writes)
        if set(owner) != set(nodes):
            raise DAGContractError("every node must be assigned to an execution group")
        dependencies = {g: set() for g in groups}
        for name, node in nodes.items():
            dependencies[owner[name]].update(owner[p] for p in node.parents if owner[p] != owner[name])
        group_order = _topological(dependencies)
        if {g.pool for g in groups.values()} != set(pools):
            raise DAGContractError("unused worker pool")
        for pool in pools.values():
            if type(pool.workers) is not int or pool.workers < 1:
                raise DAGContractError("worker count must be positive")
            if (
                not math.isfinite(pool.num_cpus)
                or pool.num_cpus <= 0
                or type(pool.num_gpus) is not int
                or pool.num_gpus not in (0, 1)
            ):
                raise DAGContractError("pools require positive CPUs and zero or one whole GPU per worker")
        # Conservative reservation: input plus every logical output may coexist.
        unit_bytes = (len(nodes) + 1) * self.max_batch_bytes
        capacity = min(self.max_inflight_units, self.object_budget_bytes // unit_bytes)
        if capacity < 1:
            raise DAGContractError("object budget cannot admit one bounded work unit")
        return dict(
            nodes=nodes,
            groups=groups,
            pools=pools,
            order=order,
            ancestors=ancestors,
            owner=owner,
            dependencies=dependencies,
            group_order=group_order,
            capacity=capacity,
            unit_reservation_bytes=unit_bytes,
        )

    def fingerprint(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True, allow_nan=False).encode()).hexdigest()


def validate_columns(columns, id_column):
    if not isinstance(columns, dict) or id_column not in columns:
        raise DAGContractError("a column batch with an ID column is required")
    if any(not isinstance(v, (list, tuple)) for v in columns.values()):
        raise DAGContractError("column values must be lists or tuples")
    ids = columns[id_column]
    if any(type(i) not in (str, int) for i in ids) or len(set(ids)) != len(ids):
        raise DAGContractError("IDs must be unique strings or integers")
    if any(len(value) != len(ids) for value in columns.values()):
        raise DAGContractError("column row counts differ")
    return ids


def payload_bytes(value, seen=None):
    """Conservative Python/array payload estimate; excludes model working memory."""
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    size = sys.getsizeof(value)
    if hasattr(value, "nbytes"):
        return size + int(value.nbytes)
    if isinstance(value, dict):
        return size + sum(payload_bytes(k, seen) + payload_bytes(v, seen) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return size + sum(payload_bytes(v, seen) for v in value)
    return size


@dataclass
class DAGPayload:
    columns: dict
    writers: Dict[str, str] = field(default_factory=dict)
    # Intermediate outputs carry their writes and references to parent data.
    origin: str = None
    bases: dict = field(default_factory=dict)


def merge_parents(payloads, node, compiled, id_column):
    """Exact keyed join. Filtering is permitted only before a common fork/join."""
    first = payloads[0]
    ids = validate_columns(first.columns, id_column)
    columns, writers = {}, {}
    for payload in payloads:
        other = validate_columns(payload.columns, id_column)
        if set(other) != set(ids):
            raise DAGContractError(f"{node.name}: branch IDs differ; implicit intersection is forbidden")
        positions = {key: i for i, key in enumerate(other)}
        indices = [positions[key] for key in ids]
        for key in node.join_keys:
            if key not in first.columns or key not in payload.columns:
                raise DAGContractError(f"{node.name}: missing join version {key}")
            if [payload.columns[key][i] for i in indices] != list(first.columns[key]):
                raise DAGContractError(f"{node.name}: branch version mismatch for {key}")
        for key, values in payload.columns.items():
            writer = payload.writers.get(key, "$input")
            if key in writers and writer != writers[key]:
                if writer in compiled["ancestors"][writers[key]]:
                    continue
                if writers[key] not in compiled["ancestors"][writer]:
                    raise DAGContractError(f"{node.name}: ambiguous branch writes to {key}")
            columns[key] = [values[i] for i in indices]
            writers[key] = writer
    return DAGPayload(columns, writers)

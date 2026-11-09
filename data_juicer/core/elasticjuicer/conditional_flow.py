"""Conditional row contracts shared by linear execution and resident forks.

Routing changes the input set of a stage, never the identity of the master
table. The existing actor-local controller sees only selected input rows.
"""

from dataclasses import dataclass

import pyarrow as pa

from .compute_flow import ComputeFlowError

ROW = "__data_juicer_compute_flow_row_id__"
PARTITION = "__data_juicer_logical_partition_id__"
ROUTES = "required_ops"
TYPES = {
    "float32": pa.float32(),
    "string": pa.string(),
    "list_float32": pa.list_(pa.float32()),
    "list_string": pa.list_(pa.string()),
}


def as_table(value):
    return value if isinstance(value, pa.Table) else pa.Table.from_pydict(value)


def specification(op):
    name = getattr(op, "conditional_stage", None)
    outputs = getattr(op, "conditional_outputs", None)
    if name is None:
        return None
    if not isinstance(name, str) or not name or not outputs:
        raise ComputeFlowError("conditional stages require a name and declared outputs")
    if any(kind not in TYPES for kind in outputs.values()):
        raise ComputeFlowError("unknown conditional output type")
    return name, dict(outputs)


def selected_indices(table, stage):
    if ROUTES not in table.column_names:
        raise ComputeFlowError("conditional stage requires a completed Topic route manifest")
    routes = table[ROUTES].to_pylist()
    for required in routes:
        if not isinstance(required, list) or len(required) != len(set(required)):
            raise ComputeFlowError("required_ops must contain unique stage names")
    return [i for i, required in enumerate(routes) if stage in required]


def check_identity(original, result):
    if result.num_rows != original.num_rows:
        raise ComputeFlowError("conditional mapper changed its selected row count")
    for key in original.column_names:
        if key not in result.column_names or not original[key].equals(result[key].cast(original[key].type)):
            raise ComputeFlowError(f"conditional mapper changed preserved field {key}")


def merge_delta(base, selected, result, outputs):
    result = as_table(result)
    source = base.take(pa.array(selected, type=pa.int64()))
    # Existing output columns may be replaced by the same logical stage on
    # explicit checkpoint replay; every other column is immutable.
    preserved = source.drop_columns([c for c in outputs if c in source.column_names])
    check_identity(preserved, result)
    for name, kind in outputs.items():
        if name not in result.column_names:
            raise ComputeFlowError(f"conditional mapper omitted output {name}")
        values = [None] * base.num_rows
        for index, value in zip(selected, result[name].to_pylist()):
            if value is None:
                raise ComputeFlowError(f"required result {name} is null")
            values[index] = value
        array = pa.array(values, type=TYPES[kind])
        base = (
            base.set_column(base.schema.get_field_index(name), name, array)
            if name in base.column_names
            else base.append_column(name, array)
        )
    return base


def invoke_selected(op, batch, call):
    spec = specification(op)
    if spec is None:
        return call(batch)
    table = as_table(batch)
    stage, outputs = spec
    selected = selected_indices(table, stage)
    if selected:
        result = call(table.take(pa.array(selected, type=pa.int64())))
    else:
        result = table.slice(0, 0)
        for name, kind in outputs.items():
            if name not in result.column_names:
                result = result.append_column(name, pa.array([], type=TYPES[kind]))
    return merge_delta(table, selected, result, outputs)


@dataclass(frozen=True)
class ConditionalPlan:
    """One root Topic, independent conditional readers and a CPU suffix."""

    topic: str
    leaves: tuple
    specs: dict

    @classmethod
    def compile(cls, stages, ops):
        if not getattr(ops[0], "conditional_topic", False):
            raise ComputeFlowError("conditional interval must start with Topic")
        leaves, specs, writers, names = [], {}, set(), set()
        for stage, op in zip(stages[1:], ops[1:]):
            spec = specification(op)
            if spec is None:
                raise ComputeFlowError("every conditional leaf must declare its routing contract")
            name, outputs = spec
            if name in names or writers.intersection(outputs):
                raise ComputeFlowError("duplicate conditional stage or overlapping output writer")
            names.add(name)
            writers.update(outputs)
            leaves.append(stage)
            specs[stage] = dict(name=name, outputs=outputs)
        if not leaves:
            raise ComputeFlowError("conditional interval needs at least one leaf")
        for op in ops[1:]:
            own_outputs = specification(op)[1]
            if set(getattr(op, "_input_columns", None) or ()).intersection(writers - own_outputs.keys()):
                raise ComputeFlowError("conditional leaves cannot read sibling outputs")
        return cls(stages[0], tuple(leaves), specs)

    def route(self, table):
        table = as_table(table)
        required = table[ROUTES].to_pylist()
        names = {spec["name"] for spec in self.specs.values()}
        if any(not isinstance(r, list) or len(r) != len(set(r)) or not set(r) <= names for r in required):
            raise ComputeFlowError("Topic produced unknown/duplicate required_ops")
        return {stage: selected_indices(table, spec["name"]) for stage, spec in self.specs.items()}

    def join(self, base, deltas):
        routes = self.route(base)
        expected = {stage for stage, indices in routes.items() if indices}
        if set(deltas) != expected:
            raise ComputeFlowError("join requires exactly the selected branch results")
        original = base
        for stage in self.leaves:
            outputs = self.specs[stage]["outputs"]
            indices = routes[stage]
            delta = deltas.get(stage)
            if delta is None:
                delta = original.slice(0, 0)
                for name, kind in outputs.items():
                    if name not in delta.column_names:
                        delta = delta.append_column(name, pa.array([], type=TYPES[kind]))
            # Each branch read the Topic snapshot, not siblings' results.
            checked = merge_delta(original, indices, delta, outputs)
            for name in outputs:
                base = (
                    base.set_column(base.schema.get_field_index(name), name, checked[name])
                    if name in base.column_names
                    else base.append_column(name, checked[name])
                )
        return base

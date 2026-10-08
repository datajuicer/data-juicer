"""Table adapter for independent Data-Juicer Mapper and Filter operators.

The adapter instantiates the operator inside its runner/actor. It supports
ordinary per-table Mapper and Filter behavior. Operators with external side
effects or cross-unit state require an additional replay contract.
"""

from typing import Callable, Optional

import pyarrow as pa

from data_juicer.core.executor.unit_protocol import ProtocolError
from data_juicer.core.executor.unit_runner import UnitStage
from data_juicer.ops import Filter, Mapper
from data_juicer.ops.base_op import TAGGING_OPS
from data_juicer.utils.constant import Fields


class _OperatorTables:
    def __init__(self, operator_type, args, kwargs, empty_schema):
        self.operator_type = operator_type
        self.args = args
        self.kwargs = kwargs
        self.empty_schema = empty_schema
        self.operator = None

    def __call__(self, table: pa.Table):
        if self.operator is None:
            self.operator = self.operator_type(*self.args, **self.kwargs)
        op = self.operator
        if isinstance(op, Mapper) and op._name in TAGGING_OPS.modules and Fields.meta not in table.column_names:
            table = table.append_column(Fields.meta, pa.array([{} for _ in range(table.num_rows)]))
        batch_size = max(1, int(getattr(op, "batch_size", 1))) if op.is_batched_op() else table.num_rows or 1
        if table.num_rows == 0:
            return (pa.Table.from_batches([], schema=self.empty_schema(table.schema)),)
        return tuple(
            self._process_batch(table.slice(start, batch_size)) for start in range(0, table.num_rows, batch_size)
        )

    def _process_batch(self, table: pa.Table):
        op = self.operator
        if isinstance(op, Mapper):
            if op.is_batched_op():
                result = op.process(table)
                if isinstance(result, pa.Table):
                    output = result
                elif isinstance(result, dict):
                    output = pa.table(result)
                else:
                    raise ProtocolError("Batched Mapper must return a table or dict of columns")
            else:
                rows = []
                for sample in table.to_pylist():
                    mapped = op.process(sample)
                    if isinstance(mapped, dict):
                        rows.append(mapped)
                    elif isinstance(mapped, list) and all(isinstance(item, dict) for item in mapped):
                        rows.extend(mapped)
                    else:
                        raise ProtocolError("Mapper must return a row or list of rows")
                output = (
                    pa.Table.from_pylist(rows)
                    if rows
                    else pa.Table.from_batches([], schema=self.empty_schema(table.schema))
                )
            if output.num_rows == 0:
                output = pa.Table.from_batches([], schema=self.empty_schema(table.schema))
            return output

        if isinstance(op, Filter):
            if Fields.stats not in table.column_names:
                table = table.append_column(Fields.stats, pa.array([{} for _ in range(table.num_rows)]))
            if op.is_batched_op():
                stats = op.compute_stats(table)
                stats_table = stats if isinstance(stats, pa.Table) else pa.table(stats)
                mask = list(op.process(stats_table.to_pydict()))
            else:
                computed = [op.compute_stats(sample) for sample in table.to_pylist()]
                stats_table = pa.Table.from_pylist(computed)
                mask = [bool(op.process(sample)) for sample in computed]
            if len(mask) != stats_table.num_rows:
                raise ProtocolError("Filter mask length differs from input rows")
            return stats_table.filter(pa.array(mask, type=pa.bool_()))

        raise TypeError("Recovery-unit adapter supports Mapper and Filter only")


def operator_unit_stage(op, *, empty_schema: Optional[Callable[[pa.Schema], pa.Schema]] = None) -> UnitStage:
    """Build a per-actor operator stage from an existing op configuration.

    Mapper schema changes must be described by ``empty_schema`` so units with
    no output can still be committed with the correct column structure.
    """
    if not isinstance(op, (Mapper, Filter)):
        raise TypeError("Recovery-unit adapter supports Mapper and Filter only")
    if isinstance(op, Filter) and op.stats_export_path is not None:
        raise ProtocolError("Filter stats export needs its own transactional sink")
    if isinstance(op, Mapper) and empty_schema is None:
        raise ValueError("Mapper requires an empty-output schema contract")
    schema_fn = empty_schema or (lambda schema: schema)
    adapter = _OperatorTables(op.__class__, op._init_args, op._init_kwargs, schema_fn)
    return UnitStage(adapter, schema_fn)

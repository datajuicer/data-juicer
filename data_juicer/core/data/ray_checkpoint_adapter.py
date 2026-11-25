"""Ray-internal integration for streaming, non-fusible checkpoint boundaries.

Keep private Ray APIs here. UDFs, actor pools and microbatching continue to be
planned by Ray; this operator only holds block references and writer metadata.
"""

import base64
import inspect
import os
from collections import deque
from dataclasses import is_dataclass
from functools import partial

import pyarrow as pa
import ray
from ray.data._internal.execution.interfaces import RefBundle
from ray.data._internal.execution.interfaces.physical_operator import MetadataOpTask, PhysicalOperator
from ray.data._internal.execution.operators.base_physical_operator import OneToOneOperator
from ray.data._internal.logical.interfaces import LogicalOperator, LogicalPlan
from ray.data._internal.planner.planner import Planner
from ray.data.block import BlockMetadata
from ray.data.datasource import Datasource, ReadTask

from data_juicer.utils.ckpt_utils import write_checkpoint_block


def validate_checkpoint_adapter():
    """Fail before execution when the installed Ray contract is unsupported."""
    supported = {"2.51.0", "2.56.1"}
    required = ("all_inputs_done", "get_active_tasks", "_do_shutdown")
    if (
        ray.__version__ not in supported
        or not all(hasattr(PhysicalOperator, name) for name in required)
        or not any(hasattr(PhysicalOperator, name) for name in ("should_add_input", "can_add_input"))
        or not hasattr(Planner, "_DEFAULT_PLAN_FNS")
        or "task_done_callback" not in inspect.signature(MetadataOpTask).parameters
    ):
        raise RuntimeError(
            f"Streaming checkpoints require the Ray 2.51.0/2.56.1 adapter contract; found {ray.__version__}"
        )


class Checkpoint(LogicalOperator):
    def __init__(self, input_op, manager, op_idx, op_name, partition_ids, partition_column, on_complete, schema):
        if is_dataclass(LogicalOperator):
            # Ray 2.56 logical operators are frozen dataclasses.
            super().__init__()
            object.__setattr__(self, "_name", f"Checkpoint[{op_idx}:{op_name}]")
            object.__setattr__(self, "_input_dependencies", [input_op])
        else:
            super().__init__(f"Checkpoint[{op_idx}:{op_name}]", [input_op])
        self.manager = manager
        self.op_idx = op_idx
        self.op_name = op_name
        self.partition_ids = partition_ids
        self.partition_column = partition_column
        self.on_complete = on_complete
        self.schema = schema

    @property
    def num_outputs(self):
        return None

    def infer_schema(self):
        return self.input_dependencies[0].infer_schema()

    def _get_args(self):
        return {"op_idx": self.op_idx, "op_name": self.op_name, "partition_ids": self.partition_ids}


class CheckpointBlockSplitter(OneToOneOperator):
    """Normalize multi-block deliveries before checkpoint admission/backpressure."""

    def __init__(self, input_op, data_context):
        super().__init__("CheckpointBlocks", input_op, data_context)
        self._bundle = None
        self._index = 0
        self._input_metric_kwargs = {"input_index": 0} if ray.__version__ == "2.56.1" else {}

    def should_add_input(self):
        return self._bundle is None

    def can_add_input(self):
        return self.should_add_input()

    def _add_input_inner(self, refs, input_index):
        assert self._bundle is None
        if refs.blocks:
            self._bundle = refs
            self._index = 0
            self._metrics.on_input_queued(refs, **self._input_metric_kwargs)

    def has_next(self):
        return self._bundle is not None

    def _get_next_inner(self):
        refs = self._bundle
        kwargs = {"slices": [refs.slices[self._index]]} if hasattr(refs, "slices") else {}
        output = RefBundle([refs.blocks[self._index]], owns_blocks=refs.owns_blocks, schema=refs.schema, **kwargs)
        self._index += 1
        if self._index == len(refs.blocks):
            self._metrics.on_input_dequeued(refs, **self._input_metric_kwargs)
            self._bundle = None
        return output

    def get_stats(self):
        return {}

    def implements_accurate_memory_accounting(self):
        return True

    def _do_shutdown(self, force):
        self._bundle = None


class StreamingCheckpointOperator(OneToOneOperator):
    """Write at most two blocks concurrently; forward successful refs in order."""

    def __init__(self, logical_op, input_op, data_context):
        super().__init__(logical_op.name, input_op, data_context)
        self.logical_op = logical_op
        self.transaction = None
        self._waiting = deque()
        self._pending = deque()
        self._outputs = deque()
        self._next_index = 0
        self._pending_bytes = 0
        self._byte_limit = 2 * data_context.target_max_block_size
        self._schema = logical_op.schema
        self._published = False
        self._input_metric_kwargs = {"input_index": 0} if ray.__version__ == "2.56.1" else {}

    def start(self, options):
        super().start(options)
        op = self.logical_op
        self.transaction = op.manager.begin_checkpoint(op.op_idx, op.op_name, op.partition_ids)

    def should_add_input(self):
        # Include completed-but-not-consumed outputs, so downstream backpressure
        # cannot turn this node into an unbounded object-reference buffer.
        return (
            not self._waiting and len(self._pending) + len(self._outputs) < 2 and self._pending_bytes < self._byte_limit
        )

    def can_add_input(self):
        # Ray 2.56 renamed the scheduler admission hook.
        return self.should_add_input()

    def _add_input_inner(self, refs, input_index):
        assert input_index == 0
        if len(refs.blocks) > 1:
            raise ValueError("Checkpoint input must pass through the block splitter")
        if refs.schema is not None:
            self._schema = refs.schema
        for index, entry in enumerate(refs.blocks):
            kwargs = {}
            if hasattr(refs, "slices"):
                kwargs["slices"] = [refs.slices[index]]
            bundle = RefBundle([entry], owns_blocks=refs.owns_blocks, schema=refs.schema, **kwargs)
            self._waiting.append(bundle)
            self._metrics.on_input_queued(bundle, **self._input_metric_kwargs)
        self._submit()

    def _submit(self):
        while self._waiting and len(self._pending) + len(self._outputs) < 2:
            bundle = self._waiting[0]
            size = bundle.size_bytes()
            # An oversized block can make progress only by itself.
            if self._pending and self._pending_bytes + size > self._byte_limit:
                break
            self._waiting.popleft()
            block = bundle.block_refs[0]
            block_slice = getattr(bundle, "slices", [None])[0]
            offsets = (block_slice.start_offset, block_slice.end_offset) if block_slice else None
            result = ray.remote(num_cpus=0, num_gpus=0, max_retries=3)(write_checkpoint_block).remote(
                block, self.transaction.directories, self.logical_op.partition_column, offsets
            )
            index = self._next_index
            self._next_index += 1
            item = {"bundle": bundle, "result": result, "done": False, "size": size}
            item["task"] = MetadataOpTask(index, result, partial(self._write_done, item))
            self._pending.append(item)
            self._pending_bytes += size

    def _write_done(self, item):
        # ray.get fetches small metadata only. A writer error fails the dataset
        # execution, and its unpublished transaction is discarded in shutdown.
        item["metadata"] = ray.get(item["result"])
        item["done"] = True
        while self._pending and self._pending[0]["done"]:
            head = self._pending.popleft()
            self.transaction.add_result(head["metadata"])
            bundle = head["bundle"]
            self._metrics.on_input_dequeued(bundle, **self._input_metric_kwargs)
            self._metrics.on_output_queued(bundle)
            self._outputs.append(bundle)
            self._pending_bytes -= head["size"]
        self._submit()
        self._maybe_publish()

    def all_inputs_done(self):
        super().all_inputs_done()
        self._maybe_publish()

    def _maybe_publish(self):
        if self._inputs_complete and not self._waiting and not self._pending and not self._published:
            schema = self._schema
            column = self.logical_op.partition_column
            if isinstance(schema, pa.Schema) and column and column in schema.names:
                schema = schema.remove(schema.get_field_index(column))
            self.transaction.publish(schema)
            self._published = True
            if self.logical_op.on_complete:
                self.logical_op.on_complete()

    def has_next(self):
        return bool(self._outputs)

    def _get_next_inner(self):
        bundle = self._outputs.popleft()
        self._metrics.on_output_dequeued(bundle)
        self._submit()
        return bundle

    def get_active_tasks(self):
        return [item["task"] for item in self._pending if not item["done"]]

    def num_active_tasks(self):
        # Prevent completion while a multi-block bundle still has queued writes.
        return len(self._pending) + len(self._waiting)

    def get_stats(self):
        return {}

    def incremental_resource_usage(self):
        from ray.data._internal.execution.interfaces import ExecutionResources

        return ExecutionResources.zero()

    def implements_accurate_memory_accounting(self):
        return True

    def _do_shutdown(self, force):
        # Stop writers before removing their directories. Published siblings are
        # outside the transaction root and survive downstream errors/cancellation.
        super()._do_shutdown(force=True)
        if self.transaction:
            self.transaction.abort()
        self._waiting.clear()
        self._pending.clear()
        self._outputs.clear()


def _plan_checkpoint(logical_op, physical_children, data_context):
    splitter = CheckpointBlockSplitter(physical_children[0], data_context)
    return StreamingCheckpointOperator(logical_op, splitter, data_context)


def insert_checkpoint(
    dataset, manager, op_idx, op_name, partition_ids, partition_column=None, on_complete=None, schema=None
):
    validate_checkpoint_adapter()
    Planner._DEFAULT_PLAN_FNS[Checkpoint] = _plan_checkpoint
    op = Checkpoint(
        dataset._logical_plan.dag, manager, op_idx, op_name, partition_ids, partition_column, on_complete, schema
    )
    # Each boundary retains upstream order even if task completions are reordered.
    context = dataset.context.copy()
    context.execution_options.preserve_order = True
    if ray.__version__ == "2.56.1":
        return ray.data.Dataset(
            LogicalPlan(op, context),
            context,
            dataset._in_stats,
            cache=dataset._cache.copy(),
            dataset_name=dataset._dataset_name,
        )
    plan = dataset._plan.copy()
    plan._context = context
    return ray.data.Dataset(plan, LogicalPlan(op, context))


def _read_file(path, entry):
    if entry["format"] == "arrow":
        with pa.memory_map(path, "r") as source:
            yield pa.ipc.open_file(source).read_all()
    else:
        import pyarrow.parquet as pq

        yield pq.read_table(path)


class CheckpointDatasource(Datasource):
    def __init__(self, path, manifest):
        self.path = path
        self.manifest = manifest

    def estimate_inmemory_data_size(self):
        return None

    def get_read_tasks(self, parallelism, per_task_row_limit=None, data_context=None):
        entries = self.manifest["files"]
        if not entries:
            schema = self.manifest.get("schema")
            schema = pa.ipc.read_schema(pa.BufferReader(base64.b64decode(schema))) if schema else pa.schema([])
            return [
                ReadTask(
                    lambda: iter([pa.Table.from_batches([], schema=schema)]),
                    BlockMetadata(num_rows=0, size_bytes=0, input_files=[], exec_stats=None),
                    schema=schema,
                )
            ]
        return [
            ReadTask(
                partial(_read_file, os.path.join(self.path, entry["path"]), entry),
                BlockMetadata(
                    num_rows=entry["rows"], size_bytes=entry["size_bytes"], input_files=[entry["path"]], exec_stats=None
                ),
            )
            for entry in entries
        ]


def read_checkpoint(path, manifest):
    validate_checkpoint_adapter()
    dataset = ray.data.read_datasource(CheckpointDatasource(path, manifest))
    dataset.context.execution_options.preserve_order = True
    return dataset

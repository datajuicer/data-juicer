import hashlib
from dataclasses import dataclass
from typing import List, Optional, Union

import pyarrow.fs as pafs
from ray.data import ActorPoolStrategy
from ray.data.dataset import MaterializedDataset
from ray.data.datasource.path_util import _resolve_paths_and_filesystem

from data_juicer.core.data.ray_batch_adapter import (
    apply_batch_operation,
    map_whole_blocks,
)


@dataclass
class BatchOperation:
    function: object
    options: dict

    def bind(self):
        """Construct a callable once per worker; plain functions need no setup."""
        if isinstance(self.function, type):
            return self.function(
                *(self.options.get("fn_constructor_args") or ()),
                **(self.options.get("fn_constructor_kwargs") or {}),
            )
        return self.function


@dataclass(frozen=True)
class Materialize:
    """A global barrier, which cannot execute within a checkpointed batch."""


@dataclass(frozen=True)
class StageResources:
    compute: object
    num_cpus: Optional[float]
    num_gpus: Optional[float]
    runtime_env: object

    def ray_options(self):
        return {
            "compute": self.compute,
            "num_cpus": self.num_cpus,
            "num_gpus": self.num_gpus,
            "runtime_env": self.runtime_env,
        }


@dataclass
class StatsExport:
    path: str
    options: dict

    def publish(self, data, identity):
        prefix = "dj-stats-" + hashlib.sha256("\0".join(identity).encode()).hexdigest() + "_"
        destinations, filesystem = _resolve_paths_and_filesystem(self.path, self.options.get("filesystem"))
        for entry in filesystem.get_file_info(pafs.FileSelector(destinations[0], allow_not_found=True)):
            if entry.is_file and entry.base_name.startswith(prefix) and entry.base_name.endswith(".json"):
                filesystem.delete_file(entry.path)
        # A stable ownership prefix permits replay without deleting unrelated exports.
        data._set_uuid(prefix)
        data.write_json(self.path, **self.options)


class RayBatchPlan:
    def __init__(self):
        self.steps: List[Union[BatchOperation, StatsExport, Materialize]] = []
        self.resources: Optional[StageResources] = None

    def map_operator(self, function, *, compute, num_cpus=None, num_gpus=None, runtime_env=None, **options):
        """Declare the operator's scheduling resources, separately from helper batches."""
        if self.resources is not None:
            raise ValueError("A batch plan can declare only one operator resource owner")
        resources = StageResources(compute, num_cpus, num_gpus, runtime_env)
        self.map_batches(function, **resources.ray_options(), **options)
        self.resources = resources
        return self

    def map_batches(self, function, **options):
        batch_size = options.get("batch_size")
        if isinstance(batch_size, int) and batch_size < 1:
            raise ValueError("Batch size can't be negative or 0")
        if (options.get("num_gpus") or 0) > 0 and batch_size in (None, "auto"):
            raise ValueError("You must provide an explicit batch_size when requesting GPUs")
        self.steps.append(BatchOperation(function, options))
        return self

    def materialize(self):
        self.steps.append(Materialize())
        return self

    def write_json(self, path, **options):
        self.steps.append(StatsExport(path, options))

    def apply(self, dataset, recovery=None):
        if recovery is None:
            materialized = isinstance(dataset, MaterializedDataset)
            for step in self.steps:
                if isinstance(step, BatchOperation):
                    dataset = dataset.map_batches(step.function, **step.options)
                    materialized = False
                elif isinstance(step, StatsExport):
                    # The statistics writer and downstream filter consume the
                    # same computed rows, including any shared operator state.
                    if not materialized:
                        dataset = dataset.materialize()
                        materialized = True
                    dataset.write_json(step.path, **step.options)
                elif isinstance(step, Materialize):
                    dataset = dataset.materialize()
                    materialized = True
                else:
                    raise TypeError(f"Unknown batch plan step: {type(step).__name__}")
            return dataset

        from data_juicer.core.executor import ray_partition_runner
        from data_juicer.core.executor.partition_protocol import ProtocolError

        if any(isinstance(step, Materialize) for step in self.steps):
            raise ProtocolError("A global materialization barrier cannot execute within a checkpointed batch")
        for step in self.steps:
            if isinstance(step, StatsExport):
                recovery.register_stats(step)
        if self.resources is None:
            raise ProtocolError("Checkpointed batch plans require explicit operator resources")
        if dataset is None:
            return None
        options = self.resources.ray_options()
        if isinstance(options["compute"], ActorPoolStrategy):
            function = ray_partition_runner.CheckpointStageActor
            options.update(fn_constructor_args=(recovery.context, self))
        else:
            function = ray_partition_runner.run_checkpoint_task
            options.update(fn_args=(recovery.context, self))
        return map_whole_blocks(dataset, function, batch_format="pyarrow", zero_copy_batch=True, **options)


class BatchPlanWorker:
    def __init__(self, plan):
        self.plan = plan
        self.functions = {}

    def _map(self, tables, index, step):
        # Defer construction until a batch actually reaches this operation.
        def invoke(batch, *args, **kwargs):
            if index not in self.functions:
                self.functions[index] = step.bind()
            return self.functions[index](batch, *args, **kwargs)

        yield from apply_batch_operation(tables, step, invoke)

    def process_partition(self, tables, *, write_stats=None):
        for index, step in enumerate(self.plan.steps):
            if isinstance(step, BatchOperation):
                tables = self._map(tables, index, step)
            elif isinstance(step, StatsExport):
                if write_stats is None:
                    raise RuntimeError("Statistics export requires an attempt-owned writer")
                tables = write_stats(tables)
            elif isinstance(step, Materialize):
                raise RuntimeError("A global materialization barrier cannot execute within a checkpointed batch")
            else:
                raise TypeError(f"Unknown batch plan step: {type(step).__name__}")
        yield from tables

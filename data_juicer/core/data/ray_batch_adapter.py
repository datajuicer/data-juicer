"""Ray-version-specific operations used by checkpointed batch execution.

Keep private Ray APIs here. Batch UDF invocation, generator handling, and
output validation use Ray's own transform rather than a second implementation.
"""

import inspect

from ray.data import Dataset
from ray.data._internal.block_batching import batch_blocks
from ray.data._internal.planner.plan_udf_map_op import (
    _generate_transform_fn_for_map_batches,
)
from ray.data.block import BlockAccessor


def map_whole_blocks(dataset, function, **options):
    """Schedule one checkpoint wrapper per block, without Ray's row bundling."""
    method = dataset._map_batches_without_batch_size_validation
    accepted = inspect.signature(method).parameters
    defaults = {
        name: parameter.default
        for name, parameter in inspect.signature(Dataset.map_batches).parameters.items()
        if name in accepted and parameter.default is not inspect.Parameter.empty
    }
    return method(function, **{**defaults, **options, "batch_size": None})


def apply_batch_operation(tables, operation, function):
    """Apply a planned operation using Ray's batch UDF transform."""
    options = operation.options
    batch_size = options.get("batch_size")
    if batch_size == "auto":
        from ray.data._internal.execution.operators.map_transformer import (
            _compute_auto_batch_size,
        )

        batch_size, tables = _compute_auto_batch_size(tables)
    zero_copy = options.get("zero_copy_batch", True)
    batches = batch_blocks(
        iter(tables),
        batch_size=batch_size,
        batch_format=options.get("batch_format", "default"),
        ensure_copy=not zero_copy,
    )

    def invoke(batch):
        # Arrow-to-NumPy conversion can recreate readonly views even after
        # copying the block. Preserve the explicit writable-batch contract.
        if not zero_copy and isinstance(batch, dict):
            batch = {key: value if value.flags.writeable else value.copy() for key, value in batch.items()}
        return function(batch, *(options.get("fn_args") or ()), **(options.get("fn_kwargs") or {}))

    transform = _generate_transform_fn_for_map_batches(invoke)
    for result in transform(batches, None):
        yield BlockAccessor.for_block(BlockAccessor.batch_to_block(result)).to_arrow()

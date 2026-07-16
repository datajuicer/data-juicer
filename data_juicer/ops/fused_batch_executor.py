"""Shared batch-local execution helpers for sequential fused operators."""

from copy import deepcopy
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Iterable, Optional

from data_juicer.ops.base_op import NON_STATS_FILTERS, TAGGING_OPS, Filter, Mapper
from data_juicer.utils.common_utils import check_op_method_param
from data_juicer.utils.constant import Fields


@dataclass(frozen=True)
class SequentialBatchExecutionPolicy:
    """Behavior owned by an outer fused operator, not by the shared loop."""

    copy_input: bool = False
    shared_context: bool = False
    use_op_wrappers: bool = True
    validate: bool = True
    ensure_fields: bool = True


GENERAL_FUSED_EXECUTION_POLICY = SequentialBatchExecutionPolicy(
    copy_input=True,
    shared_context=True,
    use_op_wrappers=False,
    validate=False,
    ensure_fields=False,
)


def get_batch_size(samples: Any) -> int:
    if not samples:
        return 0
    return len(next(iter(samples.values())))


def execute_sequential_batch(
    samples: Any,
    ops: Iterable[Any],
    *,
    rank=None,
    owner_name: str = "fused op",
    policy: SequentialBatchExecutionPolicy = SequentialBatchExecutionPolicy(),
    cleanup_columns: Optional[Iterable[str]] = None,
    on_op_complete: Optional[Callable[[Any, float], None]] = None,
) -> Any:
    """Execute mapper/filter ops in order inside one batch stage.

    Construction, resource planning, and dataset-level preparation remain the
    responsibility of the outer fused operator. This helper owns only the
    shared batch execution semantics.
    """
    batch = deepcopy(samples) if policy.copy_input else samples
    context_rows = None
    if policy.shared_context:
        context_rows = [{} for _ in range(get_batch_size(batch))]
        batch[Fields.context] = context_rows

    try:
        for op in ops:
            op_t0 = perf_counter() if on_op_complete else None
            if isinstance(op, Mapper):
                if policy.ensure_fields:
                    batch = _ensure_meta_if_needed(batch, op, owner_name)
                batch = _run_mapper(op, batch, rank, policy, owner_name)
            elif isinstance(op, Filter):
                if policy.ensure_fields:
                    batch = _ensure_meta_if_needed(batch, op, owner_name)
                    batch = _ensure_stats_if_needed(batch, op, owner_name)
                batch = _run_filter(op, batch, rank, policy, owner_name)
            else:
                raise NotImplementedError(
                    f"[{owner_name}] does not support op [{op._name}] of type "
                    f"[{type(op).__name__}]; only Mapper and Filter are supported."
                )

            if on_op_complete:
                on_op_complete(op, (perf_counter() - op_t0) * 1000.0)
            if get_batch_size(batch) == 0:
                break
    finally:
        if context_rows is not None:
            current_context_rows = batch.get(Fields.context, [])
            _cleanup_context_rows([*context_rows, *current_context_rows])
            batch.pop(Fields.context, None)

    for column in cleanup_columns or []:
        batch.pop(column, None)
    return batch


def _run_mapper(op, batch, rank, policy, owner_name):
    process_method = op.process if policy.use_op_wrappers else op.process_batched
    process_args = {"rank": rank} if _uses_cuda(op, policy.use_op_wrappers) else {}
    if policy.shared_context and check_op_method_param(op.process, "context"):
        process_args["context"] = True
    result = process_method(batch, **process_args)
    return _validate_batch(result, op, owner_name, "process", policy.validate)


def _run_filter(op, batch, rank, policy, owner_name):
    compute_method = op.compute_stats if policy.use_op_wrappers else op.compute_stats_batched
    process_method = op.process if policy.use_op_wrappers else op.process_batched
    compute_args = {"rank": rank} if _uses_cuda(op, policy.use_op_wrappers) else {}
    if policy.shared_context and check_op_method_param(op.compute_stats, "context"):
        compute_args["context"] = True
    result = compute_method(batch, **compute_args)
    result = _validate_batch(result, op, owner_name, "compute_stats", policy.validate)

    keep_mask = list(process_method(result))
    num_samples = get_batch_size(result)
    if policy.validate and len(keep_mask) != num_samples:
        raise ValueError(
            f"Filter sub-op [{op._name}] returned keep mask length "
            f"[{len(keep_mask)}], expected [{num_samples}] inside [{owner_name}]."
        )
    if policy.validate:
        kept_indices = [idx for idx, keep in enumerate(keep_mask) if keep]
        return {key: [values[idx] for idx in kept_indices] for key, values in result.items()}
    return {key: [value for value, keep in zip(values, keep_mask) if keep] for key, values in result.items()}


def _validate_batch(result, op, owner_name, method_name, validate):
    if result is None:
        raise ValueError(f"Sub-op [{op._name}] returned None from {method_name} inside [{owner_name}].")
    if validate and not isinstance(result, dict):
        raise ValueError(
            f"Sub-op [{op._name}] returned unsupported batch type "
            f"[{type(result).__name__}] from {method_name} inside [{owner_name}]."
        )
    return result


def _uses_cuda(op, use_op_wrappers):
    if use_op_wrappers:
        return op.use_cuda()
    return op.accelerator == "cuda"


def _ensure_meta_if_needed(samples, op, owner_name):
    if not _needs_meta(samples, op):
        return samples
    return _ensure_dict_column(samples, Fields.meta, op, owner_name)


def _ensure_stats_if_needed(samples, op, owner_name):
    if not _needs_stats(samples, op):
        return samples
    return _ensure_dict_column(samples, Fields.stats, op, owner_name)


def _ensure_dict_column(samples, column, op, owner_name):
    num_samples = get_batch_size(samples)
    values = samples.get(column)
    if values is None or len(values) == 0:
        samples[column] = [{} for _ in range(num_samples)]
    elif len(values) != num_samples:
        raise ValueError(
            f"Column [{column}] length [{len(values)}] does not match batch "
            f"size [{num_samples}] before sub-op [{op._name}] inside [{owner_name}]."
        )
    else:
        for idx, value in enumerate(values):
            if value is None:
                values[idx] = {}
    return samples


def _needs_meta(samples, op):
    if Fields.meta in samples:
        return True
    if getattr(op, "_requires_meta", False):
        return True
    if op._name in TAGGING_OPS.modules:
        return True
    output_columns = getattr(op, "_output_columns", []) or []
    return any(str(column).startswith(Fields.meta) for column in output_columns)


def _needs_stats(samples, op):
    if Fields.stats in samples:
        return True
    if isinstance(op, Filter) and op._name not in NON_STATS_FILTERS.modules:
        return True
    output_columns = getattr(op, "_output_columns", []) or []
    return any(str(column).startswith(Fields.stats) for column in output_columns)


def _cleanup_context_rows(context_rows):
    context_values = []
    seen_values = set()
    for context in context_rows:
        if not isinstance(context, dict):
            continue
        for value in context.values():
            value_id = id(value)
            if value_id not in seen_values:
                seen_values.add(value_id)
                context_values.append(value)
    av_values = [value for value in context_values if type(value).__module__.startswith("av.")]
    if not av_values:
        return

    from data_juicer.utils.lazy_loader import LazyLoader
    from data_juicer.utils.video_utils import setup_av

    av = LazyLoader("av", post_import=setup_av)
    for value in av_values:
        if isinstance(value, av.container.InputContainer):
            value.streams.video[0].close()
            value.close()

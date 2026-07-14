"""Run mapper/filter sub-ops with a batch-local shared context."""

from typing import Any, Dict, Iterable, List, Optional

from data_juicer.ops.base_op import (
    NON_STATS_FILTERS,
    OP,
    OPERATORS,
    TAGGING_OPS,
    Filter,
    Mapper,
)
from data_juicer.ops.load import load_ops
from data_juicer.utils.common_utils import check_op_method_param
from data_juicer.utils.constant import Fields

OP_NAME = "fused_shared_context_op"


@OPERATORS.register_module(OP_NAME)
class FusedSharedContextOp(Mapper):
    """Run mapper/filter sub-ops sequentially with shared batch context.

    The context is owned by this fused stage. It is created once per input
    row, kept aligned when filters drop rows, and removed from every batch
    before the stage returns. Context values that own PyAV containers are
    closed exactly once, including values belonging to rows dropped by an
    inner filter or batches replaced by an inner mapper.

    This operator is intentionally configured manually. Sub-ops that reuse a
    context key must agree on what that key represents. In particular, a
    mapper must not invalidate an already-cached value (for example, cached
    lines after changing the source text) unless it also updates or removes
    that context value.
    """

    _batched_op = True

    def __init__(
        self,
        batch_size: int = 1,
        fused_op_list: Optional[List[Dict[str, Dict[str, Any]]]] = None,
        fused_ops: Optional[List[Any]] = None,
        *args,
        **kwargs,
    ):
        """
        Args:
            batch_size: Outer batch size for the fused stage.
            fused_op_list: Standard Data-Juicer op config dictionaries.
            fused_ops: Pre-built op instances, primarily for tests and
                programmatic construction.
        """
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("FusedSharedContextOp: batch_size must be a positive integer.")

        kwargs["batch_size"] = batch_size
        super().__init__(*args, **kwargs)

        if fused_op_list is not None and fused_ops is not None:
            raise ValueError("FusedSharedContextOp: provide either fused_op_list or fused_ops, not both.")

        self.fused_op_list = list(fused_op_list or [])
        self.fused_ops = list(fused_ops) if fused_ops is not None else load_ops(self.fused_op_list)
        self._validate_sub_ops()

        if any(op.accelerator == "cuda" for op in self.fused_ops):
            self.accelerator = "cuda"

        runtime_nps = [op.runtime_np() for op in self.fused_ops]
        runtime_nps = [runtime_np for runtime_np in runtime_nps if runtime_np is not None]
        if runtime_nps:
            self.num_proc = min(runtime_nps)

    def process_batched(self, samples, rank=None):
        if not self.fused_ops:
            return samples

        self._validate_batch(samples, "input batch")
        if Fields.context in samples:
            raise ValueError(
                "FusedSharedContextOp owns Fields.context and does not accept "
                "an input batch that already contains it."
            )
        if self._batch_size(samples) == 0:
            return samples

        tracked_batches = []
        tracked_batch_ids = set()
        tracked_contexts = []
        tracked_context_ids = set()

        samples[Fields.context] = [{} for _ in range(self._batch_size(samples))]
        self._track_context_ownership(
            samples,
            tracked_batches,
            tracked_batch_ids,
            tracked_contexts,
            tracked_context_ids,
        )

        try:
            for op in self.fused_ops:
                previous_contexts = samples[Fields.context]
                if isinstance(op, Mapper):
                    samples = self._run_mapper(op, samples, rank=rank)
                    samples = self._prepare_sub_op_result(previous_contexts, samples, op)
                    self._track_context_ownership(
                        samples,
                        tracked_batches,
                        tracked_batch_ids,
                        tracked_contexts,
                        tracked_context_ids,
                    )
                else:
                    samples = self._compute_filter(op, samples, rank=rank)
                    samples = self._prepare_sub_op_result(previous_contexts, samples, op)
                    self._track_context_ownership(
                        samples,
                        tracked_batches,
                        tracked_batch_ids,
                        tracked_contexts,
                        tracked_context_ids,
                    )
                    keep_mask = self._filter_keep_mask(op, samples)
                    samples = self._filter_batch(samples, keep_mask, op)
                    self._track_context_ownership(
                        samples,
                        tracked_batches,
                        tracked_batch_ids,
                        tracked_contexts,
                        tracked_context_ids,
                    )

                if self._batch_size(samples) == 0:
                    break
            return samples
        finally:
            for batch in list(tracked_batches):
                self._track_context_ownership(
                    batch,
                    tracked_batches,
                    tracked_batch_ids,
                    tracked_contexts,
                    tracked_context_ids,
                )
            self._cleanup_contexts(tracked_contexts)
            for batch in tracked_batches:
                batch.pop(Fields.context, None)

    def _run_mapper(self, op, samples, rank=None):
        samples = self._ensure_meta_if_needed(samples, op)
        call_kwargs = self._build_call_kwargs(op.process, op, rank=rank, context=True)
        if op.is_batched_op():
            return op.process(samples, **call_kwargs)
        return op.process_batched(samples, **call_kwargs)

    def _compute_filter(self, op, samples, rank=None):
        samples = self._ensure_meta_if_needed(samples, op)
        samples = self._ensure_stats_if_needed(samples, op)
        call_kwargs = self._build_call_kwargs(op.compute_stats, op, rank=rank, context=True)
        if op.is_batched_op():
            return op.compute_stats(samples, **call_kwargs)

        rows = []
        for idx in range(self._batch_size(samples)):
            sample = {key: values[idx] for key, values in samples.items()}
            result = op.compute_stats_single(sample, **call_kwargs)
            if result is None or not isinstance(result, dict):
                result_type = type(result).__name__
                raise ValueError(
                    f"Filter sub-op [{op._name}] returned unsupported sample type "
                    f"[{result_type}] inside FusedSharedContextOp."
                )
            rows.append(result)
        return self._rows_to_batch(rows, op)

    def _filter_keep_mask(self, op, samples):
        if op.is_batched_op():
            return list(op.process(samples))

        keep_mask = []
        for idx in range(self._batch_size(samples)):
            sample = {key: values[idx] for key, values in samples.items()}
            keep_mask.append(op.process_single(sample))
        return keep_mask

    def _build_call_kwargs(self, method, op, rank=None, context=False):
        kwargs = {}
        if context and check_op_method_param(method, "context"):
            kwargs["context"] = True
        if op.accelerator == "cuda" and check_op_method_param(method, "rank"):
            kwargs["rank"] = rank
        return kwargs

    def _prepare_sub_op_result(self, previous_contexts, result, op):
        self._validate_batch(result, f"sub-op [{op._name}] result")
        result_size = self._batch_size(result)

        if Fields.context not in result:
            if result_size == len(previous_contexts):
                result[Fields.context] = previous_contexts
            elif result_size == 0:
                result[Fields.context] = []
            else:
                raise ValueError(
                    f"Sub-op [{op._name}] changed the batch size from "
                    f"[{len(previous_contexts)}] to [{result_size}] without "
                    f"returning an aligned Fields.context column."
                )

        contexts = result[Fields.context]
        if len(contexts) != result_size:
            raise ValueError(
                f"Fields.context length [{len(contexts)}] does not match batch "
                f"size [{result_size}] after sub-op [{op._name}]."
            )
        if any(not isinstance(context, dict) for context in contexts):
            raise ValueError(f"Sub-op [{op._name}] returned a non-dict Fields.context entry.")
        return result

    def _filter_batch(self, samples, keep_mask, op):
        num_samples = self._batch_size(samples)
        if len(keep_mask) != num_samples:
            raise ValueError(
                f"Filter sub-op [{op._name}] returned keep mask length "
                f"[{len(keep_mask)}], expected [{num_samples}] inside "
                f"FusedSharedContextOp."
            )

        kept_indices = [idx for idx, keep in enumerate(keep_mask) if keep]
        return {key: [values[idx] for idx in kept_indices] for key, values in samples.items()}

    def _rows_to_batch(self, rows, op):
        if not rows:
            return {}
        keys = list(rows[0].keys())
        key_set = set(keys)
        for row in rows[1:]:
            if set(row.keys()) != key_set:
                raise ValueError(
                    f"Filter sub-op [{op._name}] returned inconsistent fields "
                    f"across samples inside FusedSharedContextOp."
                )
        return {key: [row[key] for row in rows] for key in keys}

    def _ensure_meta_if_needed(self, samples, op):
        if not self._needs_meta(samples, op):
            return samples
        return self._ensure_dict_column(samples, Fields.meta, op)

    def _ensure_stats_if_needed(self, samples, op):
        if not self._needs_stats(samples, op):
            return samples
        return self._ensure_dict_column(samples, Fields.stats, op)

    def _ensure_dict_column(self, samples, field, op):
        num_samples = self._batch_size(samples)
        if field not in samples or samples[field] is None or len(samples[field]) == 0:
            samples[field] = [{} for _ in range(num_samples)]
        elif len(samples[field]) != num_samples:
            raise ValueError(
                f"{field} length [{len(samples[field])}] does not match batch "
                f"size [{num_samples}] before sub-op [{op._name}] inside "
                f"FusedSharedContextOp."
            )
        else:
            for idx in range(num_samples):
                if samples[field][idx] is None:
                    samples[field][idx] = {}
                elif not isinstance(samples[field][idx], dict):
                    raise ValueError(f"{field} entry [{idx}] must be a dict before sub-op [{op._name}].")
        return samples

    def _needs_meta(self, samples, op):
        if Fields.meta in samples:
            return True
        if getattr(op, "_requires_meta", False):
            return True
        if op._name in TAGGING_OPS.modules:
            return True
        output_columns = getattr(op, "_output_columns", []) or []
        return any(str(column).startswith(Fields.meta) for column in output_columns)

    def _needs_stats(self, samples, op):
        if Fields.stats in samples:
            return True
        if isinstance(op, Filter) and op._name not in NON_STATS_FILTERS.modules:
            return True
        output_columns = getattr(op, "_output_columns", []) or []
        return any(str(column).startswith(Fields.stats) for column in output_columns)

    def _validate_sub_ops(self):
        unsupported = [op for op in self.fused_ops if not isinstance(op, (Mapper, Filter))]
        if unsupported:
            names = ", ".join(f"{op._name} ({type(op).__name__})" for op in unsupported)
            raise NotImplementedError(
                f"FusedSharedContextOp supports only Mapper and Filter sub-ops; unsupported: {names}."
            )

    def _validate_batch(self, samples, description):
        if not isinstance(samples, dict):
            raise ValueError(
                f"{description} has unsupported batch type [{type(samples).__name__}] inside FusedSharedContextOp."
            )
        if not samples:
            return

        expected_size = None
        for key, values in samples.items():
            try:
                column_size = len(values)
            except TypeError as error:
                raise ValueError(f"Column [{key}] in {description} is not a sized batch column.") from error
            if expected_size is None:
                expected_size = column_size
            elif column_size != expected_size:
                raise ValueError(
                    f"Column [{key}] length [{column_size}] does not match batch "
                    f"size [{expected_size}] in {description}."
                )

    def _batch_size(self, samples):
        if not samples:
            return 0
        first_key = next(iter(samples.keys()))
        return len(samples[first_key])

    def _track_context_ownership(
        self,
        batch,
        tracked_batches,
        tracked_batch_ids,
        tracked_contexts,
        tracked_context_ids,
    ):
        batch_id = id(batch)
        if batch_id not in tracked_batch_ids:
            tracked_batch_ids.add(batch_id)
            tracked_batches.append(batch)

        for context in batch.get(Fields.context, []):
            context_id = id(context)
            if context_id not in tracked_context_ids:
                tracked_context_ids.add(context_id)
                tracked_contexts.append(context)

    def _cleanup_contexts(self, contexts: Iterable[Dict[str, Any]]):
        seen_value_ids = set()
        for context in contexts:
            for value in context.values():
                self._cleanup_context_value(value, seen_value_ids)

    def _cleanup_context_value(self, value, seen_value_ids):
        value_id = id(value)
        if value_id in seen_value_ids:
            return
        seen_value_ids.add(value_id)

        if isinstance(value, dict):
            for item in value.values():
                self._cleanup_context_value(item, seen_value_ids)
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                self._cleanup_context_value(item, seen_value_ids)
            return
        if self._looks_like_av_container(value):
            video_streams = getattr(getattr(value, "streams", None), "video", None)
            if video_streams:
                try:
                    video_streams[0].close()
                except Exception:
                    pass
            try:
                value.close()
            except Exception:
                pass

    def _looks_like_av_container(self, value):
        cls = value.__class__
        return cls.__name__ == "InputContainer" and cls.__module__.startswith("av.")

    def run(self, dataset, *, exporter=None, tracer=None):
        from data_juicer.core.data import NestedDataset
        from data_juicer.utils.model_utils import free_models

        if not isinstance(dataset, NestedDataset):
            dataset = NestedDataset(dataset)
        if not self.fused_ops:
            return dataset

        for op in self.fused_ops:
            dataset = OP.run(op, dataset)

        try:
            return dataset.map(
                self.process_batched,
                num_proc=self.runtime_np(),
                with_rank=self.use_cuda(),
                batch_size=self.batch_size,
                desc=self._name + "_process",
            )
        finally:
            free_models()

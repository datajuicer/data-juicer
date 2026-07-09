"""
FusedSharedContextOp - run batch-local ops with a shared temporary context.

This op is an explicit fused operator for cases where several mapper/filter
ops can reuse intermediate data such as words, lines, decoded images, decoded
audio, decoded video, or sampled frames. It keeps the normal sequential
pipeline semantics while making ``Fields.context`` available to every sub-op
within the same batch stage.
"""

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

    ``FusedSharedContextOp`` is intentionally manual: callers explicitly list
    the sub-ops to run. It does not attempt to discover fusible ops or parallel
    execution opportunities.
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
            batch_size: outer batch size for the fused stage.
            fused_op_list: list of standard Data-Juicer op config dicts.
            fused_ops: pre-built op instances, mainly for tests.
        """
        kwargs["batch_size"] = batch_size
        super().__init__(*args, **kwargs)

        if fused_op_list is not None and fused_ops is not None:
            raise ValueError("FusedSharedContextOp: provide either fused_op_list or fused_ops, not both.")

        self.fused_op_list = list(fused_op_list or [])
        self.fused_ops = list(fused_ops) if fused_ops is not None else load_ops(self.fused_op_list)

        if any(op.accelerator == "cuda" for op in self.fused_ops):
            self.accelerator = "cuda"

        runtime_nps = [op.runtime_np() for op in self.fused_ops]
        runtime_nps = [runtime_np for runtime_np in runtime_nps if runtime_np is not None]
        if runtime_nps:
            self.num_proc = min(runtime_nps)

    def process_batched(self, samples, rank=None):
        if not self.fused_ops:
            return samples

        if self._batch_size(samples) == 0:
            return samples

        samples[Fields.context] = [{} for _ in range(self._batch_size(samples))]
        try:
            for op in self.fused_ops:
                if isinstance(op, Mapper):
                    samples = self._run_mapper(op, samples, rank=rank)
                elif isinstance(op, Filter):
                    samples = self._run_filter(op, samples, rank=rank)
                else:
                    raise NotImplementedError(
                        f"FusedSharedContextOp does not support OP [{op._name}] "
                        f"of type [{type(op).__name__}]. Only Mapper and Filter "
                        f"sub-ops are supported."
                    )

                if self._batch_size(samples) == 0:
                    break
            return samples
        finally:
            if Fields.context in samples:
                self._cleanup_contexts(samples[Fields.context])
                samples.pop(Fields.context, None)

    def _run_mapper(self, op, samples, rank=None):
        samples = self._ensure_meta_if_needed(samples, op)
        call_kwargs = self._build_call_kwargs(op.process, op, rank=rank, context=True)
        if op.is_batched_op():
            result = op.process(samples, **call_kwargs)
        else:
            result = op.process_batched(samples, **call_kwargs)
        return self._validate_batch_result(result, op)

    def _run_filter(self, op, samples, rank=None):
        samples = self._ensure_stats_if_needed(samples, op)
        call_kwargs = self._build_call_kwargs(op.compute_stats, op, rank=rank, context=True)
        if op.is_batched_op():
            result = op.compute_stats(samples, **call_kwargs)
        else:
            result = op.compute_stats_batched(samples, **call_kwargs)
        result = self._validate_batch_result(result, op)

        if op.is_batched_op():
            keep_mask = list(op.process(result))
        else:
            keep_mask = list(op.process_batched(result))
        return self._filter_batch(result, keep_mask, op)

    def _build_call_kwargs(self, method, op, rank=None, context=False):
        kwargs = {}
        if context and check_op_method_param(method, "context"):
            kwargs["context"] = True
        if op.accelerator == "cuda" and check_op_method_param(method, "rank"):
            kwargs["rank"] = rank
        return kwargs

    def _validate_batch_result(self, result, op):
        if result is None:
            raise ValueError(f"Sub-op [{op._name}] returned None inside FusedSharedContextOp.")
        if not isinstance(result, dict):
            raise ValueError(
                f"Sub-op [{op._name}] returned unsupported batch type "
                f"[{type(result).__name__}] inside FusedSharedContextOp."
            )
        return result

    def _filter_batch(self, samples, keep_mask, op):
        num_samples = self._batch_size(samples)
        if len(keep_mask) != num_samples:
            raise ValueError(
                f"Filter sub-op [{op._name}] returned keep mask length "
                f"[{len(keep_mask)}], expected [{num_samples}] inside "
                f"FusedSharedContextOp."
            )

        dropped_contexts = []
        if Fields.context in samples:
            dropped_contexts = [ctx for ctx, keep in zip(samples[Fields.context], keep_mask) if not keep]
        self._cleanup_contexts(dropped_contexts)

        kept_indices = [idx for idx, keep in enumerate(keep_mask) if keep]
        return {key: [values[idx] for idx in kept_indices] for key, values in samples.items()}

    def _ensure_meta_if_needed(self, samples, op):
        if not self._needs_meta(samples, op):
            return samples
        num_samples = self._batch_size(samples)
        if Fields.meta not in samples or samples[Fields.meta] is None or len(samples[Fields.meta]) == 0:
            samples[Fields.meta] = [{} for _ in range(num_samples)]
        elif len(samples[Fields.meta]) != num_samples:
            raise ValueError(
                f"Fields.meta length [{len(samples[Fields.meta])}] does not "
                f"match batch size [{num_samples}] before sub-op [{op._name}] "
                f"inside FusedSharedContextOp."
            )
        else:
            for idx in range(num_samples):
                if samples[Fields.meta][idx] is None:
                    samples[Fields.meta][idx] = {}
        return samples

    def _ensure_stats_if_needed(self, samples, op):
        if not self._needs_stats(samples, op):
            return samples
        num_samples = self._batch_size(samples)
        if Fields.stats not in samples or samples[Fields.stats] is None or len(samples[Fields.stats]) == 0:
            samples[Fields.stats] = [{} for _ in range(num_samples)]
        elif len(samples[Fields.stats]) != num_samples:
            raise ValueError(
                f"Fields.stats length [{len(samples[Fields.stats])}] does not "
                f"match batch size [{num_samples}] before sub-op [{op._name}] "
                f"inside FusedSharedContextOp."
            )
        else:
            for idx in range(num_samples):
                if samples[Fields.stats][idx] is None:
                    samples[Fields.stats][idx] = {}
        return samples

    def _needs_meta(self, samples, op):
        if Fields.meta in samples:
            return True
        if getattr(op, "_requires_meta", False):
            return True
        if op._name in TAGGING_OPS.modules:
            return True
        output_columns = getattr(op, "_output_columns", []) or []
        return any(str(col).startswith(Fields.meta) for col in output_columns)

    def _needs_stats(self, samples, op):
        if Fields.stats in samples:
            return True
        if isinstance(op, Filter) and op._name not in NON_STATS_FILTERS.modules:
            return True
        output_columns = getattr(op, "_output_columns", []) or []
        return any(str(col).startswith(Fields.stats) for col in output_columns)

    def _batch_size(self, samples):
        if not samples:
            return 0
        first_key = next(iter(samples.keys()))
        return len(samples[first_key])

    def _cleanup_contexts(self, contexts: Iterable[Dict[str, Any]]):
        for context in contexts:
            if isinstance(context, dict):
                for value in context.values():
                    self._cleanup_context_value(value)

    def _cleanup_context_value(self, value):
        if isinstance(value, dict):
            for item in value.values():
                self._cleanup_context_value(item)
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                self._cleanup_context_value(item)
            return
        if self._looks_like_av_container(value):
            try:
                video_streams = getattr(getattr(value, "streams", None), "video", None)
                if video_streams:
                    video_streams[0].close()
                value.close()
            except Exception:
                pass

    def _looks_like_av_container(self, value):
        cls = value.__class__
        return cls.__name__ == "InputContainer" and cls.__module__.startswith("av.")

    def run(self, dataset, *, exporter=None, tracer=None):
        from data_juicer.core.data import NestedDataset

        if not isinstance(dataset, NestedDataset):
            dataset = NestedDataset(dataset)
        if not self.fused_ops:
            return dataset

        for op in self.fused_ops:
            dataset = OP.run(op, dataset)

        return dataset.map(
            self.process_batched,
            num_proc=self.runtime_np(),
            with_rank=self.use_cuda(),
            batch_size=self.batch_size,
            desc=self._name + "_process",
        )

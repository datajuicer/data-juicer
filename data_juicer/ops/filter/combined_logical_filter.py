"""
Combined Logical Filter Operator.

A composition operator that combines multiple filter operators with
logical operations (AND/OR). This operator is more explicit and
self-documenting than using separate filters in sequence.
"""

from typing import Dict, List

import numpy as np

from ...utils.constant import Fields
from ..base_op import OPERATORS, Filter
from ..load import load_ops

OP_NAME = "combined_logical_filter"


@OPERATORS.register_module(OP_NAME)
class CombinedLogicalFilter(Filter):
    """A combined filter operator that applies multiple filter operators
    with logical operations (AND/OR).

    This is a composition operator that combines multiple filter operators
    and applies a logical operation (AND or OR) to their results. It's
    more explicit and self-documenting than using separate filters in
    sequence.

    Features:
    - Supports AND/OR logical operations
    - Handles both batched and single-sample processing
    - Compatible with Ray executor
    - Automatically handles context variables for intermediate operations
    - Supports CUDA-accelerated filters

    组合型过滤算子，将多个过滤算子组合并应用逻辑运算（AND/OR）。

    这是一个组合算子，将多个过滤算子组合并对其结果应用逻辑运算（AND 或 OR）。
    比在序列中使用单独的过滤器更明确和自文档化。

    特性：
    - 支持 AND/OR 逻辑运算
    - 处理批处理和单样本处理
    - 兼容 Ray 执行器
    - 自动处理中间操作的上下文变量
    - 支持 CUDA 加速的过滤器
    """

    _batched_op = True

    def __init__(
        self,
        filter_ops: List[Dict[str, dict]],
        logical_op: str = "and",
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param filter_ops: A list of filter operator configurations.
            Each item should be a dict with operator name as key and its
            parameters as value. Example: [{"text_length_filter":
            {"min_len": 10, "max_len": 100}}, {"language_id_score_filter":
            {"lang": "zh", "min_score": 0.8}}]
        :param logical_op: The logical operator to combine filter results.
            Can be "and" or "or". Default is "and". When "and" is used,
            a sample is kept only if all filters return True. When "or"
            is used, a sample is kept if any filter returns True.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        if not filter_ops:
            raise ValueError("filter_ops cannot be empty. " "At least one filter operator is required.")
        if logical_op.lower() not in ["and", "or"]:
            raise ValueError(f"logical_op must be 'and' or 'or', got '{logical_op}'")

        self.logical_op = logical_op.lower()
        # Load filter operators from configuration
        self.filter_ops = load_ops(filter_ops)

        # Verify all loaded operators are Filter instances
        for i, op in enumerate(self.filter_ops):
            if not isinstance(op, Filter):
                raise ValueError(
                    f"All operators in filter_ops must be Filter instances, " f"got {type(op).__name__} at index {i}"
                )

        # Set accelerator to 'cuda' if any of the filters use it
        accelerator_methods = set([op.accelerator for op in self.filter_ops])
        if "cuda" in accelerator_methods:
            self.accelerator = "cuda"

        # Set num_proc to the minimum of all filters
        # This ensures compatibility when filters have different
        # num_proc requirements
        self.num_proc = min([op.runtime_np() for op in self.filter_ops])

    def compute_stats_batched(self, samples, rank=None, context=False):
        """Compute stats for all filter operators in batch mode.

        This method applies all filter operators sequentially to compute
        statistics. Context variables are automatically handled for
        operators that need them.

        :param samples: Batch of samples in dict-of-lists format
        :param rank: Rank for CUDA operations (used when any filter
            uses CUDA)
        :param context: Whether to use context for intermediate
            variables
        :return: Samples with computed statistics
        """
        # Context for intermediate variables
        num_samples = len(samples[Fields.stats])
        if Fields.context not in samples:
            samples[Fields.context] = [{} for _ in range(num_samples)]

        # Apply all filter operators to compute stats
        for op in self.filter_ops:
            process_args = {}
            if op.accelerator == "cuda":
                process_args["rank"] = rank
            needs_context = context or (hasattr(op, "compute_stats_batched") and op.accelerator == "cuda")
            if needs_context:
                # Check if the operator needs context
                from data_juicer.utils.common_utils import check_op_method_param

                if check_op_method_param(op.compute_stats, "context"):
                    process_args["context"] = True

            if hasattr(op, "compute_stats_batched"):
                samples = op.compute_stats_batched(samples, **process_args)
            else:
                # Fallback to single sample processing for non-batched ops
                keys = samples.keys()
                for i in range(num_samples):
                    this_sample = {key: samples[key][i] for key in keys}
                    context_flag = process_args.get("context", False)
                    res_sample = op.compute_stats_single(this_sample, context=context_flag)
                    samples[Fields.stats][i] = res_sample[Fields.stats]
                    # Preserve context if it was modified
                    if Fields.context in res_sample:
                        samples[Fields.context][i].update(res_sample[Fields.context])

        return samples

    def process_batched(self, samples):
        """Process samples by combining results from all filter operators
        in batch mode.

        This method applies all filters and combines their results using
        the specified logical operation (AND or OR).

        :param samples: Batch of samples in dict-of-lists format
        :return: List of boolean values indicating which samples to keep
        """
        # Get results from all filter operators
        all_results = []
        for op in self.filter_ops:
            if hasattr(op, "process_batched"):
                results = list(op.process_batched(samples))
            else:
                # Fallback to single sample processing for non-batched ops
                results = [op.process_single({Fields.stats: stat}) for stat in samples[Fields.stats]]
            all_results.append(np.array(results, dtype=bool))

        # Combine results based on logical operator
        if len(all_results) == 0:
            # Edge case: no filters (should not happen due to validation,
            # but handle gracefully)
            return [True] * len(samples[Fields.stats])

        combined_result = all_results[0]
        for result in all_results[1:]:
            if self.logical_op == "and":
                combined_result = np.logical_and(combined_result, result)
            else:  # or
                combined_result = np.logical_or(combined_result, result)

        return combined_result.tolist()

    def compute_stats_single(self, sample, context=False):
        """Compute stats for a single sample using all filter operators.

        :param sample: Single sample in dict format
        :param context: Whether to use context for intermediate variables
        :return: Sample with computed statistics
        """
        # Apply all filter operators to compute stats
        for op in self.filter_ops:
            if hasattr(op, "compute_stats_single"):
                sample = op.compute_stats_single(sample, context=context)
            else:
                # For batched-only operators, we cannot compute stats for
                # a single sample. This is a limitation - batched-only
                # operators should implement compute_stats_single or we need
                # to create a minimal batch. For now, we skip stats
                # computation for batched-only operators in single mode.
                # This is acceptable because process_single will handle the
                # fallback.
                pass
        return sample

    def process_single(self, sample: Dict) -> bool:
        """Process a single sample by combining results from all filter
        operators.

        :param sample: Single sample in dict format
        :return: Boolean indicating whether to keep the sample
        """
        # Get results from all filter operators
        results = []
        for op in self.filter_ops:
            if hasattr(op, "process_single"):
                result = op.process_single(sample)
            else:
                # For batched-only operators, create a minimal batch
                stat = sample.get(Fields.stats, {})
                batch_samples = {Fields.stats: [stat]}
                # Also preserve other fields if they exist
                for key in sample:
                    if key != Fields.stats:
                        batch_samples[key] = [sample[key]]
                batch_result = list(op.process_batched(batch_samples))
                result = batch_result[0] if batch_result else True
            results.append(result)

        # Combine results based on logical operator
        if self.logical_op == "and":
            return all(results)
        else:  # or
            return any(results)

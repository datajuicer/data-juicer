#!/usr/bin/env python3
"""
Operation Prober for sampling-based cost estimation.

This module provides functionality to estimate operation costs by running
them on a small sample of data. The measured costs are used by optimization
strategies (like op_reorder) to make better decisions.

Why Probing Matters
-------------------
Static heuristics categorize operations as "cheap" or "expensive" based on
operation type, but this misses actual runtime differences. For example,
all basic filters might be classified as "cheap", but their actual costs
can vary significantly (e.g., text_length_filter: 3.6s vs
character_repetition_filter: 21.6s on the same dataset).

What Probing Measures
---------------------
For each operation, probing measures:
    - Runtime: time_per_sample_ms (execution time per sample)
    - Selectivity: ratio of samples that pass (for filters only)

These are combined into an effective cost:
    effective_cost = time_per_sample_ms * selectivity

This formula favors placing highly selective filters (that remove more data)
earlier in the pipeline, reducing the data volume for subsequent operations.

Benchmark Results (C4 dataset, 356K samples, 5 filters):
    - Baseline (no optimization):     73.5s
    - Static reorder (heuristics):    73.3s  (no change - all "cheap")
    - Probed reorder (measured):      53.1s  (28% faster)

The probing overhead is minimal (~0.15s for 100 samples) compared to the
potential savings (20+ seconds in this benchmark).

Configuration
-------------
Enable probing via config options:
    enable_optimizer: true
    optimizer_strategies: ['op_reorder']
    optimizer_probe_enabled: true      # Enable probing (default: true)
    optimizer_probe_samples: 100       # Sample size (default: 100)
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from loguru import logger


@dataclass
class ProbeResult:
    """Result of probing a single operation."""

    op_name: str
    total_time_seconds: float
    samples_processed: int
    time_per_sample_ms: float
    # Selectivity for filters (ratio of samples that pass)
    selectivity: Optional[float] = None
    # Any errors encountered during probing
    error: Optional[str] = None

    @property
    def cost(self) -> float:
        """Get the cost score (higher = more expensive)."""
        return self.time_per_sample_ms


@dataclass
class FusionProbeResult:
    """Result of probing a fusion candidate group."""

    filter_names: List[str]
    unfused_time_ms: float  # Total time running filters separately
    fused_time_ms: float  # Time running filters fused
    speedup: float  # unfused_time / fused_time (>1 means fusion is faster)
    should_fuse: bool  # Recommendation based on speedup threshold
    error: Optional[str] = None

    @property
    def savings_percent(self) -> float:
        """Percentage of time saved by fusion."""
        if self.unfused_time_ms > 0:
            return (1 - self.fused_time_ms / self.unfused_time_ms) * 100
        return 0.0


@dataclass
class ProbeResults:
    """Collection of probe results for multiple operations."""

    results: Dict[str, ProbeResult] = field(default_factory=dict)
    sample_size: int = 0
    total_probe_time_seconds: float = 0.0

    def get_cost(self, op_name: str, default: float = 10.0) -> float:
        """Get the cost for an operation, with fallback to default."""
        if op_name in self.results:
            result = self.results[op_name]
            if result.error is None:
                return result.cost
        return default

    def get_selectivity(self, op_name: str, default: float = 1.0) -> float:
        """Get the selectivity for a filter operation."""
        if op_name in self.results:
            result = self.results[op_name]
            if result.selectivity is not None:
                return result.selectivity
        return default

    def get_effective_cost(self, op_name: str, default: float = 10.0) -> float:
        """Get effective cost considering selectivity.

        For filters, the effective cost considers how many samples pass through.
        A highly selective filter (low selectivity) that runs early reduces
        the data volume for subsequent operations.
        """
        cost = self.get_cost(op_name, default)
        selectivity = self.get_selectivity(op_name, 1.0)
        # Effective cost = actual cost / (1 - selectivity)
        # This favors filters that remove more data (lower selectivity)
        if selectivity < 1.0:
            # Bonus for selective filters
            return cost * selectivity
        return cost


class OpProber:
    """
    Probes operations to estimate their costs using actual execution.

    This class runs each operation on a small sample of data and measures
    the execution time. The results can be used by optimization strategies
    to make data-driven reordering decisions.
    """

    def __init__(
        self,
        sample_size: int = 100,
        timeout_per_op_seconds: float = 30.0,
        min_samples: int = 10,
    ):
        """
        Initialize the operation prober.

        Args:
            sample_size: Number of samples to use for probing (default: 100)
            timeout_per_op_seconds: Maximum time to spend probing each op
            min_samples: Minimum samples required for valid probing
        """
        self.sample_size = sample_size
        self.timeout_per_op_seconds = timeout_per_op_seconds
        self.min_samples = min_samples

    def probe_operations(
        self,
        ops: List[Any],
        dataset: Any,
    ) -> ProbeResults:
        """
        Probe all operations to estimate their costs.

        Args:
            ops: List of operation objects to probe
            dataset: Dataset to sample from for probing

        Returns:
            ProbeResults containing cost estimates for each operation
        """
        probe_results = ProbeResults()
        probe_start = time.time()

        # Get a sample of the dataset
        sample_dataset = self._get_sample(dataset)
        if sample_dataset is None:
            logger.warning("Failed to get sample dataset for probing")
            return probe_results

        probe_results.sample_size = len(sample_dataset)
        logger.info(f"Probing {len(ops)} operations on {probe_results.sample_size} samples")

        # Probe each operation
        for op in ops:
            op_name = self._get_op_name(op)
            try:
                result = self._probe_single_op(op, sample_dataset)
                probe_results.results[op_name] = result
                logger.debug(
                    f"  {op_name}: {result.time_per_sample_ms:.2f} ms/sample"
                    + (f", selectivity={result.selectivity:.2%}" if result.selectivity else "")
                )
            except Exception as e:
                logger.warning(f"Failed to probe {op_name}: {e}")
                probe_results.results[op_name] = ProbeResult(
                    op_name=op_name,
                    total_time_seconds=0,
                    samples_processed=0,
                    time_per_sample_ms=0,
                    error=str(e),
                )

        probe_results.total_probe_time_seconds = time.time() - probe_start
        logger.info(f"Probing completed in {probe_results.total_probe_time_seconds:.2f}s")

        # Clean up any state that might interfere with later runs
        # This helps prevent "subprocess abruptly died" errors
        import gc

        del sample_dataset
        gc.collect()

        return probe_results

    def _get_sample(self, dataset: Any) -> Any:
        """Get a sample from the dataset for probing."""
        try:
            # Get dataset length
            if hasattr(dataset, "__len__"):
                dataset_len = len(dataset)
            elif hasattr(dataset, "num_rows"):
                dataset_len = dataset.num_rows
            elif hasattr(dataset, "count"):
                dataset_len = dataset.count()
            else:
                logger.warning("Cannot determine dataset length")
                return None

            # Determine sample size
            actual_sample_size = min(self.sample_size, dataset_len)
            if actual_sample_size < self.min_samples:
                logger.warning(f"Dataset too small for probing ({dataset_len} < {self.min_samples})")
                return None

            # Take a sample - convert to list of dicts to completely isolate from original
            if hasattr(dataset, "select"):
                # HuggingFace datasets - convert to list and back to create completely
                # independent copy that won't interfere with multiprocessing
                from datasets import Dataset

                indices = list(range(actual_sample_size))
                sample_rows = [dataset[i] for i in indices]
                return Dataset.from_list(sample_rows)

            elif hasattr(dataset, "take"):
                # Ray datasets or similar
                return dataset.take(actual_sample_size)
            else:
                # Try slicing
                return dataset[:actual_sample_size]

        except Exception as e:
            logger.warning(f"Failed to sample dataset: {e}")
            return None

    def _probe_single_op(self, op: Any, sample_dataset: Any) -> ProbeResult:
        """Probe a single operation and return the result."""
        op_name = self._get_op_name(op)

        # Get initial sample count
        initial_count = self._get_dataset_length(sample_dataset)

        # Temporarily disable multiprocessing for probing to avoid subprocess crashes
        # on small sample sizes
        original_num_proc = getattr(op, "num_proc", None)
        if hasattr(op, "num_proc"):
            op.num_proc = 1

        try:
            # Time the operation
            start_time = time.time()

            # Run the operation
            # We need to handle different op types (Filter, Mapper, etc.)
            if hasattr(op, "run"):
                # Standard Data-Juicer operation
                result_dataset = op.run(sample_dataset)
            elif hasattr(op, "process"):
                result_dataset = op.process(sample_dataset)
            else:
                raise ValueError(f"Operation {op_name} has no run or process method")
        finally:
            # Restore original num_proc setting
            if hasattr(op, "num_proc") and original_num_proc is not None:
                op.num_proc = original_num_proc

        elapsed_time = time.time() - start_time

        # Calculate metrics
        final_count = self._get_dataset_length(result_dataset)
        samples_processed = initial_count

        # Calculate selectivity for filters
        selectivity = None
        if "filter" in op_name.lower() and initial_count > 0:
            selectivity = final_count / initial_count

        # Time per sample in milliseconds
        time_per_sample_ms = (elapsed_time * 1000) / max(samples_processed, 1)

        return ProbeResult(
            op_name=op_name,
            total_time_seconds=elapsed_time,
            samples_processed=samples_processed,
            time_per_sample_ms=time_per_sample_ms,
            selectivity=selectivity,
        )

    def _get_op_name(self, op: Any) -> str:
        """Get the name of an operation."""
        if hasattr(op, "_name"):
            return op._name
        elif hasattr(op, "name"):
            return op.name
        else:
            return type(op).__name__

    def _get_dataset_length(self, dataset: Any) -> int:
        """Get the length of a dataset."""
        if dataset is None:
            return 0
        if hasattr(dataset, "__len__"):
            return len(dataset)
        elif hasattr(dataset, "num_rows"):
            return dataset.num_rows
        elif hasattr(dataset, "count"):
            return dataset.count()
        return 0

    def probe_fusion_benefit(
        self,
        filters: List[Any],
        dataset: Any,
        fusion_speedup_threshold: float = 1.1,
    ) -> FusionProbeResult:
        """
        Probe whether fusing a group of filters provides a benefit.

        This method compares the time to run filters separately vs fused,
        and recommends fusion only if it provides meaningful speedup.

        Args:
            filters: List of filter operations to potentially fuse
            dataset: Dataset to sample from for probing
            fusion_speedup_threshold: Minimum speedup ratio to recommend fusion
                                      (default 1.1 = 10% faster)

        Returns:
            FusionProbeResult with timing comparison and recommendation
        """
        filter_names = [self._get_op_name(f) for f in filters]

        if len(filters) < 2:
            return FusionProbeResult(
                filter_names=filter_names,
                unfused_time_ms=0,
                fused_time_ms=0,
                speedup=1.0,
                should_fuse=False,
                error="Need at least 2 filters to fuse",
            )

        # Get a sample of the dataset
        sample_dataset = self._get_sample(dataset)
        if sample_dataset is None:
            return FusionProbeResult(
                filter_names=filter_names,
                unfused_time_ms=0,
                fused_time_ms=0,
                speedup=1.0,
                should_fuse=False,
                error="Failed to get sample dataset",
            )

        try:
            # 1. Probe unfused: run each filter separately on fresh sample
            unfused_total_ms = 0.0
            for f in filters:
                # Get fresh sample for each filter to simulate unfused execution
                fresh_sample = self._get_sample(dataset)
                if fresh_sample is None:
                    continue
                result = self._probe_single_op(f, fresh_sample)
                unfused_total_ms += result.time_per_sample_ms * len(fresh_sample)

            # 2. Probe fused: create FusedFilter and run once
            fused_time_ms = self._probe_fused_filters(filters, sample_dataset)

            # 3. Calculate speedup
            if fused_time_ms > 0:
                speedup = unfused_total_ms / fused_time_ms
            else:
                speedup = 1.0

            # 4. Decide whether to fuse
            should_fuse = speedup >= fusion_speedup_threshold

            result = FusionProbeResult(
                filter_names=filter_names,
                unfused_time_ms=unfused_total_ms,
                fused_time_ms=fused_time_ms,
                speedup=speedup,
                should_fuse=should_fuse,
            )

            logger.info(
                f"Fusion probe: {filter_names} - "
                f"unfused={unfused_total_ms:.1f}ms, fused={fused_time_ms:.1f}ms, "
                f"speedup={speedup:.2f}x, recommend={'FUSE' if should_fuse else 'NO FUSE'}"
            )

            return result

        except Exception as e:
            logger.warning(f"Failed to probe fusion for {filter_names}: {e}")
            return FusionProbeResult(
                filter_names=filter_names,
                unfused_time_ms=0,
                fused_time_ms=0,
                speedup=1.0,
                should_fuse=False,
                error=str(e),
            )

    def _probe_fused_filters(self, filters: List[Any], sample_dataset: Any) -> float:
        """
        Create a FusedFilter and probe its execution time.

        Args:
            filters: List of filter operations to fuse
            sample_dataset: Sample dataset to run on

        Returns:
            Total time in milliseconds
        """
        from data_juicer.core.optimizer.fused_op import FusedFilter

        # Create fused filter
        filter_names = [self._get_op_name(f) for f in filters]
        fused_name = f"fused_filter({','.join(filter_names)})"

        fused_filter = FusedFilter(
            name=fused_name,
            fused_filters=filters,
        )

        # Disable multiprocessing for probing
        fused_filter.num_proc = 1

        # Time the fused execution
        start_time = time.time()
        try:
            fused_filter.run(sample_dataset)
        except Exception as e:
            logger.warning(f"Fused filter execution failed: {e}")
            # Return a high time to discourage fusion
            return float("inf")

        elapsed_time = time.time() - start_time
        return elapsed_time * 1000  # Convert to ms

import concurrent.futures
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger

from data_juicer.ops import load_ops
from data_juicer.ops.base_op import OPERATORS, Filter, Mapper
from data_juicer.utils.constant import Fields


@OPERATORS.register_module("fused_filter")
class FusedFilter(Filter):
    """A fused operator for filters that can execute multiple filters in one pass."""

    _batched_op = True

    def __init__(
        self, name: str, fused_filters: List[Filter], analyzer_insights: Optional[Dict[str, Any]] = None, **kwargs
    ):
        """Initialize the fused filter.

        Args:
            name: Name of the fused filter
            fused_filters: List of filters to fuse
            analyzer_insights: Optional dataset analysis insights for optimization
            **kwargs: Extra config arguments (e.g., accelerator, batch_size, etc.)
        """
        super().__init__()
        self._name = name
        self.fused_filters = fused_filters
        self.analyzer_insights = analyzer_insights or {}

        # Store extra config arguments as attributes
        self.accelerator = kwargs.get("accelerator", "cpu")
        self.batch_size = kwargs.get("batch_size", None)
        self.cpu_required = kwargs.get("cpu_required", 1)  # Default to 1 CPU
        self.mem_required = kwargs.get("mem_required", 1)  # Default to 1 GB
        self.num_proc = kwargs.get("num_proc", None)
        self.skip_op_error = kwargs.get("skip_op_error", False)
        self.turbo = kwargs.get("turbo", False)
        self.text_key = kwargs.get("text_key", None)
        self.image_key = kwargs.get("image_key", None)
        self.audio_key = kwargs.get("audio_key", None)
        self.video_key = kwargs.get("video_key", None)
        self.history_key = kwargs.get("history_key", None)
        self.query_key = kwargs.get("query_key", None)
        self.response_key = kwargs.get("response_key", None)
        self.execution_strategy = kwargs.get("execution_strategy", None)
        self.has_dependencies = kwargs.get("has_dependencies", None)

        # Add recursion prevention flag
        self._in_performance_test = False

        # Set accelerator based on available methods (if not set by kwargs)
        if self.accelerator is None:
            if any(hasattr(op, "accelerator") and op.accelerator == "cuda" for op in self.fused_filters):
                self.accelerator = "cuda"
            else:
                self.accelerator = "cpu"

        # Update num_proc with the minimum of all fused filters if not set by kwargs
        if self.num_proc is None:
            self.num_proc = min([op.runtime_np() for op in self.fused_filters])

        # Store original operation configs (create simple config if not available)
        self._op_cfg = {}
        for op in self.fused_filters:
            op_name = getattr(op, "_name", None)
            op_config = getattr(op, "config", None)
            if op_name is not None and op_config:
                self._op_cfg[op_name] = op_config
            elif op_name is not None:
                # Create a simple config for filters without explicit config
                self._op_cfg[op_name] = {"inter_vars": [], "dependencies": []}

        # Analyze dependencies and determine execution strategy
        self._analyze_dependencies()
        self._determine_execution_strategy()

        # Analyze filter dependencies
        self._analyze_dependencies()

        # Pre-allocate result arrays
        self._result_cache = {}

        # Log the chosen strategy
        logger.info(f"FusedFilter '{name}' using {self.execution_strategy} execution strategy")
        if self.has_dependencies:
            logger.info("  Reason: Filters have dependencies")
        else:
            simple_count = sum(
                1
                for op in self.fused_filters
                if getattr(op, "_name", None)
                in {
                    "text_length_filter",
                    "words_num_filter",
                    "character_repetition_filter",
                    "word_repetition_filter",
                    "special_characters_filter",
                    "alphanumeric_filter",
                    "average_line_length_filter",
                    "maximum_line_length_filter",
                }
            )
            complex_count = len(self.fused_filters) - simple_count
            logger.info(f"  Reason: {simple_count} simple filters, {complex_count} complex filters")

        # Log analyzer-based insights if available
        if self.analyzer_insights:
            self._log_analyzer_insights()

    def _log_analyzer_insights(self):
        """Log insights from analyzer that influenced strategy decisions."""
        dataset_size = self.analyzer_insights.get("dataset_size", 0)
        text_length_stats = self.analyzer_insights.get("text_length", {})
        content_ratios = self.analyzer_insights.get("content_ratios", {})

        logger.info("  Analyzer Insights:")
        if dataset_size > 0:
            logger.info(f"    Dataset size: {dataset_size:,} samples")

        if text_length_stats:
            mean_len = text_length_stats.get("mean", 0)
            std_len = text_length_stats.get("std", 0)
            if mean_len > 0:
                cv = std_len / mean_len
                logger.info(f"    Text length CV: {cv:.2f} (mean: {mean_len:.1f}, std: {std_len:.1f})")

        multimodal_count = sum(
            1 for indicator in ["image_ratio", "audio_ratio", "video_ratio"] if content_ratios.get(indicator, 0) > 0.1
        )
        if multimodal_count > 0:
            logger.info(f"    Multimodal content: {multimodal_count} types detected")

    def _analyze_dependencies(self):
        """Analyze dependencies between filters to optimize execution order."""
        # Create dependency graph
        self.dependency_graph = {}
        self.independent_groups = []
        self.has_dependencies = False

        for i, op1 in enumerate(self.fused_filters):
            self.dependency_graph[op1] = set()
            for j, op2 in enumerate(self.fused_filters):
                if i != j:
                    # Check if op2 depends on op1's output
                    if self._has_dependency(op1, op2):
                        self.dependency_graph[op1].add(op2)
                        self.has_dependencies = True

        # Find independent groups
        visited = set()
        for op in self.fused_filters:
            if op not in visited:
                group = self._get_independent_group(op, visited)
                if group:
                    self.independent_groups.append(group)

        # Determine execution strategy
        self.execution_strategy = self._determine_execution_strategy()

    def _has_dependency(self, op1: Filter, op2: Filter) -> bool:
        """Check if op2 depends on op1's output."""
        # Get intermediate variables used by each operation from stored configs
        op1_vars = set(self._op_cfg.get(getattr(op1, "_name", "<unknown>"), {}).get("inter_vars", []))
        op2_vars = set(self._op_cfg.get(getattr(op2, "_name", "<unknown>"), {}).get("inter_vars", []))

        # Check if op2 uses any variables produced by op1
        return bool(op1_vars & op2_vars)

    def _get_independent_group(self, start_op: Filter, visited: set) -> List[Filter]:
        """Get a group of independent operations starting from start_op."""
        group = []
        to_visit = {start_op}

        while to_visit:
            op = to_visit.pop()
            if op not in visited:
                visited.add(op)
                group.append(op)
                # Add independent operations to visit
                for other_op in self.fused_filters:
                    if (
                        other_op not in visited
                        and other_op not in self.dependency_graph[op]
                        and op not in self.dependency_graph[other_op]
                    ):
                        to_visit.add(other_op)

        return group

    def _determine_execution_strategy(self):
        """Determine the best execution strategy based on filter characteristics and analyzer insights."""
        if self.has_dependencies:
            return "sequential"

        # Use analyzer insights for better decisions
        if self.analyzer_insights:
            return self._analyzer_based_strategy_selection()

        # Fallback to original logic
        return self._fallback_strategy_selection()

    def _analyzer_based_strategy_selection(self) -> str:
        """Select execution strategy based on analyzer insights."""
        dataset_size = self.analyzer_insights.get("dataset_size", 0)
        text_length_stats = self.analyzer_insights.get("text_length", {})
        content_ratios = self.analyzer_insights.get("content_ratios", {})

        # Factor 1: Dataset size
        if dataset_size > 500000:  # Large datasets benefit from parallel
            logger.debug("Large dataset detected - favoring parallel execution")
            return "parallel"

        # Factor 2: Text complexity
        if text_length_stats:
            mean_length = text_length_stats.get("mean", 0)
            std_length = text_length_stats.get("std", 0)
            if mean_length > 0 and std_length / mean_length > 2.0:
                logger.debug("High text length variance - using sequential for complex data")
                return "sequential"

        # Factor 3: Multimodal content
        multimodal_indicators = ["image_ratio", "audio_ratio", "video_ratio"]
        multimodal_count = sum(1 for indicator in multimodal_indicators if content_ratios.get(indicator, 0) > 0.1)

        if multimodal_count > 1:
            logger.debug(f"Multimodal content ({multimodal_count} types) - using sequential")
            return "sequential"

        # Factor 4: Filter complexity distribution
        return self._complexity_based_strategy()

    def _complexity_based_strategy(self) -> str:
        """Select strategy based on filter complexity distribution."""
        simple_filters = 0
        complex_filters = 0

        for op in self.fused_filters:
            simple_filter_names = {
                "text_length_filter",
                "words_num_filter",
                "character_repetition_filter",
                "word_repetition_filter",
                "special_characters_filter",
                "alphanumeric_filter",
                "average_line_length_filter",
                "maximum_line_length_filter",
            }

            if getattr(op, "_name", "<unknown>") in simple_filter_names:
                simple_filters += 1
            else:
                complex_filters += 1

        # Use parallel if mostly simple filters, sequential if complex filters
        if complex_filters > simple_filters:
            return "sequential"
        else:
            return "parallel"

    def _fallback_strategy_selection(self) -> str:
        """Fallback strategy selection using original logic."""
        # Check if filters are simple enough for parallel execution
        simple_filters = 0
        complex_filters = 0

        for op in self.fused_filters:
            # Simple filters: text_length, words_num, character_repetition
            # Complex filters: perplexity, stopwords, flagged_words
            simple_filter_names = {
                "text_length_filter",
                "words_num_filter",
                "character_repetition_filter",
                "word_repetition_filter",
                "special_characters_filter",
                "alphanumeric_filter",
                "average_line_length_filter",
                "maximum_line_length_filter",
            }

            if getattr(op, "_name", "<unknown>") in simple_filter_names:
                simple_filters += 1
            else:
                complex_filters += 1

        # Use parallel if mostly simple filters, sequential if complex filters
        if complex_filters > simple_filters:
            return "sequential"
        else:
            return "parallel"

    def _should_skip_fusion(self, sample_size: int = 1000) -> bool:
        """Determine if fusion should be skipped based on performance analysis and analyzer insights.

        Args:
            sample_size: Number of samples being processed

        Returns:
            True if fusion should be skipped, False if fusion is beneficial
        """
        # Prevent recursion during performance testing
        if self._in_performance_test:
            return False

        # Use analyzer insights for better decisions
        if self.analyzer_insights:
            return self._analyzer_based_fusion_decision(sample_size)

        # Fallback to original logic
        return self._fallback_fusion_decision(sample_size)

    def _analyzer_based_fusion_decision(self, sample_size: int) -> bool:
        """Make fusion decisions based on analyzer insights."""
        dataset_size = self.analyzer_insights.get("dataset_size", 0)
        text_length_stats = self.analyzer_insights.get("text_length", {})

        # Decision 1: Always use fusion for large datasets
        if dataset_size > 100000:
            logger.debug(f"Large dataset ({dataset_size:,} samples) - always use fusion")
            return False

        # Decision 2: Skip fusion for very small datasets with simple filters
        if sample_size < 100 and len(self.fused_filters) <= 2:
            simple_count = sum(
                1
                for op in self.fused_filters
                if getattr(op, "_name", "<unknown>")
                in {"text_length_filter", "words_num_filter", "character_repetition_filter"}
            )
            if simple_count == len(self.fused_filters):
                logger.debug("Small dataset with simple filters - skipping fusion")
                return True

        # Decision 3: Use fusion for complex data characteristics
        if text_length_stats:
            mean_length = text_length_stats.get("mean", 0)
            std_length = text_length_stats.get("std", 0)
            if mean_length > 0 and std_length / mean_length > 1.5:
                logger.debug("Complex text characteristics - using fusion")
                return False

        # Decision 4: Run performance test for edge cases
        return self._quick_performance_test(min(100, sample_size))

    def _fallback_fusion_decision(self, sample_size: int) -> bool:
        """Fallback fusion decision using original logic."""
        # Skip performance test for very large datasets (fusion is always beneficial)
        if sample_size > 10000:
            return False

        # Skip performance test for complex filters (always use fusion)
        complex_filter_names = {
            "perplexity_filter",
            "stopwords_filter",
            "flagged_words_filter",
            "language_id_score_filter",
            "word_repetition_filter",
        }
        complex_count = sum(1 for op in self.fused_filters if getattr(op, "_name", "<unknown>") in complex_filter_names)

        # Always use fusion for complex filters
        if complex_count > 0:
            return False

        # Skip fusion for single filters
        if len(self.fused_filters) == 1:
            return True

        # For simple filters, run a quick performance test
        try:
            return self._quick_performance_test(min(100, sample_size))
        except Exception as e:
            logger.warning(f"Performance test failed: {e}. Defaulting to fusion.")
            return False

    def _quick_performance_test(self, sample_size: int) -> bool:
        """Run a quick performance test to determine if fusion is beneficial.

        Args:
            sample_size: Number of samples to test (small sample for speed)

        Returns:
            True if fusion should be skipped, False if fusion is beneficial
        """
        import random
        import string
        import time

        from data_juicer.utils.constant import Fields

        # Set recursion prevention flag
        self._in_performance_test = True

        try:
            # Create minimal test data
            test_data = {
                "text": ["".join(random.choices(string.ascii_letters + " ", k=50)) for _ in range(sample_size)],
                Fields.stats: [{} for _ in range(sample_size)],
            }

            # Measure individual execution time
            individual_start = time.time()
            for op in self.fused_filters:
                op.compute_stats_batched(test_data.copy())
                op.process_batched(test_data.copy())
            individual_time = time.time() - individual_start

            # Measure fused execution time
            fused_start = time.time()
            self.compute_stats_batched(test_data.copy())
            self.process_batched(test_data.copy())
            fused_time = time.time() - fused_start

            # Calculate overhead ratio
            overhead_ratio = fused_time / individual_time if individual_time > 0 else float("inf")

            # Decision logic for simple filters only
            if individual_time < 0.001:
                # Very fast filters - overhead not worth it
                should_skip = True
            elif overhead_ratio > 3.0 and individual_time < 0.01:
                # Simple filters with high overhead - skip fusion
                should_skip = True
            else:
                # Default to fusion for most cases
                should_skip = False

            logger.info(
                f"Performance test: individual={individual_time:.3f}s, "
                f"fused={fused_time:.3f}s, ratio={overhead_ratio:.2f}x, "
                f"skip={should_skip}"
            )

            return should_skip

        finally:
            # Always clear the recursion prevention flag
            self._in_performance_test = False

    def compute_stats_batched(self, samples, rank=None):
        """Compute statistics for all fused filters using the best strategy."""
        import av

        # Check if we should skip fusion based on performance analysis
        if self._should_skip_fusion(len(samples[Fields.stats])):
            from loguru import logger

            logger.debug(f"Skipping fusion for {self._name} - executing filters individually")

            # Execute filters individually (no fusion)
            for op in self.fused_filters:
                if op.accelerator == "cuda":
                    samples = op.compute_stats_batched(samples, rank=rank)
                else:
                    samples = op.compute_stats_batched(samples)
            return samples

        # Initialize context for intermediate variables
        num_samples = len(samples[Fields.stats])
        samples[Fields.context] = [{} for _ in range(num_samples)]

        if self.execution_strategy == "parallel":
            # Parallel execution for independent filters
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_proc) as executor:
                futures = []
                for group in self.independent_groups:
                    for op in group:
                        if op.accelerator == "cuda":
                            futures.append(executor.submit(op.compute_stats_batched, samples, rank=rank, context=True))
                        else:
                            futures.append(executor.submit(op.compute_stats_batched, samples, context=True))

                # Wait for all operations to complete
                concurrent.futures.wait(futures)
        else:
            # Sequential execution for dependent or complex filters
            for op in self.fused_filters:
                if op.accelerator == "cuda":
                    samples = op.compute_stats_batched(samples, rank=rank)
                else:
                    samples = op.compute_stats_batched(samples)

        # Clean up contexts
        for ctx in samples[Fields.context]:
            for context_key in ctx:
                if isinstance(ctx[context_key], av.container.InputContainer):
                    ctx[context_key].streams.video[0].close()
                    ctx[context_key].close()

        # Remove context from samples
        _ = samples.pop(Fields.context)
        return samples

    def process_batched(self, samples):
        """Process samples through all fused filters using the best strategy."""
        # Check if we should skip fusion based on performance analysis
        if self._should_skip_fusion(len(samples[Fields.stats])):
            from loguru import logger

            logger.debug(f"Skipping fusion for {self._name} - processing filters individually")

            # Process filters individually (no fusion)
            result = None
            for op in self.fused_filters:
                filter_result = list(op.process_batched(samples))

                if result is None:
                    result = filter_result
                else:
                    # Combine with logical AND (sample must pass all filters)
                    result = [r1 and r2 for r1, r2 in zip(result, filter_result)]

            return result

        if self.execution_strategy == "parallel":
            # Parallel execution - all filters see original data
            return self._process_batched_parallel(samples)
        else:
            # Sequential execution - each filter sees previous filter's output
            return self._process_batched_sequential(samples)

    def _process_batched_parallel(self, samples):
        """Process filters in parallel (all see original data)."""
        # Initialize result array
        res = None

        # Process independent groups in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_proc) as executor:
            futures = []
            for group in self.independent_groups:
                group_futures = []
                for op in group:
                    future = executor.submit(op.process_batched, samples)
                    group_futures.append(future)
                futures.append(group_futures)

            # Process results in dependency order
            for group_futures in futures:
                group_results = []
                for future in group_futures:
                    this_res = np.array(list(future.result()))
                    group_results.append(this_res)

                # Combine results within group
                group_res = group_results[0]
                for this_res in group_results[1:]:
                    group_res = np.logical_and(group_res, this_res)

                # Combine with overall results
                if res is not None:
                    res = np.logical_and(res, group_res)
                else:
                    res = group_res

        return res

    def _process_batched_sequential(self, samples):
        """Process filters sequentially (each sees previous output)."""
        # Process filters sequentially to match individual execution behavior
        result = None

        for op in self.fused_filters:
            filter_result = list(op.process_batched(samples))

            if result is None:
                result = filter_result
            else:
                # Combine with logical AND (sample must pass all filters)
                result = [r1 and r2 for r1, r2 in zip(result, filter_result)]

        return result

    def run(self, dataset, *, exporter=None, tracer=None, reduce=True):
        """Run the fused filter on a dataset.

        Args:
            dataset: Dataset to process
            exporter: Optional exporter for results
            tracer: Optional tracer for monitoring
            reduce: Whether to apply filtering (True) or just compute stats (False)

        Returns:
            Processed dataset
        """
        # Prepare the dataset
        from data_juicer.core.data import NestedDataset

        if not isinstance(dataset, NestedDataset):
            dataset = NestedDataset(dataset)

        # Initialize each filter
        for op in self.fused_filters:
            dataset = Filter.run(op, dataset)

        # Compute stats for all filters
        new_dataset = dataset.map(
            self.compute_stats,
            num_proc=self.runtime_np(),
            with_rank=self.use_cuda(),
            batch_size=self.batch_size,
            desc=self._name + "_compute_stats",
        )

        # Export stats if requested
        if exporter and self.stats_export_path is not None:
            exporter.export_compute_stats(new_dataset, self.stats_export_path)

        # Apply filtering if reduce=True
        if reduce:
            new_dataset = new_dataset.filter(
                self.process, num_proc=self.runtime_np(), batch_size=self.batch_size, desc=self._name + "_process"
            )
            if tracer:
                tracer.trace_filter(self._name, dataset, new_dataset)

        # Free models to save memory
        from data_juicer.utils.model_utils import free_models

        free_models()

        return new_dataset


@OPERATORS.register_module("fused_mapper")
class FusedMapper(Mapper):
    """A fused operator for mappers that can execute multiple mappers in one pass."""

    _batched_op = True

    def __init__(self, name: str, fused_mappers: List[str], batch_size: int = 32):
        """Initialize the fused mapper.

        Args:
            name: Name of the fused mapper
            fused_mappers: List of mapper names to be fused
            batch_size: Batch size for processing
        """
        self._name = name
        super().__init__()
        self.batch_size = batch_size

        # Load the mapper operations
        self.fused_mappers = []
        for mapper_name in fused_mappers:
            # Skip if this is a fused_mapper to avoid recursive instantiation
            if mapper_name == "fused_mapper":
                logger.warning("Skipping recursive fused_mapper in FusedMapper initialization")
                continue

            mapper_config = {mapper_name: {}}
            mapper = load_ops([mapper_config])[0]
            self.fused_mappers.append(mapper)

        # Set accelerator to 'cuda' if any of the fused mappers use CUDA
        accelerator_methods = set([op.accelerator for op in self.fused_mappers])
        if "cuda" in accelerator_methods:
            self.accelerator = "cuda"

        # Update num_proc with the minimum of all fused mappers
        self.num_proc = min([op.runtime_np() for op in self.fused_mappers])

        # Store original operation configs
        self._op_cfg = {name: [op._op_cfg for op in self.fused_mappers]}

    def process_batched(self, samples, rank=None):
        """Process samples through all fused mappers.

        Args:
            samples: Batch of samples to process
            rank: Rank for distributed processing

        Returns:
            Processed samples
        """
        # Process mappers sequentially
        for op in self.fused_mappers:
            process_args = {"rank": rank} if op.accelerator == "cuda" else {}
            samples = op.process_batched(samples, **process_args)
        return samples

    def run(self, dataset, *, exporter=None, tracer=None):
        """Run the fused mapper on a dataset.

        Args:
            dataset: Dataset to process
            exporter: Optional exporter for results
            tracer: Optional tracer for monitoring

        Returns:
            Processed dataset
        """
        # Prepare the dataset
        from data_juicer.core.data import NestedDataset

        if not isinstance(dataset, NestedDataset):
            dataset = NestedDataset(dataset)

        # Initialize each mapper
        for op in self.fused_mappers:
            dataset = Mapper.run(op, dataset)

        # Process the dataset
        new_dataset = dataset.map(
            self.process_batched,
            num_proc=self.num_proc,
            with_rank=self.use_cuda(),
            batch_size=self.batch_size,
            desc=self._name + "_process",
        )

        return new_dataset

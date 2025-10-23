#!/usr/bin/env python3
"""
A/B testing framework for optimization strategies.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from loguru import logger

from ..core.benchmark_runner import BenchmarkConfig, BenchmarkRunner
from ..core.metrics_collector import BenchmarkMetrics
from ..core.report_generator import ReportGenerator
from ..core.result_analyzer import ComparisonResult, ResultAnalyzer
from ..workloads.workload_suite import WorkloadDefinition
from .strategy_library import STRATEGY_LIBRARY, StrategyConfig


@dataclass
class ABTestConfig:
    """Configuration for an A/B test."""

    name: str
    baseline_strategy: StrategyConfig
    test_strategies: List[StrategyConfig]
    workload: WorkloadDefinition
    iterations: int = 3
    warmup_runs: int = 1
    output_dir: str = "benchmark_results"
    timeout_seconds: int = 3600


class StrategyABTest:
    """A/B testing framework for optimization strategies."""

    def __init__(self, config: ABTestConfig):
        self.config = config
        self.analyzer = ResultAnalyzer()
        self.report_generator = ReportGenerator(config.output_dir)

        # Ensure output directory exists
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    def run_ab_test(self) -> Dict[str, ComparisonResult]:
        """Run the complete A/B test."""
        logger.info(f"Starting A/B test: {self.config.name}")

        # Run baseline
        logger.info("Running baseline strategy...")
        baseline_results = self._run_strategy(self.config.baseline_strategy)

        # Run test strategies
        all_results = {self.config.baseline_strategy.name: baseline_results}
        comparisons = {}

        for test_strategy in self.config.test_strategies:
            logger.info(f"Running test strategy: {test_strategy.name}")
            test_results = self._run_strategy(test_strategy)
            all_results[test_strategy.name] = test_results

            # Compare with baseline
            comparison = self.analyzer.compare_runs(
                baseline_results, test_results, self.config.baseline_strategy.name, test_strategy.name
            )
            comparisons[test_strategy.name] = comparison

        # Generate report
        report_path = self.report_generator.generate_ab_test_report(all_results, comparisons, self.config.name)

        logger.info(f"A/B test completed. Report: {report_path}")
        return comparisons

    def run_workload_suite_ab_test(
        self, strategies: List[StrategyConfig], workloads: List[WorkloadDefinition]
    ) -> Dict[str, Dict[str, ComparisonResult]]:
        """Run A/B test across multiple workloads."""
        logger.info(f"Running workload suite A/B test with {len(workloads)} workloads")

        all_results = {}

        for workload in workloads:
            logger.info(f"Testing workload: {workload.name}")

            # Create workload-specific A/B test
            workload_config = ABTestConfig(
                name=f"{self.config.name}_{workload.name}",
                baseline_strategy=self.config.baseline_strategy,
                test_strategies=self.config.test_strategies,
                workload=workload,
                iterations=self.config.iterations,
                warmup_runs=self.config.warmup_runs,
                output_dir=os.path.join(self.config.output_dir, workload.name),
                timeout_seconds=self.config.timeout_seconds,
            )

            # Run A/B test for this workload
            workload_ab_test = StrategyABTest(workload_config)
            workload_results = workload_ab_test.run_ab_test()

            all_results[workload.name] = workload_results

        # Generate comprehensive report
        report_path = self.report_generator.generate_workload_report(all_results, f"{self.config.name}_workload_suite")

        logger.info(f"Workload suite A/B test completed. Report: {report_path}")
        return all_results

    def _run_strategy(self, strategy: StrategyConfig) -> List[BenchmarkMetrics]:
        """Run a single strategy and return metrics."""

        # Create benchmark configuration
        benchmark_config = BenchmarkConfig(
            dataset_path=self.config.workload.dataset_path,
            config_path=self.config.workload.config_path,
            output_dir=os.path.join(self.config.output_dir, strategy.name),
            iterations=self.config.iterations,
            warmup_runs=self.config.warmup_runs,
            timeout_seconds=self.config.timeout_seconds,
            strategy_name=strategy.name,
            strategy_config=self._strategy_to_config_dict(strategy),
        )

        # Run benchmark
        runner = BenchmarkRunner(benchmark_config)
        return runner.run_benchmark_suite()

    def _strategy_to_config_dict(self, strategy: StrategyConfig) -> Dict[str, Any]:
        """Convert strategy to configuration dictionary."""
        config_dict = {}

        if strategy.name == "op_fusion_greedy":
            config_dict["op_fusion"] = True
            config_dict["fusion_strategy"] = "greedy"
        elif strategy.name == "op_fusion_probe":
            config_dict["op_fusion"] = True
            config_dict["fusion_strategy"] = "probe"
        elif strategy.name == "adaptive_batch_size":
            config_dict["adaptive_batch_size"] = True
        elif strategy.name == "large_batch_size":
            config_dict["batch_size"] = 1000  # Large batch size
        elif strategy.name == "memory_efficient":
            config_dict["memory_efficient"] = True
        elif strategy.name == "streaming_processing":
            config_dict["streaming"] = True
        elif strategy.name == "max_parallelism":
            config_dict["num_processes"] = -1  # Use all available cores
        elif strategy.name == "ray_optimized":
            config_dict["executor"] = "ray"
            config_dict["ray_config"] = {"num_cpus": -1}
        elif strategy.name == "aggressive_caching":
            config_dict["cache_intermediate"] = True
        elif strategy.name == "fast_algorithms":
            config_dict["use_fast_algorithms"] = True
        elif strategy.name == "vectorized_ops":
            config_dict["vectorized_operations"] = True

        # Add strategy-specific parameters
        config_dict.update(strategy.parameters)

        return config_dict

    def create_strategy_comparison(self, strategy_names: List[str], workload: WorkloadDefinition) -> "StrategyABTest":
        """Create an A/B test comparing multiple strategies."""

        if len(strategy_names) < 2:
            raise ValueError("Need at least 2 strategies for comparison")

        # Use first strategy as baseline
        baseline = STRATEGY_LIBRARY.create_strategy_config(strategy_names[0])
        test_strategies = [STRATEGY_LIBRARY.create_strategy_config(name) for name in strategy_names[1:]]

        config = ABTestConfig(
            name=f"comparison_{'_vs_'.join(strategy_names)}",
            baseline_strategy=baseline,
            test_strategies=test_strategies,
            workload=workload,
        )

        return StrategyABTest(config)

    def get_recommended_strategies(self, workload: WorkloadDefinition) -> List[str]:
        """Get recommended strategies for a specific workload."""
        recommendations = []

        # Modality-specific recommendations
        if workload.modality == "text":
            recommendations.extend(["op_fusion_greedy", "adaptive_batch_size", "vectorized_ops"])
        elif workload.modality == "image":
            recommendations.extend(["op_fusion_probe", "memory_efficient", "ray_optimized"])
        elif workload.modality == "video":
            recommendations.extend(["streaming_processing", "ray_optimized", "aggressive_caching"])
        elif workload.modality == "audio":
            recommendations.extend(["op_fusion_greedy", "adaptive_batch_size"])

        # Complexity-specific recommendations
        if workload.complexity == "complex":
            recommendations.extend(["memory_efficient", "ray_optimized"])

        # Remove duplicates while preserving order
        return list(dict.fromkeys(recommendations))

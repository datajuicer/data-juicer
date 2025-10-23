#!/usr/bin/env python3
"""
Result analysis and comparison tools for benchmark results.
"""

import statistics
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from loguru import logger

from .metrics_collector import BenchmarkMetrics


@dataclass
class ComparisonResult:
    """Results of comparing two benchmark configurations."""

    # Configuration info
    baseline_name: str
    test_name: str

    # Performance comparison
    speedup: float  # test_time / baseline_time
    throughput_improvement: float  # test_throughput / baseline_throughput
    memory_efficiency: float  # baseline_memory / test_memory

    # Statistical significance
    is_significant: bool
    confidence_level: float
    p_value: float

    # Raw metrics
    baseline_metrics: BenchmarkMetrics
    test_metrics: BenchmarkMetrics

    # Summary
    summary: str

    def is_improvement(self, threshold: float = 1.05) -> bool:
        """Check if the test shows significant improvement."""
        return self.speedup > threshold and self.is_significant

    def is_regression(self, threshold: float = 0.95) -> bool:
        """Check if the test shows significant regression."""
        return self.speedup < threshold and self.is_significant


class ResultAnalyzer:
    """Analyzes and compares benchmark results."""

    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level

    def analyze_single_run(self, metrics: BenchmarkMetrics) -> Dict[str, Any]:
        """Analyze a single benchmark run."""
        return {
            "total_time": metrics.total_wall_time,
            "throughput": metrics.samples_per_second,
            "memory_peak": metrics.peak_memory_mb,
            "cpu_avg": metrics.average_cpu_percent,
            "retention_rate": metrics.data_retention_rate,
            "efficiency_score": self._calculate_efficiency_score(metrics),
        }

    def compare_runs(
        self,
        baseline_metrics: List[BenchmarkMetrics],
        test_metrics: List[BenchmarkMetrics],
        baseline_name: str = "baseline",
        test_name: str = "test",
    ) -> ComparisonResult:
        """Compare two sets of benchmark runs."""

        if not baseline_metrics or not test_metrics:
            raise ValueError("Both baseline and test metrics must be provided")

        # Calculate aggregate metrics
        baseline_agg = self._aggregate_metrics(baseline_metrics)
        test_agg = self._aggregate_metrics(test_metrics)

        # Calculate comparisons
        speedup = baseline_agg.total_wall_time / test_agg.total_wall_time if test_agg.total_wall_time > 0 else 0
        throughput_improvement = (
            test_agg.samples_per_second / baseline_agg.samples_per_second if baseline_agg.samples_per_second > 0 else 0
        )
        memory_efficiency = baseline_agg.peak_memory_mb / test_agg.peak_memory_mb if test_agg.peak_memory_mb > 0 else 0

        # Statistical significance test
        is_significant, p_value = self._test_significance(
            [m.total_wall_time for m in baseline_metrics], [m.total_wall_time for m in test_metrics]
        )

        # Generate summary
        summary = self._generate_summary(
            speedup, throughput_improvement, memory_efficiency, is_significant, baseline_name, test_name
        )

        return ComparisonResult(
            baseline_name=baseline_name,
            test_name=test_name,
            speedup=speedup,
            throughput_improvement=throughput_improvement,
            memory_efficiency=memory_efficiency,
            is_significant=is_significant,
            confidence_level=self.confidence_level,
            p_value=p_value,
            baseline_metrics=baseline_agg,
            test_metrics=test_agg,
            summary=summary,
        )

    def analyze_ab_test(self, results: Dict[str, List[BenchmarkMetrics]]) -> Dict[str, ComparisonResult]:
        """Analyze results from an A/B test with multiple strategies."""
        if len(results) < 2:
            raise ValueError("A/B test requires at least 2 strategies")

        # Use first strategy as baseline
        baseline_name = list(results.keys())[0]
        baseline_metrics = results[baseline_name]

        comparisons = {}

        for strategy_name, strategy_metrics in results.items():
            if strategy_name == baseline_name:
                continue

            comparison = self.compare_runs(baseline_metrics, strategy_metrics, baseline_name, strategy_name)
            comparisons[strategy_name] = comparison

        return comparisons

    def _aggregate_metrics(self, metrics_list: List[BenchmarkMetrics]) -> BenchmarkMetrics:
        """Aggregate multiple metrics into a single representative metric."""
        if not metrics_list:
            raise ValueError("Cannot aggregate empty metrics list")

        # Calculate means for most metrics
        total_wall_time = statistics.mean([m.total_wall_time for m in metrics_list])
        samples_per_second = statistics.mean([m.samples_per_second for m in metrics_list])
        peak_memory_mb = max([m.peak_memory_mb for m in metrics_list])  # Use max for peak
        average_cpu_percent = statistics.mean([m.average_cpu_percent for m in metrics_list])
        peak_cpu_percent = max([m.peak_cpu_percent for m in metrics_list])
        samples_processed = sum([m.samples_processed for m in metrics_list]) // len(metrics_list)
        samples_retained = sum([m.samples_retained for m in metrics_list]) // len(metrics_list)
        data_retention_rate = statistics.mean([m.data_retention_rate for m in metrics_list])

        # Use the first metric as template and update values
        aggregated = metrics_list[0]
        aggregated.total_wall_time = total_wall_time
        aggregated.samples_per_second = samples_per_second
        aggregated.peak_memory_mb = peak_memory_mb
        aggregated.average_cpu_percent = average_cpu_percent
        aggregated.peak_cpu_percent = peak_cpu_percent
        aggregated.samples_processed = samples_processed
        aggregated.samples_retained = samples_retained
        aggregated.data_retention_rate = data_retention_rate

        return aggregated

    def _test_significance(self, baseline_times: List[float], test_times: List[float]) -> Tuple[bool, float]:
        """Test statistical significance between two sets of timing data."""
        try:
            # Simple t-test for now - could be enhanced with more sophisticated tests
            from scipy import stats

            # Perform Welch's t-test (unequal variances)
            statistic, p_value = stats.ttest_ind(test_times, baseline_times, equal_var=False)

            is_significant = p_value < (1 - self.confidence_level)
            return is_significant, p_value

        except ImportError:
            # Fallback to simple comparison if scipy not available
            baseline_mean = statistics.mean(baseline_times)
            test_mean = statistics.mean(test_times)

            # Simple threshold-based significance
            difference = abs(test_mean - baseline_mean)
            threshold = baseline_mean * 0.1  # 10% threshold

            is_significant = difference > threshold
            p_value = 0.5 if is_significant else 1.0

            return is_significant, p_value
        except Exception as e:
            logger.warning(f"Error in significance test: {e}")
            return False, 1.0

    def _calculate_efficiency_score(self, metrics: BenchmarkMetrics) -> float:
        """Calculate an overall efficiency score."""
        # Weighted combination of throughput and memory efficiency
        throughput_score = min(metrics.samples_per_second / 1000, 1.0)  # Normalize to 0-1
        memory_score = max(0, 1.0 - metrics.peak_memory_mb / 10000)  # Penalize high memory

        return throughput_score * 0.7 + memory_score * 0.3

    def _generate_summary(
        self,
        speedup: float,
        throughput_improvement: float,
        memory_efficiency: float,
        is_significant: bool,
        baseline_name: str,
        test_name: str,
    ) -> str:
        """Generate a human-readable summary of the comparison."""
        if speedup > 1.05:
            direction = "faster"
            improvement = f"{speedup:.2f}x"
        elif speedup < 0.95:
            direction = "slower"
            improvement = f"{1/speedup:.2f}x"
        else:
            direction = "similar"
            improvement = "~1x"

        significance = "statistically significant" if is_significant else "not statistically significant"

        return (
            f"{test_name} is {improvement} {direction} than {baseline_name} "
            f"({significance}). Throughput: {throughput_improvement:.2f}x, "
            f"Memory efficiency: {memory_efficiency:.2f}x"
        )

"""
Data-Juicer Performance Benchmark Framework

A comprehensive framework for A/B testing optimization strategies
across different workloads, modalities, and operation complexities.
"""

from .core.benchmark_runner import BenchmarkConfig, BenchmarkRunner
from .core.metrics_collector import MetricsCollector
from .core.report_generator import ReportGenerator
from .core.result_analyzer import ResultAnalyzer
from .strategies.ab_test import ABTestConfig, StrategyABTest
from .strategies.strategy_library import STRATEGY_LIBRARY, OptimizationStrategy
from .utils.config_manager import ConfigManager
from .workloads.workload_suite import WORKLOAD_SUITE, WorkloadDefinition, WorkloadSuite

__version__ = "1.0.0"
__all__ = [
    "BenchmarkRunner",
    "BenchmarkConfig",
    "MetricsCollector",
    "ResultAnalyzer",
    "ReportGenerator",
    "OptimizationStrategy",
    "STRATEGY_LIBRARY",
    "StrategyABTest",
    "ABTestConfig",
    "WorkloadSuite",
    "WorkloadDefinition",
    "WORKLOAD_SUITE",
    "ConfigManager",
]

#!/usr/bin/env python3
"""
Optimizer Benchmark Tool

A comprehensive benchmarking tool for testing Data-Juicer optimization strategies.
Uses the data_juicer.benchmark framework for A/B testing, statistical analysis,
and report generation.

Features:
- A/B testing between baseline and optimized pipelines using StrategyABTest
- Support for multiple optimization strategies (op_pruning, op_reorder, fusion)
- Statistical significance testing
- HTML and JSON report generation
- Multiple iteration support for reliable results

Usage:
    # Test op_pruning strategy
    python run_benchmark.py --recipe-path config.yaml --dataset-path data.jsonl --strategies op_pruning

    # Test multiple strategies
    python run_benchmark.py --recipe-path config.yaml --dataset-path data.jsonl --strategies op_pruning,op_reorder

    # A/B test with multiple iterations
    python run_benchmark.py --recipe-path config.yaml --dataset-path data.jsonl --strategies op_pruning --iterations 3
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from loguru import logger

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from data_juicer.benchmark import (  # noqa: E402
    STRATEGY_LIBRARY,
    ABTestConfig,
    StrategyABTest,
)
from data_juicer.benchmark.workloads.workload_suite import (  # noqa: E402
    WorkloadDefinition,
)

# Available optimization strategies
# Note: Probing is controlled separately via optimizer_probe_enabled config option
AVAILABLE_STRATEGIES = [
    "op_pruning",  # Remove no-op and duplicate operations
    "op_reorder",  # Reorder operations for optimal execution order
    "mapper_fusion",  # Fuse consecutive mappers
    "filter_fusion",  # Fuse filters sharing intermediate variables
    "all_optimizations",  # Enable all optimizations
]

# Default strategies if none specified
DEFAULT_STRATEGIES = ["op_pruning", "op_reorder"]


def create_workload_from_args(
    recipe_path: str, dataset_path: str, name: str = "custom_workload", executor: str = None
) -> WorkloadDefinition:
    """Create a WorkloadDefinition from command line arguments."""
    # Count samples in dataset
    try:
        import subprocess

        result = subprocess.run(["wc", "-l", dataset_path], capture_output=True, text=True, timeout=30)
        expected_samples = int(result.stdout.split()[0]) if result.returncode == 0 else 10000
    except Exception:
        expected_samples = 10000

    return WorkloadDefinition(
        name=name,
        description=f"Custom workload from {Path(recipe_path).name}",
        dataset_path=dataset_path,
        config_path=recipe_path,
        expected_samples=expected_samples,
        modality="text",  # Default to text
        complexity="medium",
        estimated_duration_minutes=5,
        resource_requirements={"cpu_cores": 4, "memory_gb": 8},
        executor_type=executor,
    )


def run_ab_test_with_framework(
    recipe_path: str,
    dataset_path: str,
    strategies: List[str],
    output_dir: str,
    iterations: int = 1,
    warmup_runs: int = 0,
    executor: str = None,
) -> Dict[str, Any]:
    """
    Run A/B test using the StrategyABTest framework.

    Args:
        recipe_path: Path to the recipe YAML file
        dataset_path: Path to the dataset file
        strategies: List of optimization strategies to test
        output_dir: Output directory for results
        iterations: Number of benchmark iterations
        warmup_runs: Number of warmup runs
        executor: Executor type ('default' or 'ray'), or None to use recipe config

    Returns:
        Dictionary containing benchmark results
    """
    logger.info("=" * 60)
    logger.info("OPTIMIZER BENCHMARK (A/B Test Framework)")
    logger.info("=" * 60)
    logger.info(f"Recipe: {recipe_path}")
    logger.info(f"Dataset: {dataset_path}")
    logger.info(f"Strategies: {strategies}")
    logger.info(f"Executor: {executor or 'from config'}")
    logger.info(f"Iterations: {iterations}")
    logger.info("=" * 60)

    # Validate strategies
    for strategy in strategies:
        if strategy not in AVAILABLE_STRATEGIES:
            logger.warning(f"Unknown strategy: {strategy}. Available: {AVAILABLE_STRATEGIES}")

    # Create workload definition
    workload = create_workload_from_args(recipe_path, dataset_path, executor=executor)

    # Create baseline strategy
    baseline_strategy = STRATEGY_LIBRARY.create_strategy_config("baseline")

    # Create test strategies
    test_strategies = []
    for strategy_name in strategies:
        try:
            strategy_config = STRATEGY_LIBRARY.create_strategy_config(strategy_name)
            test_strategies.append(strategy_config)
        except ValueError as e:
            logger.error(f"Failed to create strategy config for {strategy_name}: {e}")
            continue

    if not test_strategies:
        raise ValueError("No valid test strategies configured")

    # Create A/B test configuration
    ab_config = ABTestConfig(
        name=f"optimizer_benchmark_{'_'.join(strategies)}",
        baseline_strategy=baseline_strategy,
        test_strategies=test_strategies,
        workload=workload,
        iterations=iterations,
        warmup_runs=warmup_runs,
        output_dir=output_dir,
        timeout_seconds=3600,
    )

    # Run A/B test
    logger.info("\n>>> Running A/B test with StrategyABTest framework...")
    ab_test = StrategyABTest(ab_config)
    comparisons = ab_test.run_ab_test()

    # Build results dictionary
    results = {
        "metadata": {
            "recipe_path": recipe_path,
            "dataset_path": dataset_path,
            "strategies": strategies,
            "iterations": iterations,
            "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "framework": "StrategyABTest",
        },
        "comparisons": {},
        "validation": {
            "all_tests_passed": True,
        },
    }

    # Process comparison results
    for strategy_name, comparison in comparisons.items():
        results["comparisons"][strategy_name] = {
            "speedup": comparison.speedup,
            "throughput_improvement": comparison.throughput_improvement,
            "memory_efficiency": comparison.memory_efficiency,
            "is_significant": comparison.is_significant,
            "p_value": comparison.p_value,
            "summary": comparison.summary,
            "is_improvement": comparison.is_improvement(),
            "is_regression": comparison.is_regression(),
        }

        # Check validation
        if comparison.is_regression():
            results["validation"]["all_tests_passed"] = False
            results["validation"]["regression_detected"] = strategy_name

    # Save results JSON
    results_path = Path(output_dir) / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to: {results_path}")

    # Print summary
    _print_ab_test_summary(results, comparisons)

    return results


def _print_ab_test_summary(results: Dict[str, Any], comparisons: Dict[str, Any]):
    """Print a summary of the A/B test results."""
    logger.info("\n" + "=" * 60)
    logger.info("A/B TEST SUMMARY")
    logger.info("=" * 60)

    for strategy_name, comparison in comparisons.items():
        logger.info(f"\n{strategy_name}:")
        logger.info(f"  Speedup:     {comparison.speedup:.2f}x")
        improvement = (comparison.speedup - 1) * 100
        logger.info(f"  Improvement: {improvement:.1f}%")
        logger.info(f"  Significant: {comparison.is_significant}")
        logger.info(f"  Summary:     {comparison.summary}")

    validation = results.get("validation", {})
    logger.info(f"\nValidation: {'PASSED' if validation.get('all_tests_passed') else 'FAILED'}")
    logger.info("=" * 60)


def list_strategies():
    """List all available optimization strategies."""
    print("\nAvailable Optimization Strategies:")
    print("=" * 50)

    strategies = STRATEGY_LIBRARY.get_all_strategies()
    for strategy in strategies:
        print(f"\n  {strategy.name}")
        print(f"    {strategy.description}")
        if hasattr(strategy, "enabled_strategies"):
            print(f"    Enables: {', '.join(strategy.enabled_strategies)}")

    print("\n" + "=" * 50)
    print(f"\nDefault strategies: {', '.join(DEFAULT_STRATEGIES)}")
    print("\nUsage examples:")
    print("  # Test single strategy")
    print("  python run_benchmark.py --recipe-path config.yaml --dataset-path data.jsonl --strategies op_pruning")
    print("\n  # Test multiple strategies")
    print(
        "  python run_benchmark.py --recipe-path config.yaml --dataset-path data.jsonl --strategies op_pruning,op_reorder"
    )
    print("\n  # Test all optimizations")
    print(
        "  python run_benchmark.py --recipe-path config.yaml --dataset-path data.jsonl --strategies all_optimizations"
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Optimizer Benchmark: Compare baseline vs optimized pipeline execution using A/B testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test op_pruning strategy
  python run_benchmark.py --recipe-path config.yaml --dataset-path data.jsonl --strategies op_pruning

  # Test multiple strategies
  python run_benchmark.py --recipe-path config.yaml --dataset-path data.jsonl --strategies op_pruning,op_reorder

  # Run with multiple iterations for statistical significance
  python run_benchmark.py --recipe-path config.yaml --dataset-path data.jsonl --strategies op_pruning --iterations 3

  # List available strategies
  python run_benchmark.py --list-strategies
        """,
    )

    parser.add_argument(
        "--recipe-path",
        type=str,
        help="Path to the recipe YAML file",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        help="Path to the dataset file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/optimizer_benchmark",
        help="Output directory for results and reports (default: ./outputs/optimizer_benchmark)",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        default=None,
        help=f"Comma-separated list of optimization strategies. "
        f"Available: {', '.join(AVAILABLE_STRATEGIES)}. "
        f"Default: {', '.join(DEFAULT_STRATEGIES)}",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of benchmark iterations for statistical significance (default: 1)",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=0,
        help="Number of warmup runs before actual benchmarking (default: 0)",
    )
    parser.add_argument(
        "--executor",
        type=str,
        choices=["default", "ray"],
        default=None,
        help="Executor type: 'default' (local) or 'ray' (distributed). "
        "If not specified, uses the value from the recipe config.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--list-strategies",
        action="store_true",
        help="List all available optimization strategies and exit",
    )

    args = parser.parse_args()

    # Handle list-strategies command
    if args.list_strategies:
        list_strategies()
        sys.exit(0)

    # Validate required arguments
    if not args.recipe_path or not args.dataset_path:
        parser.error("--recipe-path and --dataset-path are required (unless using --list-strategies)")

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger.remove()
    logger.add(sys.stderr, level=log_level, format="<level>{message}</level>")

    # Add file logging
    log_dir = Path(args.output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(log_dir / "benchmark.log", rotation="10 MB", level="INFO")

    # Parse strategies
    strategies = DEFAULT_STRATEGIES
    if args.strategies:
        strategies = [s.strip() for s in args.strategies.split(",")]

    try:
        results = run_ab_test_with_framework(
            recipe_path=args.recipe_path,
            dataset_path=args.dataset_path,
            strategies=strategies,
            output_dir=args.output_dir,
            iterations=args.iterations,
            warmup_runs=args.warmup_runs,
            executor=args.executor,
        )

        # Exit with appropriate code
        validation = results.get("validation", {})
        if validation.get("all_tests_passed"):
            logger.info("Benchmark completed successfully")
            sys.exit(0)
        else:
            logger.warning("Benchmark completed but validation failed")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Benchmark failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

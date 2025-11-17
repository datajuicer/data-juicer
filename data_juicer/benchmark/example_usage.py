#!/usr/bin/env python3
"""
Example usage of the benchmark framework.
"""

import os

from data_juicer.benchmark import (
    STRATEGY_LIBRARY,
    WORKLOAD_SUITE,
    ABTestConfig,
    BenchmarkConfig,
    BenchmarkRunner,
    StrategyABTest,
)


def example_single_benchmark():
    """Example: Run a single benchmark."""
    print("=== Single Benchmark Example ===")

    # Create benchmark configuration
    config = BenchmarkConfig(
        dataset_path="demos/data/text_data.jsonl",
        config_path="configs/demo/process.yaml",
        output_dir="benchmark_results/single",
        iterations=3,
        strategy_name="op_fusion_greedy",
        strategy_config={"op_fusion": True, "fusion_strategy": "greedy"},
    )

    # Run benchmark
    runner = BenchmarkRunner(config)
    results = runner.run_benchmark_suite()

    # Print results
    for i, metrics in enumerate(results):
        print(f"Run {i+1}: {metrics.total_wall_time:.2f}s, " f"{metrics.samples_per_second:.1f} samples/sec")


def example_ab_test():
    """Example: Run A/B test between strategies."""
    print("\n=== A/B Test Example ===")

    # Get workload
    workload = WORKLOAD_SUITE.get_workload("text_simple")
    if not workload:
        print("Workload not found!")
        return

    # Create strategy configurations
    baseline = STRATEGY_LIBRARY.create_strategy_config("baseline")
    test_strategy = STRATEGY_LIBRARY.create_strategy_config("op_fusion_greedy")

    # Create A/B test configuration
    ab_config = ABTestConfig(
        name="fusion_vs_baseline",
        baseline_strategy=baseline,
        test_strategies=[test_strategy],
        workload=workload,
        iterations=3,
        output_dir="benchmark_results/ab_test",
    )

    # Run A/B test
    ab_test = StrategyABTest(ab_config)
    results = ab_test.run_ab_test()

    # Print results
    for strategy_name, comparison in results.items():
        print(f"{strategy_name}: {comparison.speedup:.2f}x speedup")
        print(f"  Summary: {comparison.summary}")


def example_workload_suite():
    """Example: Run tests across multiple workloads."""
    print("\n=== Workload Suite Example ===")

    # Get multiple workloads
    workloads = [WORKLOAD_SUITE.get_workload("text_simple"), WORKLOAD_SUITE.get_workload("image_simple")]
    workloads = [w for w in workloads if w]  # Filter out None values

    if not workloads:
        print("No workloads found!")
        return

    # Create strategies
    strategies = [
        STRATEGY_LIBRARY.create_strategy_config("baseline"),
        STRATEGY_LIBRARY.create_strategy_config("op_fusion_greedy"),
        STRATEGY_LIBRARY.create_strategy_config("adaptive_batch_size"),
    ]

    # Create A/B test
    ab_config = ABTestConfig(
        name="workload_suite_test",
        baseline_strategy=strategies[0],
        test_strategies=strategies[1:],
        workload=workloads[0],  # Will be overridden
        iterations=2,
        output_dir="benchmark_results/workload_suite",
    )

    ab_test = StrategyABTest(ab_config)
    results = ab_test.run_workload_suite_ab_test(strategies[1:], workloads)

    # Print results
    for workload_name, workload_results in results.items():
        print(f"\n{workload_name}:")
        for strategy_name, comparison in workload_results.items():
            print(f"  {strategy_name}: {comparison.speedup:.2f}x speedup")


def example_strategy_comparison():
    """Example: Compare multiple strategies."""
    print("\n=== Strategy Comparison Example ===")

    # Get workload
    workload = WORKLOAD_SUITE.get_workload("text_simple")
    if not workload:
        print("Workload not found!")
        return

    # Create strategy comparison
    strategy_names = ["baseline", "op_fusion_greedy", "adaptive_batch_size"]
    ab_test = StrategyABTest.create_strategy_comparison(strategy_names, workload)

    # Run comparison
    results = ab_test.run_ab_test()

    # Print results
    print("Strategy Comparison Results:")
    for strategy_name, comparison in results.items():
        print(f"  {strategy_name}: {comparison.speedup:.2f}x speedup")
        print(f"    {comparison.summary}")


def main():
    """Run all examples."""
    print("Data-Juicer Benchmark Framework Examples")
    print("=" * 50)

    # Ensure output directories exist
    os.makedirs("benchmark_results", exist_ok=True)

    try:
        # Run examples
        example_single_benchmark()
        example_ab_test()
        example_workload_suite()
        example_strategy_comparison()

        print("\n=== All Examples Completed ===")
        print("Check the 'benchmark_results' directory for detailed reports.")

    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

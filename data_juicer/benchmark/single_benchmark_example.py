#!/usr/bin/env python3
"""
Example: Single benchmark with custom config and dataset.
"""

from data_juicer.benchmark import BenchmarkConfig, BenchmarkRunner


def run_single_benchmark():
    """Run a single benchmark with your own config and dataset."""

    # Create benchmark configuration
    config = BenchmarkConfig(
        dataset_path="your_dataset.jsonl",  # Your dataset path
        config_path="your_config.yaml",  # Your config path
        output_dir="benchmark_results/",  # Output directory
        iterations=3,  # Number of runs
        warmup_runs=1,  # Warmup runs (not counted)
        timeout_seconds=3600,  # Timeout
        strategy_name="op_fusion_greedy",  # Strategy to test
        strategy_config={"op_fusion": True, "fusion_strategy": "greedy"},  # Strategy-specific config
    )

    # Run benchmark
    runner = BenchmarkRunner(config)
    results = runner.run_benchmark_suite()

    # Analyze results
    print("=== Single Benchmark Results ===")
    for i, metrics in enumerate(results):
        print(f"\nRun {i+1}:")
        print(f"  Time: {metrics.total_wall_time:.2f}s")
        print(f"  Throughput: {metrics.samples_per_second:.1f} samples/sec")
        print(f"  Memory: {metrics.peak_memory_mb:.0f} MB")
        print(f"  CPU: {metrics.average_cpu_percent:.1f}%")
        print(f"  Retention: {metrics.data_retention_rate:.1%}")

    # Calculate statistics
    times = [m.total_wall_time for m in results]
    throughputs = [m.samples_per_second for m in results]

    print(f"\n=== Summary Statistics ===")
    print(f"Average time: {sum(times)/len(times):.2f}s")
    print(f"Average throughput: {sum(throughputs)/len(throughputs):.1f} samples/sec")
    print(f"Time std dev: {((sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5):.2f}s")

    return results


def run_multiple_strategies_same_dataset():
    """Run multiple strategies on the same dataset for comparison."""

    strategies = [
        ("baseline", {}),
        ("op_fusion_greedy", {"op_fusion": True, "fusion_strategy": "greedy"}),
        ("adaptive_batch_size", {"adaptive_batch_size": True}),
        ("memory_efficient", {"memory_efficient": True, "streaming": True}),
    ]

    all_results = {}

    for strategy_name, strategy_config in strategies:
        print(f"\n=== Running {strategy_name} ===")

        config = BenchmarkConfig(
            dataset_path="your_dataset.jsonl",
            config_path="your_config.yaml",
            output_dir=f"benchmark_results/{strategy_name}/",
            iterations=3,
            strategy_name=strategy_name,
            strategy_config=strategy_config,
        )

        runner = BenchmarkRunner(config)
        results = runner.run_benchmark_suite()
        all_results[strategy_name] = results

        # Print quick summary
        avg_time = sum(m.total_wall_time for m in results) / len(results)
        avg_throughput = sum(m.samples_per_second for m in results) / len(results)
        print(f"  Average time: {avg_time:.2f}s")
        print(f"  Average throughput: {avg_throughput:.1f} samples/sec")

    # Compare results
    print(f"\n=== Strategy Comparison ===")
    baseline_time = sum(m.total_wall_time for m in all_results["baseline"]) / len(all_results["baseline"])

    for strategy_name, results in all_results.items():
        if strategy_name == "baseline":
            continue

        avg_time = sum(m.total_wall_time for m in results) / len(results)
        speedup = baseline_time / avg_time
        print(f"{strategy_name}: {speedup:.2f}x speedup")

    return all_results


if __name__ == "__main__":
    print("Single Benchmark Examples")
    print("=" * 50)

    # Example 1: Single strategy
    print("\n1. Single Strategy Benchmark:")
    # results = run_single_benchmark()

    # Example 2: Multiple strategies on same dataset
    print("\n2. Multiple Strategies Comparison:")
    # all_results = run_multiple_strategies_same_dataset()

    print("\nNote: Update the dataset and config paths to your actual files!")

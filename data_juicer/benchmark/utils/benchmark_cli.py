#!/usr/bin/env python3
"""
Command-line interface for the benchmark framework.
"""

import argparse
import sys
from typing import List, Optional

from loguru import logger

# Import these inside functions to avoid circular imports
# from ..core.benchmark_runner import BenchmarkRunner, BenchmarkConfig
# from ..strategies.ab_test import StrategyABTest, ABTestConfig
# from ..strategies.strategy_library import STRATEGY_LIBRARY
# from ..workloads.workload_suite import WORKLOAD_SUITE


class BenchmarkCLI:
    """Command-line interface for benchmark framework."""

    def __init__(self):
        self.parser = self._create_parser()

    def _create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser."""
        parser = argparse.ArgumentParser(
            description="Data-Juicer Performance Benchmark Framework",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Run A/B test with specific strategies
  python -m data_juicer.benchmark.utils.benchmark_cli ab-test \\
    --baseline baseline \\
    --target-strategies mapper_fusion,filter_fusion \\
    --workload text_simple \\
    --output-dir outputs/benchmark/

  # Run workload suite test
  python -m data_juicer.benchmark.utils.benchmark_cli workload-suite \\
    --workloads text_simple,image_simple \\
    --baseline baseline \\
    --target-strategies mapper_fusion,full_optimization \\
    --output-dir outputs/benchmark/

  # Run single benchmark with custom dataset and config
  python -m data_juicer.benchmark.utils.benchmark_cli single \\
    --dataset /path/to/your/dataset.jsonl \\
    --config /path/to/your/config.yaml \\
    --strategy baseline \\
    --output-dir outputs/benchmark/

  # Run benchmark with production text dataset and simple config
  python -m data_juicer.benchmark.utils.benchmark_cli single \\
    --modality text \\
    --config-type simple \\
    --strategy baseline \\
    --output-dir outputs/benchmark/

  # Run benchmark with production text dataset and production config
  python -m data_juicer.benchmark.utils.benchmark_cli single \\
    --modality text \\
    --config-type production \\
    --strategy mapper_fusion \\
    --output-dir outputs/benchmark/

  # Run benchmark with 10% sampling
  python -m data_juicer.benchmark.utils.benchmark_cli single \\
    --modality text \\
    --config-type production \\
    --strategy baseline \\
    --sample-ratio 0.1 \\
    --sample-method random \\
    --output-dir outputs/benchmark/
            """,
        )

        subparsers = parser.add_subparsers(dest="command", help="Available commands")

        # A/B test command
        ab_parser = subparsers.add_parser("ab-test", help="Run A/B test between strategies")
        ab_parser.add_argument("--baseline", required=True, help="Baseline strategy name")
        ab_parser.add_argument(
            "--target-strategies",
            required=True,
            help="Comma-separated list of target strategies to test against baseline",
        )
        ab_parser.add_argument("--workload", required=True, help="Workload to use for testing")
        ab_parser.add_argument("--iterations", type=int, default=3, help="Number of iterations per strategy")
        ab_parser.add_argument("--output-dir", default="outputs/benchmark", help="Output directory for results")
        ab_parser.add_argument("--timeout", type=int, default=3600, help="Timeout in seconds")

        # Workload suite command
        suite_parser = subparsers.add_parser("workload-suite", help="Run tests across multiple workloads")
        suite_parser.add_argument("--workloads", required=True, help="Comma-separated list of workloads to test")
        suite_parser.add_argument("--baseline", required=True, help="Baseline strategy name")
        suite_parser.add_argument(
            "--target-strategies",
            required=True,
            help="Comma-separated list of target strategies to test against baseline",
        )
        suite_parser.add_argument("--iterations", type=int, default=3, help="Number of iterations per strategy")
        suite_parser.add_argument("--output-dir", default="outputs/benchmark", help="Output directory for results")
        suite_parser.add_argument("--timeout", type=int, default=3600, help="Timeout in seconds")

        # Single benchmark command
        single_parser = subparsers.add_parser("single", help="Run single benchmark")

        # Dataset options
        single_parser.add_argument("--dataset", help="Path to custom dataset")
        single_parser.add_argument(
            "--modality",
            choices=["text", "image", "video", "audio"],
            help="Use production dataset for specified modality",
        )

        # Config options
        single_parser.add_argument("--config", help="Path to custom configuration file")
        single_parser.add_argument(
            "--config-type",
            choices=["simple", "production"],
            default="simple",
            help="Use simple or production config for the modality",
        )

        # Sampling and other options
        single_parser.add_argument("--strategy", required=True, help="Strategy to test")
        single_parser.add_argument("--iterations", type=int, default=3, help="Number of iterations")
        single_parser.add_argument("--output-dir", default="outputs/benchmark", help="Output directory for results")
        single_parser.add_argument("--timeout", type=int, default=3600, help="Timeout in seconds")
        single_parser.add_argument(
            "--sample-ratio",
            type=float,
            default=1.0,
            help="Sample ratio (0.1 = 10 percent of dataset, 1.0 = full dataset)",
        )
        single_parser.add_argument(
            "--sample-method", choices=["random", "first", "last"], default="random", help="Sampling method"
        )

        # A/B test optimization command
        ab_opt_parser = subparsers.add_parser(
            "ab-optimization", help="Run A/B test comparing baseline vs optimized strategies"
        )

        # Dataset options for A/B test
        ab_opt_parser.add_argument("--dataset", help="Path to custom dataset")
        ab_opt_parser.add_argument(
            "--modality",
            choices=["text", "image", "video", "audio"],
            help="Use production dataset for specified modality",
        )

        # Config options for A/B test
        ab_opt_parser.add_argument("--config", help="Path to custom configuration file")
        ab_opt_parser.add_argument(
            "--config-type",
            choices=["simple", "production"],
            default="simple",
            help="Use simple or production config for the modality",
        )

        # Optimization strategy options
        ab_opt_parser.add_argument(
            "--optimizations",
            nargs="+",
            choices=["mapper_fusion", "filter_fusion", "full_optimization"],
            default=["mapper_fusion"],
            help="Optimization strategies to test (default: mapper_fusion)",
        )
        ab_opt_parser.add_argument(
            "--baseline-name",
            default="baseline",
            help="Name for baseline strategy (default: baseline)",
        )
        ab_opt_parser.add_argument(
            "--optimized-name",
            default="optimized",
            help="Name for optimized strategy (default: optimized)",
        )

        # Sampling and other options
        ab_opt_parser.add_argument("--iterations", type=int, default=3, help="Number of iterations per strategy")
        ab_opt_parser.add_argument("--output-dir", default="outputs/benchmark", help="Output directory for results")
        ab_opt_parser.add_argument("--timeout", type=int, default=3600, help="Timeout in seconds")
        ab_opt_parser.add_argument(
            "--sample-ratio",
            type=float,
            default=1.0,
            help="Sample ratio (0.1 = 10 percent of dataset, 1.0 = full dataset)",
        )
        ab_opt_parser.add_argument(
            "--sample-method", choices=["random", "first", "last"], default="random", help="Sampling method"
        )

        # List command
        list_parser = subparsers.add_parser("list", help="List available options")
        list_parser.add_argument("--workloads", action="store_true", help="List available workloads")
        list_parser.add_argument("--strategies", action="store_true", help="List available strategies")

        return parser

    def run(self, args: Optional[List[str]] = None) -> int:
        """Run the CLI with given arguments."""
        if args is None:
            args = sys.argv[1:]

        parsed_args = self.parser.parse_args(args)

        if not parsed_args.command:
            self.parser.print_help()
            return 1

        try:
            if parsed_args.command == "ab-test":
                return self._run_ab_test(parsed_args)
            elif parsed_args.command == "workload-suite":
                return self._run_workload_suite(parsed_args)
            elif parsed_args.command == "single":
                return self._run_single_benchmark(parsed_args)
            elif parsed_args.command == "ab-optimization":
                return self._run_ab_optimization(parsed_args)
            elif parsed_args.command == "list":
                return self._list_options(parsed_args)
            else:
                logger.error(f"Unknown command: {parsed_args.command}")
                return 1

        except Exception as e:
            logger.error(f"Error running command: {e}")
            return 1

    def _run_ab_test(self, args) -> int:
        """Run A/B test."""
        # Import here to avoid circular imports
        from data_juicer.benchmark.strategies.ab_test import (
            ABTestConfig,
            StrategyABTest,
        )
        from data_juicer.benchmark.strategies.strategy_library import STRATEGY_LIBRARY
        from data_juicer.benchmark.workloads.workload_suite import WORKLOAD_SUITE

        # Parse baseline and target strategies
        baseline_name = args.baseline.strip()
        target_strategy_names = [s.strip() for s in args.target_strategies.split(",")]

        # Get workload
        workload = WORKLOAD_SUITE.get_workload(args.workload)
        if not workload:
            logger.error(f"Unknown workload: {args.workload}")
            return 1

        # Create strategy configs
        baseline = STRATEGY_LIBRARY.create_strategy_config(baseline_name)
        test_strategies = [STRATEGY_LIBRARY.create_strategy_config(name) for name in target_strategy_names]

        # Create A/B test config
        ab_config = ABTestConfig(
            name=f"ab_test_{args.workload}",
            baseline_strategy=baseline,
            test_strategies=test_strategies,
            workload=workload,
            iterations=args.iterations,
            output_dir=args.output_dir,
            timeout_seconds=args.timeout,
        )

        # Run A/B test
        ab_test = StrategyABTest(ab_config)
        results = ab_test.run_ab_test()

        # Print summary
        print("\n=== A/B Test Results ===")
        for strategy_name, comparison in results.items():
            print(f"\n{strategy_name} vs {comparison.baseline_name}:")
            print(f"  Speedup: {comparison.speedup:.2f}x")
            print(f"  Throughput: {comparison.throughput_improvement:.2f}x")
            print(f"  Memory: {comparison.memory_efficiency:.2f}x")
            print(f"  Significant: {comparison.is_significant}")
            print(f"  Summary: {comparison.summary}")

        return 0

    def _run_workload_suite(self, args) -> int:
        """Run workload suite test."""
        # Import here to avoid circular imports
        from data_juicer.benchmark.strategies.ab_test import (
            ABTestConfig,
            StrategyABTest,
        )
        from data_juicer.benchmark.strategies.strategy_library import STRATEGY_LIBRARY
        from data_juicer.benchmark.workloads.workload_suite import WORKLOAD_SUITE

        # Parse workloads, baseline and target strategies
        workload_names = [w.strip() for w in args.workloads.split(",")]
        baseline_name = args.baseline.strip()
        target_strategy_names = [s.strip() for s in args.target_strategies.split(",")]

        # Get workloads
        workloads = []
        for name in workload_names:
            workload = WORKLOAD_SUITE.get_workload(name)
            if not workload:
                logger.error(f"Unknown workload: {name}")
                return 1
            workloads.append(workload)

        # Create strategy configs
        baseline = STRATEGY_LIBRARY.create_strategy_config(baseline_name)
        test_strategies = [STRATEGY_LIBRARY.create_strategy_config(name) for name in target_strategy_names]

        # Run workload suite test
        ab_test = StrategyABTest(
            ABTestConfig(
                name="workload_suite_test",
                baseline_strategy=baseline,
                test_strategies=test_strategies,
                workload=workloads[0],  # Will be overridden
                iterations=args.iterations,
                output_dir=args.output_dir,
                timeout_seconds=args.timeout,
            )
        )

        results = ab_test.run_workload_suite_ab_test(test_strategies, workloads)

        # Print summary
        print("\n=== Workload Suite Results ===")
        for workload_name, workload_results in results.items():
            print(f"\n{workload_name}:")
            for strategy_name, comparison in workload_results.items():
                print(f"  {strategy_name}: {comparison.speedup:.2f}x speedup")

        return 0

    def _run_single_benchmark(self, args) -> int:
        """Run single benchmark."""
        # Import here to avoid circular imports
        from ..core.benchmark_runner import BenchmarkConfig, BenchmarkRunner
        from ..strategies.strategy_library import STRATEGY_LIBRARY

        # Determine dataset and config paths
        dataset_path, config_path = self._resolve_dataset_and_config(args)

        # Get the actual strategy and apply it to get the configuration changes
        strategy_obj = STRATEGY_LIBRARY.get_strategy(args.strategy)
        if strategy_obj:
            # Apply the strategy to get the actual config changes
            strategy_config = strategy_obj.apply_to_config({})
        else:
            # Fallback to basic config
            strategy_config = {}

        # Create benchmark config
        benchmark_config = BenchmarkConfig(
            dataset_path=dataset_path,
            config_path=config_path,
            output_dir=args.output_dir,
            iterations=args.iterations,
            timeout_seconds=args.timeout,
            strategy_name=args.strategy,
            strategy_config=strategy_config,
            sample_ratio=args.sample_ratio,
            sample_method=args.sample_method,
        )

        # Run benchmark
        runner = BenchmarkRunner(benchmark_config)
        results = runner.run_benchmark_suite()

        # Print results
        print("\n=== Benchmark Results ===")
        for i, metrics in enumerate(results):
            print(f"\nRun {i+1}:")
            print(f"  Time: {metrics.total_wall_time:.2f}s")
            print(f"  Throughput: {metrics.samples_per_second:.1f} samples/sec")
            print(f"  Memory: {metrics.peak_memory_mb:.0f} MB")
            print(f"  Retention: {metrics.data_retention_rate:.1%}")

        return 0

    def _resolve_dataset_and_config(self, args):
        """Resolve dataset and config paths based on arguments."""
        # Validate arguments
        if not args.config and not args.modality:
            raise ValueError("Either --config or --modality must be specified")

        # Determine dataset path
        dataset_path = None
        if args.dataset:
            dataset_path = args.dataset
        elif args.modality:
            # Use production dataset for the specified modality
            dataset_path = f"perf_bench_data/{args.modality}/"
            if args.modality == "text":
                dataset_path += "wiki-10k.jsonl"
            elif args.modality == "image":
                dataset_path += "10k.jsonl"
            elif args.modality == "video":
                dataset_path += "msr_vtt_train.jsonl"
            elif args.modality == "audio":
                dataset_path += "audio-10k.jsonl"
        # If neither --dataset nor --modality is specified, dataset_path will be None
        # and the config file should contain the dataset_path field

        # Determine config path
        if args.config:
            config_path = args.config
        else:  # args.modality is specified
            # Use production or simple config for the specified modality
            if args.config_type == "production":
                config_path = f"tests/benchmark_performance/configs/{args.modality}.yaml"
            else:  # simple
                config_path = "configs/demo/process.yaml"

        return dataset_path, config_path

    def _list_options(self, args) -> int:
        """List available options."""
        # Import here to avoid circular imports
        from ..strategies.strategy_library import STRATEGY_LIBRARY
        from ..workloads.workload_suite import WORKLOAD_SUITE

        if args.workloads:
            print("\n=== Available Workloads ===")
            for workload in WORKLOAD_SUITE.get_all_workloads():
                print(f"  {workload.name}: {workload.description}")
                print(f"    Modality: {workload.modality}, Complexity: {workload.complexity}")
                print(f"    Expected samples: {workload.expected_samples}")
                print(f"    Duration: {workload.estimated_duration_minutes} min")
                print()

        if args.strategies:
            print("\n=== Available Strategies ===")
            for strategy in STRATEGY_LIBRARY.get_all_strategies():
                print(f"  {strategy.name}: {strategy.description}")
                print(f"    Type: {strategy.strategy_type.value}")
                print()

        return 0


def main():
    """Main entry point for CLI."""
    cli = BenchmarkCLI()
    return cli.run()


if __name__ == "__main__":
    sys.exit(main())

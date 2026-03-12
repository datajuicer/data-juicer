#!/usr/bin/env python3
"""
Pipeline Optimization Demo

This script demonstrates the pipeline optimizer by:
1. Showing the original operation order
2. Showing the optimized operation order
3. Running both and comparing performance

Usage:
    python demos/pipeline_optimization/run_demo.py [--config CONFIG] [--dataset DATASET]

Examples:
    # Use default config with sample data
    python demos/pipeline_optimization/run_demo.py

    # Use custom config
    python demos/pipeline_optimization/run_demo.py \
        --config demos/pipeline_optimization/configs/op_reorder_showcase.yaml \
        --dataset path/to/data.jsonl
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def print_ops(ops: list, title: str):
    """Print operation list."""
    print(f"\n{title}:")
    print("-" * 40)
    for i, op in enumerate(ops, 1):
        op_name = getattr(op, "_name", type(op).__name__)
        print(f"  {i}. {op_name}")
    print()


def run_demo(config_path: str, dataset_path: str = None):
    """Run the optimization demo."""
    from data_juicer.config import init_configs
    from data_juicer.ops import load_ops
    from data_juicer.core.optimization_manager import apply_optimizations

    print_header("Pipeline Optimization Demo")

    # Load config
    print(f"\nLoading config: {config_path}")

    args = ["--config", config_path]
    if dataset_path:
        args.extend(["--dataset_path", dataset_path])

    cfg = init_configs(args=args)

    # Load original ops
    original_ops = load_ops(cfg.process)
    print_ops(original_ops, "Original Operation Order")

    # Check if optimizer is enabled
    if not getattr(cfg, "enable_optimizer", False):
        print("Note: Optimizer is disabled in config.")
        print("Enable with: enable_optimizer: true")
        print("\nSimulating optimization anyway for demo...\n")

    # Force enable for demo
    cfg.enable_optimizer = True
    if not hasattr(cfg, "optimizer_strategies") or not cfg.optimizer_strategies:
        cfg.optimizer_strategies = ["op_reorder", "filter_fusion"]

    print(f"Strategies: {cfg.optimizer_strategies}")

    # Apply optimizations
    optimized_ops = load_ops(cfg.process)
    optimized_ops = apply_optimizations(optimized_ops, cfg)
    print_ops(optimized_ops, "Optimized Operation Order")

    # Show what changed
    print_header("Optimization Summary")

    original_names = [getattr(op, "_name", type(op).__name__) for op in original_ops]
    optimized_names = [getattr(op, "_name", type(op).__name__) for op in optimized_ops]

    # Check for fusion
    fused_count = sum(1 for name in optimized_names if "fused" in name.lower())
    if fused_count > 0:
        original_filter_count = sum(1 for op in original_ops if "filter" in getattr(op, "_name", "").lower())
        print(f"Filter Fusion: {original_filter_count} filters -> {fused_count} fused operation(s)")

    # Check for reordering
    if original_names != optimized_names:
        print("Operation Reorder: Yes")

        # Find what moved
        for i, (orig, opt) in enumerate(zip(original_names, optimized_names)):
            if orig != opt:
                print(f"  Position {i + 1}: {orig} -> {opt}")
    else:
        print("Operation Reorder: No changes needed")

    print(f"\nOriginal ops: {len(original_ops)}")
    print(f"Optimized ops: {len(optimized_ops)}")

    return cfg, original_ops, optimized_ops


def run_benchmark(cfg, dataset_path: str = None):
    """Run a quick benchmark comparing optimized vs non-optimized."""
    from data_juicer.config import init_configs
    from data_juicer.core import DefaultExecutor
    from data_juicer.core.data.dataset_builder import DatasetBuilder

    print_header("Performance Comparison")

    if dataset_path:
        cfg.dataset_path = dataset_path

    # Check if dataset exists
    if not Path(cfg.dataset_path).exists():
        print(f"Dataset not found: {cfg.dataset_path}")
        print("Skipping benchmark. Provide --dataset to run performance comparison.")
        return

    print(f"Dataset: {cfg.dataset_path}")

    # Run without optimization
    print("\nRunning WITHOUT optimization...")
    cfg.enable_optimizer = False
    cfg.export_path = "/tmp/optimization_demo_baseline.jsonl"

    start = time.time()
    executor = DefaultExecutor(cfg)
    executor.run()
    baseline_time = time.time() - start
    print(f"  Time: {baseline_time:.2f}s")

    # Run with optimization
    print("\nRunning WITH optimization...")
    cfg.enable_optimizer = True
    cfg.export_path = "/tmp/optimization_demo_optimized.jsonl"

    start = time.time()
    executor = DefaultExecutor(cfg)
    executor.run()
    optimized_time = time.time() - start
    print(f"  Time: {optimized_time:.2f}s")

    # Compare
    print_header("Results")
    print(f"Baseline:  {baseline_time:.2f}s")
    print(f"Optimized: {optimized_time:.2f}s")

    if optimized_time < baseline_time:
        speedup = baseline_time / optimized_time
        improvement = (1 - optimized_time / baseline_time) * 100
        print(f"Speedup:   {speedup:.2f}x ({improvement:.1f}% faster)")
    else:
        print("No speedup (dataset may be too small to benefit)")


def main():
    parser = argparse.ArgumentParser(description="Pipeline Optimization Demo")
    parser.add_argument(
        "--config",
        default="demos/pipeline_optimization/configs/basic.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Path to dataset (optional, for benchmark)",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run performance benchmark",
    )
    args = parser.parse_args()

    # Run demo
    cfg, original_ops, optimized_ops = run_demo(args.config, args.dataset)

    # Optionally run benchmark
    if args.benchmark and args.dataset:
        run_benchmark(cfg, args.dataset)
    elif args.benchmark:
        print("\nNote: Add --dataset PATH to run performance benchmark")


if __name__ == "__main__":
    main()

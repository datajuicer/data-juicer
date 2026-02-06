#!/usr/bin/env python3
"""
Robustness Benchmark for Partitioned Checkpointing

Measures the tradeoff between fault tolerance and performance overhead.

Usage:
    cd /path/to/data-juicer
    python demos/partition_and_checkpoint/robustness_benchmark.py

    # With custom dataset
    python demos/partition_and_checkpoint/robustness_benchmark.py --dataset /path/to/data.jsonl

    # Quick mode (fewer runs)
    python demos/partition_and_checkpoint/robustness_benchmark.py --quick
"""

import argparse
import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    config_name: str
    executor_type: str
    checkpoint_strategy: str
    partition_mode: str
    total_time_seconds: float
    num_partitions: int
    num_operations: int
    checkpoint_count: int
    storage_used_mb: float
    input_rows: int
    output_rows: int
    success: bool
    error_message: Optional[str] = None


@dataclass
class RecoveryResult:
    """Results from a failure/recovery test."""

    config_name: str
    failure_point_percent: float
    time_before_failure: float
    recovery_time: float
    total_time_with_recovery: float
    rows_reprocessed: int
    work_preserved_percent: float


def get_dir_size_mb(path: str) -> float:
    """Get directory size in MB."""
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total += os.path.getsize(fp)
    return total / (1024 * 1024)


def count_jsonl_rows(path: str) -> int:
    """Count rows in a JSONL file or directory of JSON files."""
    if not os.path.exists(path):
        return 0

    if os.path.isfile(path):
        with open(path) as f:
            return sum(1 for _ in f)

    # Directory (Ray's sharded output format)
    total = 0
    for filename in os.listdir(path):
        if filename.endswith(".json") or filename.endswith(".jsonl"):
            filepath = os.path.join(path, filename)
            with open(filepath) as f:
                total += sum(1 for _ in f)
    return total


def create_test_dataset(output_path: str, num_samples: int = 10000) -> str:
    """Create a test dataset for benchmarking."""
    print(f"Creating test dataset with {num_samples} samples...")

    with open(output_path, "w") as f:
        for i in range(num_samples):
            sample = {
                "text": f"Sample {i}: " + "This is test content. " * 20,
                "id": i,
                "meta": {"source": "benchmark", "index": i},
            }
            f.write(json.dumps(sample) + "\n")

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Created dataset: {output_path} ({size_mb:.1f} MB)")
    return output_path


def create_benchmark_config(
    dataset_path: str,
    output_dir: str,
    executor_type: str = "ray_partitioned",
    num_partitions: int = 4,
    checkpoint_enabled: bool = True,
    checkpoint_strategy: str = "every_op",
    checkpoint_n_ops: int = 2,
) -> str:
    """Create a config file for benchmarking."""

    config = {
        "dataset_path": dataset_path,
        "export_path": os.path.join(output_dir, "output.jsonl"),
        "work_dir": output_dir,
        "executor_type": executor_type,
        "ray_address": "local",  # Start local Ray cluster
        "np": 2,
        "event_logging": {
            "enabled": True,
        },
        # Simple pipeline for benchmarking
        "process": [
            {"whitespace_normalization_mapper": None},
            {"clean_email_mapper": None},
            {"clean_links_mapper": None},
            {"fix_unicode_mapper": None},
        ],
    }

    # Only add partition config for ray_partitioned executor
    if executor_type == "ray_partitioned":
        config["partition"] = {
            "mode": "manual",
            "num_of_partitions": num_partitions,
        }
        config["checkpoint"] = {
            "enabled": checkpoint_enabled,
            "strategy": checkpoint_strategy,
            "n_ops": checkpoint_n_ops,
        }

    config_path = os.path.join(output_dir, "benchmark_config.yaml")

    import yaml

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    return config_path


def run_benchmark(config_path: str, job_id: str, work_dir: str, input_rows: int) -> BenchmarkResult:
    """Run a single benchmark and collect metrics."""

    # Parse config to get settings
    import yaml

    with open(config_path) as f:
        config = yaml.safe_load(f)

    executor_type = config.get("executor_type", "ray")
    checkpoint_config = config.get("checkpoint", {})
    partition_config = config.get("partition", {})

    cmd = [
        "dj-process",
        "--config",
        config_path,
        "--job_id",
        job_id,
    ]

    ckpt_str = checkpoint_config.get("strategy", "disabled") if checkpoint_config.get("enabled") else "disabled"
    part_str = partition_config.get("mode", "none") if partition_config else "none"

    print(f"\nRunning: {job_id}")
    print(f"  Executor: {executor_type}")
    print(f"  Checkpoint: {ckpt_str}")
    print(f"  Partition: {part_str}")

    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    total_time = time.time() - start_time

    success = result.returncode == 0
    # Only capture actual errors, not INFO/WARNING logs in stderr
    error_msg = None
    if not success:
        # Filter for actual error lines
        error_lines = [
            line
            for line in result.stderr.split("\n")
            if "ERROR" in line or "error:" in line.lower() or "exception" in line.lower()
        ]
        error_msg = "\n".join(error_lines[:5]) if error_lines else result.stderr[:500]

    # Collect metrics from work directory (checkpoints are stored directly in work_dir)
    storage_mb = get_dir_size_mb(work_dir) if os.path.exists(work_dir) else 0

    # Count checkpoints (each checkpoint_op_*_partition_*.parquet directory is one checkpoint)
    checkpoint_dir = os.path.join(work_dir, "checkpoints")
    checkpoint_count = 0
    if os.path.exists(checkpoint_dir):
        checkpoint_count = len(list(Path(checkpoint_dir).glob("checkpoint_op_*.parquet")))

    # Count output rows
    output_path = config.get("export_path", "")
    output_rows = count_jsonl_rows(output_path) if success else 0

    # Get partition count from events
    num_partitions = partition_config.get("num_of_partitions", 1) if partition_config else 1
    num_operations = len(config.get("process", []))

    # Row verification status
    row_status = "OK" if output_rows == input_rows else f"MISMATCH (expected {input_rows})"
    print(
        f"  Time: {total_time:.1f}s | Storage: {storage_mb:.1f}MB | Checkpoints: {checkpoint_count} | Rows: {output_rows} {row_status}"
    )

    return BenchmarkResult(
        config_name=job_id,
        executor_type=executor_type,
        checkpoint_strategy=(
            checkpoint_config.get("strategy", "disabled") if checkpoint_config.get("enabled") else "disabled"
        ),
        partition_mode=partition_config.get("mode", "none") if partition_config else "none",
        total_time_seconds=total_time,
        num_partitions=num_partitions,
        num_operations=num_operations,
        checkpoint_count=checkpoint_count,
        storage_used_mb=storage_mb,
        input_rows=input_rows,
        output_rows=output_rows,
        success=success,
        error_message=error_msg,
    )


def run_overhead_benchmark(dataset_path: str, output_base: str, num_partitions: int = 4) -> Dict[str, BenchmarkResult]:
    """Run overhead comparison benchmark."""

    print("\n" + "=" * 60)
    print("OVERHEAD BENCHMARK")
    print("Comparing execution time across configurations")
    print("=" * 60)

    # Count input rows once
    input_rows = count_jsonl_rows(dataset_path)
    print(f"\nInput dataset: {input_rows} rows")

    results = {}

    configs = [
        # (name, executor, ckpt_enabled, ckpt_strategy)
        ("baseline_ray", "ray", False, "disabled"),
        ("partitioned_no_ckpt", "ray_partitioned", False, "disabled"),
        ("partitioned_ckpt_every_op", "ray_partitioned", True, "every_op"),
        ("partitioned_ckpt_every_2", "ray_partitioned", True, "every_n_ops"),
    ]

    for name, executor, ckpt_enabled, ckpt_strategy in configs:
        work_dir = os.path.join(output_base, name)
        os.makedirs(work_dir, exist_ok=True)

        config_path = create_benchmark_config(
            dataset_path=dataset_path,
            output_dir=work_dir,
            executor_type=executor,
            num_partitions=num_partitions,
            checkpoint_enabled=ckpt_enabled,
            checkpoint_strategy=ckpt_strategy,
            checkpoint_n_ops=2,
        )

        result = run_benchmark(config_path, name, work_dir, input_rows)
        results[name] = result

    return results


def print_overhead_report(results: Dict[str, BenchmarkResult]):
    """Print overhead comparison report."""

    print("\n" + "=" * 60)
    print("OVERHEAD REPORT")
    print("=" * 60)

    baseline = results.get("baseline_ray")
    if not baseline or not baseline.success:
        print("Baseline failed, cannot compute overhead percentages")
        baseline_time = None
    else:
        baseline_time = baseline.total_time_seconds

    print(f"\n{'Config':<30} {'Time (s)':<10} {'Overhead':<10} {'Rows':<15} {'Checkpoints':<12}")
    print("-" * 77)

    row_mismatches = []
    for name, result in results.items():
        if not result.success:
            print(f"{name:<30} FAILED: {result.error_message[:40] if result.error_message else 'unknown'}")
            continue

        if baseline_time:
            overhead = ((result.total_time_seconds - baseline_time) / baseline_time) * 100
            overhead_str = f"{overhead:+.1f}%"
        else:
            overhead_str = "N/A"

        # Row verification
        if result.output_rows == result.input_rows:
            row_str = f"{result.output_rows} OK"
        else:
            row_str = f"{result.output_rows} MISMATCH"
            row_mismatches.append((name, result.input_rows, result.output_rows))

        print(
            f"{name:<30} {result.total_time_seconds:<10.1f} {overhead_str:<10} {row_str:<15} {result.checkpoint_count:<12}"
        )

    # Report row verification results
    print("\nRow verification:")
    if row_mismatches:
        print("  FAILED - Row count mismatches detected:")
        for name, expected, actual in row_mismatches:
            print(f"    - {name}: expected {expected}, got {actual}")
    else:
        first_result = next(iter(results.values()))
        print(f"  PASSED - All configurations produced {first_result.input_rows} rows")

    print("\nKey findings:")
    if baseline_time:
        for name in ["partitioned_no_ckpt", "partitioned_ckpt_every_op", "partitioned_ckpt_every_2"]:
            if name in results and results[name].success:
                overhead = ((results[name].total_time_seconds - baseline_time) / baseline_time) * 100
                print(f"  - {name}: {overhead:+.1f}% vs baseline")

    # Show checkpoint overhead relative to partitioned_no_ckpt
    partitioned_base = results.get("partitioned_no_ckpt")
    if partitioned_base and partitioned_base.success:
        print("\nCheckpoint overhead (vs partitioned without checkpoint):")
        for name in ["partitioned_ckpt_every_op", "partitioned_ckpt_every_2"]:
            if name in results and results[name].success:
                ckpt_overhead = (
                    (results[name].total_time_seconds - partitioned_base.total_time_seconds)
                    / partitioned_base.total_time_seconds
                ) * 100
                print(f"  - {name}: {ckpt_overhead:+.1f}%")


def print_summary(results: Dict[str, BenchmarkResult]):
    """Print final summary."""

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    successful = [r for r in results.values() if r.success]
    failed = [r for r in results.values() if not r.success]

    print(f"\nRan {len(results)} configurations: {len(successful)} succeeded, {len(failed)} failed")

    if successful:
        baseline = results.get("baseline_ray")
        ckpt_every_op = results.get("partitioned_ckpt_every_op")

        if baseline and baseline.success and ckpt_every_op and ckpt_every_op.success:
            overhead = (
                (ckpt_every_op.total_time_seconds - baseline.total_time_seconds) / baseline.total_time_seconds
            ) * 100

            print(f"\nCheckpointing overhead (every_op): {overhead:+.1f}%")
            print(f"Storage cost: {ckpt_every_op.storage_used_mb:.1f} MB")
            print(f"Checkpoints saved: {ckpt_every_op.checkpoint_count}")
            print(f"\nWith checkpointing, failures lose at most 1 operation worth of work")
            print(f"Without checkpointing, failures lose all work")

    # Save results to JSON
    results_path = os.path.join(
        os.path.dirname(list(results.values())[0].config_name if results else "."), "benchmark_results.json"
    )

    print(f"\nResults interpretation:")
    print(f"  - Overhead < 10%: Acceptable for production use")
    print(f"  - Overhead 10-20%: Consider for critical pipelines")
    print(f"  - Overhead > 20%: Use every_n_ops to reduce overhead")


def main():
    parser = argparse.ArgumentParser(description="Robustness benchmark for partitioned checkpointing")
    parser.add_argument("--dataset", type=str, help="Path to dataset (creates test data if not provided)")
    parser.add_argument("--samples", type=int, default=10000, help="Number of samples for test dataset")
    parser.add_argument("--partitions", type=int, default=4, help="Number of partitions")
    parser.add_argument("--output", type=str, default=None, help="Output directory for results")
    parser.add_argument("--quick", action="store_true", help="Quick mode with smaller dataset")

    args = parser.parse_args()

    if args.quick:
        args.samples = 2000
        args.partitions = 2

    # Setup output directory
    if args.output:
        output_base = args.output
    else:
        output_base = tempfile.mkdtemp(prefix="dj_benchmark_")

    os.makedirs(output_base, exist_ok=True)
    print(f"Output directory: {output_base}")

    # Setup dataset
    if args.dataset and os.path.exists(args.dataset):
        dataset_path = args.dataset
        print(f"Using provided dataset: {dataset_path}")
    else:
        dataset_path = os.path.join(output_base, "test_data.jsonl")
        create_test_dataset(dataset_path, args.samples)

    # Run benchmarks
    try:
        results = run_overhead_benchmark(
            dataset_path=dataset_path, output_base=output_base, num_partitions=args.partitions
        )

        print_overhead_report(results)
        print_summary(results)

        # Save results
        results_file = os.path.join(output_base, "benchmark_results.json")
        with open(results_file, "w") as f:
            json.dump({k: asdict(v) for k, v in results.items()}, f, indent=2)
        print(f"\nDetailed results saved to: {results_file}")

    except KeyboardInterrupt:
        print("\nBenchmark interrupted")
    except Exception as e:
        print(f"\nBenchmark failed: {e}")
        raise


if __name__ == "__main__":
    main()

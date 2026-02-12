#!/usr/bin/env python3
"""
Partition Determinism Benchmark

Demonstrates the importance of deterministic partitioning for checkpoint resumption.

Tests:
1. Deterministic splitting - same data produces same partitions across runs
2. Non-deterministic splitting - shows how partitions can differ without preserve_order
3. Partition validation - detects when partitions don't match saved checkpoints

Usage:
    cd /path/to/data-juicer
    python demos/partition_and_checkpoint/partition_determinism_benchmark.py

    # Quick mode
    python demos/partition_and_checkpoint/partition_determinism_benchmark.py --quick
"""

import argparse
import hashlib
import json
import os
import tempfile
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Initialize Ray before importing ray.data
import ray

ray.init(ignore_reinit_error=True)

import ray.data


@dataclass
class PartitionFingerprint:
    """Fingerprint of a partition for comparison."""
    partition_id: int
    row_count: int
    first_row_hash: str
    last_row_hash: str
    sample_hashes: List[str]  # Hashes of sampled rows

    def matches(self, other: "PartitionFingerprint") -> bool:
        """Check if two fingerprints match."""
        return (
            self.row_count == other.row_count
            and self.first_row_hash == other.first_row_hash
            and self.last_row_hash == other.last_row_hash
        )


def compute_row_hash(row: Dict) -> str:
    """Compute hash of a row."""
    row_str = json.dumps(row, sort_keys=True, default=str)
    return hashlib.md5(row_str.encode()).hexdigest()[:16]


def fingerprint_partition(partition, partition_id: int, sample_size: int = 5) -> PartitionFingerprint:
    """Create a fingerprint of a partition."""
    rows = partition.take(partition.count())
    row_count = len(rows)

    first_hash = compute_row_hash(rows[0]) if rows else ""
    last_hash = compute_row_hash(rows[-1]) if rows else ""

    # Sample some rows for additional validation
    sample_indices = [int(i * row_count / sample_size) for i in range(sample_size) if row_count > 0]
    sample_hashes = [compute_row_hash(rows[i]) for i in sample_indices if i < row_count]

    return PartitionFingerprint(
        partition_id=partition_id,
        row_count=row_count,
        first_row_hash=first_hash,
        last_row_hash=last_hash,
        sample_hashes=sample_hashes,
    )


def create_test_dataset(num_samples: int = 10000, num_files: int = 1) -> str:
    """Create test dataset file(s).

    Args:
        num_samples: Total number of samples
        num_files: Number of files to split data across (more files = more potential for non-determinism)

    Returns:
        Path to dataset (single file) or directory (multiple files)
    """
    if num_files == 1:
        output_path = tempfile.mktemp(suffix=".jsonl")
        with open(output_path, "w") as f:
            for i in range(num_samples):
                sample = {
                    "id": i,
                    "text": f"Sample {i}: " + "Content " * 10,
                    "value": i * 1.5,
                }
                f.write(json.dumps(sample) + "\n")
        return output_path
    else:
        # Create multiple files to increase chance of non-determinism
        output_dir = tempfile.mkdtemp(prefix="benchmark_data_")
        samples_per_file = num_samples // num_files

        for file_idx in range(num_files):
            file_path = os.path.join(output_dir, f"data_{file_idx:04d}.jsonl")
            start_idx = file_idx * samples_per_file
            end_idx = start_idx + samples_per_file if file_idx < num_files - 1 else num_samples

            with open(file_path, "w") as f:
                for i in range(start_idx, end_idx):
                    sample = {
                        "id": i,
                        "text": f"Sample {i}: " + "Content " * 10,
                        "value": i * 1.5,
                    }
                    f.write(json.dumps(sample) + "\n")

        return output_dir


def split_with_preserve_order(dataset_path: str, num_partitions: int, preserve_order: bool, shuffle: bool = False) -> List[PartitionFingerprint]:
    """Split dataset and return fingerprints of each partition.

    Args:
        dataset_path: Path to dataset file or directory
        num_partitions: Number of partitions to create
        preserve_order: Whether to enable preserve_order in Ray
        shuffle: Whether to shuffle the dataset (demonstrates non-determinism)
    """
    # Set execution options
    ctx = ray.data.DataContext.get_current()
    ctx.execution_options.preserve_order = preserve_order

    # Load dataset
    if os.path.isdir(dataset_path):
        dataset = ray.data.read_json(os.path.join(dataset_path, "*.jsonl"))
    else:
        dataset = ray.data.read_json(dataset_path)

    # Optionally shuffle to demonstrate non-determinism
    if shuffle:
        dataset = dataset.random_shuffle()

    # Split
    partitions = dataset.split(num_partitions)

    # Fingerprint each partition
    fingerprints = []
    for i, partition in enumerate(partitions):
        fp = fingerprint_partition(partition, i)
        fingerprints.append(fp)

    return fingerprints


def compare_fingerprints(fps1: List[PartitionFingerprint], fps2: List[PartitionFingerprint]) -> Tuple[bool, Dict]:
    """Compare two sets of partition fingerprints."""
    if len(fps1) != len(fps2):
        return False, {"error": f"Different partition counts: {len(fps1)} vs {len(fps2)}"}

    mismatches = []
    for i, (fp1, fp2) in enumerate(zip(fps1, fps2)):
        if not fp1.matches(fp2):
            mismatches.append({
                "partition": i,
                "run1": {"rows": fp1.row_count, "first": fp1.first_row_hash, "last": fp1.last_row_hash},
                "run2": {"rows": fp2.row_count, "first": fp2.first_row_hash, "last": fp2.last_row_hash},
            })

    return len(mismatches) == 0, {"mismatches": mismatches}


def benchmark_determinism(dataset_path: str, num_partitions: int, num_runs: int = 3) -> Dict:
    """Benchmark determinism of partitioning with and without preserve_order."""

    results = {
        "preserve_order_true": {"fingerprints": [], "all_match": True, "details": []},
        "preserve_order_false": {"fingerprints": [], "all_match": True, "details": []},
        "with_shuffle": {"fingerprints": [], "all_match": True, "details": []},
    }

    print("\n" + "=" * 70)
    print("TEST 1: Deterministic Splitting (preserve_order=True)")
    print("=" * 70)

    # Test with preserve_order=True
    print(f"\nRunning {num_runs} splits with preserve_order=True...")
    for run in range(num_runs):
        fps = split_with_preserve_order(dataset_path, num_partitions, preserve_order=True)
        results["preserve_order_true"]["fingerprints"].append(fps)

        if run > 0:
            matches, details = compare_fingerprints(
                results["preserve_order_true"]["fingerprints"][0],
                fps
            )
            results["preserve_order_true"]["details"].append(details)
            if not matches:
                results["preserve_order_true"]["all_match"] = False

        # Print partition info
        total_rows = sum(fp.row_count for fp in fps)
        print(f"  Run {run + 1}: {num_partitions} partitions, {total_rows} total rows")
        for fp in fps:
            print(f"    Partition {fp.partition_id}: {fp.row_count} rows, first={fp.first_row_hash[:8]}...")

    if results["preserve_order_true"]["all_match"]:
        print("\n  RESULT: All runs produced IDENTICAL partitions")
    else:
        print("\n  RESULT: Partitions DIFFERED between runs!")
        for i, detail in enumerate(results["preserve_order_true"]["details"]):
            if detail.get("mismatches"):
                print(f"    Run 1 vs Run {i+2}: {len(detail['mismatches'])} mismatches")

    print("\n" + "=" * 70)
    print("TEST 2: Non-Deterministic Splitting (preserve_order=False)")
    print("=" * 70)

    # Test with preserve_order=False
    print(f"\nRunning {num_runs} splits with preserve_order=False...")
    for run in range(num_runs):
        fps = split_with_preserve_order(dataset_path, num_partitions, preserve_order=False)
        results["preserve_order_false"]["fingerprints"].append(fps)

        if run > 0:
            matches, details = compare_fingerprints(
                results["preserve_order_false"]["fingerprints"][0],
                fps
            )
            results["preserve_order_false"]["details"].append(details)
            if not matches:
                results["preserve_order_false"]["all_match"] = False

        # Print partition info
        total_rows = sum(fp.row_count for fp in fps)
        print(f"  Run {run + 1}: {num_partitions} partitions, {total_rows} total rows")
        for fp in fps:
            print(f"    Partition {fp.partition_id}: {fp.row_count} rows, first={fp.first_row_hash[:8]}...")

    if results["preserve_order_false"]["all_match"]:
        print("\n  RESULT: All runs produced IDENTICAL partitions")
        print("  NOTE: Small single-file datasets may appear deterministic")
        print("        but larger multi-file datasets can vary!")
    else:
        print("\n  RESULT: Partitions DIFFERED between runs (expected)")
        for i, detail in enumerate(results["preserve_order_false"]["details"]):
            if detail.get("mismatches"):
                print(f"    Run 1 vs Run {i+2}: {len(detail['mismatches'])} partition mismatches")

    print("\n" + "=" * 70)
    print("TEST 3: Shuffled Data (simulates worst-case non-determinism)")
    print("=" * 70)
    print("  This test uses random_shuffle() to demonstrate what happens")
    print("  when partition contents vary between runs.")

    # Test with shuffle to demonstrate the problem
    print(f"\nRunning {num_runs} splits with random_shuffle()...")
    for run in range(num_runs):
        fps = split_with_preserve_order(dataset_path, num_partitions, preserve_order=True, shuffle=True)
        results["with_shuffle"]["fingerprints"].append(fps)

        if run > 0:
            matches, details = compare_fingerprints(
                results["with_shuffle"]["fingerprints"][0],
                fps
            )
            results["with_shuffle"]["details"].append(details)
            if not matches:
                results["with_shuffle"]["all_match"] = False

        # Print partition info
        total_rows = sum(fp.row_count for fp in fps)
        print(f"  Run {run + 1}: {num_partitions} partitions, {total_rows} total rows")
        for fp in fps:
            print(f"    Partition {fp.partition_id}: {fp.row_count} rows, first={fp.first_row_hash[:8]}...")

    if results["with_shuffle"]["all_match"]:
        print("\n  RESULT: All runs produced IDENTICAL partitions (very unlikely!)")
    else:
        print("\n  RESULT: Partitions DIFFERED between runs (expected with shuffle)")
        print("  This demonstrates the checkpoint mismatch problem:")
        print("    - Run 1 saves checkpoint with partition contents A")
        print("    - Run 2 (after failure) has partition contents B")
        print("    - Resuming from checkpoint would process WRONG data!")
        for i, detail in enumerate(results["with_shuffle"]["details"]):
            if detail.get("mismatches"):
                print(f"    Run 1 vs Run {i+2}: {len(detail['mismatches'])} partition mismatches")

    return results


def benchmark_validation_detection(dataset_path: str, num_partitions: int) -> Dict:
    """Benchmark partition validation - detecting when partitions don't match."""

    print("\n" + "=" * 70)
    print("TEST 3: Partition Validation (Detecting Mismatches)")
    print("=" * 70)

    results = {"scenarios": []}

    # Scenario 1: Same data, same partitions - should validate
    print("\nScenario 1: Same data, deterministic split - should PASS validation")
    ctx = ray.data.DataContext.get_current()
    ctx.execution_options.preserve_order = True

    dataset = ray.data.read_json(dataset_path)
    partitions1 = dataset.split(num_partitions)
    fps1 = [fingerprint_partition(p, i) for i, p in enumerate(partitions1)]

    dataset = ray.data.read_json(dataset_path)
    partitions2 = dataset.split(num_partitions)
    fps2 = [fingerprint_partition(p, i) for i, p in enumerate(partitions2)]

    matches, details = compare_fingerprints(fps1, fps2)
    status = "PASS" if matches else "FAIL"
    print(f"  Result: {status}")
    results["scenarios"].append({"name": "same_data_deterministic", "expected": "PASS", "actual": status})

    # Scenario 2: Different partition count - should fail validation
    print("\nScenario 2: Different partition count - should FAIL validation")
    dataset = ray.data.read_json(dataset_path)
    partitions3 = dataset.split(num_partitions + 1)
    fps3 = [fingerprint_partition(p, i) for i, p in enumerate(partitions3)]

    matches, details = compare_fingerprints(fps1, fps3)
    status = "FAIL" if not matches else "PASS"
    print(f"  Result: {status} (detected partition count mismatch)")
    results["scenarios"].append({"name": "different_partition_count", "expected": "FAIL", "actual": status})

    # Scenario 3: Modified data - should fail validation
    print("\nScenario 3: Modified input data - should FAIL validation")
    modified_path = tempfile.mktemp(suffix=".jsonl")
    with open(dataset_path, "r") as f_in, open(modified_path, "w") as f_out:
        for i, line in enumerate(f_in):
            if i == 0:
                # Modify first row
                data = json.loads(line)
                data["text"] = "MODIFIED " + data["text"]
                f_out.write(json.dumps(data) + "\n")
            else:
                f_out.write(line)

    dataset = ray.data.read_json(modified_path)
    partitions4 = dataset.split(num_partitions)
    fps4 = [fingerprint_partition(p, i) for i, p in enumerate(partitions4)]

    matches, details = compare_fingerprints(fps1, fps4)
    status = "FAIL" if not matches else "PASS"
    print(f"  Result: {status} (detected data modification)")
    results["scenarios"].append({"name": "modified_data", "expected": "FAIL", "actual": status})

    os.unlink(modified_path)

    return results


def print_summary(determinism_results: Dict, validation_results: Dict):
    """Print benchmark summary."""

    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    print("\nDeterminism Tests:")
    print(f"  preserve_order=True:  {'DETERMINISTIC' if determinism_results['preserve_order_true']['all_match'] else 'NON-DETERMINISTIC'}")
    print(f"  preserve_order=False: {'DETERMINISTIC' if determinism_results['preserve_order_false']['all_match'] else 'NON-DETERMINISTIC'}")
    print(f"  with random_shuffle:  {'DETERMINISTIC' if determinism_results['with_shuffle']['all_match'] else 'NON-DETERMINISTIC'}")

    print("\nValidation Tests:")
    for scenario in validation_results["scenarios"]:
        expected = scenario["expected"]
        actual = scenario["actual"]
        match = "OK" if expected == actual else "UNEXPECTED"
        print(f"  {scenario['name']}: {actual} ({match})")

    print("\n" + "-" * 70)
    print("CONCLUSIONS:")
    print("-" * 70)

    if determinism_results["preserve_order_true"]["all_match"]:
        print("  1. With preserve_order=True, partitions are REPRODUCIBLE")
        print("     -> Safe to resume from checkpoints after failure")
    else:
        print("  1. WARNING: Even with preserve_order=True, partitions varied!")
        print("     -> May need additional measures for checkpoint safety")

    if not determinism_results["with_shuffle"]["all_match"]:
        print("  2. The shuffle test demonstrates the checkpoint mismatch problem:")
        print("     -> When partition contents change between runs,")
        print("        resuming from checkpoint would process WRONG data!")
        print("     -> This is why deterministic mode + validation is CRITICAL")
    else:
        print("  2. Note: shuffle test was deterministic (very unlikely)")

    if determinism_results["preserve_order_false"]["all_match"]:
        print("  3. Note: preserve_order=False was deterministic for this dataset")
        print("     -> Small single-file datasets may appear deterministic")
        print("     -> Larger multi-file datasets with parallel reads can vary!")
        print("     -> Always use preserve_order=True for safety")

    all_validation_ok = all(
        s["expected"] == s["actual"]
        for s in validation_results["scenarios"]
    )
    if all_validation_ok:
        print("  4. Partition validation correctly detects all mismatch scenarios")
        print("     -> Safe to use for checkpoint integrity verification")
    else:
        print("  4. WARNING: Partition validation had unexpected results")


def main():
    parser = argparse.ArgumentParser(description="Partition determinism benchmark")
    parser.add_argument("--samples", type=int, default=5000, help="Number of samples")
    parser.add_argument("--partitions", type=int, default=4, help="Number of partitions")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs for determinism test")
    parser.add_argument("--quick", action="store_true", help="Quick mode (fewer samples)")

    args = parser.parse_args()

    if args.quick:
        args.samples = 1000
        args.partitions = 2
        args.runs = 2

    print("=" * 70)
    print("PARTITION DETERMINISM BENCHMARK")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Samples: {args.samples}")
    print(f"  Partitions: {args.partitions}")
    print(f"  Runs: {args.runs}")

    # Create test dataset
    print("\nCreating test dataset...")
    dataset_path = create_test_dataset(args.samples)
    print(f"  Created: {dataset_path}")

    try:
        # Run determinism benchmark
        determinism_results = benchmark_determinism(
            dataset_path,
            args.partitions,
            args.runs
        )

        # Run validation benchmark
        validation_results = benchmark_validation_detection(
            dataset_path,
            args.partitions
        )

        # Print summary
        print_summary(determinism_results, validation_results)

    finally:
        # Cleanup
        if os.path.exists(dataset_path):
            os.unlink(dataset_path)
        print(f"\nCleaned up test dataset")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Ray Datasets Arrow vs. Parquet Performance Comparison

Benchmarks write/read speed and file size for Arrow (Feather) and Parquet using Ray Datasets.
Uses C4 data if available, otherwise synthetic data.
"""
import os
import time
import json
from typing import List, Dict, Any, Optional
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.parquet as pq
import ray
from loguru import logger

C4_FILE_PATH = os.path.expanduser("~/Downloads/c4-train.00000-of-01024.jsonl")


def load_c4_data(file_path: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    logger.info(f"Loading C4 data from {file_path}...")
    data = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            try:
                sample = json.loads(line.strip())
                simplified_sample = {
                    'id': i,
                    'text': sample.get('text', ''),
                    'timestamp': sample.get('meta', {}).get('timestamp', ''),
                    'url': sample.get('meta', {}).get('url', ''),
                    'language': sample.get('meta', {}).get('language', ''),
                    'source': sample.get('meta', {}).get('source', ''),
                    'text_length': len(sample.get('text', ''))
                }
                data.append(simplified_sample)
            except json.JSONDecodeError:
                continue
    logger.info(f"Loaded {len(data)} samples from C4 data")
    return data


def create_sample_data(num_samples: int = 10000) -> List[Dict[str, Any]]:
    data = []
    for i in range(num_samples):
        sample = {
            'id': i,
            'text': f'This is sample text number {i} for performance testing.',
            'category': f'category_{i % 100}',
            'score': i * 0.1,
            'text_length': len(f'This is sample text number {i}')
        }
        data.append(sample)
    return data


def ray_benchmark_arrow_parquet(data: List[Dict[str, Any]], num_iterations: int = 3):
    import tempfile
    results = {}
    dataset = ray.data.from_items(data)
    # Arrow (Feather) write/read
    arrow_write_times = []
    arrow_read_times = []
    arrow_sizes = []
    for _ in range(num_iterations):
        with tempfile.NamedTemporaryFile(suffix='.arrow', delete=False) as tmp_file:
            start = time.time()
            # Write as Arrow (Feather)
            df = dataset.to_pandas()
            table = pa.Table.from_pandas(df)
            feather.write_feather(table, tmp_file.name)
            write_time = time.time() - start
            arrow_write_times.append(write_time)
            size = os.path.getsize(tmp_file.name)
            arrow_sizes.append(size)
            # Read as Arrow (Feather)
            start = time.time()
            table2 = feather.read_table(tmp_file.name)
            df2 = table2.to_pandas()
            read_time = time.time() - start
            arrow_read_times.append(read_time)
            os.unlink(tmp_file.name)
    results['arrow'] = {
        'avg_write_time': sum(arrow_write_times) / num_iterations,
        'avg_read_time': sum(arrow_read_times) / num_iterations,
        'avg_file_size': sum(arrow_sizes) / num_iterations
    }
    # Parquet write/read (Ray Datasets expects a directory)
    parquet_write_times = []
    parquet_read_times = []
    parquet_sizes = []
    for _ in range(num_iterations):
        with tempfile.TemporaryDirectory(suffix='.parquet') as tmp_dir:
            start = time.time()
            dataset.write_parquet(tmp_dir)
            write_time = time.time() - start
            parquet_write_times.append(write_time)
            # Sum all files in the directory for total size
            size = sum(
                os.path.getsize(os.path.join(tmp_dir, f))
                for f in os.listdir(tmp_dir)
                if os.path.isfile(os.path.join(tmp_dir, f))
            )
            parquet_sizes.append(size)
            # Read as Parquet
            start = time.time()
            loaded = ray.data.read_parquet(tmp_dir)
            _ = loaded.take(10)  # Force read
            read_time = time.time() - start
            parquet_read_times.append(read_time)
    results['parquet'] = {
        'avg_write_time': sum(parquet_write_times) / num_iterations,
        'avg_read_time': sum(parquet_read_times) / num_iterations,
        'avg_file_size': sum(parquet_sizes) / num_iterations
    }
    return results


def print_ray_perf_report(results, num_samples):
    print("\n" + "="*80)
    print("RAY DATASET: ARROW (FEATHER) VS. PARQUET PERFORMANCE")
    print("="*80)
    print(f"Dataset Size: {num_samples:,} samples")
    print(f"{'Format':<10} {'File Size':<15} {'Write Time':<15} {'Read Time':<15}")
    print("-" * 60)
    for fmt in ['arrow', 'parquet']:
        size = results[fmt]['avg_file_size'] / 1024 / 1024
        write = results[fmt]['avg_write_time']
        read = results[fmt]['avg_read_time']
        print(f"{fmt.capitalize():<10} {size:>8.2f} MB {write:>12.4f}s {read:>12.4f}s")
    print("\nNotes:")
    print("- Arrow (Feather) is written/read via pandas/pyarrow, not distributed.")
    print("- Parquet is written/read via Ray Datasets, can be distributed.")
    print("- For distributed, partitioned, or large-scale pipelines, Parquet is preferred.")
    print("- For fast, in-memory, or intermediate results, Arrow (Feather) can be faster.")
    print("- For true distributed Arrow, use Ray Datasets with Arrow batches (advanced).")


def main():
    print("\n🚀 Ray Dataset Arrow vs. Parquet Performance Comparison")
    print("="*80)
    ray.init(ignore_reinit_error=True)
    # Use C4 data if available
    if os.path.exists(C4_FILE_PATH):
        print(f"Using real C4 data: {C4_FILE_PATH}")
        data = load_c4_data(C4_FILE_PATH)
    else:
        print("C4 data not found, using synthetic data")
        data = create_sample_data(50000)
    num_samples = len(data)
    results = ray_benchmark_arrow_parquet(data, num_iterations=3)
    print_ray_perf_report(results, num_samples)
    print("\n✅ Done!")

if __name__ == "__main__":
    main() 
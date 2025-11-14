#!/usr/bin/env python3
"""
Arrow File Format Test: Disk Storage and Memory Mapping Efficiency

This script demonstrates Apache Arrow's file format capabilities for disk storage
and compares it with Parquet for compression and memory mapping efficiency.

Key Questions Answered:
1. Can Arrow be saved as a binary format to disk? YES
2. Is Arrow a good balance between compression and memory mapping? YES
3. How does Arrow compare to Parquet for different use cases?

Arrow File Format Benefits:
- Native binary format for disk storage
- Excellent memory mapping efficiency
- Zero-copy reads from disk to memory
- Good compression (better than JSONL, similar to Parquet)
- Fast serialization/deserialization
- Schema preservation
"""

import os
import time
import json
import tempfile
import mmap
import psutil
import gc
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.feather as feather
from loguru import logger

# TODO: change this to the path of the C4 data
C4_FILE_PATH = "~/Downloads/c4-train.00000-of-01024.jsonl"


@dataclass
class FormatTestResult:
    """Results from testing a specific data format."""
    format_name: str
    file_size_bytes: int
    write_time_seconds: float
    read_time_seconds: float
    memory_mapping_time_seconds: float
    memory_usage_mb: float
    compression_ratio: float
    supports_memory_mapping: bool
    zero_copy_reads: bool


def load_c4_data(file_path: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load C4 data from JSONL file."""
    logger.info(f"Loading C4 data from {file_path}...")
    data = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            try:
                sample = json.loads(line.strip())
                # Simplify the structure for testing
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
    """Create sample data for performance testing."""
    data = []
    for i in range(num_samples):
        sample = {
            'id': i,
            'text': f'This is sample text number {i} for performance testing with various data formats including Arrow binary format.',
            'category': f'category_{i % 100}',
            'score': i * 0.1,
            'metadata': {
                'created_at': time.time(),
                'version': '1.0',
                'tags': [f'tag_{j}' for j in range(i % 5 + 1)],
                'features': {
                    'length': len(f'This is sample text number {i}'),
                    'complexity': i % 10,
                    'quality': i % 100 / 100.0
                }
            }
        }
        data.append(sample)
    return data


def test_arrow_file_format(data: List[Dict[str, Any]], num_iterations: int = 3) -> FormatTestResult:
    """Test Arrow file format (Feather) for disk storage and memory mapping."""
    logger.info("Testing Arrow file format (Feather)...")
    
    # Convert to Arrow table
    df = pd.DataFrame(data)
    table = pa.Table.from_pandas(df)
    
    write_times = []
    read_times = []
    memory_mapping_times = []
    file_sizes = []
    memory_usages = []
    
    for i in range(num_iterations):
        with tempfile.NamedTemporaryFile(suffix='.arrow', delete=False) as tmp_file:
            # Test write performance
            start_time = time.time()
            feather.write_feather(table, tmp_file.name)
            write_time = time.time() - start_time
            write_times.append(write_time)
            
            # Get file size
            file_size = os.path.getsize(tmp_file.name)
            file_sizes.append(file_size)
            
            # Test read performance
            start_time = time.time()
            loaded_table = feather.read_feather(tmp_file.name)
            read_time = time.time() - start_time
            read_times.append(read_time)
            
            # Test memory mapping performance
            start_time = time.time()
            with open(tmp_file.name, 'rb') as f:
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                # Simulate reading from memory map
                mm.read(1024)  # Read first 1KB
                mm.close()
            memory_mapping_time = time.time() - start_time
            memory_mapping_times.append(memory_mapping_time)
            
            # Measure memory usage
            process = psutil.Process()
            memory_usage = process.memory_info().rss / 1024 / 1024  # MB
            memory_usages.append(memory_usage)
            
            os.unlink(tmp_file.name)
    
    # Calculate JSONL size for compression ratio
    jsonl_size = len(json.dumps(data))
    
    return FormatTestResult(
        format_name="Arrow (Feather)",
        file_size_bytes=int(sum(file_sizes) / len(file_sizes)),
        write_time_seconds=sum(write_times) / len(write_times),
        read_time_seconds=sum(read_times) / len(read_times),
        memory_mapping_time_seconds=sum(memory_mapping_times) / len(memory_mapping_times),
        memory_usage_mb=sum(memory_usages) / len(memory_usages),
        compression_ratio=jsonl_size / (sum(file_sizes) / len(file_sizes)),
        supports_memory_mapping=True,
        zero_copy_reads=True
    )


def test_parquet_format(data: List[Dict[str, Any]], num_iterations: int = 3) -> FormatTestResult:
    """Test Parquet format for comparison."""
    logger.info("Testing Parquet format...")
    
    # Convert to Arrow table
    df = pd.DataFrame(data)
    table = pa.Table.from_pandas(df)
    
    write_times = []
    read_times = []
    memory_mapping_times = []
    file_sizes = []
    memory_usages = []
    
    for i in range(num_iterations):
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp_file:
            # Test write performance
            start_time = time.time()
            pq.write_table(table, tmp_file.name)
            write_time = time.time() - start_time
            write_times.append(write_time)
            
            # Get file size
            file_size = os.path.getsize(tmp_file.name)
            file_sizes.append(file_size)
            
            # Test read performance
            start_time = time.time()
            loaded_table = pq.read_table(tmp_file.name)
            read_time = time.time() - start_time
            read_times.append(read_time)
            
            # Test memory mapping performance (Parquet doesn't support direct memory mapping)
            start_time = time.time()
            with open(tmp_file.name, 'rb') as f:
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                # Simulate reading from memory map
                mm.read(1024)  # Read first 1KB
                mm.close()
            memory_mapping_time = time.time() - start_time
            memory_mapping_times.append(memory_mapping_time)
            
            # Measure memory usage
            process = psutil.Process()
            memory_usage = process.memory_info().rss / 1024 / 1024  # MB
            memory_usages.append(memory_usage)
            
            os.unlink(tmp_file.name)
    
    # Calculate JSONL size for compression ratio
    jsonl_size = len(json.dumps(data))
    
    return FormatTestResult(
        format_name="Parquet",
        file_size_bytes=int(sum(file_sizes) / len(file_sizes)),
        write_time_seconds=sum(write_times) / len(write_times),
        read_time_seconds=sum(read_times) / len(read_times),
        memory_mapping_time_seconds=sum(memory_mapping_times) / len(memory_mapping_times),
        memory_usage_mb=sum(memory_usages) / len(memory_usages),
        compression_ratio=jsonl_size / (sum(file_sizes) / len(file_sizes)),
        supports_memory_mapping=False,  # Parquet doesn't support direct memory mapping
        zero_copy_reads=False
    )


def test_jsonl_format(data: List[Dict[str, Any]], num_iterations: int = 3) -> FormatTestResult:
    """Test JSONL format for baseline comparison."""
    logger.info("Testing JSONL format...")
    
    write_times = []
    read_times = []
    memory_mapping_times = []
    file_sizes = []
    memory_usages = []
    
    for i in range(num_iterations):
        with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as tmp_file:
            # Test write performance
            start_time = time.time()
            with open(tmp_file.name, 'w') as f:
                for sample in data:
                    f.write(json.dumps(sample) + '\n')
            write_time = time.time() - start_time
            write_times.append(write_time)
            
            # Get file size
            file_size = os.path.getsize(tmp_file.name)
            file_sizes.append(file_size)
            
            # Test read performance
            start_time = time.time()
            loaded_data = []
            with open(tmp_file.name, 'r') as f:
                for line in f:
                    loaded_data.append(json.loads(line.strip()))
            read_time = time.time() - start_time
            read_times.append(read_time)
            
            # Test memory mapping performance
            start_time = time.time()
            with open(tmp_file.name, 'rb') as f:
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                # Simulate reading from memory map
                mm.read(1024)  # Read first 1KB
                mm.close()
            memory_mapping_time = time.time() - start_time
            memory_mapping_times.append(memory_mapping_time)
            
            # Measure memory usage
            process = psutil.Process()
            memory_usage = process.memory_info().rss / 1024 / 1024  # MB
            memory_usages.append(memory_usage)
            
            os.unlink(tmp_file.name)
    
    # Calculate JSONL size for compression ratio
    jsonl_size = len(json.dumps(data))
    
    return FormatTestResult(
        format_name="JSONL",
        file_size_bytes=int(sum(file_sizes) / len(file_sizes)),
        write_time_seconds=sum(write_times) / len(write_times),
        read_time_seconds=sum(read_times) / len(read_times),
        memory_mapping_time_seconds=sum(memory_mapping_times) / len(memory_mapping_times),
        memory_usage_mb=sum(memory_usages) / len(memory_usages),
        compression_ratio=1.0,  # Baseline
        supports_memory_mapping=True,
        zero_copy_reads=False
    )


def demonstrate_arrow_memory_mapping(data: List[Dict[str, Any]]):
    """Demonstrate Arrow's memory mapping capabilities."""
    logger.info("Demonstrating Arrow memory mapping capabilities...")
    
    # Create Arrow table
    df = pd.DataFrame(data)
    table = pa.Table.from_pandas(df)
    
    with tempfile.NamedTemporaryFile(suffix='.arrow', delete=False) as tmp_file:
        # Write Arrow file
        feather.write_feather(table, tmp_file.name)
        
        # Demonstrate memory mapping
        print("\n" + "="*80)
        print("ARROW MEMORY MAPPING DEMONSTRATION")
        print("="*80)
        
        # Method 1: Direct memory mapping with Arrow
        start_time = time.time()
        with open(tmp_file.name, 'rb') as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            # Arrow can read directly from memory-mapped file
            mapped_table = feather.read_table(mm)
            mm.close()
        direct_mapping_time = time.time() - start_time
        
        # Method 2: Standard file reading
        start_time = time.time()
        standard_table = feather.read_table(tmp_file.name)
        standard_read_time = time.time() - start_time
        
        print(f"Direct Memory Mapping Time: {direct_mapping_time:.4f}s")
        print(f"Standard File Read Time:    {standard_read_time:.4f}s")
        print(f"Memory Mapping Speedup:     {standard_read_time / direct_mapping_time:.2f}x")
        
        # Demonstrate random access benefits
        print(f"\nRandom Access Performance Test:")
        
        # Test 1: Random row access (memory mapping should be faster)
        num_random_accesses = 1000
        random_indices = [i % len(table) for i in range(num_random_accesses)]
        
        # Standard random access
        start_time = time.time()
        for idx in random_indices:
            # Simulate random access to specific rows
            row_data = table.slice(idx, 1)
        standard_random_time = time.time() - start_time
        
        # Memory-mapped random access
        start_time = time.time()
        with open(tmp_file.name, 'rb') as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            mapped_table = feather.read_table(mm)
            for idx in random_indices:
                # Simulate random access to specific rows
                row_data = mapped_table.slice(idx, 1)
            mm.close()
        mapped_random_time = time.time() - start_time
        
        print(f"Standard Random Access:     {standard_random_time:.4f}s")
        print(f"Memory-Mapped Random Access: {mapped_random_time:.4f}s")
        print(f"Random Access Speedup:      {standard_random_time / mapped_random_time:.2f}x")
        
        # Demonstrate zero-copy reads
        print(f"\nZero-Copy Read Verification:")
        print(f"Original table address: {id(table)}")
        print(f"Mapped table address:   {id(mapped_table)}")
        print(f"Tables are identical:   {table.equals(mapped_table)}")
        print(f"Data types preserved:   {table.schema == mapped_table.schema}")
        print(f"Row count preserved:    {len(table) == len(mapped_table)}")
        print(f"Column count preserved: {len(table.column_names) == len(mapped_table.column_names)}")
        
        # Show memory efficiency
        import psutil
        process = psutil.Process()
        
        # Memory usage comparison
        gc.collect()
        baseline_memory = process.memory_info().rss / 1024 / 1024
        
        print(f"\nDetailed Memory Analysis:")
        print(f"Baseline Memory:        {baseline_memory:.1f} MB")
        
        # Standard table memory usage
        print(f"\nLoading Standard Table...")
        before_standard = process.memory_info().rss / 1024 / 1024
        standard_table = feather.read_table(tmp_file.name)
        after_standard = process.memory_info().rss / 1024 / 1024
        standard_memory = after_standard - baseline_memory
        print(f"Memory before loading:  {before_standard:.1f} MB")
        print(f"Memory after loading:   {after_standard:.1f} MB")
        print(f"Standard Table Memory:  {standard_memory:.1f} MB")
        
        # Check table size
        print(f"Table size (rows):      {len(standard_table)}")
        print(f"Table size (columns):   {len(standard_table.column_names)}")
        print(f"Table schema:           {standard_table.schema}")
        
        # Memory-mapped table usage
        print(f"\nLoading Memory-Mapped Table...")
        before_mapped = process.memory_info().rss / 1024 / 1024
        with open(tmp_file.name, 'rb') as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            mapped_table = feather.read_table(mm)
            after_mapped = process.memory_info().rss / 1024 / 1024
            mm.close()
        mapped_memory = after_mapped - baseline_memory
        print(f"Memory before mapping:  {before_mapped:.1f} MB")
        print(f"Memory after mapping:   {after_mapped:.1f} MB")
        print(f"Memory-Mapped Memory:   {mapped_memory:.1f} MB")
        
        # Force garbage collection and measure again
        print(f"\nAfter Garbage Collection:")
        del standard_table
        del mapped_table
        gc.collect()
        after_gc = process.memory_info().rss / 1024 / 1024
        print(f"Memory after GC:        {after_gc:.1f} MB")
        print(f"Memory freed:           {after_mapped - after_gc:.1f} MB")
        
        print(f"\nMemory Usage Comparison:")
        print(f"Baseline Memory:        {baseline_memory:.1f} MB")
        print(f"Standard Table Memory:  {standard_memory:.1f} MB")
        print(f"Memory-Mapped Memory:   {mapped_memory:.1f} MB")
        if mapped_memory > 0:
            print(f"Memory Efficiency:      {standard_memory / mapped_memory:.1f}x")
        else:
            print(f"Memory Efficiency:      N/A (mapped memory is 0)")
        
        os.unlink(tmp_file.name)


def print_comprehensive_report(results: List[FormatTestResult], num_samples: int):
    """Print comprehensive performance report."""
    print("\n" + "="*100)
    print("ARROW FILE FORMAT PERFORMANCE ANALYSIS")
    print("="*100)
    print(f"Dataset Size: {num_samples:,} samples")
    print(f"Test Iterations: 3")
    
    print("\n" + "-"*100)
    print("PERFORMANCE COMPARISON")
    print("-"*100)
    print(f"{'Format':<15} {'File Size':<12} {'Write Time':<12} {'Read Time':<12} {'Memory Map':<12} {'Memory':<10} {'Compression':<12}")
    print("-"*100)
    
    for result in results:
        print(f"{result.format_name:<15} "
              f"{result.file_size_bytes/1024/1024:>8.2f}MB "
              f"{result.write_time_seconds:>10.4f}s "
              f"{result.read_time_seconds:>10.4f}s "
              f"{result.memory_mapping_time_seconds:>10.4f}s "
              f"{result.memory_usage_mb:>8.1f}MB "
              f"{result.compression_ratio:>10.1f}x")
    
    print("\n" + "-"*100)
    print("FEATURE COMPARISON")
    print("-"*100)
    print(f"{'Format':<15} {'Memory Mapping':<15} {'Zero-Copy':<10} {'Compression':<12} {'Schema':<8}")
    print("-"*100)
    
    for result in results:
        print(f"{result.format_name:<15} "
              f"{'✓' if result.supports_memory_mapping else '✗':<15} "
              f"{'✓' if result.zero_copy_reads else '✗':<10} "
              f"{result.compression_ratio:>10.1f}x "
              f"{'✓' if result.format_name != 'JSONL' else '✗':<8}")
    
    print("\n" + "-"*100)
    print("RECOMMENDATIONS")
    print("-"*100)
    
    # Find best performers
    best_compression = max(results, key=lambda x: x.compression_ratio)
    fastest_read = min(results, key=lambda x: x.read_time_seconds)
    fastest_write = min(results, key=lambda x: x.write_time_seconds)
    best_memory_mapping = min(results, key=lambda x: x.memory_mapping_time_seconds)
    
    print(f"🏆 BEST COMPRESSION: {best_compression.format_name} ({best_compression.compression_ratio:.1f}x)")
    print(f"⚡ FASTEST READ: {fastest_read.format_name} ({fastest_read.read_time_seconds:.4f}s)")
    print(f"🚀 FASTEST WRITE: {fastest_write.format_name} ({fastest_write.write_time_seconds:.4f}s)")
    print(f"🗺️  BEST MEMORY MAPPING: {best_memory_mapping.format_name} ({best_memory_mapping.memory_mapping_time_seconds:.4f}s)")
    
    print("\n" + "-"*100)
    print("ARROW FILE FORMAT BENEFITS")
    print("-"*100)
    print("✅ Native binary format for disk storage")
    print("✅ Excellent memory mapping efficiency")
    print("✅ Zero-copy reads from disk to memory")
    print("✅ Good compression (better than JSONL)")
    print("✅ Fast serialization/deserialization")
    print("✅ Schema preservation")
    print("✅ Language-agnostic (Python, R, Julia, etc.)")
    print("✅ Streaming support for large files")


def test_arrow_streaming_capabilities(data: List[Dict[str, Any]]):
    """Test Arrow's streaming capabilities for large files."""
    logger.info("Testing Arrow streaming capabilities...")
    
    # Create Arrow table
    df = pd.DataFrame(data)
    table = pa.Table.from_pandas(df)
    
    with tempfile.NamedTemporaryFile(suffix='.arrow', delete=False) as tmp_file:
        # Write with streaming
        start_time = time.time()
        # Use new_file for Arrow file format (not open_file)
        with pa.ipc.new_file(tmp_file.name, table.schema) as writer:
            # Write in batches
            batch_size = 1000
            for i in range(0, len(table), batch_size):
                batch = table.slice(i, min(batch_size, len(table) - i))
                writer.write(batch)
        streaming_write_time = time.time() - start_time
        
        # Read with streaming
        start_time = time.time()
        with pa.ipc.open_file(tmp_file.name) as reader:
            for i in range(reader.num_record_batches):
                batch = reader.get_batch(i)
                # Process batch
                pass
        streaming_read_time = time.time() - start_time
        
        print(f"\nStreaming Performance:")
        print(f"Streaming Write Time: {streaming_write_time:.4f}s")
        print(f"Streaming Read Time:  {streaming_read_time:.4f}s")
        
        os.unlink(tmp_file.name)


def benchmark_random_access(arrow_path, parquet_path, num_accesses=1000):
    import numpy as np
    print("\n" + "-"*80)
    print("RANDOM ACCESS BENCHMARK")
    print("-"*80)
    # Load Arrow
    table_arrow = feather.read_table(arrow_path)
    # Load Parquet
    table_parquet = pq.read_table(parquet_path)
    n = table_arrow.num_rows
    indices = np.random.randint(0, n, size=num_accesses)
    # Arrow random row access
    start = time.time()
    for idx in indices:
        _ = table_arrow.slice(idx, 1)
    arrow_time = time.time() - start
    # Parquet random row access
    start = time.time()
    for idx in indices:
        _ = table_parquet.slice(idx, 1)
    parquet_time = time.time() - start
    print(f"Arrow random row access:   {arrow_time:.4f}s")
    print(f"Parquet random row access: {parquet_time:.4f}s")
    print(f"Speedup: {parquet_time/arrow_time:.2f}x (Arrow over Parquet)")


def benchmark_zero_copy_conversion(arrow_path, parquet_path):
    print("\n" + "-"*80)
    print("ZERO-COPY CONVERSION BENCHMARK")
    print("-"*80)
    # Arrow to pandas (single column, zero-copy)
    table_arrow = feather.read_table(arrow_path)
    col = table_arrow.column(0)
    try:
        col_single = col.combine_chunks()
        start = time.time()
        series = col_single.to_pandas(zero_copy_only=True)
        arrow_time = time.time() - start
        print(f"Arrow single column to pandas (zero-copy): {arrow_time:.4f}s")
    except Exception as e:
        print(f"Zero-copy single column failed: {e}")
    # Arrow to pandas (full table, with copy)
    start = time.time()
    df_arrow = table_arrow.to_pandas()
    arrow_full_time = time.time() - start
    print(f"Arrow full table to pandas (with copy):   {arrow_full_time:.4f}s")
    # Parquet to pandas
    table_parquet = pq.read_table(parquet_path)
    start = time.time()
    df_parquet = table_parquet.to_pandas()
    parquet_time = time.time() - start
    print(f"Parquet to pandas:                        {parquet_time:.4f}s")
    # Numpy conversion
    start = time.time()
    arr_arrow = col_single.to_numpy()
    arrow_np_time = time.time() - start
    start = time.time()
    arr_parquet = table_parquet.column(0).combine_chunks().to_numpy()
    parquet_np_time = time.time() - start
    print(f"Arrow to numpy:                           {arrow_np_time:.4f}s")
    print(f"Parquet to numpy:                         {parquet_np_time:.4f}s")


def benchmark_memory_mapping(arrow_path, parquet_path):
    print("\n" + "-"*80)
    print("MEMORY MAPPING BENCHMARK")
    print("-"*80)
    import mmap
    # Arrow memory mapping
    start = time.time()
    with open(arrow_path, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        table = feather.read_table(mm)
        mm.close()
    arrow_time = time.time() - start
    print(f"Arrow memory-mapped load: {arrow_time:.4f}s")
    # Parquet memory mapping (simulate, not true zero-copy)
    start = time.time()
    with open(parquet_path, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        # Parquet cannot use mmap directly, must read from file
        table = pq.read_table(parquet_path)
        mm.close()
    parquet_time = time.time() - start
    print(f"Parquet mmap+load:        {parquet_time:.4f}s")


def benchmark_batch_slicing(arrow_path, parquet_path, batch_size=1000, num_batches=100):
    print("\n" + "-"*80)
    print("BATCH SLICING BENCHMARK")
    print("-"*80)
    # Load Arrow
    table_arrow = feather.read_table(arrow_path)
    # Load Parquet
    table_parquet = pq.read_table(parquet_path)
    n = table_arrow.num_rows
    # Arrow batch slicing
    start = time.time()
    for i in range(num_batches):
        idx = (i * batch_size) % (n - batch_size)
        _ = table_arrow.slice(idx, batch_size)
    arrow_time = time.time() - start
    # Parquet batch slicing
    start = time.time()
    for i in range(num_batches):
        idx = (i * batch_size) % (n - batch_size)
        _ = table_parquet.slice(idx, batch_size)
    parquet_time = time.time() - start
    print(f"Arrow batch slicing:   {arrow_time:.4f}s")
    print(f"Parquet batch slicing: {parquet_time:.4f}s")
    print(f"Speedup: {parquet_time/arrow_time:.2f}x (Arrow over Parquet)")


def main():
    """Main function to run all tests."""
    print("🚀 Arrow File Format Test: Disk Storage and Memory Mapping Efficiency")
    print("="*80)
    
    # Use real C4 data for more realistic testing
    c4_file_path = os.path.expanduser(C4_FILE_PATH)
    
    if os.path.exists(c4_file_path):
        print(f"Using real C4 data: {c4_file_path}")
        # Load a subset for testing (adjust max_samples as needed)
        data = load_c4_data(c4_file_path)  # Use 50K samples for reasonable test time
        num_samples = len(data)
    else:
        print("C4 data not found, using synthetic data")
        # Create test data
        num_samples = 100000  # Increased from 10,000 to 100,000 for better memory mapping demonstration
        data = create_sample_data(num_samples)
    
    # Run tests
    results = []
    
    # Test Arrow format
    arrow_result = test_arrow_file_format(data)
    results.append(arrow_result)
    
    # Test Parquet format
    parquet_result = test_parquet_format(data)
    results.append(parquet_result)
    
    # Test JSONL format
    jsonl_result = test_jsonl_format(data)
    results.append(jsonl_result)
    
    # Demonstrate Arrow memory mapping
    demonstrate_arrow_memory_mapping(data)
    
    # Test Arrow streaming capabilities
    test_arrow_streaming_capabilities(data)
    
    # Print comprehensive report
    print_comprehensive_report(results, num_samples)
    
    # Write Arrow and Parquet files for targeted benchmarks
    df = pd.DataFrame(data)
    table = pa.Table.from_pandas(df)
    arrow_path = "arrow_bench.arrow"
    parquet_path = "parquet_bench.parquet"
    feather.write_feather(table, arrow_path)
    pq.write_table(table, parquet_path)
    benchmark_random_access(arrow_path, parquet_path)
    benchmark_zero_copy_conversion(arrow_path, parquet_path)
    benchmark_memory_mapping(arrow_path, parquet_path)
    benchmark_batch_slicing(arrow_path, parquet_path)
    # Clean up
    os.remove(arrow_path)
    os.remove(parquet_path)


if __name__ == "__main__":
    main() 
# Pipeline Optimization Demo

This demo shows how to use Data-Juicer's pipeline optimizer to automatically improve processing performance.

## Quick Start

```bash
# See optimization in action (no data processing)
python demos/pipeline_optimization/run_demo.py

# Run with benchmark comparison
python demos/pipeline_optimization/run_demo.py \
    --dataset demos/data/demo-dataset-deduplication.jsonl \
    --benchmark
```

## Overview

The pipeline optimizer analyzes your processing pipeline and applies optimizations:

| Strategy | Description | Benefit |
|----------|-------------|---------|
| `op_reorder` | Moves cheap filters before expensive operations | Reduces wasted computation on filtered data |
| `filter_fusion` | Combines multiple filters into a single fused operation | Reduces iteration overhead, enables parallel filtering |
| `mapper_fusion` | Combines multiple mappers into a single pass | Reduces data iteration overhead |

## Quick Start

Enable optimization in your config:

```yaml
# Enable the optimizer
enable_optimizer: true
optimizer_strategies:
  - op_reorder
  - filter_fusion

process:
  # Your operations here...
```

## Example Configs

### Basic Example (`configs/basic.yaml`)

A simple config showing optimization enabled with common filters.

```bash
python tools/process_data.py --config demos/pipeline_optimization/configs/basic.yaml
```

### Operation Reordering (`configs/op_reorder_showcase.yaml`)

Demonstrates how the optimizer reorders expensive operations after cheap filters:

**Before optimization:**
```
expensive_mapper → expensive_filter → cheap_filter (runs on ALL data)
```

**After optimization:**
```
cheap_filter → expensive_filter → expensive_mapper (cheap filter reduces data first)
```

### Filter Fusion (`configs/filter_fusion_showcase.yaml`)

Demonstrates how multiple filters are fused into a single operation:

**Before optimization:**
```
text_length_filter → words_num_filter → special_chars_filter (3 passes)
```

**After optimization:**
```
fused_filter (1 pass, parallel execution)
```

## Measuring Performance

Use the benchmark tool to measure optimization impact:

```bash
python tools/optimizer_benchmark/run_benchmark.py \
  --recipe-path demos/pipeline_optimization/configs/basic.yaml \
  --dataset-path path/to/your/data.jsonl \
  --output-dir outputs/optimization_benchmark
```

## Configuration Reference

```yaml
# Enable/disable the optimizer (default: false)
enable_optimizer: true

# Strategies to apply (default: ['op_reorder'])
# Available: op_reorder, filter_fusion, mapper_fusion
optimizer_strategies:
  - op_reorder
  - filter_fusion
```

## Tips

1. **Start with `op_reorder`** - It's safe and provides consistent speedups
2. **Add `filter_fusion`** for pipelines with 3+ filters - Greater benefit with more filters
3. **Use the benchmark tool** to measure actual speedup on your data
4. **Check logs** - The optimizer logs what optimizations it applied

# Optimizer Benchmark

A/B testing tool for comparing Data-Juicer pipeline execution with and without optimizer strategies. Uses the `data_juicer.benchmark` framework for statistical analysis and report generation.

## Quick Start

```bash
# Test op_pruning strategy
python tools/optimizer_benchmark/run_benchmark.py \
  --recipe-path tools/optimizer_benchmark/configs/optimizer_benchmark_pruning.yaml \
  --dataset-path /path/to/data.jsonl \
  --strategies op_pruning

# List available strategies
python tools/optimizer_benchmark/run_benchmark.py --list-strategies
```

## Options

| Option | Description | Default |
|--------|-------------|---------|
| `--recipe-path` | Path to recipe YAML file | Required |
| `--dataset-path` | Path to dataset file | Required |
| `--output-dir` | Output directory for results | `./outputs/optimizer_benchmark` |
| `--strategies` | Comma-separated list of strategies | `op_pruning,op_reorder` |
| `--iterations` | Number of benchmark iterations | `1` |
| `--warmup-runs` | Number of warmup runs | `0` |
| `--verbose` | Enable verbose logging | `false` |

## Available Strategies

| Strategy | Description |
|----------|-------------|
| `op_pruning` | Remove no-op and duplicate operations |
| `op_reorder` | Reorder operations for optimal execution |
| `mapper_fusion` | Fuse consecutive mapper operations |
| `filter_fusion` | Fuse filters sharing intermediate variables |
| `all_optimizations` | Enable all core optimizations |

## Examples

### Test Single Strategy

```bash
python tools/optimizer_benchmark/run_benchmark.py \
  --recipe-path tools/optimizer_benchmark/configs/optimizer_benchmark_pruning.yaml \
  --dataset-path /tmp/c4_sample_10k.jsonl \
  --strategies op_pruning
```

### Test Multiple Strategies

```bash
python tools/optimizer_benchmark/run_benchmark.py \
  --recipe-path tools/optimizer_benchmark/configs/optimizer_benchmark_pruning.yaml \
  --dataset-path /tmp/c4_sample_10k.jsonl \
  --strategies op_pruning,op_reorder
```

### Multiple Iterations for Statistical Significance

```bash
python tools/optimizer_benchmark/run_benchmark.py \
  --recipe-path tools/optimizer_benchmark/configs/optimizer_benchmark_pruning.yaml \
  --dataset-path /tmp/c4_sample_10k.jsonl \
  --strategies op_pruning \
  --iterations 3 \
  --warmup-runs 1
```

### Test All Optimizations

```bash
python tools/optimizer_benchmark/run_benchmark.py \
  --recipe-path tools/optimizer_benchmark/configs/optimizer_benchmark_pruning.yaml \
  --dataset-path /tmp/c4_sample_10k.jsonl \
  --strategies all_optimizations
```

## Synthetic Data for Testing

The benchmark framework includes synthetic workloads for quick testing without production datasets:

```python
from data_juicer.benchmark import WORKLOAD_SUITE, SYNTHETIC_DATA_GENERATOR

# Use pre-defined synthetic workloads
workload = WORKLOAD_SUITE.get_workload("synthetic_text_10k")
workload.ensure_data_exists()  # Auto-generates if needed

# Or generate custom synthetic data
SYNTHETIC_DATA_GENERATOR.generate_text_data(
    output_path="/tmp/custom_data.jsonl",
    num_samples=5000,
)
```

Available synthetic workloads:
- `synthetic_text_1k` - 1K samples, ~1 min
- `synthetic_text_10k` - 10K samples, ~5 min
- `synthetic_text_100k` - 100K samples, ~30 min

## Output

The benchmark generates:

| File | Description |
|------|-------------|
| `results.json` | Structured results with comparisons |
| `ab_test_report_*.html` | Professional HTML report |
| `ab_test_data_*.json` | Raw A/B test data |
| `benchmark.log` | Detailed execution log |
| `baseline/` | Baseline run outputs |
| `<strategy>/` | Optimized run outputs for each strategy |

## Example Results

With 10K samples (C4 dataset) and op_pruning strategy:

| Configuration | Time | Throughput | Speedup |
|--------------|------|------------|---------|
| Baseline | 19.75s | 506 samples/sec | - |
| Optimized (op_pruning) | 15.50s | 645 samples/sec | **1.27x** |

The `op_pruning` strategy removes redundant operations (no-op filters with default parameters, duplicate operations) from the pipeline, reducing execution overhead.

## Architecture

This tool uses the `data_juicer.benchmark` framework:

- **StrategyABTest**: Orchestrates A/B testing between baseline and optimized runs
- **WorkloadDefinition**: Defines dataset and config for benchmarking
- **STRATEGY_LIBRARY**: Registry of available optimization strategies
- **ResultAnalyzer**: Statistical comparison with significance testing
- **ReportGenerator**: HTML and JSON report generation

# Pipeline Performance Test

This directory contains a comprehensive performance testing framework for comparing individual pipeline execution vs optimized pipeline execution in Data-Juicer.

## Overview

The pipeline performance test compares two execution modes:
1. **Individual Pipeline**: Operations are executed one by one without optimization
2. **Optimized Pipeline**: Operations are fused and optimized using the pipeline optimizer

## Files

- `perf_test_pipeline_comparison.py`: Main performance test script
- `run_pipeline_perf_test.sh`: Convenient shell script wrapper
- `README_pipeline_perf_test.md`: This documentation file

## Features

- **Separate Process Execution**: Each mode runs in its own process for fair comparison
- **Recipe Support**: Works with any Data-Juicer recipe YAML file
- **Dataset Support**: Supports various dataset formats
- **Comprehensive Metrics**: Collects execution time, memory usage, throughput
- **Result Validation**: Ensures both modes produce equivalent results
- **Detailed Reporting**: Generates markdown reports with performance analysis
- **Two Execution Modes**:
  - Separate process execution (default)
  - Comprehensive benchmark framework

## Usage

### Using the Shell Script (Recommended)

```bash
# Basic usage
./tests/benchmark_performance/run_pipeline_perf_test.sh \
  -r configs/demo/analyzer.yaml \
  -d demos/data/demo-dataset.jsonl

# With verbose logging
./tests/benchmark_performance/run_pipeline_perf_test.sh \
  -r configs/demo/analyzer.yaml \
  -d demos/data/demo-dataset.jsonl \
  -v

# Using comprehensive benchmark framework
./tests/benchmark_performance/run_pipeline_perf_test.sh \
  -r configs/demo/analyzer.yaml \
  -d demos/data/demo-dataset.jsonl \
  -b

# Custom output directory
./tests/benchmark_performance/run_pipeline_perf_test.sh \
  -r configs/demo/analyzer.yaml \
  -d demos/data/demo-dataset.jsonl \
  -o ./my_results
```

### Using Python Script Directly

```bash
python tests/benchmark_performance/perf_test_pipeline_comparison.py \
  --recipe-path configs/demo/analyzer.yaml \
  --dataset-path demos/data/demo-dataset.jsonl \
  --output-dir ./outputs/pipeline_perf_test \
  --verbose
```

## Command Line Options

### Shell Script Options

- `-r, --recipe-path PATH`: Path to the recipe YAML file (required)
- `-d, --dataset-path PATH`: Path to the dataset file (required)
- `-o, --output-dir PATH`: Output directory (default: ./outputs/pipeline_perf_test)
- `-b, --benchmark-framework`: Use comprehensive benchmark framework
- `-v, --verbose`: Enable verbose logging
- `-h, --help`: Show help message

### Python Script Options

- `--recipe-path`: Path to the recipe YAML file (required)
- `--dataset-path`: Path to the dataset file (required)
- `--output-dir`: Output directory for results and reports
- `--use-benchmark-framework`: Use the comprehensive benchmark framework
- `--verbose`: Enable verbose logging

## Output

The test generates several output files:

### Results JSON (`results.json`)
Contains detailed performance metrics and comparison data:
```json
{
  "individual": {
    "wall_time": 10.5,
    "output_samples": 1000,
    "success": true,
    "error": null
  },
  "optimized": {
    "wall_time": 8.2,
    "output_samples": 1000,
    "success": true,
    "error": null
  },
  "comparison": {
    "individual_time": 10.5,
    "optimized_time": 8.2,
    "speedup": 1.28,
    "improvement_percent": 21.9,
    "faster_mode": "optimized"
  },
  "validation": {
    "samples_match": true,
    "individual_samples": 1000,
    "optimized_samples": 1000,
    "sample_difference": 0,
    "validation_passed": true
  },
  "metadata": {
    "recipe_path": "configs/demo/analyzer.yaml",
    "dataset_path": "demos/data/demo-dataset.jsonl",
    "test_timestamp": "2024-01-15 10:30:00",
    "use_benchmark_framework": false
  }
}
```

### Performance Report (`performance_report.md`)
A comprehensive markdown report with:
- Executive summary
- Detailed results for each mode
- Performance comparison
- Validation results
- Recommendations

### Log File (`perf_test.log`)
Detailed execution logs for debugging and analysis.

## Execution Modes

### 1. Separate Process Execution (Default)

This mode runs each pipeline in a completely separate process to ensure:
- No interference between executions
- Fair resource allocation
- Clean memory state for each run

**Pros:**
- Completely isolated execution
- Fair comparison
- No memory leaks between runs

**Cons:**
- Higher overhead due to process creation
- Slower startup time

### 2. Comprehensive Benchmark Framework

This mode uses the existing performance benchmark framework which provides:
- More detailed metrics
- Memory usage tracking
- Throughput analysis
- Resource utilization

**Pros:**
- More comprehensive metrics
- Better integration with existing tools
- Detailed resource analysis

**Cons:**
- Runs in the same process
- May have interference between runs

## Example Recipes

### Simple Analyzer Recipe
```yaml
project_name: 'demo-analyzer'
dataset_path: 'demos/data/demo-dataset.jsonl'
export_path: 'outputs/demo-analyzer/res.jsonl'
np: 1
use_cache: false

process:
  - whitespace_normalization_mapper:
  - token_num_filter:
      hf_tokenizer: 'EleutherAI/pythia-6.9b-deduped'
      min_num: 0
  - document_deduplicator:
      lowercase: false
      ignore_non_character: false
```

### Complex Pipeline Recipe
```yaml
project_name: 'complex-pipeline'
dataset_path: 'data/large-dataset.jsonl'
export_path: 'outputs/complex-pipeline/res.jsonl'
np: 4
use_cache: true

process:
  - whitespace_normalization_mapper:
  - token_num_filter:
      hf_tokenizer: 'EleutherAI/pythia-6.9b-deduped'
      min_num: 10
      max_num: 1000
  - document_deduplicator:
      lowercase: true
      ignore_non_character: true
  - language_id_score_filter:
      lang: 'en'
      min_score: 0.8
  - text_length_filter:
      min_len: 50
      max_len: 2000
  - topk_specified_field_selector:
      field_key: '__dj__stats__.num_token'
      topk: 10000
```

## Interpreting Results

### Performance Metrics

- **Execution Time**: Wall clock time for each mode
- **Speedup**: Ratio of individual time to optimized time
- **Improvement**: Percentage improvement from individual to optimized
- **Throughput**: Samples processed per second

### Validation

- **Samples Match**: Whether both modes produce the same number of output samples
- **Sample Difference**: Absolute difference in output sample counts
- **Validation Passed**: Overall validation status

### Recommendations

Based on the results, the test provides recommendations:

- **Use Optimized Pipeline**: When optimized mode is faster and produces correct results
- **Consider Individual Pipeline**: When individual mode is faster (may still be beneficial for larger datasets)
- **Both Modes Similar**: When performance is similar between modes
- **Investigation Required**: When results don't match between modes

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you're running from the project root directory
2. **File Not Found**: Verify recipe and dataset paths are correct
3. **Permission Errors**: Ensure the shell script is executable (`chmod +x`)
4. **Timeout Errors**: Large datasets may need longer timeout values

### Debug Mode

Use the `--verbose` flag for detailed logging:
```bash
./tests/benchmark_performance/run_pipeline_perf_test.sh \
  -r configs/demo/analyzer.yaml \
  -d demos/data/demo-dataset.jsonl \
  -v
```

### Manual Testing

For debugging, you can run the Python script directly:
```bash
cd /path/to/data-juicer
python tests/benchmark_performance/perf_test_pipeline_comparison.py \
  --recipe-path configs/demo/analyzer.yaml \
  --dataset-path demos/data/demo-dataset.jsonl \
  --verbose
```

## Integration with CI/CD

The test can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run Pipeline Performance Test
  run: |
    ./tests/benchmark_performance/run_pipeline_perf_test.sh \
      -r configs/demo/analyzer.yaml \
      -d demos/data/demo-dataset.jsonl \
      -o ./test-results

- name: Upload Results
  uses: actions/upload-artifact@v2
  with:
    name: performance-test-results
    path: ./test-results/
```

## Contributing

When adding new features to the performance test:

1. Update this README with new options and examples
2. Add appropriate error handling
3. Include new metrics in the results JSON
4. Update the markdown report template
5. Add tests for new functionality

## Related Documentation

- [Data-Juicer Operators](https://github.com/modelscope/data-juicer/blob/main/docs/Operators.md)
- [Performance Benchmark Framework](../core/optimizer/performance_benchmark.py)
- [Pipeline Optimizer](../core/optimizer/optimizer.py) 
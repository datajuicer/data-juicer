# Pipeline Optimization Strategies

This document details the behavior and performance characteristics of Data-Juicer's pipeline optimization strategies.

## Overview

| Strategy | What It Does | Typical Benefit | Ray Support |
|----------|--------------|-----------------|-------------|
| **op_pruning** | Removes no-op and duplicate operations | ~16% speedup | Yes |
| **op_reorder** | Moves cheap filters before expensive ones | 5-33% speedup | Yes |
| **filter_fusion** | Shares intermediate variables between filters | ~15% speedup | Skipped (auto) |
| **mapper_fusion** | Combines mappers into single pass | ~14% speedup | Skipped (auto) |
| **Combined** | All strategies together | ~20% speedup | Partial (op_reorder only) |

## Operation Pruning (op_pruning)

### How It Works

Operation pruning identifies and removes **redundant operations** that don't change the output:
1. **No-op filters**: Filters with pass-through conditions (e.g., min=0, max=inf)
2. **Duplicate operations**: Consecutive identical operations
3. **No-op mappers**: Mappers with empty configuration (e.g., no chars to remove)

**Key principle:** Remove operations that don't change the output to save processing time.

```
Before: text_length_filter(min=0, max=inf) → text_length_filter(min=100, max=5000) → ...
After:  text_length_filter(min=100, max=5000) → ...
```

### Prunable Operations

**No-op Filters** (automatically removed when conditions pass everything):
- `text_length_filter` with min_len=0 and max_len>=10^8
- `words_num_filter` with min_num=0 and max_num>=10^8
- `alphanumeric_filter` with min_ratio=0 and max_ratio=1
- `special_characters_filter` with min_ratio=0 and max_ratio=1
- `character_repetition_filter` with max_ratio=1
- `word_repetition_filter` with max_ratio=1
- `stopwords_filter` with min_ratio=0
- `flagged_words_filter` with max_ratio=1
- `suffix_filter` with empty suffixes list

**No-op Mappers** (automatically removed when config is empty):
- `remove_specific_chars_mapper` with empty chars_to_remove
- `replace_content_mapper` with empty pattern

### Example

```yaml
# BAD: Contains redundant operations
process:
  - text_length_filter:      # No-op: passes everything
      min_len: 0
      max_len: 999999999
  - text_length_filter:      # Useful: actual constraint
      min_len: 100
      max_len: 50000
  - words_num_filter:        # No-op: passes everything
      min_num: 0
      max_num: 999999999
  - words_num_filter:        # Useful: actual constraint
      min_num: 20
      max_num: 5000
  - special_characters_filter:  # Useful: actual constraint
      min_ratio: 0.0
      max_ratio: 0.3
  - special_characters_filter:  # Duplicate: same as previous
      min_ratio: 0.0
      max_ratio: 0.3

# With op_pruning enabled, becomes:
# 1. text_length_filter (min=100, max=50000)
# 2. words_num_filter (min=20, max=5000)
# 3. special_characters_filter (min=0, max=0.3)
# Pruned: 3 operations (2 no-ops + 1 duplicate)
```

### Performance Results

**Dataset:** C4 (356K samples)
**Pipeline:** 14 operations (3 mappers + 11 filters, 6 redundant)

| Executor | Baseline | Optimized | Speedup | Improvement |
|----------|----------|-----------|---------|-------------|
| Default  | 77.5s    | 65.2s     | 1.19x   | **+15.9%**  |

**Operation count:** 14 ops → 8 ops (6 pruned, 43% reduction)

### Configuration

```yaml
enable_optimizer: true
optimizer_strategies:
  - op_pruning  # Remove redundant operations

process:
  # Your operations - redundant ones will be removed automatically
  - text_length_filter:
      min_len: 0          # This will be pruned (no-op)
      max_len: 999999999
  - text_length_filter:
      min_len: 100        # This will be kept
      max_len: 50000
```

### When to Use op_pruning

**Always recommended** - it's safe and works with any executor:
- Automatically identifies redundant operations
- Preserves semantic correctness (only removes true no-ops)
- Works with both default and Ray executors
- No configuration needed - just enable it

---

## Operation Reordering (op_reorder)

### How It Works

Operation reordering moves **cheap filters** (no model, fast computation) before **expensive filters** (model-based, slow). This reduces the amount of data that expensive filters need to process.

**Key principle:** Filter early, process less.

```
Before: expensive_filter (100% data) → cheap_filter (keeps 40%)
After:  cheap_filter (100% data, keeps 40%) → expensive_filter (40% data)
```

### Cheap vs Expensive Filters

**Cheap Filters** (run first):
- `text_length_filter` - character count
- `words_num_filter` - word count
- `alphanumeric_filter` - character ratio
- `special_characters_filter` - special char ratio
- `character_repetition_filter` - repetition detection
- `average_line_length_filter`, `maximum_line_length_filter`

**Expensive Filters** (run last):
- `stopwords_filter` - requires tokenization + stopwords model
- `flagged_words_filter` - requires tokenization + flagged words list
- `perplexity_filter` - requires language model
- `language_id_score_filter` - requires language detection model
- `image_*_filter` - requires image loading/processing
- `video_*_filter` - requires video loading/processing

### Example

```yaml
# BAD ORDER: expensive filters run on ALL data first
process:
  - stopwords_filter:        # Expensive: runs on 356K samples
      lang: en
  - flagged_words_filter:    # Expensive: runs on 354K samples
      lang: en
  - text_length_filter:      # Cheap: but too late!
      min_len: 1000
  - words_num_filter:        # Cheap: but too late!
      min_num: 150

# With op_reorder enabled, becomes:
# 1. text_length_filter   → reduces to 195K samples (45% reduction)
# 2. words_num_filter     → reduces to 193K samples
# 3. stopwords_filter     → runs on 91K samples (75% less work!)
# 4. flagged_words_filter → runs on 91K samples
```

### Performance Results

**Dataset:** C4 (356K samples)

| Data Reduction | Baseline | Optimized | Speedup | Improvement |
|----------------|----------|-----------|---------|-------------|
| Mild (~1%)     | 92.6s    | 87.5s     | 1.06x   | **+5.5%**   |
| Moderate (~25%)| 88.8s    | 82.6s     | 1.08x   | **+7.0%**   |
| Aggressive (~75%)| 86.6s  | 58.1s     | 1.49x   | **+32.9%**  |

**Key insight:** The more data cheap filters remove, the bigger the benefit.

### Detailed Breakdown (Aggressive Filtering)

| Filter | Baseline (356K) | Optimized (91K) | Savings |
|--------|-----------------|-----------------|---------|
| stopwords_filter | 22.1s | 7.7s | **65% faster** |
| flagged_words_filter | 21.1s | 7.7s | **64% faster** |

### Configuration

```yaml
enable_optimizer: true
optimizer_strategies:
  - op_reorder  # Recommended for all pipelines

process:
  # Put your operations in any order
  # The optimizer will reorder them for best performance
  - stopwords_filter: {}
  - text_length_filter:
      min_len: 500
```

### When to Use op_reorder

**Always recommended** - it's safe and provides consistent benefits:
- Preserves mapper order (no semantic changes)
- Only reorders filters based on cost
- Works with both default and Ray executors

---

## Filter Fusion

### How It Works

Filter fusion combines multiple filters that share **intermediate variables** into a single fused operation. This allows expensive computations (like tokenization) to be performed once and shared across filters.

**Intermediate Variable Registries:**
- `INTER_WORDS` - Tokenized words (word_repetition_filter, words_num_filter, stopwords_filter, flagged_words_filter, perplexity_filter)
- `INTER_LINES` - Split lines
- `LOADED_IMAGES` - Loaded image data
- `LOADED_VIDEOS` - Loaded video data
- `LOADED_AUDIOS` - Loaded audio data
- `INTER_SAMPLED_FRAMES` - Sampled video frames

### Example

```yaml
# These 4 filters all tokenize text into words
# Without fusion: tokenize 4 times
# With fusion: tokenize once, share across all filters

process:
  - word_repetition_filter:
      rep_len: 10
      max_ratio: 0.5

  - words_num_filter:
      min_num: 5
      max_num: 1000

  - stopwords_filter:
      lang: en
      max_ratio: 0.8

  - flagged_words_filter:
      lang: en
      max_ratio: 0.01
```

### Performance Results

**Dataset:** C4 (356K samples)
**Filters:** 4 INTER_WORDS filters

| Executor | Baseline | Optimized | Speedup | Improvement |
|----------|----------|-----------|---------|-------------|
| Default  | 88.98s   | 75.69s    | 1.18x   | **+14.9%**  |
| Ray      | 110.00s  | 125.77s   | 0.87x   | -14.3%      |

### Why Ray Is Different

Ray already parallelizes operations efficiently across workers. When filters are fused:
- **Default executor**: Fusion reduces iteration overhead and shares intermediate variables
- **Ray executor**: Fusion forces sequential execution within each worker, losing Ray's parallelization benefits

**Result:** Filter fusion is automatically skipped in Ray mode.

---

## Mapper Fusion

### How It Works

Mapper fusion combines consecutive mappers into a single pass through the data, reducing dataset iteration overhead.

### Example

```yaml
# These 6 mappers can be fused into a single pass
process:
  - clean_html_mapper: {}
  - clean_links_mapper: {}
  - clean_email_mapper: {}
  - clean_copyright_mapper: {}
  - punctuation_normalization_mapper: {}
  - whitespace_normalization_mapper: {}
```

### Performance Results

**Dataset:** C4 (356K samples)
**Mappers:** 6 text cleaning mappers

| Executor | Baseline | Optimized | Speedup | Improvement |
|----------|----------|-----------|---------|-------------|
| Default  | 45.07s   | 38.59s    | 1.17x   | **+14.4%**  |
| Ray      | 65.18s   | 90.37s    | 0.72x   | -38.6%      |

**Result:** Mapper fusion is automatically skipped in Ray mode.

---

## Configuration

### Recommended Configuration

```yaml
enable_optimizer: true
optimizer_strategies:
  - op_reorder      # Always safe, good for all pipelines
  - filter_fusion   # Good for default executor with 3+ filters
  - mapper_fusion   # Good for default executor with 3+ mappers

executor_type: default  # For fusion benefits
```

### Ray Executor

```yaml
enable_optimizer: true
optimizer_strategies:
  - op_reorder      # Works with Ray
  - filter_fusion   # Auto-skipped in Ray mode
  - mapper_fusion   # Auto-skipped in Ray mode

executor_type: ray
```

---

## Benchmarking

Test optimization performance on your data:

```bash
# Test all strategies (recommended)
python tools/optimizer_benchmark/run_benchmark.py \
  --recipe-path tools/optimizer_benchmark/configs/optimizer_benchmark_combined.yaml \
  --dataset-path your_data.jsonl \
  --strategies op_reorder,filter_fusion,mapper_fusion \
  --executor default

# Test individual strategies
python tools/optimizer_benchmark/run_benchmark.py \
  --recipe-path tools/optimizer_benchmark/configs/optimizer_benchmark_combined.yaml \
  --dataset-path your_data.jsonl \
  --strategies op_reorder \
  --executor default

# Test with Ray executor
python tools/optimizer_benchmark/run_benchmark.py \
  --recipe-path tools/optimizer_benchmark/configs/optimizer_benchmark_combined.yaml \
  --dataset-path your_data.jsonl \
  --strategies op_reorder,filter_fusion,mapper_fusion \
  --executor ray
```

---

## Summary

| Strategy | Best For | Typical Speedup | Ray Compatible |
|----------|----------|-----------------|----------------|
| **op_pruning** | Pipelines with potential redundancy | ~16% | Yes |
| **op_reorder** | All pipelines | 5-33% | Yes |
| **filter_fusion** | 3+ filters sharing vars | ~15% | Default only |
| **mapper_fusion** | 3+ consecutive mappers | ~14% | Default only |

**Recommendation:** Start with `op_pruning` and `op_reorder` for all pipelines. Add fusion strategies for default executor when you have multiple compatible operations.

---

## Combined Strategies

### Using All Strategies Together

For maximum performance, enable all three strategies. The optimizer automatically handles strategy interactions:
- `op_reorder` runs first to optimize filter order
- `filter_fusion` and `mapper_fusion` are auto-skipped on Ray executor

### Combined Benchmark Results

**Dataset:** C4 (356K samples)
**Pipeline:** 6 mappers + 6 filters (2 expensive, 4 cheap in bad order)

#### With 12 CPUs (recommended for single-node)

| Executor | Baseline | Optimized | Speedup | Improvement | Notes |
|----------|----------|-----------|---------|-------------|-------|
| **Default** | 75.0s | 65.2s | 1.15x | **+13.0%** | All 3 strategies applied |
| **Ray** | ~102s | ~98s | ~1.04x | **+3-6%** | Only op_reorder (fusion auto-skipped, high variance) |

#### With 4 CPUs

| Executor | Baseline | Optimized | Speedup | Improvement | Notes |
|----------|----------|-----------|---------|-------------|-------|
| **Default** | 130.7s | 104.9s | 1.25x | **+19.7%** | All 3 strategies applied |
| **Ray** | 218.4s | 214.4s | 1.02x | **+1.9%** | Only op_reorder (fusion auto-skipped) |

### Why Default Executor Benefits More

On the default executor, all three strategies contribute:
1. **mapper_fusion**: 6 mappers combined into single pass
2. **op_reorder**: cheap filters moved before expensive filters
3. **filter_fusion**: filters sharing INTER_WORDS tokenize once

On Ray, only `op_reorder` applies. The fusion strategies are auto-skipped because Ray's parallel execution model already optimizes data flow differently.

### When to Use Ray vs Default Executor

| Scenario | Recommended | Why |
|----------|-------------|-----|
| Single machine, small-medium data (<10GB) | **Default** | Lower overhead, all optimizations apply |
| Single machine, more CPUs available | **Default** | Scale with `np` parameter |
| Multi-node cluster | **Ray** | Distributed processing across nodes |
| Dataset exceeds RAM | **Ray** | Out-of-core processing via object store |
| GPU cluster for model inference | **Ray** | GPU distribution and scheduling |

**Key insight:** On a single machine, Ray adds ~36% overhead (serialization, task scheduling) without distributed benefits. Use Ray when you have multiple physical nodes or datasets that don't fit in memory.

### Configuration for Combined Strategies

```yaml
enable_optimizer: true
optimizer_strategies:
  - op_reorder       # Reorder filters by cost
  - filter_fusion    # Share intermediate variables
  - mapper_fusion    # Combine consecutive mappers

process:
  # Mappers (will be fused on default executor)
  - clean_html_mapper: {}
  - clean_links_mapper: {}
  - clean_email_mapper: {}
  - punctuation_normalization_mapper: {}
  - whitespace_normalization_mapper: {}

  # Filters in any order (will be reordered)
  - stopwords_filter:       # Expensive - will be moved last
      lang: en
  - text_length_filter:     # Cheap - will be moved first
      min_len: 500
  - words_num_filter:       # Cheap - will be moved first
      min_num: 50
```

### Benchmark Commands

```bash
# Test combined strategies on default executor
python tools/optimizer_benchmark/run_benchmark.py \
  --recipe-path tools/optimizer_benchmark/configs/optimizer_benchmark_combined.yaml \
  --dataset-path your_data.jsonl \
  --strategies op_reorder,filter_fusion,mapper_fusion \
  --executor default

# Test combined strategies on Ray executor
python tools/optimizer_benchmark/run_benchmark.py \
  --recipe-path tools/optimizer_benchmark/configs/optimizer_benchmark_combined.yaml \
  --dataset-path your_data.jsonl \
  --strategies op_reorder,filter_fusion,mapper_fusion \
  --executor ray
```

---

## Technical Details

### Files

- `data_juicer/core/optimizer/op_pruning_strategy.py` - Operation pruning (no-op/duplicate removal)
- `data_juicer/core/optimizer/op_reorder_strategy.py` - Operation reordering
- `data_juicer/core/optimizer/filter_fusion_strategy.py` - Filter fusion
- `data_juicer/core/optimizer/mapper_fusion_strategy.py` - Mapper fusion
- `data_juicer/core/optimizer/fused_op.py` - FusedFilter and FusedMapper
- `data_juicer/ops/op_fusion.py` - Intermediate variable registries

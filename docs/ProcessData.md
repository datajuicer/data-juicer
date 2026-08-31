# Processing Data

This guide covers running data processing pipelines with Data-Juicer—both CLI and Python API.

> If you haven't run your first pipeline yet, start with the [Quick Start](tutorial/QuickStart.md).
> For the full parameter list, see [Global Configuration Reference](GlobalConfig.md).

---

## CLI

### Basic Usage

```bash
dj-process --config my-recipe.yaml
```

Data-Juicer reads the recipe, executes operators in the listed order, and writes results to `export_path`.

### Command-Line Overrides

Override any recipe parameter without modifying the YAML:

```bash
dj-process --config recipe.yaml --np 8 --export_path ./out/result.parquet
dj-process --config recipe.yaml --language_id_score_filter.lang=en
```

### Auto-Install Operator Dependencies

```bash
dj-install --config my-recipe.yaml
```

Resolves and installs all per-operator dependencies in one pass. See [Installation §5](tutorial/Installation.md#5-installation-for-specific-ops).

---

## Python API

The Python API provides finer control than YAML recipes—ideal for training scripts, notebooks, or automated pipelines.

### Option 1: Load a Recipe

```python
from data_juicer.config import init_configs
from data_juicer.core import DefaultExecutor

cfg = init_configs(args=['--config', 'my-recipe.yaml'])
executor = DefaultExecutor(cfg)
dataset = executor.run()
```

If you already hold a dataset object in memory (e.g. assembled or sampled upstream), you can skip the recipe's data source and reuse only its operator pipeline:

```python
dataset = executor.run(dataset=my_dataset, skip_export=True)
```

### Option 2: Instantiate Operators from Config

No YAML needed—assemble an operator chain in Python:

```python
from data_juicer.ops import load_ops
from data_juicer.core import NestedDataset

# Load ops from dict config (same format as YAML process list)
ops = load_ops([
    {'language_id_score_filter': {'lang': 'en', 'min_score': 0.8}},
    {'text_length_filter': {'min_len': 10, 'max_len': 50000}},
    {'document_minhash_deduplicator': {'tokenization': 'space', 'window_size': 5}},
])

dataset = NestedDataset(NestedDataset.from_json('my-data.jsonl'))
dataset = dataset.process(ops)
```

### Option 3: Fine-Grained Single-Operator Control

When you need conditional logic, loops, or intermediate inspection:

```python
from data_juicer.ops.filter import LanguageIDScoreFilter, TextLengthFilter
from data_juicer.ops.deduplicator import DocumentMinhashDeduplicator
from data_juicer.core import NestedDataset

dataset = NestedDataset(NestedDataset.from_json('my-data.jsonl'))

# Step 1: Language filter
lang_filter = LanguageIDScoreFilter(lang='en', min_score=0.8)
dataset = lang_filter.run(dataset=dataset)
print(f"After language filter: {len(dataset)} samples")

# Step 2: Conditional dedup — only when dataset is large
if len(dataset) > 10000:
    dedup = DocumentMinhashDeduplicator(tokenization='space', window_size=5)
    dataset = dedup.run(dataset=dataset)
    print(f"After dedup: {len(dataset)} samples")

# Step 3: Length filter
length_filter = TextLengthFilter(min_len=10, max_len=50000)
dataset = length_filter.run(dataset=dataset)
```

### Option 4: Dynamic Operator Composition

Choose operators programmatically based on data characteristics—useful for automated pipelines:

```python
from data_juicer.ops import load_ops
from data_juicer.core import NestedDataset

dataset = NestedDataset(NestedDataset.from_json('input.jsonl'))

# Inspect data to decide processing strategy
sample = dataset[0]
ops_config = []

# Add language filter if text field exists
if 'text' in sample:
    ops_config.append({'language_id_score_filter': {'lang': 'en', 'min_score': 0.5}})

# Add image filter if images present
if 'images' in sample and sample['images']:
    ops_config.append({'image_shape_filter': {'min_width': 256, 'min_height': 256}})

# Common cleaning
ops_config.append({'clean_html_mapper': {}})
ops_config.append({'text_length_filter': {'min_len': 10}})

ops = load_ops(ops_config)
dataset = dataset.process(ops)
```

---

## Operator Execution Order

Operators run **top-to-bottom sequentially**. Order matters:

1. **Cheap filters first**: Text length, language ID—reduce sample count early
2. **Dedup in the middle**: Requires global state; run after initial filtering
3. **Expensive operators last**: GPU inference only sees the filtered subset

```yaml
process:
  # Cheap
  - text_length_filter: { min_len: 10, max_len: 50000 }
  - language_id_score_filter: { lang: en, min_score: 0.5 }
  # Dedup
  - document_minhash_deduplicator: { tokenization: space, window_size: 5 }
  # Expensive
  - clean_html_mapper: {}
  - perplexity_filter: { lang: en, max_ppl: 1500 }
```

---

## Performance Tuning

### Op Fusion

Merges adjacent compatible operators into a single data pass—**2–10× throughput** (default mode only):

```yaml
op_fusion: true
fusion_strategy: probe   # probe (reorder by speed) | greedy (keep order)
```

### GPU Mapper Fusion

Fuses consecutive GPU Mappers into one GPU pass:

```yaml
op_fusion: true
mapper_fusion: true
adaptive_batch_size: true
```

`adaptive_batch_size` is applied only by the `default` executor, and updates only batched operators. It does not enable automatic batch-size tuning in the Ray executors.

### Sampling Dry Run

For a source with at least 1,000 rows, validate a recipe on 1,000 samples with the default executor by replacing its `dataset_path` with a structured `dataset` configuration. Keep its `process` list:

```yaml
dataset:
  max_sample_num: 1000
  configs:
    - type: local
      path: path/to/your/dataset.jsonl
```

Replace the path with your dataset and choose a budget no larger than its row count. `max_sample_num` is a sample-count budget, not a percentage or a truncation-only limit: if the budget exceeds the available rows, the sampler repeats rows to fill it. Sampling happens after loading, so this does not guarantee reduced input I/O. Alternatively, prepare a small input file first and use it directly. Remove the budget for the full run.

`data_probe_ratio` and `data_probe_algo` belong to the Sandbox model-inference probe. A normal `dj-process` run does not apply them automatically; setting `data_probe_ratio: 0.01` alone still processes the full input. See [Analysis and tool-specific parameters](GlobalConfig.md).

---

## Checkpointing & Resumption

```yaml
use_checkpoint: true
```

Re-running the same config skips completed operators. `use_checkpoint` is mutually exclusive with `op_fusion`.

For `ray_partitioned`, finer strategies are available; resume with `--resume <job_id>`. See [Global Config](GlobalConfig.md#cache--checkpointing).

---

## Tracing & Debugging

```yaml
open_tracer: true
trace_num: 10
```

Writes before/after comparisons per operator. See [Tracing](Tracing.md).

---

## Next Steps

- [Data Analysis](AnalyzeData.md)—understand your data distribution before processing
- [Dataset Configuration](DatasetCfg.md)—input formats, mixing, remote datasets
- [Global Configuration Reference](GlobalConfig.md)—full parameter list
- [Distributed Processing](Distributed.md)—scale to Ray clusters
- [Operator Library](Operators.md)—browse 200+ available operators

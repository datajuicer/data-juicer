# Data Analysis

Before deciding on filter thresholds, it helps to understand the statistical profile of your dataset. `dj-analyze` computes distributions and correlations for all operator-produced statistics, enabling data-driven threshold decisions.

> For the full parameter list, see [Global Configuration Reference](GlobalConfig.md).

---

## CLI

### Basic Usage

Run the analyzer with an existing recipe:

```bash
dj-analyze --config path/to/your-recipe.yaml
```

### Auto Mode

No dedicated analysis recipe needed—automatically uses all stats-producing Filters on a dataset subset:

```bash
dj-analyze --auto --dataset_path your-dataset.jsonl --auto_num 1000
```

- `--auto_num`: Number of samples to analyze (default 1000). Good for quick distribution overview.

---

## Python API

### Basic: Load Config and Run

```python
from data_juicer.config import init_configs
from data_juicer.core import Analyzer

cfg = init_configs(args=['--config', 'my-recipe.yaml'])
analyzer = Analyzer(cfg)
dataset = analyzer.run()

# Access results
print(analyzer.overall_result)
```

### Analyze an Existing Dataset

Wrap the loaded data explicitly: with current dependencies, `from_json()` returns a plain Hugging Face Dataset, while Analyzer needs the `.process()` method.

```python
from data_juicer.core import Analyzer, NestedDataset
from data_juicer.config import init_configs

cfg = init_configs(args=[
    '--config', 'my-recipe.yaml',
    '--export_path', './analysis-output/stats.jsonl',
])
analyzer = Analyzer(cfg)

dataset = NestedDataset(NestedDataset.from_json('my-data.jsonl'))
analyzed = analyzer.run(dataset=dataset)
```

### Compute Stats Without Exporting Results

Call a Filter directly with `reduce=False` to compute statistics without filtering or exporting a dataset or analysis reports. Dataset caching may still write cache files.

```python
from data_juicer.core import NestedDataset
from data_juicer.ops.filter import TextLengthFilter
from data_juicer.utils.constant import Fields, StatsKeys

dataset = NestedDataset(NestedDataset.from_json('my-data.jsonl'))
analyzed = TextLengthFilter().run(dataset=dataset, reduce=False)
stats = analyzed[Fields.stats]
avg_len = sum(s[StatsKeys.text_len] for s in stats) / len(stats) if len(stats) else 0
print(f"Average text length: {avg_len:.2f}")
```

`Analyzer.run(..., skip_export=True)` has a different scope: it skips the overall tables and plots, but currently still exports the statistics dataset and creates working directories/configuration backups. It is not a no-export mode.

### Manual Analysis Pipeline

For full control over the analysis logic, use the underlying components directly:

```python
from data_juicer.ops.filter import LanguageIDScoreFilter, TextLengthFilter
from data_juicer.analysis import OverallAnalysis, ColumnWiseAnalysis
from data_juicer.core import NestedDataset

dataset = NestedDataset(NestedDataset.from_json('my-data.jsonl'))

# Compute stats only: reduce=False below disables filtering
filters = [
    TextLengthFilter(min_len=0, max_len=999999),
    LanguageIDScoreFilter(lang='en', min_score=0.0),
]

for f in filters:
    dataset = f.run(dataset=dataset, reduce=False)  # compute stats only, no filtering

# Run analysis
output_dir = './my-analysis'
overall = OverallAnalysis(dataset, output_dir)
result = overall.analyze()
print(result)

column_wise = ColumnWiseAnalysis(dataset, output_dir, overall_result=result)
column_wise.analyze()
```

### Dynamic Analysis Dimensions

Programmatically choose analysis operators based on data modality—useful for automated pipelines:

```python
from data_juicer.core import Analyzer, NestedDataset
from data_juicer.config import init_configs

dataset = NestedDataset(NestedDataset.from_json('input.jsonl'))
sample = dataset[0]

# Build analysis config based on data modality
process_config = []

# Text statistics
if 'text' in sample:
    process_config.extend([
        {'text_length_filter': {'min_len': 0, 'max_len': 999999}},
        {'language_id_score_filter': {'lang': 'en', 'min_score': 0.0}},
        {'alphanumeric_filter': {'min_ratio': 0.0}},
    ])

# Image statistics
if 'images' in sample and sample['images']:
    process_config.extend([
        {'image_shape_filter': {'min_width': 0, 'min_height': 0}},
        {'image_aspect_ratio_filter': {'min_ratio': 0.0, 'max_ratio': 999}},
    ])

cfg = init_configs(args=[
    '--auto',
    '--dataset_path', 'input.jsonl',
    '--export_path', './analysis/stats.jsonl',
], allow_auto=True)
cfg.process = process_config
# Auto mode remains enabled: analyze at most auto_num rows (default 1000).

analyzer = Analyzer(cfg)
analyzed = analyzer.run(dataset=dataset)
```

---

## Analysis Output

The analyzer produces:

- **Overall statistics table**: count, mean, std, min, max for each metric
- **Distribution plots**: histogram for each metric
- **Correlation analysis**: heatmap of metric correlations

Plots and overall tables are saved in `analyzer.analysis_path` (`<cfg.work_dir>/analysis`); the resolved `work_dir` includes `job_id`. The statistics dataset is exported according to `export_path`.

---

## Which Operators Participate

The Analyzer processes two types of operators:
- **Filter operators** that produce stats in the `__dj__stats__` field (most Filters do)
- **Tagging operators** that produce labels in the `__dj__meta__` field

Registry markers:
- `NON_STATS_FILTERS`: Filters that do NOT produce stats
- `TAGGING_OPS`: Operators that produce tags

---

## Distributed Analysis

Set `executor_type: ray` to use `RayAnalyzer` with native Ray aggregation:

```bash
dj-analyze --config demos/analyze_simple/ray_analyzer.yaml
```

> RayAnalyzer does not produce per-column distribution plots or correlation analysis. See [Distributed Processing](Distributed.md).

---

## Font Configuration

If distribution plots show "Glyph missing" warnings:

```bash
export ANALYZER_FONT="Heiti SC"  # default; supports CJK characters
```

---

## Next Steps

- Adjust thresholds interactively? Use [Web Playground](Playground.md)
- Ready to process? See [Processing Data](ProcessData.md)
- Large-scale analysis? See [Distributed Processing](Distributed.md)

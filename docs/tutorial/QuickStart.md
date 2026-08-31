# Quickstart

This guide walks you through a complete Data-Juicer workflow: install, write a recipe, run the pipeline, and inspect the output.

> **Note:** Some operators download model weights on first use (for example, `language_id_score_filter` downloads a fastText language identification model). The initial run may take a few extra minutes while these assets are fetched. Subsequent runs can reuse the local model cache.

---

## 1. Install Data-Juicer

First install Python, Git, and uv as described in [Installation](Installation.md). The commands below use a Linux/macOS shell.

Clone the repository and install from source. This guide uses the demo recipes and sample data included in the repo:

```bash
git clone https://github.com/datajuicer/data-juicer.git --depth 1
cd data-juicer
uv venv
source .venv/bin/activate  # Linux / macOS
uv pip install -e ".[nlp]"
```

Verify the CLI is available:

```bash
dj-process --help
```

> **Tip:** If you don't need the demo files, `uv pip install "py-data-juicer[nlp]"` installs the same package from PyPI without cloning. For all installation methods (extras, Docker, etc.), see [Installation](Installation.md).

---

## 2. Understand the input data

Data-Juicer supports JSONL, Parquet, CSV/TSV, plain text, and more out of the box. This guide uses the built-in sample dataset [`demos/data/demo-dataset.jsonl`](https://github.com/datajuicer/data-juicer/blob/main/demos/data/demo-dataset.jsonl):

```json
{"text": "Today is Sunday and it's a happy day!", "meta": {"src": "Arxiv"}}
{"text": "Do you need a cup of coffee?", "meta": {"src": "code"}}
{"text": "你好，请问你是谁", "meta": {"src": "customized"}}
{"text": "Sur la plateforme MT4, plusieurs manières...", "meta": {"src": "Oscar"}}
{"text": "欢迎来到阿里巴巴！", "meta": {"src": "customized"}}
{"text": "This paper proposed a novel method on LLM pretraining.", "meta": {"src": "customized"}}
```

Each line is a JSON object. Text is in the `"text"` field by default; other fields are preserved as metadata.

> For advanced input configuration (format options, data mixing, remote datasets like Hugging Face), see [Dataset Configuration](../DatasetCfg.md). For raw data that requires extra extraction or conversion (arXiv tar archives, Stack Exchange 7z files, etc.), the [preprocessing tools](../../tools/preprocess/README.md) can transform them into formats Data-Juicer reads directly.

---

## 3. Write a recipe

A **recipe** is a YAML config file declaring which operators to run and in what order. Use the built-in demo recipe [`demos/process_simple/process.yaml`](https://github.com/datajuicer/data-juicer/blob/main/demos/process_simple/process.yaml):

```yaml
# Global parameters
project_name: 'demo-process'
dataset_path: './demos/data/demo-dataset.jsonl'
np: 4

export_path: './outputs/demo-process/demo-processed.jsonl'

# Operators to apply
process:
  - language_id_score_filter:
      lang: 'zh'
      min_score: 0.8
```

Each entry under `process` is one operator, executed in listed order—the output of each becomes the input to the next. This recipe keeps only Chinese samples with high language confidence.

> For full recipe syntax, see [Global Configuration Reference](../GlobalConfig.md). For the complete list of 200+ built-in operators, see [Operators](../Operators.md).

---

## 4. Run the pipeline

Pass the recipe to the `dj-process` CLI:

```bash
dj-process --config demos/process_simple/process.yaml
```

Data-Juicer loads the dataset, executes each operator in sequence, and writes filtered results to `export_path`. Per-operator statistics are printed to the console as the pipeline runs.

Override any recipe parameter from the command line without editing the YAML:

```bash
dj-process --config demos/process_simple/process.yaml --language_id_score_filter.lang=en \
  --export_path ./outputs/demo-process/demo-processed-en.jsonl
```

> The English results use a separate file so the Chinese output inspected below is preserved. `dj-install --config your-recipe.yaml` can preinstall dependencies identified in operator source, but does not guarantee all runtime dependencies or model files are ready. See [Installation](Installation.md).

---

## 5. Inspect the output

The processed dataset is at your configured `export_path`:

```bash
cat ./outputs/demo-process/demo-processed.jsonl
```

You should see only the Chinese samples that passed the filter—English and French lines have been removed.

To understand your dataset's quality distribution before committing to a full run, use the analyzer:

```bash
dj-analyze --config demos/process_simple/process.yaml
```

> For the full analyzer usage (auto mode, distributed analysis, custom metrics), see [Data Analysis Guide](../AnalyzeData.md). For interactive threshold tuning with sliders, see [Web Playground](../Playground.md).

---

## 6. Use from Python (optional)

To embed Data-Juicer in a training script or notebook:

```python
from data_juicer.config import init_configs
from data_juicer.core import DefaultExecutor

cfg = init_configs(args=['--config', 'my-recipe.yaml'])
executor = DefaultExecutor(cfg)
executor.run()
```

Chain-style single-operator calls are also supported:

```python
dataset = dataset.process([op1, op2])
```

> For complete Python API usage, see the [Processing Data Guide](../ProcessData.md).

---

## Next steps

You have a working pipeline. Explore further based on your needs:

| Topic | Description | Link |
|-------|-------------|------|
| **Processing Data** | Full CLI & Python API usage, performance tuning | [Processing Data Guide](../ProcessData.md) |
| **Data Analysis** | Profile dataset distributions before processing | [Analysis Guide](../AnalyzeData.md) |
| **Operator Zoo** | Browse 200+ operators across text/image/audio/video | [Operators Overview](../Operators.md) |
| **Visual Tuning** | Drag sliders to tune filter thresholds interactively | [Web Playground](../Playground.md) |
| **Distributed** | Scale to multi-node clusters with Ray | [Distributed Processing](../Distributed.md) |
| **Sandbox** | Small-scale experiments with data-model co-optimization | [DJ-Sandbox](https://datajuicer.github.io/data-juicer-sandbox/en/main/index.html) |
| **Export & Cache** | Control output format, speed up repeated runs | [Export](../Export.md) · [Cache](../Cache.md) |
| **Custom Operators** | Write your own operators and contribute code | [Developer Guide](../DeveloperGuide.md) |
| **DJ-Cookbook** | Community recipes and tutorial resources | [DJ-Cookbook](DJ-Cookbook.md) |

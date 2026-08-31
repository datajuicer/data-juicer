# Dataset Configuration

This guide covers how to configure input datasets in your Data-Juicer recipe. You will learn how to point at local files, remote Hugging Face or arXiv datasets, mix multiple sources, validate data, and handle edge cases.

## Supported dataset formats

Data-Juicer auto-detects file formats for local files. Supported formats include `parquet`, `jsonl`, `json`, `csv`, `tsv`, `txt`, and `jsonl.gz`.

### Local dataset

Point at a file or directory on your local filesystem. The `format` field is optional — Data-Juicer detects it from the file extension.

```yaml
dataset:
  configs:
    - type: local
      path: path/to/your/local/dataset.json
      format: json    # optional
```

```yaml
dataset:
  configs:
    - type: local
      path: path/to/your/local/dataset.parquet
      format: parquet
```

See [local_json.yaml](https://github.com/datajuicer/data-juicer-hub/blob/main/dataset_config/local_json.yaml) for a complete example.

### Remote Hugging Face dataset

Load any dataset from the Hugging Face Hub. Set `type` to `remote` and `source` to `huggingface`.

```yaml
dataset:
  configs:
    - type: 'remote'
      source: 'huggingface'
      path: "HuggingFaceFW/fineweb"
      name: "CC-MAIN-2024-10"   # optional: dataset config name
      split: "train"             # optional: which split to load
      limit: 1000                # optional: cap the number of samples
```

See [remote_huggingface.yaml](https://github.com/datajuicer/data-juicer-hub/blob/main/dataset_config/remote_huggingface.yaml) for a complete example.

### arXiv data

For arXiv papers, use the [preprocessing tools](../tools/preprocess/README.md) to download and convert arXiv tar archives into JSONL format that Data-Juicer can process directly.

### Other formats

For the full list of supported formats and loading strategies, see [load_strategy.py](https://github.com/datajuicer/data-juicer/blob/main/data_juicer/core/data/load_strategy.py).

---

## Data mixture

With the default executor, combine multiple datasets by listing them under `dataset.configs`. Weights determine sample allocation only when `dataset.max_sample_num` is set. Without this budget, the sources are concatenated in full and weights do not change their proportions. Ray currently supports a single source through this dataset builder.

Despite its name, `max_sample_num` is a sampling budget, not a truncation-only limit. If a source's allocated count exceeds its available rows, the sampler repeats rows to fill that allocation.

```yaml
dataset:
  max_sample_num: 10000    # total sample budget; required for weights to take effect
  configs:
    - type: 'local'
      weight: 1.0
      path: 'path/to/json/file'
    - type: 'local'
      weight: 1.0
      path: 'path/to/csv/file'
```

See [mixture.yaml](https://github.com/datajuicer/data-juicer-hub/blob/main/dataset_config/mixture.yaml) for a complete example.

---

## Data validation

Validate your dataset before processing by adding `validators` to your recipe. Each validator checks a specific aspect of the data.

```yaml
dataset:
  configs:
    - type: local
      path: path/to/data.json

validators:
  - type: swift_messages
    min_turns: 2
    max_turns: 20
    sample_size: 1000
  - type: required_fields
    required_fields:
      - "text"
      - "metadata"
      - "language"
    field_types:
      text: "str"
      metadata: "dict"
      language: "str"
```

See [data_validator.py](https://github.com/datajuicer/data-juicer/blob/main/data_juicer/core/data/data_validator.py) for the full list of supported validators.

---

## Troubleshooting

### JSONL per-line fault tolerance

If your JSONL file contains a few corrupted lines, enable **lenient JSONL loading** to skip bad lines instead of failing the entire job:

Use the environment variable; the current global config parser does not accept `load_jsonl_lenient` as a top-level YAML option:

```bash
DATA_JUICER_JSONL_LENIENT=1 dj-process --config path/to/config.yaml
```

> **Note:** Only `.jsonl` / `.jsonl.gz` / `.jsonl.zst` shards are read. Other files in the same directory (e.g. `.json`) are skipped with a warning. Search logs for `[lenient jsonl]` to see which lines were skipped.

### `Value is too big!` error

When loading local JSONL, HuggingFace `datasets` may parse with `ujson`, which cannot handle very large integers. If you see `ValueError: Value is too big!`:

| Fix | How |
| --- | --- |
| **Use stdlib json** (recommended) | `DATA_JUICER_USE_STDLIB_JSON=1 dj-process --config path/to/config.yaml` |
| **Export as strings** | Quote the problematic numeric fields in your JSON source. |
| **Switch to Parquet** | Parquet uses Arrow, which avoids this code path entirely. |

---

## Legacy `dataset_path` configuration

The `dataset_path` key is the original, simpler way to specify input. It works but lacks the flexibility of the `dataset.configs` approach above.

```yaml
# YAML
dataset_path: path/to/your/dataset.json
```

```bash
# CLI
dj-process --dataset_path path/to/your/dataset.json

# CLI with mixture weights
dj-process --dataset_path 0.5 path/to/dataset1.json 0.5 path/to/dataset2.json
```

---

## What's next

- [Processing Data](ProcessData.md) — learn how to run pipelines and chain operators.
- [Operator Schemas](Operators.md) — understand the operator types available for your data.
- [Export Guide](Export.md) — control the output format and path.

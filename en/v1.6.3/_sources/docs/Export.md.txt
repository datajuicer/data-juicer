# Dataset Export

After processing, Data-Juicer writes the result dataset to the path you specify in `export_path`. This page covers supported output formats, sharding large datasets into multiple files, parallel export, writing directly to S3, and controlling which intermediate fields (stats, hashes) are kept in the output.

## Overview

Data-Juicer exports via `Exporter` (default mode) or `RayExporter` (Ray mode). The export system supports:

- **Multiple output formats** — JSONL, JSON, Parquet, and more in Ray mode
- **Shard export** — split large datasets into multiple files by size
- **Parallel export** — speed up single-file export with multiprocessing
- **S3 export** — write results directly to Amazon S3 or S3-compatible storage
- **Stats and hash management** — control which intermediate fields are kept in the output

## Configuration

### Basic Settings

```yaml
export_path: ./outputs/result.jsonl       # Output file path (required)
export_type: jsonl                         # Format type (auto-detected from path if omitted)
export_shard_size: 0                       # Local: one file; Ray: use dataset block layout
export_in_parallel: false                  # Local mode: parallel writing to one file
keep_stats_in_res_ds: false                # Keep computed stats in output
keep_hashes_in_res_ds: false               # Keep computed hashes in output
export_extra_args: {}                      # Additional format-specific arguments
export_aws_credentials: null               # For S3 export, see S3 section for details
```

### Command Line

```bash
# Basic export
dj-process --config config.yaml --export_path ./outputs/result.jsonl

# Export as Parquet
dj-process --config config.yaml --export_path ./outputs/result.parquet

# Export with sharding (256MB per shard)
dj-process --config config.yaml --export_shard_size 268435456

# Keep stats in output
dj-process --config config.yaml --keep_stats_in_res_ds true
```

## Supported Formats

### Default Mode (Exporter)

| Format | Suffix | Description |
|--------|--------|-------------|
| JSONL | `.jsonl` | JSON Lines — one JSON object per line (default) |
| JSON | `.json` | Standard JSON array |
| Parquet | `.parquet` | Columnar format, efficient for large datasets |

### Ray Mode (RayExporter)

| Format | Suffix | Description |
|--------|--------|-------------|
| JSONL | `.jsonl` | JSON Lines |
| JSON | `.json` | Standard JSON |
| Parquet | `.parquet` | Columnar format |
| CSV | `.csv` | Comma-separated values |
| TFRecords | `.tfrecords` | TensorFlow record format |
| WebDataset | `webdataset` | WebDataset tar-based format |
| Lance | `.lance` | Lance columnar format |

In local mode, `export_path` is a file path. Include a `.jsonl`, `.json`, or `.parquet` extension, or set `export_type` explicitly.

In Ray mode, `export_path` is a directory for output files, even when the path ends in `.jsonl`. Ray writes files according to the dataset block layout; `export_shard_size: 0` uses that default layout.

## Shard Export (Local Mode)

For large datasets, split the output into multiple shard files based on size:

```yaml
export_path: ./outputs/result.jsonl
export_shard_size: 268435456              # 256 MB per shard
```

This produces files like:
```
outputs/
├── result-00-of-04.jsonl
├── result-01-of-04.jsonl
├── result-02-of-04.jsonl
└── result-03-of-04.jsonl
```

Data-Juicer estimates the dataset size, splits rows into contiguous shards, and writes the shards with multiple workers. The configured size is a target; the encoded files may be larger or smaller.

**Recommended shard sizes:**

| Dataset Size | Recommended Shard Size | Notes |
|-------------|----------------------|-------|
| < 1 GB | 0 (single file) | No need to shard |
| 1-10 GB | 256 MB - 512 MB | Good balance |
| 10-100 GB | 512 MB - 1 GB | Fewer files |
| > 100 GB | 1 GB - 10 GB | Avoid too many shards |

Shard sizes below 1 MiB or above 1 TiB will trigger warnings.

## Parallel Export (Local Mode)

For single-file export (`export_shard_size: 0`), enable parallel writing to speed up the process:

```yaml
export_path: ./outputs/result.jsonl
export_shard_size: 0
export_in_parallel: true
np: 4                                     # Number of parallel processes
```

**Important**: Parallel export can sometimes be **slower** than sequential export due to IO blocking, especially for very large datasets. If you observe this, set `export_in_parallel: false`.

When `export_shard_size > 0`, shards are always exported in parallel regardless of this setting.

## S3 Export

Both local and Ray modes can write results directly to S3. Set `export_path` to an S3 location:

```yaml
export_path: "s3://my-bucket/outputs/result.jsonl"
```

Provide credentials through environment variables:

```bash
export AWS_ACCESS_KEY_ID="your-access-key-id"
export AWS_SECRET_ACCESS_KEY="your-secret-access-key"
export AWS_DEFAULT_REGION="us-east-1"
```

For temporary credentials, also set `AWS_SESSION_TOKEN`. Alternatively, supply credentials through `export_aws_credentials` in your recipe. Both local and Ray modes support this configuration:

```yaml
export_path: "s3://my-bucket/outputs/result.jsonl"
export_aws_credentials:
  aws_access_key_id: "your-access-key-id"
  aws_secret_access_key: "your-secret-access-key"
  aws_region: "us-east-1"
  endpoint_url: "https://s3.example.com"  # Set for S3-compatible storage
```

Access keys, session tokens, and region are read from environment variables first, then from explicit configuration, field by field. When no access keys are supplied, the storage client uses its default AWS credential chain, such as an IAM role or a local credentials file.

Ray mode also accepts credentials in `export_extra_args`; values in `export_aws_credentials` take precedence over matching entries. Local mode accesses S3 through s3fs, while Ray mode uses PyArrow.

In local mode, set `export_shard_size: 268435456` for a target shard size of approximately 256 MiB. Output objects have names such as `result-00-of-04.jsonl`. Ray mode writes files under the directory prefix specified by `export_path`.

## Stats and Hash Management

During processing, DataJuicer computes intermediate fields:
- **Stats** (`__dj__stats__`, `__dj__meta__`): computed by Filter operators
- **Hashes** (`__dj__hash__`, `__dj__minhash__`, `__dj__simhash__`, etc.): computed by Deduplicator operators

By default, these fields are **removed** from the exported dataset. To keep them:

```yaml
keep_stats_in_res_ds: true                # Keep stats and meta fields
keep_hashes_in_res_ds: true               # Keep hash fields
```

### Stats Export

In local mode, a dataset containing `__dj__stats__` or `__dj__meta__` columns also produces a separate statistics file:

```text
outputs/
├── result.jsonl
└── result_stats.jsonl
```

The statistics file contains only the stats and meta columns present in the dataset. When using the Python `Exporter` directly, set `export_stats=False` to turn off this additional export.

Ray mode exports statistics as part of the main dataset. Set `keep_stats_in_res_ds: true` to retain them; Ray does not produce a separate `_stats.jsonl` file.

## WebDataset Export (Ray Mode)

In Ray mode, you can export to WebDataset format with custom field mapping:

```yaml
export_path: ./outputs/webdataset
export_type: webdataset
export_extra_args:
  field_mapping:
    txt: "text"
    png: "images"
    json: "metadata"
```

## API Reference

### Exporter (Default Mode)

```python
from data_juicer.core.exporter import Exporter

exporter = Exporter(
    export_path="./outputs/result.jsonl",
    export_type="jsonl",
    export_shard_size=0,
    export_in_parallel=True,
    num_proc=4,
    keep_stats_in_res_ds=False,
    keep_hashes_in_res_ds=False,
)

exporter.export(dataset)
```

### RayExporter (Ray Mode)

```python
from data_juicer.core.ray_exporter import RayExporter

exporter = RayExporter(
    export_path="./outputs/result.jsonl",
    export_type="jsonl",
    export_shard_size=268435456,
    keep_stats_in_res_ds=False,
    keep_hashes_in_res_ds=False,
)

exporter.export(ray_dataset)
```

## Troubleshooting

**Export format not supported:**
```bash
# Check supported formats
# Default mode: jsonl, json, parquet
# Ray mode: jsonl, json, parquet, csv, tfrecords, webdataset, lance
```

**Parallel export is slower than expected:**
```yaml
# Disable parallel export
export_in_parallel: false
```

**S3 export fails with permission error:**
```bash
# Verify credentials
aws s3 ls s3://your-bucket/

# Check that export_aws_credentials is configured
```

**Too many shard files generated:**
```yaml
# Increase shard size
export_shard_size: 1073741824             # 1 GB
```

**Stats missing from exported dataset:**
```yaml
# Keep stats in the result dataset
keep_stats_in_res_ds: true
# In local mode, also check the separate stats file: result_stats.jsonl
```

---

## What's next

- [Cache Management](Cache.md) — speed up re-runs by caching intermediate results.
- [Data Tracing](Tracing.md) — debug sample-level changes through the pipeline.
- [Distributed Processing](Distributed.md) — scale export across a Ray cluster.

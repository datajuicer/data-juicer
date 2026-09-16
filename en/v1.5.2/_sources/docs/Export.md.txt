# Dataset Export

This document describes how DataJuicer exports processed datasets, including supported formats, sharding, parallel export, S3 export, and stats/hash management.

## Overview

After processing, DataJuicer exports the result dataset to disk using the `Exporter` (default mode) or `RayExporter` (Ray mode). The export system supports:

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
export_shard_size: 0                       # Shard size in bytes (0 = single file)
export_in_parallel: false                  # Parallel export for single-file mode
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

## Shard Export

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

**How shard size is calculated:**
1. The total dataset size in bytes is estimated
2. Number of shards = `ceil(dataset_bytes / export_shard_size)`
3. The dataset is split into contiguous shards
4. Each shard is exported in parallel using multiprocessing

**Recommended shard sizes:**

| Dataset Size | Recommended Shard Size | Notes |
|-------------|----------------------|-------|
| < 1 GB | 0 (single file) | No need to shard |
| 1-10 GB | 256 MB - 512 MB | Good balance |
| 10-100 GB | 512 MB - 1 GB | Fewer files |
| > 100 GB | 1 GB - 10 GB | Avoid too many shards |

Shard sizes below 1 MiB or above 1 TiB will trigger warnings.

## Parallel Export

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

Export results directly to Amazon S3 or S3-compatible storage.

### Default Mode

```yaml
export_path: "s3://my-bucket/outputs/result.jsonl"
export_aws_credentials:
  aws_access_key_id: "AKIA..."
  aws_secret_access_key: "secret..."
  aws_region: "us-east-1"
  endpoint_url: "https://s3.example.com"   # Optional: for S3-compatible storage
```

The default exporter uses HuggingFace's `storage_options` with `fsspec`/`s3fs` for S3 access.

### Ray Mode

```yaml
export_path: "s3://my-bucket/outputs/result.jsonl"
export_extra_args:
  aws_access_key_id: "AKIA..."
  aws_secret_access_key: "secret..."
  aws_region: "us-east-1"
```

The Ray exporter uses PyArrow's S3 filesystem for S3 access.

### S3 with Sharding

When using S3 with shard export, shard files are written directly to S3:

```yaml
export_path: "s3://my-bucket/outputs/result.jsonl"
export_shard_size: 268435456
export_aws_credentials:
  aws_access_key_id: "AKIA..."
  aws_secret_access_key: "secret..."
```

This produces S3 objects like:
```
s3://my-bucket/outputs/result-00-of-04.jsonl
s3://my-bucket/outputs/result-01-of-04.jsonl
...
```

### Credential Resolution

AWS credentials are resolved in priority order:
1. `export_aws_credentials` config (default mode) or `export_extra_args` (Ray mode)
2. Environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)
3. Default credential chain (IAM role, `~/.aws/credentials`)

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

Regardless of `keep_stats_in_res_ds`, DataJuicer always exports a separate stats file alongside the main dataset:

```
outputs/
├── result.jsonl                          # Main dataset (stats removed by default)
└── result_stats.jsonl                    # Stats-only file (always exported)
```

The stats file contains only the `__dj__stats__` and `__dj__meta__` columns.

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
# Or check the separate stats file: result_stats.jsonl
```

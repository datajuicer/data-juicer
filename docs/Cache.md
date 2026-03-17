# Cache Management

This document describes DataJuicer's cache management system, including HuggingFace dataset caching, cache directory configuration, cache compression, and temporary storage.

## Overview

DataJuicer provides a caching mechanism based on HuggingFace Datasets to avoid redundant computation. When enabled, each operator generates cache files with a unique **fingerprint** based on:
- The fingerprint of the input data
- The operator name and parameters
- The hash of the processing function

That is: same input + same operator configuration = same fingerprint = cache hit. Therefore, re-running the same pipeline on the same data will skip already-computed steps. For more details, please refer to our paper: [Data-Juicer: A One-Stop Data Processing System for Large Language Models](https://arxiv.org/abs/2309.02033).

The cache system also provides:
- **Configurable cache directories** via environment variables or config options
- **Cache compression** to reduce disk usage for large-scale datasets
- **Temporary storage** for intermediate files in non-cache mode
- **Fine-grained cache control** via context managers and decorators

## Configuration

### Basic Cache Settings

```yaml
use_cache: true           # Enable/disable HuggingFace dataset caching
ds_cache_dir: null         # Custom cache directory (overrides HF_DATASETS_CACHE)
cache_compress: null       # Compression method: 'gzip', 'zstd', 'lz4', or null
temp_dir: null             # Temp directory for intermediate files when cache is disabled
```

### Command Line

```bash
# Enable caching (default)
dj-process --config config.yaml --use_cache true

# Disable caching
dj-process --config config.yaml --use_cache false

# Enable cache compression
dj-process --config config.yaml --cache_compress zstd

# Custom cache directory
dj-process --config config.yaml --ds_cache_dir /fast-storage/dj-cache
```

## Cache Directory Structure

DataJuicer organizes cache files in a hierarchical directory structure controlled by environment variables:

```
~/.cache/                              # CACHE_HOME (default)
└── data_juicer/                       # DATA_JUICER_CACHE_HOME
    ├── assets/                        # DATA_JUICER_ASSETS_CACHE
    │   └── (extracted frames, stopwords, flagged words, etc.)
    └── models/                        # DATA_JUICER_MODELS_CACHE
        └── (downloaded model files)
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CACHE_HOME` | `~/.cache` | Root cache directory |
| `DATA_JUICER_CACHE_HOME` | `$CACHE_HOME/data_juicer` | DataJuicer cache root |
| `DATA_JUICER_ASSETS_CACHE` | `$DATA_JUICER_CACHE_HOME/assets` | Assets cache (frames, word lists, etc.) |
| `DATA_JUICER_MODELS_CACHE` | `$DATA_JUICER_CACHE_HOME/models` | Downloaded models cache |
| `DATA_JUICER_EXTERNAL_MODELS_HOME` | `None` | External models directory |

Override defaults by setting environment variables:

```bash
export DATA_JUICER_CACHE_HOME=/data/dj-cache
export DATA_JUICER_MODELS_CACHE=/models/dj-models
dj-process --config config.yaml
```

## Cache Compression

For large-scale datasets (tens of GB or more), cache files can consume significant disk space. Cache compression reduces storage requirements by compressing intermediate cache files after each operator completes.

### Supported Algorithms

| Algorithm | Library | Speed | Compression Ratio | Recommended For |
|-----------|---------|-------|-------------------|-----------------|
| `zstd` | zstandard | Fast | High | General use (default) |
| `lz4` | lz4 | Fastest | Moderate | Speed-critical workloads |
| `gzip` | gzip | Slow | High | Compatibility needs |

### Configuration

```yaml
use_cache: true
cache_compress: zstd    # Enable zstd compression
```

```bash
dj-process --config config.yaml --cache_compress zstd
```

### Multi-Process Compression

Cache compression supports parallel processing. The number of compression worker processes is controlled by the `np` parameter:

```yaml
np: 4                   # Number of parallel workers (also used for compression)
cache_compress: zstd
```

## Cache Control API

### DatasetCacheControl

A context manager to temporarily enable or disable HuggingFace dataset caching within a specific scope:

```python
from data_juicer.utils.cache_utils import DatasetCacheControl

# Temporarily disable caching
with DatasetCacheControl(on=False):
    # Operations here will not use cache
    result = dataset.map(my_function)

# Temporarily enable caching
with DatasetCacheControl(on=True):
    # Operations here will use cache
    result = dataset.map(my_function)
```

### dataset_cache_control Decorator

A decorator for functions that need to control cache state:

```python
from data_juicer.utils.cache_utils import dataset_cache_control

@dataset_cache_control(on=False)
def process_without_cache(dataset):
    return dataset.map(my_function)
```

### CompressionOff

A context manager to temporarily disable cache compression:

```python
from data_juicer.utils.compress import CompressionOff

with CompressionOff():
    # Cache compression is disabled in this scope
    result = dataset.map(my_function)
```

### CompressManager

Low-level API for manual compression/decompression:

```python
from data_juicer.utils.compress import CompressManager

manager = CompressManager(compressor_format="zstd")

# Compress a file
manager.compress("input.arrow", "input.arrow.zstd")

# Decompress a file
manager.decompress("input.arrow.zstd", "input.arrow")
```

### CacheCompressManager

High-level API for managing HuggingFace dataset cache compression:

```python
from data_juicer.utils.compress import CacheCompressManager

manager = CacheCompressManager(compressor_format="zstd")

# Compress previous dataset's cache files
manager.compress(prev_ds=previous_dataset, this_ds=current_dataset, num_proc=4)

# Decompress cache files for a dataset
manager.decompress(ds=dataset, num_proc=4)

# Clean up all compressed cache files
manager.cleanup_cache_files(ds=dataset)
```

## Cache vs Checkpoint

Cache and checkpoint are mutually exclusive — enabling checkpoint automatically disables cache:

| Feature | Cache | Checkpoint |
|---------|-------|------------|
| **Purpose** | Accelerate repeated runs with same configuration | Fault recovery and resumption |
| **Granularity** | Per-operator result | Full dataset snapshot |
| **Storage Location** | HuggingFace cache directory | Work directory |
| **Recovery Method** | Automatic (hash-based) | Manual (config-based) |
| **Compression** | Supported (`cache_compress`) | Not applicable |
| **Scenario** | Iterative development, parameter tuning | Long-running production tasks |

```yaml
# Cache mode (default)
use_cache: true
use_checkpoint: false

# Checkpoint mode (cache auto-disabled)
use_cache: true           # Will be overridden to false
use_checkpoint: true
```

## Disabling Cache and Temporary Directory

When `use_cache: false` or checkpoint mode is enabled (`use_checkpoint: true`), HuggingFace dataset caching is fully disabled. In this mode, DataJuicer writes intermediate files produced during operator processing to a temporary directory, and cleans them up automatically when processing completes. The `temp_dir` parameter controls where these intermediate files are stored.

### Behavior

- **Defaults to `null`**: When not set, the operating system determines the temporary directory location (typically `/tmp`), equivalent to Python's `tempfile.gettempdir()`.
- **Takes effect automatically when cache is disabled**: Once caching is disabled, `temp_dir` is applied as the global temporary directory for the entire process via Python's `tempfile.tempdir`, affecting all temporary files created through `tempfile` in the process.
- **Cache compression is automatically disabled**: When caching is disabled, `cache_compress` is automatically ignored and reset to `null`.

### Configuration

```yaml
use_cache: false
temp_dir: /data/dj-temp    # Custom temp directory; null means system default
```

```bash
dj-process --config config.yaml --use_cache false --temp_dir /data/dj-temp
```

### Safety Notes

> **Set `temp_dir` with caution — an unsafe path can cause unexpected program behavior.**

- **Do not point to critical system directories** (e.g., `/`, `/usr`, `/etc`). Automatic cleanup of temporary files may accidentally delete important files.
- **Do not point to directories containing important data**. Temporary file writes and cleanup operations may conflict with existing files.
- **Ensure sufficient disk space**: When cache is disabled, intermediate files are written and deleted dynamically during processing. Peak disk usage is approximately equal to the output size of a single operator.
- **The directory is created automatically if it does not exist**: DataJuicer calls `os.makedirs` to create the specified path if it is missing.
- **`temp_dir` affects the entire process's `tempfile` behavior**: Because it sets the global `tempfile.tempdir` variable, this setting influences all components in the process that rely on `tempfile`, including third-party libraries.

## Performance Considerations

### When to Enable Cache

- **Enable**: For iterative development where you frequently re-run pipelines with minor changes
- **Enable**: When operators are computationally expensive and you want to skip already-computed steps
- **Disable**: For one-shot processing to avoid disk overhead

### When to Enable Compression

- **Enable**: When dataset size exceeds tens of GB and disk space is limited
- **Enable** `zstd`: For the best balance of speed and compression ratio
- **Enable** `lz4`: When compression speed is critical
- **Disable**: When disk space is abundant and you want maximum processing speed

## Troubleshooting

**Cache files consuming too much disk space:**
```bash
# Check cache directory size
du -sh ~/.cache/data_juicer/

# Enable compression
dj-process --config config.yaml --cache_compress zstd
```

**Stale cache causing unexpected results:**
```bash
# Clear HuggingFace dataset cache
rm -rf ~/.cache/huggingface/datasets/

# Or specify a fresh cache directory
dj-process --config config.yaml --ds_cache_dir /tmp/fresh-cache
```

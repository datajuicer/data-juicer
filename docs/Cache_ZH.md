# 缓存管理

本文档描述 DataJuicer 的缓存管理系统，包括 HuggingFace 数据集缓存、缓存目录配置、缓存压缩和临时存储。

## 概述

DataJuicer 提供了基于 HuggingFace Datasets 的缓存机制来避免重复计算。启用后，每个算子根据以下内容生成具有唯一 **fingerprint（指纹）** 的缓存文件：
- 输入数据的指纹
- 算子名称和参数
- 处理函数的哈希值

即：相同输入 + 相同算子配置 = 相同指纹 = 缓存命中。因此在相同数据上重新运行相同的管道时会跳过已计算的步骤。更多细节请参考我们的论文：[Data-Juicer: A One-Stop Data Processing System for Large Language Models](https://arxiv.org/abs/2309.02033) 。

缓存系统还提供：
- **可配置的缓存目录**，通过环境变量或配置选项设置
- **缓存压缩**，减少大规模数据集的磁盘占用
- **临时存储**，用于非缓存模式下的中间文件
- **细粒度缓存控制**，通过上下文管理器和装饰器实现

## 配置

### 基本缓存设置

```yaml
use_cache: true           # 启用/禁用 HuggingFace 数据集缓存
ds_cache_dir: null         # 自定义缓存目录（覆盖 HF_DATASETS_CACHE）
cache_compress: null       # 压缩方法：'gzip'、'zstd'、'lz4' 或 null
temp_dir: null             # 缓存禁用时的中间文件临时目录
```

### 命令行

```bash
# 启用缓存（默认）
dj-process --config config.yaml --use_cache true

# 禁用缓存
dj-process --config config.yaml --use_cache false

# 启用缓存压缩
dj-process --config config.yaml --cache_compress zstd

# 自定义缓存目录
dj-process --config config.yaml --ds_cache_dir /fast-storage/dj-cache
```

## 缓存目录结构

DataJuicer 通过环境变量控制的层级目录结构来组织缓存文件：

```
~/.cache/                              # CACHE_HOME（默认）
└── data_juicer/                       # DATA_JUICER_CACHE_HOME
    ├── assets/                        # DATA_JUICER_ASSETS_CACHE
    │   └── （提取的帧、停用词、标记词等）
    └── models/                        # DATA_JUICER_MODELS_CACHE
        └── （下载的模型文件）
```

### 环境变量

| 变量 | 默认值 | 描述 |
|------|--------|------|
| `CACHE_HOME` | `~/.cache` | 根缓存目录 |
| `DATA_JUICER_CACHE_HOME` | `$CACHE_HOME/data_juicer` | DataJuicer 缓存根目录 |
| `DATA_JUICER_ASSETS_CACHE` | `$DATA_JUICER_CACHE_HOME/assets` | 资产缓存（帧、词表等） |
| `DATA_JUICER_MODELS_CACHE` | `$DATA_JUICER_CACHE_HOME/models` | 下载的模型缓存 |
| `DATA_JUICER_EXTERNAL_MODELS_HOME` | `None` | 外部模型目录 |

通过设置环境变量覆盖默认值：

```bash
export DATA_JUICER_CACHE_HOME=/data/dj-cache
export DATA_JUICER_MODELS_CACHE=/models/dj-models
dj-process --config config.yaml
```

## 缓存压缩

对于大规模数据集（数十 GB 或更大），缓存文件可能占用大量磁盘空间。缓存压缩通过在每个算子完成后压缩中间缓存文件来减少存储需求。

### 支持的算法

| 算法 | 依赖库 | 速度 | 压缩率 | 推荐场景 |
|------|--------|------|--------|----------|
| `zstd` | zstandard | 快 | 高 | 通用场景（默认） |
| `lz4` | lz4 | 最快 | 中等 | 速度敏感的工作负载 |
| `gzip` | gzip | 慢 | 高 | 需要兼容性的场景 |

### 配置

```yaml
use_cache: true
cache_compress: zstd    # 启用 zstd 压缩
```

```bash
dj-process --config config.yaml --cache_compress zstd
```

### 多进程压缩

缓存压缩支持并行处理。压缩工作进程数由 `np` 参数控制：

```yaml
np: 4                   # 并行工作进程数（也用于压缩）
cache_compress: zstd
```

## 缓存控制 API

### DatasetCacheControl

上下文管理器，用于在特定范围内临时启用或禁用 HuggingFace 数据集缓存：

```python
from data_juicer.utils.cache_utils import DatasetCacheControl

# 临时禁用缓存
with DatasetCacheControl(on=False):
    # 此处的操作不会使用缓存
    result = dataset.map(my_function)

# 临时启用缓存
with DatasetCacheControl(on=True):
    # 此处的操作会使用缓存
    result = dataset.map(my_function)
```

### dataset_cache_control 装饰器

用于需要控制缓存状态的函数的装饰器：

```python
from data_juicer.utils.cache_utils import dataset_cache_control

@dataset_cache_control(on=False)
def process_without_cache(dataset):
    return dataset.map(my_function)
```

### CompressionOff

上下文管理器，用于临时禁用缓存压缩：

```python
from data_juicer.utils.compress import CompressionOff

with CompressionOff():
    # 此范围内缓存压缩被禁用
    result = dataset.map(my_function)
```

### CompressManager

底层 API，用于手动压缩/解压：

```python
from data_juicer.utils.compress import CompressManager

manager = CompressManager(compressor_format="zstd")

# 压缩文件
manager.compress("input.arrow", "input.arrow.zstd")

# 解压文件
manager.decompress("input.arrow.zstd", "input.arrow")
```

### CacheCompressManager

高层 API，用于管理 HuggingFace 数据集缓存压缩：

```python
from data_juicer.utils.compress import CacheCompressManager

manager = CacheCompressManager(compressor_format="zstd")

# 压缩前一个数据集的缓存文件
manager.compress(prev_ds=previous_dataset, this_ds=current_dataset, num_proc=4)

# 解压数据集的缓存文件
manager.decompress(ds=dataset, num_proc=4)

# 清理所有压缩的缓存文件
manager.cleanup_cache_files(ds=dataset)
```

## 缓存与检查点

缓存和检查点是互斥的——启用检查点会自动禁用缓存：

| 特性 | 缓存 | 检查点 |
|------|------|--------|
| **用途** | 加速相同配置的重复运行 | 故障恢复和断点续跑 |
| **粒度** | 每个算子的结果 | 完整数据集快照 |
| **存储位置** | HuggingFace 缓存目录 | 工作目录 |
| **恢复方式** | 自动（基于哈希） | 手动（基于配置） |
| **压缩** | 支持（`cache_compress`） | 不适用 |
| **场景** | 迭代开发、参数调优 | 长时间运行的生产任务 |

```yaml
# 缓存模式（默认）
use_cache: true
use_checkpoint: false

# 检查点模式（缓存自动禁用）
use_cache: true           # 将被覆盖为 false
use_checkpoint: true
```

## 缓存禁用与临时目录

当 `use_cache: false` 或启用检查点（`use_checkpoint: true`）时，HuggingFace 数据集缓存会被完全禁用。此时，DataJuicer 会将算子处理过程中产生的中间文件写入一个临时目录，并在处理完成后自动清理。`temp_dir` 参数用于指定这些中间文件的存放位置。

### 行为说明

- **默认值为 `null`**：此时由操作系统决定临时目录位置（通常为 `/tmp`），等价于 Python 的 `tempfile.gettempdir()` 返回值。
- **禁用缓存时自动生效**：只要缓存被禁用，`temp_dir` 就会被设置为 Python `tempfile` 模块的全局临时目录（`tempfile.tempdir`），影响整个进程中所有通过 `tempfile` 创建的临时文件。
- **缓存压缩自动禁用**：禁用缓存后，`cache_compress` 配置会被自动忽略并置为 `null`。

### 配置示例

```yaml
use_cache: false
temp_dir: /data/dj-temp    # 指定临时目录，null 则由系统决定
```

```bash
dj-process --config config.yaml --use_cache false --temp_dir /data/dj-temp
```

### 安全须知

> **请谨慎设置 `temp_dir`，错误的路径可能导致不可预期的程序行为。**

- **不要指向系统关键目录**（如 `/`、`/usr`、`/etc`），因为临时文件的自动清理可能误删重要文件。
- **不要指向已有重要数据的目录**，临时文件写入和清理操作可能与现有文件产生冲突。
- **确保目录有足够的磁盘空间**：禁用缓存时，中间文件会在处理过程中动态写入和删除，峰值占用约等于单个算子输出的数据量。
- **目录不存在时会自动创建**：若指定路径不存在，DataJuicer 会自动调用 `os.makedirs` 创建该目录。

## 性能考虑

### 何时启用缓存

- **启用**：迭代开发中频繁重新运行管道且仅有少量更改时
- **启用**：算子计算成本高，希望跳过已计算步骤时
- **禁用**：一次性处理，避免磁盘开销

### 何时启用压缩

- **启用**：数据集大小超过数十 GB 且磁盘空间有限时
- **启用** `zstd`：速度和压缩率的最佳平衡
- **启用** `lz4`：压缩速度至关重要时
- **禁用**：磁盘空间充足且希望最大化处理速度时

## 故障排除

**缓存文件占用过多磁盘空间：**
```bash
# 检查缓存目录大小
du -sh ~/.cache/data_juicer/

# 启用压缩
dj-process --config config.yaml --cache_compress zstd
```

**过期缓存导致意外结果：**
```bash
# 清除 HuggingFace 数据集缓存
rm -rf ~/.cache/huggingface/datasets/

# 或指定一个新的缓存目录
dj-process --config config.yaml --ds_cache_dir /tmp/fresh-cache
```
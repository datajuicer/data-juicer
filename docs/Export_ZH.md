# 数据集导出

本文档描述 DataJuicer 如何导出处理后的数据集，包括支持的格式、分片、并行导出、S3 导出以及统计信息/哈希管理。

## 概述

处理完成后，DataJuicer 使用 `Exporter`（默认模式）或 `RayExporter`（Ray 模式）将结果数据集导出到磁盘。导出系统支持：

- **多种输出格式** — JSONL、JSON、Parquet，Ray 模式下支持更多格式
- **分片导出** — 按大小将大型数据集拆分为多个文件
- **并行导出** — 使用多进程加速单文件导出
- **S3 导出** — 将结果直接写入 Amazon S3 或 S3 兼容存储
- **统计信息和哈希管理** — 控制输出中保留哪些中间字段

## 配置

### 基本设置

```yaml
export_path: ./outputs/result.jsonl       # 输出文件路径（必需）
export_type: jsonl                         # 格式类型（省略时从路径自动检测）
export_shard_size: 0                       # 分片大小（字节），0 = 单文件
export_in_parallel: false                  # 单文件模式下的并行导出
keep_stats_in_res_ds: false                # 在输出中保留计算的统计信息
keep_hashes_in_res_ds: false               # 在输出中保留计算的哈希值
export_extra_args: {}                      # 额外的格式特定参数
export_aws_credentials: null               # S3 导出专用，详见 S3 导出章节
```

### 命令行

```bash
# 基本导出
dj-process --config config.yaml --export_path ./outputs/result.jsonl

# 导出为 Parquet
dj-process --config config.yaml --export_path ./outputs/result.parquet

# 分片导出（每片 256MB）
dj-process --config config.yaml --export_shard_size 268435456

# 在输出中保留统计信息
dj-process --config config.yaml --keep_stats_in_res_ds true
```

## 支持的格式

### 默认模式（Exporter）

| 格式 | 后缀 | 描述 |
|------|------|------|
| JSONL | `.jsonl` | JSON Lines — 每行一个 JSON 对象（默认） |
| JSON | `.json` | 标准 JSON 数组 |
| Parquet | `.parquet` | 列式格式，适合大型数据集 |

### Ray 模式（RayExporter）

| 格式 | 后缀 | 描述 |
|------|------|------|
| JSONL | `.jsonl` | JSON Lines |
| JSON | `.json` | 标准 JSON |
| Parquet | `.parquet` | 列式格式 |
| CSV | `.csv` | 逗号分隔值 |
| TFRecords | `.tfrecords` | TensorFlow 记录格式 |
| WebDataset | `webdataset` | WebDataset tar 格式 |
| Lance | `.lance` | Lance 列式格式 |

## 分片导出

对于大型数据集，按大小将输出拆分为多个分片文件：

```yaml
export_path: ./outputs/result.jsonl
export_shard_size: 268435456              # 每片 256 MB
```

生成的文件如下：
```
outputs/
├── result-00-of-04.jsonl
├── result-01-of-04.jsonl
├── result-02-of-04.jsonl
└── result-03-of-04.jsonl
```

**分片大小计算方式：**
1. 估算数据集的总字节大小
2. 分片数 = `ceil(dataset_bytes / export_shard_size)`
3. 数据集被拆分为连续的分片
4. 每个分片使用多进程并行导出

**推荐的分片大小：**

| 数据集大小 | 推荐分片大小 | 说明 |
|-----------|-------------|------|
| < 1 GB | 0（单文件） | 无需分片 |
| 1-10 GB | 256 MB - 512 MB | 良好平衡 |
| 10-100 GB | 512 MB - 1 GB | 更少文件 |
| > 100 GB | 1 GB - 10 GB | 避免过多分片 |

分片大小低于 1 MiB 或高于 1 TiB 将触发警告。

## 并行导出

对于单文件导出（`export_shard_size: 0`），启用并行写入以加速导出过程：

```yaml
export_path: ./outputs/result.jsonl
export_shard_size: 0
export_in_parallel: true
np: 4                                     # 并行进程数
```

**重要提示**：并行导出有时可能比顺序导出**更慢**，因为 IO 阻塞，特别是对于非常大的数据集。如果观察到这种情况，请设置 `export_in_parallel: false`。

当 `export_shard_size > 0` 时，无论此设置如何，分片始终并行导出。

## S3 导出

将结果直接导出到 Amazon S3 或 S3 兼容存储。

### 默认模式

```yaml
export_path: "s3://my-bucket/outputs/result.jsonl"
export_aws_credentials:
  aws_access_key_id: "AKIA..."
  aws_secret_access_key: "secret..."
  aws_region: "us-east-1"
  endpoint_url: "https://s3.example.com"   # 可选：用于 S3 兼容存储
```

默认导出器使用 HuggingFace 的 `storage_options` 配合 `fsspec`/`s3fs` 进行 S3 访问。

### Ray 模式

```yaml
export_path: "s3://my-bucket/outputs/result.jsonl"
export_extra_args:
  aws_access_key_id: "AKIA..."
  aws_secret_access_key: "secret..."
  aws_region: "us-east-1"
```

Ray 导出器使用 PyArrow 的 S3 文件系统进行 S3 访问。

### S3 分片导出

使用 S3 进行分片导出时，分片文件直接写入 S3：

```yaml
export_path: "s3://my-bucket/outputs/result.jsonl"
export_shard_size: 268435456
export_aws_credentials:
  aws_access_key_id: "AKIA..."
  aws_secret_access_key: "secret..."
```

生成的 S3 对象如下：
```
s3://my-bucket/outputs/result-00-of-04.jsonl
s3://my-bucket/outputs/result-01-of-04.jsonl
...
```

### 凭证解析

AWS 凭证按以下优先级解析：
1. `export_aws_credentials` 配置（默认模式）或 `export_extra_args`（Ray 模式）
2. 环境变量（`AWS_ACCESS_KEY_ID`、`AWS_SECRET_ACCESS_KEY`）
3. 默认凭证链（IAM 角色、`~/.aws/credentials`）

## 统计信息和哈希管理

在处理过程中，DataJuicer 会计算中间字段：
- **统计信息**（`__dj__stats__`、`__dj__meta__`）：由 Filter 算子计算
- **哈希值**（`__dj__hash__`、`__dj__minhash__`、`__dj__simhash__` 等）：由 Deduplicator 算子计算

默认情况下，这些字段会从导出的数据集中**移除**。要保留它们：

```yaml
keep_stats_in_res_ds: true                # 保留统计信息和元数据字段
keep_hashes_in_res_ds: true               # 保留哈希字段
```

### 统计信息导出

无论 `keep_stats_in_res_ds` 如何设置，DataJuicer 始终会在主数据集旁边导出一个单独的统计信息文件：

```
outputs/
├── result.jsonl                          # 主数据集（默认移除统计信息）
└── result_stats.jsonl                    # 仅统计信息文件（始终导出）
```

统计信息文件仅包含 `__dj__stats__` 和 `__dj__meta__` 列。

## WebDataset 导出（Ray 模式）

在 Ray 模式下，可以使用自定义字段映射导出为 WebDataset 格式：

```yaml
export_path: ./outputs/webdataset
export_type: webdataset
export_extra_args:
  field_mapping:
    txt: "text"
    png: "images"
    json: "metadata"
```

## API 参考

### Exporter（默认模式）

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

### RayExporter（Ray 模式）

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

## 故障排除

**导出格式不支持：**
```bash
# 检查支持的格式
# 默认模式：jsonl, json, parquet
# Ray 模式：jsonl, json, parquet, csv, tfrecords, webdataset, lance
```

**并行导出比预期慢：**
```yaml
# 禁用并行导出
export_in_parallel: false
```

**S3 导出权限错误：**
```bash
# 验证凭证
aws s3 ls s3://your-bucket/

# 检查 export_aws_credentials 是否已配置
```

**生成的分片文件过多：**
```yaml
# 增大分片大小
export_shard_size: 1073741824             # 1 GB
```

**导出的数据集中缺少统计信息：**
```yaml
# 在结果数据集中保留统计信息
keep_stats_in_res_ds: true
# 或检查单独的统计信息文件：result_stats.jsonl
```

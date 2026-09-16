# 数据集导出

处理完成后，Data-Juicer 将结果数据集写入你在 `export_path` 中指定的路径。本页介绍支持的输出格式、将大数据集分片为多个文件、并行导出、直接写入 S3，以及控制哪些中间字段（统计信息、哈希）保留在输出中。

## 概述

Data-Juicer 通过 `Exporter`（默认模式）或 `RayExporter`（Ray 模式）导出。导出系统支持：

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
export_shard_size: 0                       # 本地模式：写入单文件；Ray：按数据块布局写入
export_in_parallel: false                  # 本地模式下并行写入单文件
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

在本地模式下，`export_path` 是文件路径。请提供 `.jsonl`、`.json` 或 `.parquet` 扩展名，或显式设置 `export_type`。

在 Ray 模式下，`export_path` 是存放输出文件的目录，即使路径以 `.jsonl` 结尾也是如此。Ray 按数据块布局写入文件；`export_shard_size: 0` 使用默认布局。

## 分片导出（本地模式）

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

Data-Juicer 估算数据集大小，将连续的数据行分成多个分片，再使用多个进程写入。配置的大小是目标值，编码后的文件可能更大或更小。

**推荐的分片大小：**

| 数据集大小 | 推荐分片大小 | 说明 |
|-----------|-------------|------|
| < 1 GB | 0（单文件） | 无需分片 |
| 1-10 GB | 256 MB - 512 MB | 良好平衡 |
| 10-100 GB | 512 MB - 1 GB | 更少文件 |
| > 100 GB | 1 GB - 10 GB | 避免过多分片 |

分片大小低于 1 MiB 或高于 1 TiB 将触发警告。

## 并行导出（本地模式）

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

本地和 Ray 模式都可以直接将结果写入 S3。先设置 `export_path`，例如：

```yaml
export_path: "s3://my-bucket/outputs/result.jsonl"
```

可以通过环境变量提供凭证：

```bash
export AWS_ACCESS_KEY_ID="your-access-key-id"
export AWS_SECRET_ACCESS_KEY="your-secret-access-key"
export AWS_DEFAULT_REGION="us-east-1"
```

使用临时凭证时，还需要设置 `AWS_SESSION_TOKEN`。也可以在配方的 `export_aws_credentials` 中提供凭证；本地和 Ray 模式均支持此配置：

```yaml
export_path: "s3://my-bucket/outputs/result.jsonl"
export_aws_credentials:
  aws_access_key_id: "your-access-key-id"
  aws_secret_access_key: "your-secret-access-key"
  aws_region: "us-east-1"
  endpoint_url: "https://s3.example.com"  # 使用 S3 兼容存储时设置
```

访问密钥、会话令牌和区域按字段依次从环境变量、显式配置读取。未提供访问密钥时，存储客户端会使用默认 AWS 凭证链，例如 IAM 角色或本地凭证文件。

Ray 模式也接受 `export_extra_args` 中的凭证；同名配置以 `export_aws_credentials` 为准。本地模式通过 s3fs 访问 S3，Ray 模式通过 PyArrow 访问 S3。

在本地模式中，设置 `export_shard_size: 268435456` 可按约 256 MiB 的目标大小分片，输出对象形如 `result-00-of-04.jsonl`。Ray 模式将文件写入 `export_path` 对应的目录前缀。

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

本地模式下，数据包含 `__dj__stats__` 或 `__dj__meta__` 列时，会额外导出一份统计文件：

```text
outputs/
├── result.jsonl
└── result_stats.jsonl
```

统计文件只包含数据中已有的统计和元数据列。直接使用 Python `Exporter` 时，可以通过 `export_stats=False` 关闭该文件的导出。

Ray 模式将统计信息与主数据集一起导出。需要保留它们时，请设置 `keep_stats_in_res_ds: true`；Ray 不会额外生成独立的 `_stats.jsonl` 文件。

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
# 本地模式也可查看独立的统计文件：result_stats.jsonl
```

---

## 下一步

- [缓存管理](Cache_ZH.md)——通过缓存中间结果加速重复运行。
- [数据追踪](Tracing_ZH.md)——调试流水线中样本级别的变化。
- [分布式处理](Distributed_ZH.md)——在 Ray 集群上扩展导出。

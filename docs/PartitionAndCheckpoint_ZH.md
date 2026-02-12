# 分区处理与检查点

本文档描述 DataJuicer 的容错处理系统，包括分区、检查点和事件日志。

## 概述

`ray_partitioned` 执行器将数据集分割成分区，并使用可配置的检查点进行处理。失败的作业可以从最后一个检查点恢复。

**检查点策略：**
- `every_n_ops` - 每 N 个操作检查点（默认，平衡方案）
- `every_op` - 每个操作后检查点（最高容错性，影响性能）
- `manual` - 仅在指定操作后检查点（适合已知的耗时操作）
- `disabled` - 不检查点（最佳性能）

## 目录结构

```
{work_dir}/{job_id}/
├── job_summary.json              # 作业元数据（完成时创建）
├── events_{timestamp}.jsonl      # 机器可读事件日志
├── dag_execution_plan.json       # DAG 执行计划
├── checkpoints/                  # 检查点数据
├── partitions/                   # 输入分区
├── logs/                         # 人类可读日志
└── metadata/                     # 作业元数据
```

## 配置

### 分区模式

**自动模式**（推荐）- 分析数据和资源以确定最佳分区：

```yaml
executor_type: ray_partitioned

partition:
  mode: "auto"
  target_size_mb: 256    # 目标分区大小（128、256、512 或 1024）
  size: 5000             # 自动分析失败时的回退值
  max_size_mb: 256       # 回退最大大小
```

**手动模式** - 指定确切的分区数量：

```yaml
partition:
  mode: "manual"
  num_of_partitions: 8
```

### 检查点

```yaml
checkpoint:
  enabled: true
  strategy: every_n_ops  # every_n_ops（默认）, every_op, manual, disabled
  n_ops: 5               # 默认：每 5 个操作检查点
  op_names:              # 用于 manual 策略 - 在耗时操作后检查点
    - document_deduplicator
    - embedding_mapper
```

### 中间存储

```yaml
intermediate_storage:
  format: "parquet"              # parquet, arrow, jsonl
  compression: "snappy"          # snappy, gzip, none
  preserve_intermediate_data: true
  retention_policy: "keep_all"   # keep_all, keep_failed_only, cleanup_all
```

## 使用方法

### 运行作业

```bash
# 自动分区模式
dj-process --config config.yaml --partition.mode auto

# 手动分区模式
dj-process --config config.yaml --partition.mode manual --partition.num_of_partitions 4

# 自定义作业 ID
dj-process --config config.yaml --job_id my_experiment_001
```

### 恢复作业

```bash
dj-process --config config.yaml --job_id my_experiment_001
```

### 检查点策略

```bash
# 每个操作
dj-process --config config.yaml --checkpoint.strategy every_op

# 每 N 个操作
dj-process --config config.yaml --checkpoint.strategy every_n_ops --checkpoint.n_ops 3

# 手动
dj-process --config config.yaml --checkpoint.strategy manual --checkpoint.op_names op1,op2
```

## 自动配置

在自动模式下，优化器会：
1. 采样数据集以检测模态（文本、图像、音频、视频、多模态）
2. 测量每个样本的内存使用
3. 分析管道复杂性
4. 计算目标为配置的 `target_size_mb` 的分区大小

按模态的默认分区大小：

| 模态 | 默认大小 | 最大大小 | 内存倍数 |
|------|----------|----------|----------|
| 文本 | 10000 | 50000 | 1.0x |
| 图像 | 2000 | 10000 | 5.0x |
| 音频 | 1000 | 4000 | 8.0x |
| 视频 | 400 | 2000 | 20.0x |
| 多模态 | 1600 | 6000 | 10.0x |

## 作业管理工具

### 监控器

```bash
# 显示进度
python -m data_juicer.utils.job.monitor {job_id}

# 详细视图
python -m data_juicer.utils.job.monitor {job_id} --detailed

# 监视模式
python -m data_juicer.utils.job.monitor {job_id} --watch --interval 10
```

```python
from data_juicer.utils.job.monitor import show_job_progress

data = show_job_progress("job_id", detailed=True)
```

### 停止器

```bash
# 优雅停止
python -m data_juicer.utils.job.stopper {job_id}

# 强制停止
python -m data_juicer.utils.job.stopper {job_id} --force

# 列出运行中的作业
python -m data_juicer.utils.job.stopper --list
```

```python
from data_juicer.utils.job.stopper import stop_job

stop_job("job_id", force=True, timeout=60)
```

### 通用工具

```python
from data_juicer.utils.job.common import JobUtils, list_running_jobs

running_jobs = list_running_jobs()

job_utils = JobUtils("job_id")
summary = job_utils.load_job_summary()
events = job_utils.load_event_logs()
```

## 事件类型

- `job_start`, `job_complete`, `job_failed`
- `partition_start`, `partition_complete`, `partition_failed`
- `op_start`, `op_complete`, `op_failed`
- `checkpoint_save`, `checkpoint_load`

## 性能考虑

### 检查点与 Ray 优化的权衡

**关键洞察：检查点会干扰 Ray 的自动优化。**

Ray 通过融合操作和流水线处理数据来优化执行。每个检查点都会强制物化，从而打破优化窗口：

```
无检查点：          op1 → op2 → op3 → op4 → op5
                    |___________________________|
                         Ray 优化整个窗口

every_op：          op1 | op2 | op3 | op4 | op5
                    每个 | 处物化（5 个屏障）

every_n_ops(5)：    op1 → op2 → op3 → op4 → op5 |
                    |_____________________________|
                         Ray 优化全部 5 个操作
```

### 检查点成本分析

| 成本类型 | 典型值 |
|----------|--------|
| 检查点写入 | ~2-5 秒 |
| 轻量操作执行 | ~1-2 秒 |
| 耗时操作执行 | 分钟到小时 |

**对于轻量操作，检查点的成本比失败后重新执行更高。**

管道分析示例：
```
filter(1秒) → mapper(2秒) → deduplicator(300秒) → filter(1秒)

策略              | 开销    | 保护价值
------------------|---------|------------------
every_op          | ~20秒   | 失败时节省 1-304秒
仅在 dedup 后     | ~5秒    | 失败时节省 300秒
disabled          | 0秒     | 重新执行全部
```

### 策略建议

| 作业时长 | 建议策略 | 理由 |
|----------|----------|------|
| < 10 分钟 | `disabled` | 重新执行成本低 |
| 10-60 分钟 | `every_n_ops` (n=5) | 平衡保护 |
| > 60 分钟且有耗时操作 | `manual` | 仅在耗时操作后检查点 |
| 不稳定的基础设施 | `every_n_ops` (n=2-3) | 接受开销换取可靠性 |

### 操作分类

**耗时操作（建议在这些操作后检查点）：**
- `*_deduplicator` - 全局状态，计算耗时
- `*_embedding_*` - 模型推理
- `*_model_*` - 模型推理
- `*_vision_*` - 图像/视频处理
- `*_audio_*` - 音频处理

**轻量操作（可跳过检查点）：**
- `*_filter` - 简单过滤
- `clean_*` - 文本清理
- `remove_*` - 字段移除

### 存储建议

- 事件日志：快速存储（SSD）
- 检查点：大容量存储
- 分区：本地存储

### 分区大小权衡

- 较小分区：更好的容错性，更多调度开销
- 较大分区：更少开销，更粗粒度的恢复

## 故障排除

**作业恢复失败：**
```bash
ls -la ./outputs/{work_dir}/{job_id}/job_summary.json
ls -la ./outputs/{work_dir}/{job_id}/checkpoints/
```

**检查 Ray 状态：**
```bash
ray status
```

**查看日志：**
```bash
cat ./outputs/{work_dir}/{job_id}/events_*.jsonl
tail -f ./outputs/{work_dir}/{job_id}/logs/*.txt
```

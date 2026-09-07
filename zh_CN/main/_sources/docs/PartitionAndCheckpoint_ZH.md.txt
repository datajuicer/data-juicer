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
│   ├── partitioning_info.json    # 保存的行号边界和分区内容 hash
│   └── checkpoint_op_*.parquet/  # 各操作、各分区的检查点
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
  max_concurrent_partitions: "auto"  # 资源感知的 Driver 并发上限
  target_size_mb: 256    # 自动模式规划使用的目标大小（MB）
  size: null             # 可选：优化器无法给出有效建议时使用的样本数回退值
```

**手动模式** - 指定确切的分区数量：

```yaml
partition:
  mode: "manual"
  num_of_partitions: 8
  max_concurrent_partitions: "auto"
```

手动模式也可以通过样本数目标推导最接近的分区数量。手动模式下，`size` 与
`num_of_partitions` 互斥。执行器会物化一次输入数据，再按照行边界切分。除最后
一个分区吸收按最接近数量规划产生的余数外，其余分区各包含 `size` 条样本。如果
推导结果为一个分区，则整个数据集作为一个已物化分区处理：

```yaml
partition:
  mode: "manual"
  size: 5000
```

`target_size_mb` 是优化器的规划输入，并非内存或输出文件大小的硬限制。该参数
接受 0 和负数；如果据此算出的建议低于优化器内部下限，Data-Juicer 会应用下限
并记录 warning。

`max_concurrent_partitions: "auto"` 是默认值。该值会在 Operator 资源规划完成后解析：含 GPU Operator 的 pipeline 根据 Ray 集群可容纳的最小 CPU/GPU worker 数确定，纯 CPU pipeline 的外层并发保守限制为 4。实际并发还会受到 Partition 数量和显式全局 Actor `num_proc` 预算的限制。需要手动调优时，可将其设置为正整数来覆盖自动值。

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

`strategy` 只接受 `every_op`、`every_n_ops`、`manual` 和 `disabled`，其他取值（包括 `every_partition`）会在解析配置时被拒绝。`n_ops` 必须为正整数。

启用检查点后，首次运行会保存
`checkpoints/partitioning_info.json`。该文件为每个逻辑分区记录：

- 输入数据中的 `start_row`（包含）和 `end_row`（不包含）；
- 分区样本行数；
- 覆盖完整分区内容的稳定 hash；
- 写入 metadata 时使用的分区 hash 算法。

即使新进程中的 Ray 物理 block 布局发生变化，显式续跑也可以用这些信息重建首次运行的逻辑分区。完整分区 hash 对样本顺序敏感、不依赖 Ray batch 边界，并且会在复用任何 checkpoint 前完成校验。

### 检查点与临时文件

检查点由 `checkpoint.enabled`、`checkpoint.strategy`、`checkpoint.n_ops` 和 `checkpoint.op_names` 控制，保存为 Parquet，压缩使用底层写入器的默认设置。分区通过 Ray Dataset 拆分；检查点记录所选算子执行后的数据。

执行器退出运行上下文时会尝试清理 `work_dir/.tmp/<Ray job id>`，正常完成和异常退出都会触发。检查点保存在单独的目录中，可用于续跑。

## 使用方法

### 运行作业

```bash
# 自动分区模式
dj-process --config config.yaml --partition.mode auto

# 手动分区模式
dj-process --config config.yaml --partition.mode manual --partition.num_of_partitions 4

# 可选：使用自定义作业 ID 启动新任务
dj-process --config config.yaml --job_id my_experiment_001
```

启用检查点后，新建的 `ray_partitioned` 作业会打印 resume token：

```text
Resume token: 20260805_115141_81270d. Rerun the original command with
--resume 20260805_115141_81270d to resume this job.
```

### 恢复作业

```bash
# 使用首次运行打印的 token，并保持输入数据和 recipe 不变
dj-process --config config.yaml --resume 20260805_115141_81270d

# 首次运行使用的自定义 ID 也可以作为 resume token
dj-process --config config.yaml --resume my_experiment_001
```

`--resume` 仅支持 `ray_partitioned` 执行器。它会依次执行严格续跑流程：

1. 定位原任务的工作目录和检查点目录；
2. 确认当前配置与首次运行配置一致；
3. 读取保存的分区数、行号边界和内容 hash；
4. 使用保存的行号边界重建首次运行的逻辑分区；
5. 校验所有分区的完整内容 hash；
6. 加载已完成的 checkpoint，仅处理尚未完成的部分。

如果 metadata 缺失、行号边界非法、输入内容发生变化或内容 hash 不一致，显式续跑会报错停止，并保留已有 checkpoint，不会将其删除。

`--job_id` 仍可用于自定义任务名称和向后兼容。需要进行容错续跑时，应优先使用 `--resume`：旧的 `--job_id` 续跑路径保持原有行为，在分区不匹配时可能清除 checkpoint 并重新开始。旧版 Data-Juicer 创建的 metadata 仍可读取，但其中没有显式续跑所需的行号边界和完整内容 hash；因此，`--resume` 会拒绝使用这种旧 metadata，同时保留已有 checkpoint。

如果同时提供两个参数，它们的值必须相同：

```bash
dj-process --config config.yaml \
  --job_id my_experiment_001 \
  --resume my_experiment_001
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
cat ./outputs/{work_dir}/{job_id}/checkpoints/partitioning_info.json
```

请使用首次运行打印的 resume token，并通过 `--resume` 重新执行相同的 recipe。可以在错误日志中检查配置不一致、分区 metadata 缺失、行号边界非法或分区内容 hash 不匹配。显式续跑校验失败时不会删除已有 checkpoint。

**检查 Ray 状态：**
```bash
ray status
```

**查看日志：**
```bash
cat ./outputs/{work_dir}/{job_id}/events_*.jsonl
tail -f ./outputs/{work_dir}/{job_id}/logs/*.txt
```

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
├── gpu_probe_results.json         # 自动探测的 GPU Operator 资源
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
  max_gpu_workers_per_device: 5       # 每张 GPU 的保守模型副本上限
  max_concurrent_gpu_probes: "auto"   # 默认逐个探测，设置正整数才开启并发探测
  gpu_preflight_enabled: true          # false 时跳过 preflight，直接使用显式资源和 Actor 数
  gpu_probe_timeout_seconds: null     # 可选的单任务超时；null 表示不主动终止
  gpu_probe_warmup_batches: 1         # 稳态测速前的 warmup batch 数
  gpu_probe_steady_batches: 3         # 用于稳态吞吐估计的 batch 数
  gpu_probe_sample_offset: 0          # preflight 取样时跳过的开头行数
  gpu_probe_sample_shuffle: false     # true 时先随机化 block 顺序再取样
  gpu_probe_sample_seed: 42           # 随机取样的种子；null 表示每次运行都重新取样，因此永不复用报告
  execution_group_size: "auto"       # 共用一次 GPU Actor 生命周期的逻辑分区数
  max_initialization_overhead_ratio: 0.1  # 每个执行组允许的模型初始化时间占比
  target_size_mb: 256    # 目标分区大小（128、256、512 或 1024）
  size: 5000             # 自动分析失败时的回退值
  max_size_mb: 256       # 回退最大大小
```

**手动模式** - 指定确切的分区数量：

```yaml
partition:
  mode: "manual"
  num_of_partitions: 8
  max_concurrent_partitions: "auto"
```

分区按精确行号边界切分，各分区行数最多相差一行。这里特意没有使用 Ray 的 `Dataset.split(n)`：它按整块（block）分配，当数据块数少于 `num_of_partitions` 时会产生空的尾部分区（单块 10 行切 4 份会得到 `[10, 0, 0, 0]`），块大小不均时即使块数足够也会造成分区严重倾斜。空分区并非零成本——每个空分区仍会构建算子图、占用一次 Actor 生命周期并写出 checkpoint；而倾斜会破坏 execution group 定容所依赖的“各分区行数均匀”假设。当分区数超过总行数时，会在任何 Actor 启动前直接报错，而不是静默下调：分区数是续跑校验的依据，必须与请求值一致。

逻辑分区数与 GPU 执行并发现在相互独立：逻辑分区只决定 checkpoint 粒度、恢复边界和单分区数据上限；GPU Actor 数由 preflight 的稳态吞吐和集群资源统一规划。多个逻辑分区会组成 execution group，共用一次 Ray Actor pool 生命周期，但在每个 checkpoint 点仍按原逻辑分区分别落盘。`execution_group_size: "auto"` 会根据模型初始化时间、稳态吞吐、数据量和 `max_initialization_overhead_ratio` 选择组大小；显式正整数可覆盖该选择。续跑时可能存在不同 checkpoint 位置，因此会安全回退到逐分区串行执行。

自动 Actor 规划先给每个待规划 GPU 阶段分配一个 Actor，然后反复给当前 pipeline 瓶颈阶段增加 Actor。每次增加都必须同时满足集群 CPU、Ray GPU 调度份额、每张卡的实测显存份额、每卡 Actor 上限以及有效 batch 数上限。若连“每阶段一个 Actor”的最低方案都无法放入集群，会在正式任务启动前直接报错，不会通过增加分区或超配显存来规避。

#### GPU 显存预探测

若希望运行固定资源的对照实验，可设置 `gpu_preflight_enabled: false`。此时不会读取小样本、初始化一次性探测 Actor 或生成 `gpu_probe_results.json`；CUDA 算子的 `num_gpus`、`memory` 和固定 `num_proc` 应由 recipe 显式给出，并建议同时关闭 `auto_op_parallelism`。

在 `ray_partitioned` 模式下（包括手动分区），普通单卡 CUDA Mapper/Filter 会在正式实验启动前完成小样本 preflight。未配置 `memory`/`num_gpus` 的算子同时获得资源估计；显式值保持优先，但仍会测量吞吐用于自动 `num_proc` 规划：

1. 只读取输入开头的固定样本，数量为所有待探测 GPU Operator 的最大 `batch_size`；若数据按长度、分辨率或来源排序，开头样本不具代表性，实测显存和吞吐都会有偏差。设置 `gpu_probe_sample_shuffle: true` 会先随机化 block 顺序，从随机 block 而不是第一个 block 取样；该操作只重排 block 引用，preflight 开销仍与样本量成正比，而不是与数据集规模成正比。也可以用 `gpu_probe_sample_offset` 跳过已知不具代表性的开头。两者是二选一而不是叠加：shuffle 成功后样本已经离开开头，此时 offset 会被忽略并给出告警。修改任一选项都会使 `gpu_probe_results.json` 失效，因为来自数据集不同位置的测量结果不可比较；未设置 `gpu_probe_sample_seed` 的 shuffle 每次取样都不同，其报告只用于观测，不会被当作缓存复用；
2. Operator 可以通过 `input_columns` 和 `output_columns` 声明读写字段，支持 `__dj__meta__.quality_score` 这样的嵌套路径；执行器据此构建保守的数据依赖 DAG；
3. 能证明相互独立、且祖先只包含兼容 CPU Mapper/Filter 的目标可以并行探测。每个一次性 Ray worker 接收轻量原始样本，在 worker 内重放所需 CPU 祖先，并为目标独占一张 GPU；因此不会再把大型 NumPy 中间值经 Driver 往返。`max_concurrent_gpu_probes: "auto"` 默认逐个探测：并发探测意味着同时初始化多个模型，而 checkpoint 存储和主机内存带宽都不是 Ray 资源，自动策略无法判断节点是否扛得住。checkpoint 在本地或已知 I/O 有余量时，可以设置正整数开启并发；该值仍受依赖安全的 GPU/CPU 槽位约束；
   worker 会分别记录依赖重放和目标测量耗时，Driver 在每个任务完成时立即记录，并每 30 秒输出一次存活任务心跳；可通过 `gpu_probe_timeout_seconds` 让卡住的目标带算子名超时失败，默认 `null` 不主动终止；
4. 缺少字段契约、存在 GPU-to-GPU 依赖、runtime environment 不兼容或包含 Dataset 级 Operator 时，安全回退到原有 recipe 有序重放。若前序 Filter 导致样本不足，则循环补齐目标的一个 batch；
5. 在同一个一次性 worker 内只初始化一次算子，分别记录模型初始化、warmup 和多个稳态 batch 的耗时，并统计稳态输入吞吐与输出比例；默认 warmup 1 个 batch、测量 3 个 batch。依赖重放和目标测量都沿用 recipe 的 `skip_op_error`，少量非法样本会像正式运行一样被跳过，而不是在 preflight 阶段直接终止任务；但由于测量只重复同一个 batch，若该 batch 完全没有输出，则测量结果不可用：Mapper 返回空一定会带算子名让 preflight 失败（`skip_op_error` 开启时说明整批走了错误路径，关闭时说明算子本该抛异常却返回了空），Filter 全部过滤则只告警说明吞吐为最坏情况。preflight 不再设置 `DATA_JUICER_GPU_PREFLIGHT_OP` 标记（曾有算子据此增加 CUDA 同步，把一次正常的模型搬运拖成数分钟停顿），自定义算子不应再依赖读取该环境变量；
6. probe 不会在模型初始化期间从后台线程轮询 CUDA，以免 `cudaMemGetInfo` 与大量参数的 `model.to(cuda)` 竞争同一 CUDA context。一次性 worker 会像正式 Actor 一样让 Operator 自行初始化 CUDA，调用完成后合并 PyTorch allocator 的全过程峰值和设备持久占用（后者也覆盖 Paddle 等非 PyTorch runtime），再增加 10% 余量得到 `memory_fraction`；缺省资源时 Ray 使用的 `num_gpus` 为 `max(memory_fraction, 1 / max_gpu_workers_per_device)`，默认每张卡最多调度 5 个自动探测的模型 Actor；
7. 将测量值、调度值、分阶段耗时、吞吐、输出比例、探测模式、重放依赖以及样本实际来源保存到 `{work_dir}/gpu_probe_results.json`。Operator 配置、GPU 型号/容量、测速 batch 数、每卡 worker 上限或取样策略变化时会重新探测。原始 YAML 不会被覆盖。

例如，共享一个 CPU resize 的多个独立图像打标算子可以只在 recipe 中声明字段契约，无需修改 Python 类：

```yaml
process:
  - bucket_resize_mapper:
      input_columns: [images]
      output_columns: [_bucket_img]
  - image_quality_mapper:
      input_columns: [images, _bucket_img]
      output_columns: [__dj__meta__.quality_score]
  - image_rotation_mapper:
      input_columns: [images, _bucket_img]
      output_columns: [__dj__meta__.rotation_*]
```

省略字段契约表示“未知”，而不是“没有读写字段”；因此旧 recipe 会继续使用有序探测，直到相关 Operator 补齐契约。

显式配置的 `memory` 或 `num_gpus` 优先于实测值，但当显式 `num_gpus` 低于实测显存份额时会被抬升到该份额并给出告警：Ray 只按 `num_gpus` 打包 Actor，更小的请求会让它在单卡上放入超出显存容量的 Actor。当前 preflight 仅支持能装入单张 GPU 的普通 Mapper/Filter；GPU `Pipeline`、待探测 Operator 之前的 `Pipeline`，以及需要多张 GPU 的 Operator 必须显式配置资源和并发。空输入、探测异常、OOM 或无有效显存峰值都会在正式 partition worker 启动前终止任务。

### 检查点

```yaml
checkpoint:
  enabled: true
  strategy: every_n_ops  # every_n_ops（默认）, every_op, manual, disabled
  n_ops: 5               # 默认：每 5 个操作检查点
  op_names:              # 用于 manual 策略 - 在耗时操作后检查点
    - ray_document_deduplicator
    - extract_keyword_mapper
```

选择 `every_op`、`every_n_ops`、`manual` 或 `disabled` 作为保存策略。使用 `every_n_ops` 时，将 `n_ops` 设置为正整数；使用 `manual` 时，在 `op_names` 中填写配方里的算子名。

启用检查点后，首次运行会保存
`checkpoints/partitioning_info.json`。该文件为每个逻辑分区记录：

- 输入数据中的 `start_row`（包含）和 `end_row`（不包含）；
- 分区样本行数；
- 覆盖完整分区内容的稳定 hash；
- 写入 metadata 时使用的分区 hash 算法。

即使新进程中的 Ray 物理 block 布局发生变化，显式续跑也可以用这些信息重建首次运行的逻辑分区。完整分区 hash 对样本顺序敏感、不依赖 Ray batch 边界，并且会在复用任何 checkpoint 前完成校验。

### 检查点与临时文件

使用 `checkpoint.enabled` 开关检查点，用 `checkpoint.strategy` 选择保存时机。检查点以 Parquet 数据集的形式保存在 `checkpoint_dir`，默认目录为 `<work_dir>/checkpoints`。保留该目录即可恢复中断的作业。

运行退出时，执行器会清理临时工作文件。检查点单独存储，仍可用于恢复作业。

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

`--job_id` 用于指定作业名称。恢复作业时，在原命令上添加 `--resume`，并使用首次运行的作业 ID。

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

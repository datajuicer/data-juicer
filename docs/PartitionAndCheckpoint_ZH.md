# 分区处理与检查点

本文档描述 DataJuicer 的容错处理系统，包括分区、检查点和事件日志。

## 概述

`ray_partitioned` 执行器在 Ray Data 上处理数据集，并可以为每个算子保存检查点。失败的作业可以从已提交的结果继续。

**检查点策略：**
- `every_op` - 每个 Mapper/Filter batch 的输出都提交（默认）
- `disabled` - 不检查点（最佳性能）

## 目录结构

```
{work_dir}/{job_id}/
├── job_summary.json              # 作业元数据（完成时创建）
├── gpu_probe_results.json         # 自动探测的 GPU Operator 资源
├── events_{timestamp}.jsonl      # 机器可读事件日志
├── dag_execution_plan.json       # DAG 执行计划
├── checkpoints/                  # checkpoint_dir
│   └── partition_recovery/
│       ├── run.json              # 输入指纹和 recipe 身份
│       ├── commits/              # 每次提交一个不可变 manifest
│       ├── exports/              # 每次导出尝试一条记录
│       └── data/                 # manifest 引用的 Parquet 文件
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
  target_size_mb: 256    # 目标分区大小（128、256、512 或 1024）
  size: 5000             # 自动分析失败时的回退值；启用检查点时为每次提交的输入行数上限
  max_size_mb: 256       # 回退最大大小
```

**手动模式** - 指定确切的分区数量：

```yaml
partition:
  mode: "manual"
  num_of_partitions: 8
  max_concurrent_partitions: "auto"
```

分区按精确行号边界切分，各分区行数最多相差一行。这里特意没有使用 Ray 的 `Dataset.split(n)`：它按整块（block）分配，当数据块数少于 `num_of_partitions` 时会产生空的尾部分区（单块 10 行切 4 份会得到 `[10, 0, 0, 0]`），块大小不均时即使块数足够也会造成分区严重倾斜。当分区数超过总行数时，会在任何 Actor 启动前直接报错。

分区只在没有算子被检查点覆盖时生效。启用检查点的运行会把整个数据集放进同一个 Ray Data 计划流式处理，恢复提交在 batch 完成时才形成（见[检查点](#检查点)），因此 `partition.mode` 和 `num_of_partitions` 不影响它。

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
  strategy: every_op     # every_op（默认）或 disabled
partition:
  size: 5000             # 每次提交的输入行数上限
checkpoint_dir: s3://bucket/jobs/my_job   # 可选；默认 <work_dir>/checkpoints
```

recipe 开头连续的每个 Mapper/Filter 都是保存点。数据始终留在 Ray Data 上：算子像不加检查点时一样经 object store 读写 block，每个 worker 在把输出 batch 往下游传之前，顺带把一份副本写到 `checkpoint_dir`。恢复提交在这一刻形成，而不是在读入数据时预先切好：一次提交记录这个 batch 消费了哪些输入行、产出了哪些 Parquet 文件，因此 Mapper 扩行或 Filter 全部过滤都能被准确记录。被检查点覆盖的阶段向 Ray 传入 `batch_size=None`，不会为凑满 `partition.size` 合并小 block；每个 Ray block 内切为最多 `partition.size` 输入行的提交，算子内部推理的 `batch_size` 不变。未设置 `partition.size` 时，该上限按 preflight 吞吐取约 30 秒的工作量（限制在 64–2048 行），没有吞吐数据时为 512。

遇到第一个 Deduplicator、`Pipeline` 或其他需要一次看到全部数据的算子时，检查点前缀结束。该算子及其后的所有算子在合并后的输出上原样运行、不保存检查点，并会打印告警指出是哪个算子。

提交由一个固定在 driver 节点上的协调 Ray actor 负责，它只处理元数据：分配提交序号，并拒绝输入行与另一个已提交 batch 重叠的提交。重试时，worker 在运行算子前按阶段及已消费血缘查找持久化提交；若已存在，直接读取原输出，不再运行算子。并发尝试以先提交者为准，后提交者复用其输出。每次提交以“仅在不存在时创建”的 manifest 发布，因此提交永远不会被覆盖。

`checkpoint_dir` 必须对所有 Ray worker 可见。支持共享 POSIX 路径和 `s3://`（AWS S3 或 MinIO 等 S3 兼容服务，通过 `boto3` 的 `If-None-Match` 条件写入）；其他协议会在启动时报错。

### 检查点与临时文件

作业结束后检查点会保留，以便续跑。运行退出时，执行器会清理临时工作文件。

检查点清理只删除本次尝试拥有、且已确认未提交的文件。进程被强制终止、或远端写入结果不明时，仍可能留下孤立文件；系统不会扫描并删除所有“尚未被引用”的文件，因为其他 worker 可能仍在写入或提交它们。具体边界见[提交协议](RecoveryPartitionProtocol.md#storage)。空的检查点导出保持目录形式，以便继续恢复。

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
3. 重新读取输入并计算指纹，与 `run.json` 中的指纹、行数、schema 和被检查点覆盖的算子配置比较；
4. 以最多 32 个并行存储 `get` 读取提交 manifest，按 stage 顺序登记提交，算出每个算子还有哪些输入行没有消费；
5. 每个算子只处理这些行，加上上一个算子已提交、但本算子尚未消费的行（由 worker 从存储读取）；
6. 导出后校验每一条输入行都被恰好消费和交付一次；
7. 校验通过后，才删除之前中断的运行留下的导出文件。

`partition.size` 不属于身份信息，续跑时可以修改。如果记录的输入指纹、行数、schema 或被检查点覆盖的算子配置不匹配，显式续跑会报错停止，并保留已有 checkpoint。用已有检查点的作业 ID 新建运行会被拒绝，而不会与旧检查点混用。

用户需要保证源输入的行身份和读取顺序稳定：同一源输入行号在续跑时仍应指向同一条逻辑记录。输入指纹只是额外的误用检查，不承诺核验所有 payload 字节完全一致。非 JSON 值使用展示字符串，可能省略 tensor 中间的元素；引用的媒体文件、模型文件内容也不在哈希范围内。内容一致性由用户通过数据集和版本管理保证，逻辑输入改变时应创建新作业。

导出与算子同时进行，所以作业中断时导出目录里通常已有一部分结果。续跑会用新的文件名重新写出完整结果，只有在自己的导出写完并通过校验后，才删除之前的文件，因此续跑失败不会丢失已导出的内容。导出目录中不是本作业写出的文件会保留，并打印警告。Lance 格式的部分导出需要先手动删除再续跑。

`--job_id` 用于指定作业名称。恢复作业时，在原命令上添加 `--resume`，并使用首次运行的作业 ID。

如果同时提供两个参数，它们的值必须相同：

```bash
dj-process --config config.yaml \
  --job_id my_experiment_001 \
  --resume my_experiment_001
```

### 检查点策略

```bash
# 每个 Mapper/Filter（启用检查点时的默认值）
dj-process --config config.yaml --checkpoint.strategy every_op

# 不保存检查点
dj-process --config config.yaml --checkpoint.strategy disabled
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

### 检查点成本

检查点不会在算子之间插入物化屏障：Ray Data 仍像不加检查点时一样融合、流水线化和调度算子。额外开销来自：每个算子输出 batch 的旁路写入、每次提交一个小 manifest、启动时对输入做一次指纹扫描，以及导出前去掉内部血缘列。对于很轻的文本算子，写入成本可能高于失败后重跑；短作业建议关闭检查点。

两个可选需求需要保留最终计算结果：设置 `export_shard_size` 时，会物化导出输入以精确统计大小；Python API 请求返回数据集时，会物化检查点前缀之后的算子结果，供导出和后续读取共同使用。CLI 不使用返回值，显式设置 `skip_return=True`。检查点前缀外的 Filter 如果配置了统计导出，也会保留一次统计计算的结果，供统计写入和过滤共同使用。

### 存储建议

- 事件日志：快速存储（SSD）
- 检查点：所有 worker 共享、容量足以容纳每个算子输出的存储

### 提交大小权衡

- 较小的 `partition.size`：失败时丢失的工作更少，manifest 和文件更多
- 较大的 `partition.size`：文件更少，失败后需要重算的行更多

## 故障排除

**作业恢复失败：**
```bash
ls -la ./outputs/{work_dir}/{job_id}/job_summary.json
ls -la ./outputs/{work_dir}/{job_id}/checkpoints/partition_recovery/commits/
cat ./outputs/{work_dir}/{job_id}/checkpoints/partition_recovery/run.json
```

请使用首次运行打印的 resume token，并通过 `--resume` 重新执行相同的 recipe。可以在错误日志中检查配置不一致、输入内容/读取顺序/算子配置变化，或存在提交但缺少 `run.json`。显式续跑校验失败时不会删除已有 checkpoint。

**检查 Ray 状态：**
```bash
ray status
```

**查看日志：**
```bash
cat ./outputs/{work_dir}/{job_id}/events_*.jsonl
tail -f ./outputs/{work_dir}/{job_id}/logs/*.txt
```

# 基于共享存储的多机弹性分片

## 项目简介

这个示例为大规模 JSONL 数据处理提供一种“节点间分片、节点内 Ray”的执行模式：

1. 先把完整输入预切成固定数量、顺序稳定的 JSONL 分片。
2. 多个独立 Worker 通过共享 POSIX/NAS/CPFS 目录动态认领分片。
3. Worker 认领一片后，在当前节点使用 Data-Juicer `ray` executor 并行处理。
4. 分片完成后发布可校验的完成记录；空闲 Worker 继续认领下一片。
5. 全部分片成功后，按原始分片顺序验证并合并结果。

它不要求维护一个跨节点 Ray 集群。共享文件系统只负责节点间协调，每个节点的 Ray
运行时相互独立。

该能力现已通过主库命令 `dj-process-sharded` 提供。单机只需执行一次；多机时让调度器
在所有节点上使用相同的 `job-dir`、`run-id` 和参数执行同一条命令：

```bash
dj-process-sharded run \
  --config /mnt/shared/recipes/process.yaml \
  --job-dir /mnt/shared/data-juicer-jobs/job-001 \
  --num-shards 32 \
  --run-id submission-001
```

命令会先检查 recipe。仅包含 Mapper/Filter 且使用本地 JSONL 输入输出时执行弹性
分片；包含 Deduplicator、Selector、Grouper、Aggregator、Pipeline 等全数据语义
算子，或输入输出不满足条件时，会打印原因并由一个协调节点执行一次原始
`dj-process`。分片已经开始后的运行错误不会降级，以免重复处理完整数据。

分片任务可用 `dj-process-sharded status --job-dir <job-dir>` 查看。终态失败分片可用
`dj-process-sharded retry --job-dir <job-dir> --all-failed` 重新入队，之后应使用新的
`run-id` 再次执行 `run`。同一 `job-dir/run-id` 的参数必须保持不变。

> **必须区分 DLC 启动拓扑。** `dlc_job.py dlc` 要求所选作业类型把启动命令广播到
> 每个 Worker。PAI-DLC MPIJob 并不会这样做：启动命令只在 Launcher 上运行，
> Launcher 必须通过 `mpirun` 和 DLC 生成的 `/etc/mpi/hostfile` 在每个 GPU
> Worker 上各拉起一个进程。两种拓扑中的 GPU Worker 都继续使用节点内独立 Ray。

```text
                 Worker 广播型 DLC 作业提交
                                  │
             ┌────────────────────┼────────────────────┐
             │                    │                    │
       DLC Worker 0         DLC Worker 1         DLC Worker N
       认领 shard A          认领 shard B          认领 shard C
       本节点 Ray 处理        本节点 Ray 处理        本节点 Ray 处理
             │                    │                    │
             └────────────────────┼────────────────────┘
                                  │
                        共享 NAS/CPFS job-dir
                manifest / shards / locks / done / attempts
                                  │
                          校验并按顺序合并
```

## 主要优势

- **一次提交，多节点自动工作**：使用 Worker 广播，或由 MPIJob Launcher 调用
  `mpirun`，都不需要逐台登录机器。
- **动态负载均衡**：节点处理完一片后继续认领下一片，快节点自然承担更多工作。
- **节点内继续使用 Ray**：每个分片仍由 Data-Juicer Ray executor 利用当前节点的
  CPU/GPU 并行能力。
- **不依赖 rank**：分片协调不需要 `RANK`、`WORLD_SIZE` 或固定 hostname 到文件的
  静态映射。
- **任务可审计、可恢复**：manifest 固定输入、recipe、Data-Juicer commit、Ray
  配置和分片顺序；每次 attempt 都保留元数据与日志。
- **避免重复认领**：`O_CREAT|O_EXCL` 原子创建锁，同一时刻只有一个 Worker 能持有
  一个分片。成功或终态失败后，claim 会保留在原路径作为持久 fence，NAS/CPFS
  短暂的元数据可见性延迟不会让已完成分片重新开放。
- **支持失败与过期接管**：失败可自动重试；超过锁超时的认领会被其他 Worker 接管。
- **结果完整性校验**：完成记录包含行数、字节数和 SHA256；合并前会重新校验。
- **顺序确定**：目录输入按相对路径排序，分片连续，最终结果按 manifest 顺序合并。
- **低侵入**：状态机位于 `tools/elastic_sharding.py`，不修改现有 executor、recipe
  schema 或算子行为；本目录保留调度平台示例和兼容包装器。

## 与现有方式的区别

| 方式 | 负责切分 | 动态认领 | 失败恢复 | 节点内 Ray | 跨节点 Ray 集群 |
| --- | --- | --- | --- | --- | --- |
| `tools/data_resplit.py` | 是 | 否 | 否 | 由用户决定 | 否 |
| `ray_partitioned` | 运行时切分 | 由 Ray 调度 | 由 Ray 作业管理 | 是 | 是 |
| `dj-process-sharded` | 预切分 | 共享文件系统 | 锁超时、重试、retry | 是 | 否 |

本示例适合：

- 输入很大，希望先生成可检查、可复用的固定分片；
- 多个 DLC Worker 共享 NAS/CPFS，但不希望组成一个跨节点 Ray 集群；
- 希望节点增减、速度差异或部分失败时仍由空闲节点继续认领；
- 希望保留每个分片的运行日志、状态和可验证结果。

## 文件说明

```text
demos/elastic_sharding/
├── shard_job.py             # 通用分片任务：prepare/worker/status/retry/merge
├── dlc_job.py               # 适用于命令广播到每个 Worker 的 DLC 编排器
├── two_node_test.py         # 向后兼容的严格两节点包装器
├── configs/
│   ├── demo.yaml            # 纯 CPU Mapper/Filter recipe
│   ├── gpu_demo.yaml        # 每节点一张 GPU 的 CPU + GPU 冒烟测试
│   └── gpu_demo_4gpu.yaml   # 单机四卡冒烟测试
├── data/
│   └── gpu-demo-dataset.jsonl
├── README.md
└── README_ZH.md
```

`shard_job.py` 是主库 `dj-process-sharded` 的兼容包装器。对于 Worker 广播型作业，
`dlc_job.py` 为任意数量的 DLC Worker 协调一次性 prepare 和 finalize。
`two_node_test.py` 保留原来的严格两节点默认值，仅用于向后兼容。MPIJob Launcher
应改为在 `prepare` 和 `merge` 之间，通过 `mpirun -np N -npernode 1` 拉起
底层 `worker` 命令。

## 核心路径概念

这三个路径不要混淆：

- `--config`：Data-Juicer YAML recipe。
- `--dataset-path`：要预切分的 JSONL 文件或目录；指定后覆盖 recipe 中的
  `dataset_path`。
- `--job-dir`：所有 Worker 共享的任务工作目录，保存分片、锁、attempt、日志、
  状态和结果；它不是原始数据目录。

例如：

```bash
python demos/elastic_sharding/dlc_job.py dlc \
  --config /mnt/shared/recipes/my_process.yaml \
  --dataset-path /mnt/shared/input/my_dataset.jsonl \
  --job-dir /mnt/shared/data-juicer-jobs/my-job-001 \
  --nodes 4 \
  --num-shards 16
```

## 前置条件与限制

- 所有 Worker 必须使用相同 Data-Juicer 代码版本和依赖环境。
- `job-dir` 必须是支持原子创建、原子重命名、硬链接和 `fcntl` 建议锁的共享 POSIX
  文件系统，例如正常配置的 NAS/NFS/CPFS。
- 所有 Worker 必须以相同路径访问 `job-dir`、输入 JSONL 和本地媒体文件。
- 输入仅支持本地 `.jsonl` 文件，或递归包含 `.jsonl` 文件的本地目录。
- 每行必须是 UTF-8 JSON object；不允许空行、数组行或损坏 JSON。
- 当前只接受可逐分片独立执行的 Mapper 和 Filter。
- Deduplicator、Selector、Grouper、Aggregator、Pipeline 和其他全数据集语义算子会在
  `prepare` 阶段被拒绝。
- 锁采用静态超时，不发送心跳；锁超时必须大于最长单片处理时间。
- `prepare` 会保存完整分片，attempt 会保存处理结果，需为 `job-dir` 预留足够空间。
  媒体文件只改写路径，不会复制到 `job-dir`。

## PAI-DLC Worker 广播型作业快速开始

本节**不是** MPIJob 的启动流程。只有确认所选 DLC 作业类型会在每个 Worker
执行启动命令时才能使用。MPIJob 应只在 Launcher 配置一条命令，并由该命令依次
执行 prepare、`mpirun` 每 Worker 一个进程、merge。

### 1. DLC 任务配置

在 DLC 控制台创建任务：

- 选择会在每个 Worker 启动用户命令的 `PyTorch` 类型作业，但不使用
  `torchrun`。
- Worker 数量可以是任意正整数，例如 `4`；每个 Worker 启动一个脚本进程。
- 不要选择 DLC 的 `Ray` 框架；它会建立跨节点 Ray 集群，与本示例的节点内独立 Ray
  拓扑不同。
- 将同一份 Data-Juicer 代码挂载到所有 Worker 的相同路径，例如
  `/mnt/data/data-juicer`。
- 将同一个可读写 NAS/CPFS 挂载到相同路径，例如 `/mnt/shared`。
- 所有 Worker 使用相同镜像，镜像已安装 Data-Juicer 的 Ray 依赖。

PAI-DLC 的任务页面只需配置一份启动命令和 Worker 数量，但所选作业类型必须把
这条命令分发到每个 Worker，参考
[创建训练任务](https://help.aliyun.com/zh/pai/create-a-training-task)。

### 2. 配置一条启动命令

在 DLC 的“启动命令”中只填写一次：

```bash
cd /mnt/data/data-juicer && \
python demos/elastic_sharding/dlc_job.py dlc \
  --job-dir /mnt/shared/data-juicer-jobs/multi-node-job-001 \
  --nodes 4 \
  --num-shards 16 \
  --ray-address local
```

默认使用：

- recipe：`demos/elastic_sharding/configs/demo.yaml`
- 输入：recipe 中的 `demos/data/demo-dataset.jsonl`
- 分片数：未指定时为 4
- DLC Worker 数：默认弹性模式不强制检查
- Ray：每个节点独立 `local`
- 合并结果：`<job-dir>/merged.jsonl`

### 3. 自动执行流程

Worker 广播型作业会让每个实例执行相同的 `dlc` 逻辑：

1. 通过共享目录原子竞选 prepare coordinator。
2. coordinator 验证 recipe 和输入，只执行一次预切分。
3. 每个存活 Worker 不设单节点上限，持续动态认领分片。
4. 每片通过 `tools/process_data.py` 强制使用
   `--executor_type ray --ray_address local`。
5. Worker 达到上限后等待整个任务完成或失败。
6. 全部成功后原子竞选 finalize coordinator。
7. finalizer 汇总实际参与的 hostname，然后验证并合并。
8. 其他实例读取相同 finalize 结果，以相同退出码结束。

成功时可以看到：

```text
elected as DLC prepare coordinator
Starting DLC worker on hostname=...
elected as DLC finalize coordinator
PASS: 16 shards were completed by 4 node(s)
```

查看状态：

```bash
python demos/elastic_sharding/dlc_job.py status \
  --job-dir /mnt/shared/data-juicer-jobs/multi-node-job-001
```

最终结果：

```text
/mnt/shared/data-juicer-jobs/multi-node-job-001/merged.jsonl
```

### 弹性模式与严格参与模式

默认是**弹性模式**：

- `--nodes` 可省略，只用于日志提示；
- 不限制单个 Worker 最多处理多少分片；
- 即使实际启动的 Worker 少于预期，存活 Worker 仍可接管所有剩余分片；
- finalize 只要求所有分片完成，不强制特定节点数。

这是推荐的生产运行方式：

```bash
python demos/elastic_sharding/dlc_job.py dlc \
  --job-dir /mnt/shared/data-juicer-jobs/elastic-job-001 \
  --num-shards 32 \
  --ray-address local
```

只有在需要验证所有 DLC Worker 都实际处理过至少一个分片时，才使用
**严格参与模式**：

```bash
python demos/elastic_sharding/dlc_job.py dlc \
  --job-dir /mnt/shared/data-juicer-jobs/strict-job-001 \
  --nodes 4 \
  --num-shards 16 \
  --require-all-nodes \
  --ray-address local
```

严格模式会限制每个 Worker 的 claim 数，并检查至少有 `--nodes` 个不同 hostname
完成过分片。因此缺少任一 Worker 都会让严格任务等待并最终失败。建议让分片数是
节点数的整数倍。

## GPU 冒烟测试：一个 recipe 同时运行 CPU 和 GPU 算子

`configs/gpu_demo.yaml` 用来验证节点认领分片后，节点内 Ray 能让同一分片依次经过
CPU 和 GPU 算子：

| 顺序 | 算子 | 向 Ray 申请的资源 |
| --- | --- | --- |
| 1 | `whitespace_normalization_mapper` | 1 CPU |
| 2 | `query_sentiment_detection_mapper` | 1 CPU + 1 GPU |
| 3 | `query_topic_detection_mapper` | 1 CPU + 1 GPU |
| 4 | `text_pair_similarity_filter` | 1 CPU + 1 GPU |
| 5 | `text_length_filter` | 1 CPU |

两个 GPU Mapper 会在 `meta` 中写入情感和主题标签；GPU Filter 使用
`openai/clip-vit-base-patch32`，并在输出的 `__dj__stats__` 中写入
`text_pair_similarity`。三个 GPU 算子都显式设置了 `num_gpus: 1`，所以 Ray
必须将相应 task 调度到真实 GPU；这不是允许回退到 CPU 的测试。

提交任务前需要：

- 每个 DLC Worker 至少有一张对进程可见的 NVIDIA GPU；
- 镜像包含支持 CUDA 的 PyTorch、Ray、Transformers 和 Data-Juicer 依赖；
- 每个 Worker 都能读取三个模型：
  `mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis`、
  `dstefa/roberta-base_topic_classification_nyt_news` 和
  `openai/clip-vit-base-patch32`；
- 第一次运行默认从 Hugging Face 下载，也可以把 recipe 中对应模型字段改为所有
  Worker 都能访问的预下载路径；
- 多节点测试建议把模型直接放进镜像或预热缓存，不要让所有 Worker 同时下载模型。

先在一个 Worker 镜像中检查：

```bash
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
```

### 单机四卡命令

一台机器有 4 张可见 GPU 时，使用 `gpu_demo_4gpu.yaml`：

```bash
cd /mnt/data/data-juicer && \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python demos/elastic_sharding/dlc_job.py dlc \
  --config demos/elastic_sharding/configs/gpu_demo_4gpu.yaml \
  --job-dir /mnt/shared/data-juicer-jobs/gpu-4card-smoke-001 \
  --nodes 1 \
  --num-shards 1 \
  --ray-address local \
  --output /mnt/shared/data-juicer-jobs/gpu-4card-smoke-001/merged.jsonl
```

四卡 recipe 设置了 `override_num_blocks: 4`、`batch_size: 1`、
`num_proc: 4`，且每个 GPU task 设置 `num_gpus: 1`。这样同一个节点内 Ray
可以并发调度 4 个各占一张卡的 task。

对于内置 4 行数据，请保持 `--num-shards 1`。单个 DLC Worker 会顺序处理分片，
如果改成 `--num-shards 4`，会产生 4 个只有一行的 Ray 作业，通常每次仍只使用
一张 GPU。换成更大输入时，也要让每个分片至少包含 4 个 Ray block。

可以在另一个终端观察 GPU 调度：

```bash
watch -n 1 nvidia-smi
```

内置输入很小，GPU 利用率可能只短暂出现；测试持续利用率和吞吐量时，建议通过
`--dataset-path` 换成更大的数据。

### 多节点、每节点一张 GPU 命令

严格验证两个 GPU 节点时，在 DLC 中配置两个 Worker，并只填写一次下面的启动命令：

```bash
cd /mnt/data/data-juicer && \
python demos/elastic_sharding/dlc_job.py dlc \
  --config demos/elastic_sharding/configs/gpu_demo.yaml \
  --job-dir /mnt/shared/data-juicer-jobs/gpu-smoke-001 \
  --nodes 2 \
  --num-shards 4 \
  --require-all-nodes \
  --ray-address local \
  --output /mnt/shared/data-juicer-jobs/gpu-smoke-001/merged.jsonl
```

内置测试数据只有 4 行，因此最多只能切出 4 个非空分片。测试更多节点或真实吞吐量时，
通过 `--dataset-path` 换成同时包含 `text` 和 `target_text` 字段的更大 JSONL：

```bash
python demos/elastic_sharding/dlc_job.py dlc \
  --config demos/elastic_sharding/configs/gpu_demo.yaml \
  --dataset-path /mnt/shared/input/text-pairs.jsonl \
  --job-dir /mnt/shared/data-juicer-jobs/gpu-large-smoke-001 \
  --nodes 8 \
  --num-shards 32 \
  --require-all-nodes \
  --ray-address local
```

完成后检查节点认领情况以及 GPU 算子生成的元数据和统计字段：

```bash
python demos/elastic_sharding/dlc_job.py status \
  --job-dir /mnt/shared/data-juicer-jobs/gpu-smoke-001

rg -n 'query_(sentiment|topic)_label|text_pair_similarity' \
  /mnt/shared/data-juicer-jobs/gpu-smoke-001/merged.jsonl
```

每个 Worker 同一时间处理一个分片 attempt。内置 recipe 为每个 GPU 算子设置
`ray_execution_mode: task`、`num_proc: 1` 和 `num_gpus: 1`，因此三个阶段可以
依次复用节点上的同一张 GPU，而不要求每个节点同时提供三张 GPU。生产 recipe 应
根据硬件调整算子并发数、GPU 比例、batch size、模型大小和分片大小。

## 重点：使用现有 Mapper/Filter recipe 和自己的 JSONL

这个功能不是只能运行 demo recipe。推荐的真实使用方式就是复用已有的 Data-Juicer
Mapper/Filter recipe，通过 `--dataset-path` 指向自己的大规模 JSONL。

### Recipe 必须满足的条件

1. `executor_type` 可以是 `default` 或 `ray`。Worker 最终都会显式覆盖成 `ray`。
2. 必须有明确的 `process` 列表。
3. 每个算子必须能被识别为 Mapper 或 Filter，并且不是全局操作。
4. 算子不能设置 `stats_export_path`，否则多个分片会写入相同统计文件。
5. 算子使用固定 `save_dir` 时可以运行，但必须确保不同分片生成的文件名不会冲突。
6. `custom_operator_paths` 支持加载自定义算子；相对路径按 Data-Juicer 仓库根目录
   解析。自定义算子也必须继承 Mapper 或 Filter。
7. 算子配置了 `index_key` 时，预切分会为缺失值填入完整输入范围内的全局行号；
   已有值保持不变。

以下类型通常适合：

- 文本清洗 Mapper；
- 文本、图像、音频或视频属性 Mapper；
- 基于单条样本统计值的 Filter；
- 不依赖其他样本、不写共享固定文件的自定义 Mapper/Filter。

以下类型不适合直接逐分片运行：

- 全局去重；
- 需要全局排序或全局采样的 Selector；
- Grouper、Aggregator、Pipeline；
- 需要跨样本共享状态或生成唯一全局输出的算子。

示例 recipe：

```yaml
project_name: my-elastic-job
dataset_path: /mnt/shared/input/default.jsonl
export_path: /mnt/shared/output/ignored-by-shard-worker.jsonl
executor_type: ray
ray_address: local

text_key: text
image_key: images
audio_key: audios
video_key: videos

process:
  - whitespace_normalization_mapper:
      text_key: text
  - text_length_filter:
      text_key: text
      min_len: 10
      max_len: 10000
```

Worker 会用当前分片覆盖 recipe 的 `dataset_path`，并把 `export_path` 覆盖为当前
attempt 的隔离输出目录。因此同一份 recipe 可以安全地重复用于多个分片。

### 自有 JSONL 的要求

单文件输入：

```text
/mnt/shared/input/my_dataset.jsonl
```

目录输入：

```text
/mnt/shared/input/my_dataset/
├── 000.jsonl
├── 001.jsonl
└── nested/
    └── 002.jsonl
```

目录会被递归扫描，并按相对路径排序。每行必须是 object：

```json
{"id": 1, "text": "example", "images": ["media/1.jpg"]}
```

媒体路径处理：

- 绝对路径保持不变。
- `http://`、`https://`、`s3://`、`gs://`、`hdfs://` 保持不变。
- 相对路径会转换成绝对路径：
  - 单 JSONL 文件：相对于 JSONL 所在目录；
  - JSONL 目录：相对于传入的数据集根目录。
- `images`、`audios`、`videos` 字段必须是字符串列表或 `null`。字段名可以通过
  recipe 的 `image_key`、`audio_key`、`video_key` 修改。

### 在任意数量的广播启动 DLC Worker 上运行自己的 recipe 和数据

在 Worker 广播型 DLC 作业中仍然只配置这一条命令：

```bash
cd /mnt/data/data-juicer && \
python demos/elastic_sharding/dlc_job.py dlc \
  --config /mnt/shared/recipes/my_process.yaml \
  --dataset-path /mnt/shared/input/my_dataset.jsonl \
  --job-dir /mnt/shared/data-juicer-jobs/my-dataset-001 \
  --nodes 4 \
  --num-shards 4 \
  --ray-address local \
  --output /mnt/shared/output/my_dataset.processed.jsonl
```

这里：

- `--config` 复用现有 Mapper/Filter recipe；
- `--dataset-path` 覆盖 recipe 原来的 `dataset_path`；
- `--job-dir` 保存这一次任务的所有状态和中间结果；
- `--output` 指定最终合并文件；
- 4 个分片由 4 个存活 Worker 动态认领，每个 Worker 内部使用 Ray。

建议先用少量数据、每个节点一个分片完成冒烟测试，再扩大输入规模。

### 如何选择分片数

- 必须满足 `1 <= num_shards <= JSONL 总记录数`。
- 为获得最佳吞吐，分片数应不大于 Worker 节点数，最好与节点数相等，即每节点一个分片。
- 分片数少于节点数会让部分节点空闲；分片数多于节点数会让同一 Worker 顺序处理多个
  分片，并为每个分片启动新的 Data-Juicer 进程。
- 只有在负载明显不均衡，或需要更细的失败重试粒度时，才建议让分片数多于节点数；
  此时应确保单片计算量足以摊薄进程和节点本地 Ray 的启动成本。
- 单片处理时间应明显小于 `lock_timeout_secs`。
- 分片太多还会增加 Ray 启动、元数据和小文件开销。
- 默认弹性模式不限制单节点 claim 数，所以分片数不必是 Worker 数的整数倍。
- 使用 `--require-all-nodes` 时，建议直接让 `--num-shards` 等于 `--nodes`。

### Recipe 或输入变化时

`prepare` 会记录 recipe SHA256、输入文件 SHA256、大小、mtime、行数、规范化内容
SHA256 和 Data-Juicer commit。

- 完全相同的请求重复执行是幂等 no-op。
- 修改 recipe、输入内容、分片数或 Ray 地址后，不要复用旧 `job-dir`。
- 推荐每次独立任务使用新目录，例如 `my-dataset-001`、`my-dataset-002`。

## `dlc_job.py` 参数参考

### `dlc`

完整的 Worker 广播型 DLC 流程。

| 参数 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--job-dir` | 是 | 无 | 所有 Worker 共享的任务目录 |
| `--config` | 否 | `configs/demo.yaml` | 现有的 shard-safe Data-Juicer recipe |
| `--dataset-path` | 否 | recipe 中的值 | 覆盖 recipe 的 JSONL 文件或目录 |
| `--nodes` | 否 | 不强制 | 弹性模式仅用于日志；严格模式必填 |
| `--num-shards` | 否 | `4` | 精确的非空分片数 |
| `--require-all-nodes` | 否 | false | 限制 claim，并要求所有 `--nodes` hostname 参与 |
| `--ray-address` | 否 | `local` | 每个节点内部使用的 Ray 地址 |
| `--output` | 否 | `<job-dir>/merged.jsonl` | 最终合并 JSONL |
| `--run-id` | 否 | DLC Job ID | 同一次提交中所有 Worker 共享的标识 |
| `--wait-timeout-secs` | 否 | `126000`（35 小时） | 等待 prepare、全分片完成或 finalize 的最长时间 |
| `--poll-interval-secs` | 否 | `2` | DLC 协调状态轮询间隔 |

`--wait-timeout-secs` 是 DLC 包装器等待其他实例的超时，不是分片锁超时。

启动器会依次从 `PAI_JOB_ID`、`DLC_JOB_ID`、`JOB_ID` 读取 submission 标识。
在 DLC 之外运行时，应给所有 Worker 传入相同的 `--run-id`，并在每次新提交时
更换它。这样即使复用同一个 `job-dir`，也不会读取上一次提交留下的终态协调文件。

### 其他包装器子命令

| 子命令 | 参数 | 默认值与说明 |
| --- | --- | --- |
| `prepare` | `--job-dir` | 必填；共享任务目录 |
|  | `--config` | demo recipe |
|  | `--dataset-path` | 可选；覆盖 recipe 输入 |
|  | `--num-shards` | `4` |
|  | `--ray-address` | `local` |
| `worker` | `--job-dir` | 必填 |
|  | `--max-shards` | 默认不限制；可显式设置本次 claim 上限 |
|  | `--ray-address` | 可选；覆盖 manifest 的 Ray 地址 |
| `status` | `--job-dir` | 必填；显示全部分片状态 |
| `verify` | `--job-dir` | 必填 |
|  | `--output` | 默认写到 job-dir |
|  | `--expect-nodes` | `1`；完成记录中要求的最少不同 hostname 数 |

## `shard_job.py` 完整参数参考

### `prepare`

验证 recipe、扫描输入并原子发布任务目录。

| 参数 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--config` | 是 | 无 | Data-Juicer YAML recipe |
| `--dataset-path` | 否 | recipe 中的值 | 覆盖输入 JSONL 文件或目录 |
| `--job-dir` | 是 | 无 | 新的共享 POSIX 任务目录 |
| `--num-shards` | 是 | 无 | 精确的非空分片数 |
| `--lock-timeout-secs` | 否 | `126000` | 保存到 manifest 的分片锁超时 |
| `--max-retries` | 否 | `3` | 首次失败后的重试次数；默认最多 4 次失败 attempt |
| `--poll-interval-secs` | 否 | `20` | Worker 无可认领分片时的等待间隔 |
| `--ray-address` | 否 | `local` | 保存给 Worker 的 Ray 地址 |

### `worker`

循环认领和处理分片。

| 参数 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--job-dir` | 是 | 无 | prepare 创建的共享任务目录 |
| `--max-shards` | 否 | 不限制 | 本次进程最多处理的 claim 数；失败 attempt 也计数 |
| `--lock-timeout-secs` | 否 | manifest 值 | 覆盖本次 Worker 的锁超时 |
| `--max-retries` | 否 | manifest 值 | 覆盖本次 Worker 的失败重试次数 |
| `--poll-interval-secs` | 否 | manifest 值 | 无 claim 可用时的轮询间隔 |
| `--ray-address` | 否 | manifest 值 | 覆盖本次 Worker 的 Ray 地址 |
| `--allow-version-mismatch` | 否 | false | 允许 Worker commit 与 manifest 不同，仅应有意使用 |

不设置 `--max-shards` 时，Worker 会持续认领，直到全部成功或任务进入终态失败。

### `status`

只读查看任务状态。

| 参数 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--job-dir` | 是 | 无 | 任务目录 |
| `--lock-timeout-secs` | 否 | manifest 值 | 仅用于判断现有锁是否显示为 stale |
| `--json` | 否 | false | 输出机器可读 JSON |
| `--all` | 否 | false | 文本模式下逐片显示状态和 owner |

### `retry`

归档终态失败并重新入队。必须在 `--all-failed` 与一个或多个 `--shard-id` 中选择一种。

| 参数 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--job-dir` | 是 | 无 | 任务目录 |
| `--all-failed` | 条件必填 | false | 重新入队所有失败分片 |
| `--shard-id` | 条件必填 | 无 | 指定分片 ID，可重复传入 |

retry 会把旧 failed 元数据、attempts 和终态 claim fence 移到 `state/history`，不会
覆盖历史记录；移走 fence 是重新入队的最后一步。

### `merge`

重新验证并按 manifest 顺序合并所有完成结果。

| 参数 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--job-dir` | 是 | 无 | 已全部成功的任务目录 |
| `--output` | 是 | 无 | 最终 JSONL 路径 |
| `--lock-timeout-secs` | 否 | manifest 值 | 计算合并前状态时使用 |
| `--overwrite` | 否 | false | 允许替换已存在的输出 |

## 通用手工/调度器流程

如果不使用 DLC 一键包装器，可以直接调用底层命令。

### 1. 预切分

```bash
python demos/elastic_sharding/shard_job.py prepare \
  --config /mnt/shared/recipes/my_process.yaml \
  --dataset-path /mnt/shared/input/my_dataset.jsonl \
  --job-dir /mnt/shared/data-juicer-jobs/my-job-001 \
  --num-shards 16 \
  --lock-timeout-secs 126000 \
  --max-retries 3 \
  --poll-interval-secs 20 \
  --ray-address local
```

`prepare` 两遍顺序扫描输入：

- 第一遍校验 JSONL、计算指纹、规范化媒体路径和全局 `index_key`；
- 第二遍按规范化字节数生成连续、近似均衡、非空分片；
- 输入在两遍之间变化会报错；
- 完整 stage 目录准备好后才原子重命名为 `job-dir`。

### 2. 每个节点启动 Worker

由调度器在每个节点启动：

```bash
python demos/elastic_sharding/shard_job.py worker \
  --job-dir /mnt/shared/data-juicer-jobs/my-job-001
```

### 3. 查看状态

```bash
python demos/elastic_sharding/shard_job.py status \
  --job-dir /mnt/shared/data-juicer-jobs/my-job-001 --all

python demos/elastic_sharding/shard_job.py status \
  --job-dir /mnt/shared/data-juicer-jobs/my-job-001 --json
```

状态包括：

- `pending`：尚未认领；
- `running`：存在未超时锁；
- `stale`：锁年龄超过超时，下一次认领会尝试接管；
- `committing`：claim 已进入终态，但当前文件系统客户端尚未看到终态标记；
- `conflict`：标记与 claim 元数据冲突；Worker 会停止，避免重复处理；
- `done`：完成记录已发布；
- `failed`：超过重试上限的终态失败。

### 4. 重新入队失败分片

修复环境问题后：

```bash
python demos/elastic_sharding/shard_job.py retry \
  --job-dir /mnt/shared/data-juicer-jobs/my-job-001 \
  --all-failed
```

或只重试指定分片：

```bash
python demos/elastic_sharding/shard_job.py retry \
  --job-dir /mnt/shared/data-juicer-jobs/my-job-001 \
  --shard-id part-00003-of-00016 \
  --shard-id part-00007-of-00016
```

然后重新启动 Worker。

### 5. 合并

```bash
python demos/elastic_sharding/shard_job.py merge \
  --job-dir /mnt/shared/data-juicer-jobs/my-job-001 \
  --output /mnt/shared/output/my_dataset.processed.jsonl
```

合并会逐片重新解析 JSONL，并验证完成记录里的行数和 SHA256。目标已存在时默认拒绝
覆盖；确需替换时使用 `--overwrite`。

## Ray 运行模式

### 默认：每个 attempt 使用本节点 local Ray

```bash
--ray-address local
```

每个分片 attempt 启动独立 Ray 实例，Data-Juicer 进程退出后清理。优点是配置简单、
节点相互隔离；代价是每片有 Ray 启动开销。

### 可选：每个节点预启动持久 Ray head

如果调度系统能保证每个节点只运行一个 Worker，可以在每个节点分别执行：

```bash
ray start --head
python demos/elastic_sharding/shard_job.py worker \
  --job-dir /mnt/shared/data-juicer-jobs/my-job-001 \
  --ray-address auto
ray stop
```

不要让所有节点的 `auto` 指向同一个 Ray 集群，否则拓扑会变成跨节点共享 Ray。

## `job-dir` 结构

```text
my-job-001/
├── manifest.json
├── recipe.yaml
├── shards/
│   └── part-xxxxx-of-xxxxx.jsonl
├── cache/
├── attempts/<shard-id>/<attempt-id>/
│   ├── attempt.json
│   ├── process.log
│   ├── logs/
│   ├── checkpoints/
│   ├── partitions/
│   ├── ray-output.jsonl/
│   └── processed.jsonl
├── state/
│   ├── locks/
│   ├── stale_locks/
│   ├── done/
│   ├── failed/
│   └── history/
│       ├── failed/
│       ├── attempts/
│       └── claims/
└── merge.json
```

DLC 一键入口还会在 `job-dir` 同级创建：

```text
.<job-dir-name>.dlc-coordination/
└── <submission-id-hash>/
    ├── prepare.lock
    ├── prepare-result.json
    ├── abort.lock
    ├── abort.json
    ├── finalize.lock
    └── finalize-result.json
```

如果环境没有显式设置 `XDG_CACHE_HOME` 或 `HF_HOME`，Worker 会使用
`<job-dir>/cache`。

## 故障语义

- 同一分片只允许一个可见 claim，但超时接管可能造成旧 attempt 与新 attempt 短暂
  重叠。
- 成功和终态失败的 claim 会在原文件中改写为 `done` 或 `failed` 并作为 fence 保留；
  只有显式 `retry` 会归档失败 fence 并重新开放该分片。
- 每个 attempt 使用独立目录，不会互相覆盖结果。
- 只有第一个成功原子发布 done 元数据的结果会被接受。
- `max_retries=3` 表示首次失败后再重试 3 次，共允许 4 次失败 attempt。
- 分片达到失败上限后写入 `state/failed`，任务退出码为 `2`。
- `retry` 只应在修复根因后执行；它会拒绝非终态 claim。
- 修改 recipe 或输入不是 retry 场景，应新建 `job-dir` 并重新 prepare。
- DLC prepare/finalize coordinator 在写出阶段结果前被强制终止时，其他实例会等待
  `wait_timeout_secs` 后失败；新的 submission 会使用新的协调 generation，因此在
  recipe 和输入不变时可以安全复用同一个 `job-dir` 重试。

## 退出码

- `0`：操作成功；Worker 也可能表示已达到 `--max-shards`。
- `1`：`status` 检测到任务尚未完成且没有终态失败。
- `2`：参数、数据、recipe、版本或运行错误，或存在终态失败。

DLC 中任一 Worker 非零退出都应视为整个任务失败。

## 排障

### prepare 阶段失败

检查：

- JSONL 是否为 UTF-8、每行 object 且无空行；
- `num_shards` 是否超过记录数；
- recipe 是否包含全局算子或 `stats_export_path`；
- recipe/数据相对路径是否从 Data-Juicer 仓库根目录正确解析；
- 新请求是否错误复用了不匹配的 `job-dir`。

### Worker 或 Ray 失败

查看：

```text
<job-dir>/attempts/<shard-id>/<attempt-id>/process.log
<job-dir>/attempts/<shard-id>/<attempt-id>/attempt.json
```

同时检查：

- 镜像是否安装 Ray 依赖；
- 当前节点 CPU、GPU、共享内存和临时空间；
- `ray_address` 是否为预期的 `local` 或本节点 `auto`；
- `lock_timeout_secs` 是否短于实际单片耗时；
- 所有 Worker 是否使用同一 Data-Juicer commit。

### DLC 一直等待

检查：

- 当前是否确实为 Worker 广播型作业，而不是启动命令只落到 Launcher 的 MPIJob；
- 严格模式下 DLC Worker 数是否等于 `--nodes`；弹性模式可以省略
  `--nodes`；
- 所有 Worker 是否真的启动了同一命令；
- `job-dir` 是否是同一个共享挂载，而不是各节点本地同名目录；
- 所有 Worker 解析到的 `--run-id` 或 DLC Job ID 是否一致；
- 同级隐藏协调目录下当前 generation 中的 `prepare-result.json`、`abort.json`
  和 `finalize-result.json`；
- 使用 `--require-all-nodes` 时，失败 attempt 是否消耗了严格模式的单 Worker
  claim 上限。

## 测试

```bash
python -m pytest -q tests/demos/test_elastic_sharding.py
```

测试覆盖：

- 确定性分片、输入顺序和媒体路径规范化；
- recipe 安全校验和 default-to-Ray 覆盖；
- 并发原子认领与 stale lock 接管；
- attempt 隔离、Ray 命令和 Ray 多文件输出物化；
- 重试上限、指定 retry 和顺序合并；
- 两个不同 hostname 的结果验证；
- DLC prepare/finalize 选主与失败传播。

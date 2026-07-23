# 基于共享存储的多机弹性分片

这个示例先把大型 JSONL 数据集切成固定数量的分片，再让多台机器通过共享
POSIX/NAS 目录动态认领分片。节点认领分片后，使用 Data-Juicer `ray`
executor 在该节点内部并行处理；默认 `ray_address=local`，各节点的 Ray
实例互相独立。

它与已有能力的区别：

- `tools/data_resplit.py` 只负责拆文件，不管理认领、失败恢复和结果合并。
- `ray_partitioned` 在一个跨节点 Ray 作业内切分数据并由一个 driver 调度。
- 本示例允许多台互不隶属的机器运行相同 worker 命令，处理完一片后继续认领下一片。
  节点之间由共享文件系统协调，节点内部由 Ray 调度。

## 限制

- 共享目录必须是支持原子创建、原子重命名和 `fcntl` 建议锁的 POSIX 文件系统，
  例如正常配置的 NAS/NFS。
- 输入仅支持本地 `.jsonl` 文件或包含 `.jsonl` 的目录。
- 首版只接受可以逐分片独立运行的 Mapper 和 Filter。去重、Selector、Grouper、
  Aggregator 和 Pipeline 会在 `prepare` 阶段被拒绝。
- 所有节点都必须安装带 Ray 依赖的 Data-Juicer。使用默认 `local` 模式时建议
  每个节点只运行一个 worker，避免多个本地 Ray 实例争抢同一节点资源。
- 锁采用静态超时，不发送心跳。`lock_timeout_secs` 必须大于最长单片处理时间。
- 所有节点需要能访问相同的输入数据、媒体文件和任务目录，并使用相同的
  Data-Juicer commit。

## 底层命令（非 DLC 手工模式）

以下命令都从 Data-Juicer 仓库根目录运行。任务目录必须位于所有节点都能访问的
共享存储上。DLC 用户可以跳过这一节，直接使用下一节的一条启动命令。

```bash
python demos/elastic_sharding/shard_job.py prepare \
  --config demos/elastic_sharding/configs/demo.yaml \
  --job-dir /shared/data-juicer-jobs/demo \
  --num-shards 4 \
  --ray-address local
```

`prepare` 会严格解析 JSONL，通过两遍顺序扫描生成 4 个连续、非空且按字节数近似
均衡的分片。目录输入按相对路径排序。图片、音频和视频字段中的相对路径会相对于
原数据根目录转为绝对路径。算子通过 `index_key` 请求的字段会在完整输入范围内
按全局输入序号补齐；输入中已有的值会保留。

## DLC 两节点冒烟测试（一条启动命令）

仓库提供了一个很薄的测试入口 `two_node_test.py`。它默认使用现有的
`configs/demo.yaml` 和 Data-Juicer 自带的 `demos/data/demo-dataset.jsonl`，
切成 4 片，并限制每个节点最多处理 2 片，从而保证两台节点都需要参与。

在 DLC 控制台创建一个任务：

- 框架选择 `PyTorch`，不需要使用 `torchrun`。这里借助的是 DLC 在每个 Worker
  上执行同一条启动命令的能力；节点内部的计算仍由 Ray 完成。不要选择 DLC 的
  `Ray` 框架，否则 DLC 会建立跨节点 Ray 集群，与本示例“共享存储协调节点、
  每个节点独立运行 Ray”的拓扑不同。
- Worker 节点数量设置为 `2`，每个 Worker 只启动一个本脚本进程。
- 将 Data-Juicer 代码挂载到两台 Worker 的相同路径，例如
  `/mnt/data/data-juicer`。
- 将同一个 NAS/CPFS 目录以读写方式挂载到两台 Worker，例如 `/mnt/shared`。
  不要把任务目录放在各节点本地盘或不支持 POSIX 原子操作的对象存储挂载上。
- 两台 Worker 应使用相同镜像，且镜像已安装 Data-Juicer 的 Ray 依赖。

PAI-DLC 的创建任务页面只需配置一份启动命令和 Worker 节点数量，参考
[创建训练任务](https://help.aliyun.com/zh/pai/create-a-training-task)。

在 DLC 的“启动命令”中只填写一次下面的命令。DLC 会在两个 Worker 实例上分别
启动它，不需要手工登录节点，也不依赖 `RANK`、`WORLD_SIZE` 等环境变量：

```bash
cd /mnt/data/data-juicer && \
python demos/elastic_sharding/two_node_test.py dlc \
  --job-dir /mnt/shared/data-juicer-jobs/two-node-test-001 \
  --nodes 2 \
  --num-shards 4 \
  --ray-address local
```

`dlc` 子命令会自动完成整个流程：

1. 两个实例通过共享目录原子竞选，只由一个实例执行 JSONL 分片。
2. 两个实例各自动态认领分片；认领后使用本节点独立的
   `executor_type=ray, ray_address=local` 处理。
3. 每个实例在本测试中最多处理 2 片，所以 4 片必须由两个实例共同完成。
4. 全部分片完成后只由一个实例验证至少有两个不同 hostname，并合并结果；另一个
   实例读取相同的最终状态并以相同退出码结束。

两个 DLC Worker 的日志中会分别出现自己的 hostname。成功时还能看到类似下面的
关键日志：

```text
elected as DLC prepare coordinator
Starting test worker on hostname=...
elected as DLC finalize coordinator
PASS: 4 shards were completed by 2 node(s)
```

中途可以从任意能挂载该共享目录的环境查看状态：

```bash
python demos/elastic_sharding/two_node_test.py status \
  --job-dir /mnt/shared/data-juicer-jobs/two-node-test-001
```

默认合并结果为
`/mnt/shared/data-juicer-jobs/two-node-test-001/two-node-merged.jsonl`。
同一次 DLC 作业中的两个 Worker 必须使用完全相同的 `--job-dir`。重复执行一个
已经成功的任务是幂等的；更换输入、recipe 或重新进行一次独立测试时，应换一个
全新的 `--job-dir`。任务目录及同级的隐藏协调目录会保留已发布的阶段状态。

也可以换成其他现有 Mapper/Filter recipe 和自己的 JSONL：

```bash
python demos/elastic_sharding/two_node_test.py dlc \
  --job-dir /shared/data-juicer-jobs/my-test \
  --nodes 2 \
  --num-shards 4 \
  --config demos/process_simple/process.yaml \
  --dataset-path /shared/input/my-data.jsonl
```

`--nodes` 应与 DLC Worker 数量一致。脚本会根据节点数限制单实例最多处理的分片数，
并拒绝无法保证每个节点都参与的分片/节点组合。生产任务如果不要求强制每个节点都
处理到分片，可直接使用后文的 `shard_job.py worker`，不设置 `--max-shards`。

### DLC 运行前检查

- `--job-dir`、输入 JSONL 及其中引用的本地媒体文件必须能被两个 Worker 以相同路径
  访问。
- `--num-shards` 不能超过输入记录数；两节点冒烟测试建议保持为 `4`。
- `--ray-address local` 表示每个分片 attempt 启动本节点 Ray，不会连接另一个节点。
- 默认等待超时为 35 小时。协调节点在写出阶段结果前被强制终止时，其他实例最终会
  以退出码 `2` 失败；重新提交时使用新的任务目录。
- DLC 任一 Worker 返回非零退出码时，任务应视为失败。检查
  `<job-dir>/attempts/*/*/process.log` 和同级隐藏协调目录中的 `abort.json`。

### 手工或其他调度器运行

如果不是 DLC，也可以分别执行显式的 `prepare`、`worker` 和 `verify` 子命令：

```bash
python demos/elastic_sharding/two_node_test.py prepare \
  --job-dir /shared/data-juicer-jobs/manual-test

# 由调度器在每个节点启动一次：
python demos/elastic_sharding/two_node_test.py worker \
  --job-dir /shared/data-juicer-jobs/manual-test

python demos/elastic_sharding/two_node_test.py verify \
  --job-dir /shared/data-juicer-jobs/manual-test
```

在每台机器上运行相同的 worker 命令：

```bash
python demos/elastic_sharding/shard_job.py worker \
  --job-dir /shared/data-juicer-jobs/demo
```

Worker 使用原子锁一次认领一片，处理完成后继续认领，直到所有分片完成。可以用
`--max-shards 1` 限制本次调用最多处理一个分片。默认锁超时为 126000 秒
（35 小时），首次处理失败后最多自动重试 3 次；这些默认值可以在 `prepare` 或
`worker` 命令中覆盖。若环境中没有显式设置 Hugging Face 或 XDG 缓存目录，
worker 会复用共享任务目录中的 `cache/`。

Worker 无论 recipe 原来写的是 `default` 还是 `ray`，实际处理命令都会显式覆盖为
`--executor_type ray`。默认的 `--ray-address local` 会为每个分片 attempt 启动
节点本地 Ray 实例并在处理进程退出时清理。如果已经在每个节点分别启动了持久化
Ray head，可以在 prepare 或 worker 时传入 `--ray-address auto` 来复用它，降低
逐分片启动 Ray 的开销。不要把所有节点指向同一个 Ray 地址，否则会变成共享 Ray
集群，而不再是“节点间分片、节点内 Ray”的模型。

持久化模式需要在每个节点分别执行：

```bash
ray start --head
python demos/elastic_sharding/shard_job.py worker \
  --job-dir /shared/data-juicer-jobs/demo \
  --ray-address auto
# 本节点不再处理任务后：
ray stop
```

查看状态：

```bash
python demos/elastic_sharding/shard_job.py status \
  --job-dir /shared/data-juicer-jobs/demo --all

python demos/elastic_sharding/shard_job.py status \
  --job-dir /shared/data-juicer-jobs/demo --json
```

失败分片达到重试上限后不会自动重跑。修复运行环境后，可以归档失败历史并重新
入队；若需要修改 recipe，则应使用新的任务目录重新执行 `prepare`：

```bash
python demos/elastic_sharding/shard_job.py retry \
  --job-dir /shared/data-juicer-jobs/demo --all-failed
```

全部分片完成后按原分片顺序合并：

```bash
python demos/elastic_sharding/shard_job.py merge \
  --job-dir /shared/data-juicer-jobs/demo \
  --output /shared/data-juicer-results/demo.jsonl
```

合并会重新验证每个结果的 JSONL、行数和 SHA256，并通过临时文件原子发布。
目标文件已存在时默认拒绝覆盖；确需替换时显式传入 `--overwrite`。

## 任务目录

```text
demo/
├── manifest.json
├── recipe.yaml
├── shards/
├── cache/
├── attempts/<shard-id>/<attempt-id>/
│   ├── attempt.json
│   ├── process.log
│   ├── ray-output.jsonl/
│   └── processed.jsonl
└── state/
    ├── locks/
    ├── stale_locks/
    ├── done/
    ├── failed/
    └── history/
```

`manifest.json` 固定输入文件指纹、recipe 哈希、Data-Juicer commit、Ray 地址和
分片顺序。Ray 会把一个分片输出为包含若干文件的 `ray-output.jsonl/` 目录；
worker 校验并按文件名顺序物化成单一 `processed.jsonl`，供状态记录和最终合并。
每个 attempt 都有独立的输出、日志和 Data-Juicer 工作目录。

manifest schema 已升级到版本 2；旧版本任务目录需要重新执行 `prepare`。

DLC 一键入口还会在任务目录同级创建
`.two-node-test-001.dlc-coordination/`，其中的 `prepare.lock`、
`prepare-result.json`、`abort.json` 和 `finalize-result.json` 用于跨实例选主和
传播统一退出状态。若协调实例在写出阶段结果前被强制终止，其他实例会等待
`--wait-timeout-secs`（默认 35 小时）后失败；重新提交时建议使用新的任务目录。

## 退出码

- `0`：命令完成；worker 已完成全部任务或达到 `--max-shards`。
- `1`：`status` 检测到任务仍在运行或等待处理。
- `2`：参数/数据错误，或任务存在终态失败。

静态超时可能让超长任务被另一节点接管。每个 attempt 使用不同结果路径，只有第一
个成功发布 `.done` 元数据的结果会被 `merge` 接受，因此不会互相覆盖。

# Data-Juicer 分布式数据处理

## 概览

Data-Juicer 支持基于 [Ray](https://github.com/ray-project/ray) 和阿里巴巴 [PAI](https://www.aliyun.com/product/bigdata/learn) 的大规模分布式数据处理。

经过专门的设计后，几乎所有在单机模式下实现的 Data-Juicer 算子都可以无缝地运行在 Ray 的分布式模式下。对于大规模场景，我们继续进行了针对计算引擎的特定优化，例如用于平衡文件和进程数目的数据子集分割策略，针对 Ray 和 Apache Arrow的 JSON 文件流式 I/O 补丁等。

作为参考，我们在 25 到 100 个阿里云节点上进行了实验，使用 Ray 模式下的 Data-Juicer 处理不同的数据集。在 6,400 个 CPU 核上处理包含 700 亿条样本的数据集只需要花费 2 小时，在 3,200 个 CPU 核上处理包含 70 亿条样本的数据集只需要花费 0.45 小时。此外，在 Ray 模式下，对 TB 大小级别的数据集，Data-Juicer 的 MinHash-LSH 去重算子在 1,280 个 CPU 核的 8 节点集群上进行去重只需 3 小时。 

更多细节请参考我们的论文：[Data-Juicer 2.0: Cloud-Scale Adaptive Data Processing for Foundation Models](https://arxiv.org/abs/2501.14755) 。

<img src="https://img.alicdn.com/imgextra/i2/O1CN01EteoQ31taUweAW1UE_!!6000000005918-2-tps-4034-4146.png" align="center" width="600" />

## 实现与优化

### Data-Juicer 的 Ray 处理模式

- 对于 Data-Juicer 的大部分[算子](Operators.md)实现，其核心处理函数是引擎无关的。[RayDataset](../data_juicer/core/ray_data.py) 和 [RayExecutor](../data_juicer/core/executor/ray_executor.py) 封装了与Ray引擎的具体互操作，它们分别是基类 `DJDataset` 和 `BaseExecutor` 的子类，并且都支持 Ray [Tasks](https://docs.ray.io/en/latest/ray-core/tasks.html) 和 [Actors](https://docs.ray.io/en/latest/ray-core/actors.html) 。
- 其中，去重算子是例外。它们在单机模式下很难规模化。因此我们提供了针对它们的 Ray 优化版本算子，并以特殊前缀开头：[`ray_xx_deduplicator`](../data_juicer/ops/deduplicator/) 。

### 数据子集分割

当在上万个节点中处理仅有若干个文件的数据集时， Ray 会根据可用资源分割数据集文件，并将它们分发到所有节点上，这可能带来极大的网络通信开销并减少 CPU 利用率。更多细节可以参考文档 [Ray's autodetect_parallelism](https://github.com/ray-project/ray/blob/2dbd08a46f7f08ea614d8dd20fd0bca5682a3078/python/ray/data/_internal/util.py#L201-L205) 和 [tuning output blocks for Ray](https://docs.ray.io/en/latest/data/performance-tips.html#tuning-output-blocks-for-read) 。

这种默认执行计划可能非常低效，尤其是在节点数量较多的情况下。为了优化此类情况的性能，我们考虑到 Ray 和 Arrow 的特性，提前将原始数据集自动拆分为较小的文件。当用户遇到此类性能问题时，他们可以利用此功能或根据偏好自己拆分数据集。在我们的自动拆分策略中，单个文件大小设置为 128MB，且结果应确保 拆分后的子文件数量 至少是 集群中可用CPU核心总数 的两倍。对应工具可在tools/data_resplit.py获取。


### JSON 文件的流式读取

为了解决 Ray Dataset 类底层框架 Arrow 对流式读取 JSON 数据的原生支持的缺失，我们开发了一个流式载入的接口并贡献到了一个针对 Apache Arrow 的内部 [补丁](https://github.com/datajuicer/data-juicer/pull/515)（ [相关 PR](https://github.com/apache/arrow/pull/45084) ） 。这个补丁可以缓解内存不够的问题。


流式读取 JSON 文件是基础模型数据处理中的常见要求，因为许多数据集都以 JSONL 格式存储，并且尺寸巨大。
但是，Ray Datasets 中当前的实现不支持流式读取 JSON 文件，根因来源于其底层 Arrow 库（截至 Ray 版本 2.40 和 Arrow 版本 18.1.0）。

为了解决不支持流式 JSON 数据的原生读取问题，我们开发了一个流式加载接口，并为 Apache Arrow 贡献了一个第三方 [补丁](https://github.com/datajuicer/data-juicer/pull/515)（[PR 到 repo](https://github.com/apache/arrow/pull/45084)）。这将有助于缓解内存不足问题。使用此补丁后， Data-Juicer 的Ray模式将默认使用流式加载接口加载 JSON 文件。此外，如果输入变为 CSV 和 Parquet 文件，Ray模式下流式读取已经会自动开启。

### 去重

在 Ray 模式下，我们提供了一个优化过的基于 MinHash-LSH 的去重算子。我们使用 Ray Actors 实现了一个多进程的并查集和一个负载均衡的分布式算法 [BTS](https://ieeexplore.ieee.org/document/10598116) 来完成等价类合并操作。这个算子在 1,280 个CPU核上对 TB 大小级别的数据集去重只需要 3 个小时。我们的消融实验还表明相比于这个去重算子的初始实现版本，这些专门的优化项可以带来 2-3 倍的提速。

## 性能结果

### 不同数据规模的数据处理

我们在十亿样本规模的数据集上进行了实验。我们先准备了一个 56 万条样本的多模态数据集，并用不同的倍数（1-125,000倍）将其扩展来创建不同大小的数据集。下图的实验结果展示出了 Data-Juicer 的高扩展性。

![Overview](https://img.alicdn.com/imgextra/i3/O1CN01JV8wcC1oxn0G2xnBT_!!6000000005292-0-tps-1328-1742.jpg)

### 大规模数据集分布式去重

我们在 200GB、1TB、5TB 的数据集上测试了我们的基于 MinHash 的 Ray 去重算子，测试机器的 CPU 核数从 640 核到 1280 核。如下表所示，当数据集大小增长 5 倍，处理时间增长 4.02 到 5.62 倍。当 CPU 核数翻倍，处理时间较原来减少了 58.9% 到 67.1%。

| CPU 核数  | 200GB 耗时 | 1TB 耗时   | 5TB 耗时    |
|---------|----------|----------|-----------|
| 4 * 160 | 11.13 分钟 | 50.83 分钟 | 285.43 分钟 |
| 8 * 160 | 7.47 分钟  | 30.08 分钟 | 168.10 分钟 |

## 快速开始

在开始前，你应该安装 Data-Juicer 以及它的 `dist` 依赖需求：

```shell
uv pip install -v -e .  # 安装 Data-Juicer 的最小依赖需求
uv pip install -v -e ".[dist]"  # 包括 Ray 以及其他分布式相关的依赖库
```

然后启动一个 Ray 集群（参考 [Ray 文档](https://docs.ray.io/en/latest/ray-core/starting-ray.html) ）：

```shell
# 启动一个集群并作为头节点
ray start --head

# （可选）在其他节点或机器上连接集群
ray start --address='{head_ip}:6379'
```

我们在目录 `demos/process_on_ray/` 中准备了简单的例子，包括 2 个配置文件和 2 个测试数据集。

```text
demos/process_on_ray
├── configs
│   ├── demo.yaml
│   └── dedup.yaml
└── data
    ├── demo-dataset.json
    └── demo-dataset.jsonl
```

> [!Important]
> 如果你要在多个节点上运行这些例子，你需要将示例数据集放置与一个共享磁盘（如 NAS）上，并且将结果数据集导出到那里。你可以通过修改配置文件中的 `dataset_path` 和 `export_path` 参数来实现。

### 运行 Ray 模式样例

在配置文件 `demo.yaml` 中，我们将执行器类型设置为 "ray" 并且指定了自动的 Ray 地址。

```yaml
...
dataset_path: './demos/process_on_ray/data/demo-dataset.jsonl'
export_path: './outputs/demo/demo-processed'

executor_type: 'ray'  # 将执行器类型设置为 "ray"
ray_address: 'auto'  # 设置为自动 Ray 地址
...
```

运行这个例子，以使用 12 个常规算子处理测试数据集：

```shell
# 从源码运行处理工具
python tools/process_data.py --config demos/process_on_ray/configs/demo.yaml

# 使用命令行工具
dj-process --config demos/process_on_ray/configs/demo.yaml
```

Data-Juicer 会使用示例配置文件处理示例数据集，并将结果数据集导出到配置文件中 `export_path` 参数指定的目录中。

### 运行分布式去重样例

在配置文件 `dedup.yaml` 中，我们将执行器类型设置为 "ray" 并且指定了自动的 Ray 地址。我们使用了 MinHash 去重算子专门的分布式版本来对数据集去重。

```yaml
project_name: 'demo-dedup'
dataset_path: './demos/process_on_ray/data/'
export_path: './outputs/demo-dedup/demo-ray-bts-dedup-processed'

executor_type: 'ray'  # 将执行器类型设置为 "ray"
ray_address: 'auto'  # 设置为自动 Ray 地址

# process schedule
# a list of several process operators with their arguments
process:
  - ray_bts_minhash_deduplicator:  # minhash 去重算子的分布式版本
      tokenization: 'character'
```

运行该实例来对数据集去重：

```shell
# 从源码运行处理工具
python tools/process_data.py --config demos/process_on_ray/configs/dedup.yaml

# 使用命令行工具
dj-process --config demos/process_on_ray/configs/dedup.yaml
```

Data-Juicer 会使用示例配置文件对示例数据集去重，并将结果数据集导出到配置文件中 `export_path` 参数指定的目录中。

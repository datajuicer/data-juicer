# 数据集配置

本指南介绍如何在 Data-Juicer 菜谱中配置输入数据集。你将学习如何指向本地文件、远程 Hugging Face 或 arXiv 数据集、混合多个数据源、校验数据，以及处理边界情况。

## 支持的数据集格式

Data-Juicer 会自动检测本地文件的格式。支持的格式包括 `parquet`、`jsonl`、`json`、`csv`、`tsv`、`txt` 和 `jsonl.gz`。

### 本地数据集

指向本地文件系统上的文件或目录。`format` 字段是可选的——Data-Juicer 根据文件扩展名自动检测。

```yaml
dataset:
  configs:
    - type: local
      path: path/to/your/local/dataset.json
      format: json    # 可选
```

```yaml
dataset:
  configs:
    - type: local
      path: path/to/your/local/dataset.parquet
      format: parquet
```

完整示例参见 [local_json.yaml](https://github.com/datajuicer/data-juicer-hub/blob/main/dataset_config/local_json.yaml)。

### 远程 Hugging Face 数据集

从 Hugging Face Hub 加载任意数据集。将 `type` 设为 `remote`，`source` 设为 `huggingface`。

```yaml
dataset:
  configs:
    - type: 'remote'
      source: 'huggingface'
      path: "HuggingFaceFW/fineweb"
      name: "CC-MAIN-2024-10"   # 可选：数据集配置名
      split: "train"             # 可选：加载哪个 split
      limit: 1000                # 可选：限制加载的样本数
```

完整示例参见 [remote_huggingface.yaml](https://github.com/datajuicer/data-juicer-hub/blob/main/dataset_config/remote_huggingface.yaml)。

### arXiv 数据

arXiv 论文请使用[预处理工具](../tools/preprocess/README_ZH.md)下载并将 arXiv tar 包转为 Data-Juicer 可直接处理的 JSONL 格式。

### 其他格式

完整的支持格式和加载策略列表，参见 [load_strategy.py](https://github.com/datajuicer/data-juicer/blob/main/data_juicer/core/data/load_strategy.py)。

---

## 数据混合

默认执行器支持在 `dataset.configs` 中配置多个数据源，并按以下方式合并：

- 设置 `dataset.max_sample_num`：按各源权重分配总样本预算，再采样并合并。分配量超过某个源的样本数时，会重复采样以补足预算。
- 省略 `dataset.max_sample_num`：全量拼接各数据源。

Ray 执行器通过该配置加载单个数据源。

```yaml
dataset:
  max_sample_num: 10000    # 按各数据源权重分配的总样本数
  configs:
    - type: 'local'
      weight: 1.0
      path: 'path/to/json/file'
    - type: 'local'
      weight: 1.0
      path: 'path/to/csv/file'
```

完整示例参见 [mixture.yaml](https://github.com/datajuicer/data-juicer-hub/blob/main/dataset_config/mixture.yaml)。

---

## 数据校验

在菜谱中添加 `validators` 即可在处理前校验数据集。每个校验器检查数据的一个特定方面。

```yaml
dataset:
  configs:
    - type: local
      path: path/to/data.json

validators:
  - type: swift_messages
    min_turns: 2
    max_turns: 20
    sample_size: 1000
  - type: required_fields
    required_fields:
      - "text"
      - "metadata"
      - "language"
    field_types:
      text: "str"
      metadata: "dict"
      language: "str"
```

完整的校验器列表参见 [data_validator.py](https://github.com/datajuicer/data-juicer/blob/main/data_juicer/core/data/data_validator.py)。

---

## 读取选项

本地处理和分析时，可以用 `load_dataset_kwargs` 设置文件格式相关的读取选项。例如，只读取 Parquet 中的指定列：

```yaml
load_dataset_kwargs:
  columns: ['text', 'meta']
```

读取以制表符分隔的 CSV 时，改用 `load_dataset_kwargs: {delimiter: "\t"}`。请根据输入文件的格式选择对应选项。

### 加载进程数

`np` 同时设置加载和处理的默认进程数。要减少加载阶段使用的进程数：

```yaml
np: 16
load_dataset_kwargs:
  num_proc: 4
```

此配置使用 4 个进程加载数据，处理阶段默认使用 16 个进程。在 Python 中，显式传给 `DatasetBuilder.load_dataset(num_proc=...)` 的参数优先于这些配置。

### Ray 输入

使用 `ray`、`ray_partitioned` 或 RayAnalyzer 时，通过 `override_num_blocks` 设置所需的输入数据块数量。读取较大的 JSON 记录时，可以增大 `block_size`：

```yaml
read_options:
  block_size: 268435456  # 256 MiB
override_num_blocks: 64
```

`read_options` 接受 PyArrow JSON 读取选项，留空即使用默认值。Ray 将加载任务分配到集群；`load_dataset_kwargs.num_proc` 仅控制本地加载。

---

## 故障排除

### JSONL 逐行容错

本地处理和分析时，在配方中设置 `load_jsonl_lenient: true`，即可跳过格式错误的 JSONL 行并继续处理其余数据：

```yaml
load_jsonl_lenient: true
```

也可以只为一次运行开启此选项：

```bash
DATA_JUICER_JSONL_LENIENT=1 dj-process --config path/to/config.yaml
```

> **注意：** 仅读取 `.jsonl` / `.jsonl.gz` / `.jsonl.zst` 分片。同目录下的其它文件（如 `.json`）会被跳过并打警告。搜索日志中的 `[lenient jsonl]` 可查看哪些行被跳过。

### `Value is too big!` 报错

加载本地 JSONL 时，HuggingFace `datasets` 可能使用 `ujson` 解析，它无法处理超大整数。如果看到 `ValueError: Value is too big!`：

| 修复方式 | 做法 |
|---------|------|
| **使用标准库 json**（推荐） | `DATA_JUICER_USE_STDLIB_JSON=1 dj-process --config path/to/config.yaml` |
| **导出为字符串** | 在 JSON 源数据中将问题数值字段加引号。 |
| **改用 Parquet** | Parquet 使用 Arrow，完全不经过此代码路径。 |

---

## 旧版 `dataset_path` 配置

`dataset_path` 是最初的、更简单的输入指定方式。它可以用但缺乏上面 `dataset.configs` 方式的灵活性。

```yaml
# YAML
dataset_path: path/to/your/dataset.json
```

```bash
# 命令行
dj-process --dataset_path path/to/your/dataset.json

# 带权重的命令行
dj-process --dataset_path "0.5 path/to/dataset1.json 0.5 path/to/dataset2.json"
```

---

## 下一步

- [处理数据](ProcessData_ZH.md)——学习如何运行流水线和串联算子。
- [算子提要](Operators.md)——了解可用于数据的算子类型。
- [导出指南](Export_ZH.md)——控制输出格式和路径。

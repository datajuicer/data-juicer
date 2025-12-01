# 数据集配置指南
中文 | [EN](DatasetCfg.md)

本指南概述了如何在 Data-Juicer 框架中使用 YAML 格式配置数据集。允许您指定本地和远程数据集以及数据验证规则。

## 支持的数据集格式

### 本地数据集

`local_json.yaml` 配置文件用于指定以 JSON 格式本地存储的数据集。*path* 是必需的，用于指定本地数据集路径，可以是单个文件或目录。*format* 是可选的，用于指定数据集格式。
对于本地文件，DJ 将自动检测文件格式并相应地加载数据集。支持 parquet、jsonl、json、csv、tsv、txt 和 jsonl.gz 等格式
有关更多详细信息，请参阅 [local_json.yaml](https://github.com/datajuicer/data-juicer-hub/blob/main/dataset_config/local_json.yaml)。
```yaml
dataset:
  configs:
    - type: local
      path: path/to/your/local/dataset.json
      format: json
```

```yaml
dataset:
  configs:
    - type: local
      path: path/to/your/local/dataset.parquet
      format: parquet
```

### Remote Huggingface 数据集

`remote_huggingface.yaml` 配置文件用于指定 huggingface 数据集。*type* 和 *source* 固定为 'remote' 和 'huggingface'，以定位 huggingface 加载逻辑。*path* 是必需的，用于标识 huggingface 数据集。*name*、*split* 和 *limit* 是可选的，用于指定数据集名称/拆分并限制要加载的样本数量。
更多详细信息请参阅 [remote_huggingface.yaml](https://github.com/datajuicer/data-juicer-hub/blob/main/dataset_config/remote_huggingface.yaml)。

```yaml
dataset:
  configs:
    - type: 'remote'
      source: 'huggingface'
      path: "HuggingFaceFW/fineweb"
      name: "CC-MAIN-2024-10"
      split: "train"
      limit: 1000
```

### 远程 Arxiv 数据集

`remote_arxiv.yaml` 配置文件用于指定以 JSON 格式远程存储的数据集。*type* 和 *source* 固定为 'remote' 和 'arxiv'，以定位 arxiv 加载逻辑。 *lang*、*dump_date*、*force_download* 和 *url_limit* 是可选的，用于指定数据集语言、转储日期、强制下载和 URL 限制。
有关更多详细信息，请参阅 [remote_arxiv.yaml](https://github.com/datajuicer/data-juicer-hub/blob/main/dataset_config/remote_arxiv.yaml)。

```yaml
dataset:
  configs:
    - type: 'remote'
      source: 'arxiv'
      lang: 'en'
      dump_date: 'latest'
      force_download: false
      url_limit: 2
```

### 其他支持的数据集格式

有关更多详细信息和支持的数据集格式，请参阅 [load_strategy.py](https://github.com/datajuicer/data-juicer/blob/main/data_juicer/core/data/load_strategy.py)。

## 其他功能

### 数据混合

`mixture.yaml` 配置文件演示了如何指定数据混合规则。DJ 将通过对数据集的一部分进行采样并应用适当的权重来混合数据集。
有关更多详细信息，请参阅 [mixture.yaml](https://github.com/datajuicer/data-juicer-hub/blob/main/dataset_config/mixture.yaml)。
```yaml
dataset:
  max_sample_num: 10000
  configs:
    - type: 'local'
      weight: 1.0
      path: 'path/to/json/file'
    - type: 'local'
      weight: 1.0
      path: 'path/to/csv/file'
```

### 数据验证

`validator.yaml` 配置文件演示了如何指定数据验证规则。DJ 将通过对数据集的一部分进行采样并应用验证规则来验证数据集。
有关更多详细信息和支持的验证器，请参阅 [data_validator.py](https://github.com/datajuicer/data-juicer/blob/main/data_juicer/core/data/data_validator.py)。
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

### 旧版 dataset_path 配置

`dataset_path` 配置是指定数据集路径的历史版本方式。它简单易用，但缺乏灵活性。它可以在 yaml 或命令行输入中使用。一些示例：

命令行输入：
```bash
# 命令行输入
dj-process --dataset_path path/to/your/dataset.json

# 带权重的命令行输入
dj-process --dataset_path 0.5 path/to/your/dataset1.json 0.5 path/to/your/dataset2.json
```

Yaml 输入：
```yaml
dataset_path：path/to/your/dataset.json
```
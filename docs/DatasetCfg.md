# Dataset Configuration Guide
EN | [中文](DatasetCfg_ZH.md)

This guide provides an overview of how to configure datasets using YAML format in the Data-Juicer framework. The configurations allow you to specify local and remote datasets, with data validation rules.

## Supported Dataset Formats

### Local Dataset

The `local_json.yaml` configuration file is used to specify datasets stored locally in JSON format. *path* is required to specify the local dataset path, either a single file or a directory. *format* is optional to specify the dataset format.
For local files, DJ will automatically detect the file format and load the dataset accordingly. Formats like parquet, jsonl, json, csv, tsv, txt, and jsonl.gz are supported
Refer to [local_json.yaml](https://github.com/datajuicer/data-juicer-hub/blob/main/dataset_config/local_json.yaml) for more details.
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

### Remote Huggingface Dataset

The `remote_huggingface.yaml` configuration file is used to specify huggingface datasets. *type* and *source* are fixed to 'remote' and 'huggingface' to locate huggingface loading logic. *path* is required to identify the huggingface dataset. *name*, *split* and *limit* are optional to specify the dataset name/split and limit the number of samples to load.
Refer to [remote_huggingface.yaml](https://github.com/datajuicer/data-juicer-hub/blob/main/dataset_config/remote_huggingface.yaml) for more details.

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

### Remote Arxiv Dataset

The `remote_arxiv.yaml` configuration file is used to specify datasets stored remotely in JSON format. *type* and *source* are fixed to 'remote' and 'arxiv' to locate arxiv loading logic. *lang*, *dump_date*, *force_download* and *url_limit* are optional to specify the dataset language, dump date, force download and url limit.
Refer to [remote_arxiv.yaml](https://github.com/datajuicer/data-juicer-hub/blob/main/dataset_config/remote_arxiv.yaml) for more details.

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

### Other Supported Dataset Formats

Refer to [load_strategy.py](https://github.com/datajuicer/data-juicer/blob/main/data_juicer/core/data/load_strategy.py) for more details and supported dataset formats.


## Other features 

### Data Mixture  

The `mixture.yaml` configuration file demonstrates how to specify data mixture rules. DJ will mix the datasets by sampling a portion of the dataset and applying proper weights.
Refer to [mixture.yaml](https://github.com/datajuicer/data-juicer-hub/blob/main/dataset_config/mixture.yaml) for more details.
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


### Data Validation 

The `validator.yaml` configuration file demonstrates how to specify data validation rules. DJ will validate the dataset by sampling a portion of the dataset and applying the validation rules.
Refer to [data_validator.py](https://github.com/datajuicer/data-juicer/blob/main/data_juicer/core/data/data_validator.py) for more details and supported validators.
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


### Legacy dataset_path Configuration

The `dataset_path` configuration is the original way to specify the dataset path. It's simplistic and easy to use, but lacks flexibility. It can be used in yaml or command line input. Some examples:

Command line input:
```bash
# command line input
dj-process --dataset_path path/to/your/dataset.json

# command line input with weights
dj-process --dataset_path 0.5 path/to/your/dataset1.json 0.5 path/to/your/dataset2.json
```

Yaml input:
```yaml
dataset_path: path/to/your/dataset.json
```

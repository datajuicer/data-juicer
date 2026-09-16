# Evaluation Results Recorder

使用 [`wandb_writer.py`](wandb_writer.py) 将评测结果记录到 [W&B](https://wandb.ai/) (wandb) 并可视化展示。

`wandb_writer.py` 能够:

- 可视化模型在训练过程中各项评测指标的变化
![Metrics](../../../docs/imgs/eval-02.png "指标变化")
- 制作排行榜来比较不同模型的各项评测指标
![Leaderboard](../../../docs/imgs/eval-01.png "排行榜")

## 用法

```shell
python wandb_writer.py --config <config_file> [--print-only]
```

- `config_file`: yaml 配置文件路径（配置项细节请见[配置](#配置)）
- `--print-only`: 仅将结果打印到命令行，不执行写 wandb 操作，用于调试

## 配置

我们在 [`config`](config) 文件夹中为三种不同的情况提供了三个示例文件，其中通用的配置项格式如下:

```yaml
project: <str>   # wandb 项目名
base_url: <str>  # wandb 实例 url
# other specific configuration items
```

其他配置项根据实际需要填写。

### 从 HELM 输出目录中提取评测结果

如下配置项用于从 HELM 的输出目录中提取评测结果并记录到 wandb 中。

```yaml
# general configurations
# ...

evals:  # evaluations to record
  - eval_type: helm    # 目前仅支持 helm
    model_name: <str>  # 模型名字
    source: helm  # helm 或 file，这里使用 helm 来从 helm 输出目录提取评测结果
    helm_output_dir: <your helm output dir path>
    helm_suite_name: <your helm suite name>
    token_per_iteration: <tokens per iteration in billions>
    benchmarks:  # 需要记录到 wandb 的评测指标，如下是一些样例
      - name: mmlu
        metrics:
          - EM
      - name: boolq
        metrics:
          - EM
      - name: narrative_qa
        metrics:
          - F1
      - name: hellaswag
        metrics:
          - EM
      - ...
```

> 本工具使用 HELM 的 16 组核心指标作为默认评测指标。如果配置中没有提供 benchmarks 域，则会自动使用如下16个评测指标：
>  ```
>  mmlu.EM, raft.EM, imdb.EM, truthful_qa.EM, summarization_cnndm.ROUGE-2, summarization_xsum.ROUGE-2, boolq.EM, msmarco_trec.NDCG@10, msmarco_regular.RR@10, narrative_qa.F1, natural_qa_closedbook.F1, natural_qa_openbook_longans.F1, civil_comments.EM, hellaswag.EM, openbookqa.EM
>  ```

### 从配置文件中读取评测结果

评测结果可以直接写在配置文件中，该选项主要用于快速向 wandb 记录已有的评测结果。

```yaml
# general configurations
# ...

evals:  # evaluations to record
  - eval_type: helm
    model_name: llama-7B  # 模型名字
    source: file  # helm 或 file，这里使用 file 来直接从配置文件提取评测结果
    token_num: 1000 # 需要提供该模型训练时使用的 token 数量（单位：B）
    eval_result:  # 需要被记录的评测结果，如下为一些样例
      mmlu:
        EM: 0.345
      boolq:
        EM: 0.751
      narrative_qa:
        F1: 0.524
      hellaswag:
        EM: 0.747
      ...
```

### 构建排行榜

如下配置用于对在同一个 wandb 项目中的数据生成排行榜。

```yaml
# general configurations
# ...
leaderboard: True
leaderboard_metrics:  # 排行榜中需要统计的指标（仅有包含全部指标评测结果的模型才会进入榜单）
  - mmlu.EM
  - boolq.EM
  - quac.F1
  - hellaswag.EM
  - ...
excluded_models:   # 不参与排行榜的模型名称
  - <model to exclude>
  - ...
```

> 工具使用 HELM 的 16 组核心指标作为默认的排行榜指标。如果没有提供 `leaderboard_metrics` 域，则会自动使用这 16 组核心指标。

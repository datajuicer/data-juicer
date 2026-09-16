# Auto Evaluation Toolkit

在训练过程中自动评测您的模型并持续监控指标的变化。

## 准备工作

1. 多台GPU机器（至少2台，一台用于运行评测，其他机器用于训练模型）。

2. 将共享文件系统（例如NAS）挂载到上述机器上的相同路径（例如 `/mnt/shared`）。

3. 在共享文件系统中安装 Data-Juicer（例如 `/mnt/shared/code/data-juicer`）。

4. 根据 [thirdparty/README_ZH.md](../../thirdparty/README_ZH.md) 在每台机器上安装第三方依赖项（Megatron-LM 和 HELM）。

5. 准备数据集和 tokenizer，在共享文件系统（例如 `/mnt/shared/dataset`）中使用 Megatron-LM 提供的预处理工具将数据集预处理为 mmap 格式（更多详细信息，请参阅 Megatron-LM 的 [README](../../thirdparty/Megatron-LM/README.md)）。

6. 在训练机器上运行 Megatron-LM 并将检查点保存在共享文件系统中（例如 `/mnt/shared/checkpoints`）。

## 用法

通过 [`evaluator.py`](evaluator.py) 来使用 HELM 或 OpenAI API 自动评估您的模型。

```shell
python tools/evaluator.py  \
    --config <config>      \
    --begin-iteration     <begin_iteration>     \
    [--end-iteration      <end_iteration>]      \
    [--iteration-interval <iteration_interval>] \
    [--check-interval <check_interval>]         \
    [--model-type     <model_type>]             \
    [--eval-type      <eval_type>]
```

- `config`: 包含运行评估所需的各种设置的 yaml 文件（详细信息请参阅[配置](#配置)）。
- `begin_iteration`: 首个需要评估的检查点的 iteration。
- `end_iteration`: 最后一个需要评估的检查点的 iteration。如果没有设置，该进程将持续监控训练过程中产生的检查点。
- `iteration_interval`: 两次评测之间的 iteration 间隔，默认为 1000。
- `check_interval`: 两次检查是否有满足条件检查点的时间间隔，默认为 30 分钟。
- `model_type`: 被评测的模型类型，当前支持 `megatron` 和 `huggingface`。
    - `megatron`: 即 Megatron-LM 检查点，默认为此项。
    - `huggingface`: 即 HuggingFace 模型。
- `eval-type`: 运行的评估类型，当前支持 `helm` 和 `gpt`。
    - `helm`: 使用 HELM 评测，默认为此项，当前仅支持评测 Megatron-LM 模型。
    - `gpt`: 使用 OpenAI API 评测，更多细节请见 [gpt_eval/README_ZH.md](gpt_eval/README_ZH.md)。

> 例如：
> ```shell
> python evaluator.py --config <config_file> --begin-iteration 2000 --iteration-interval 1000 --check-interval 10
> ```
> 将会使用 HELM 从 Megatron-LM 训练到 2000 iteration 开始每隔 1000 iterations 评测一个检查点，并会每隔 30 分钟检测一次是否有新的检查点生成

在运行 [evaluator.py](evaluator.py) 之后, 可以使用 [recorder/wandb_writer.py](recorder/wandb_writer.py) 将评测结果记录到 wandb 并可视化展示，更多细节请参考 [recorder/README_ZH.md](recorder/README_ZH.md)。

## 配置

`config_file` 文件格式如下:

```yaml
auto_eval:
  project_name: <str> # 项目名称
  model_name: <str>   # 模型名称
  cache_dir: <str>    # 缓存目录路径
  megatron:
    process_num: <int>     # 运行 megatron-lm 所需的进程数
    megatron_home: <str>   # Megatron-LM 代码根目录
    checkpoint_path: <str> # 检查点保存根目录
    tokenizer_type: <str>  # 目前支持 gpt2 或 sentencepiece
    vocab_path: <str>      # 针对 gpt2 tokenizer 的配置项， vocab 文件的路径
    merge_path: <str>      # 针对 gpt2 tokenizer 的配置项， merge 文件的路径
    tokenizer_path: <str>  # 针对 sentencepiece tokenizer 的配置项， model 文件的路径
    max_tokens: <int>      # 在执行生成任务时最大生成的 token 数量
    token_per_iteration: <float> # 训练时每次迭代所使用的 token 数量（单位：B）
  helm:
    helm_spec_template_path: <str> # helm 评测模版文件， 默认为 tools/evaluator/config/helm_spec_template.conf，可通过修改此文件来调整运行的评测
    helm_output_path: <str>  # helm 输出目录路径
    helm_env_name: <str>     # helm 的 conda 环境名
  gpt_evaluation:
    # openai config
    openai_api_key: <str>
    openai_organization: <str>
    # files config
    question_file: <str>  # 默认为 tools/evaluator/gpt_eval/config/question.jsonl
    baseline_file: <str>  # 默认为 tools/evaluator/gpt_eval/answer/openai/gpt-3.5-turbo.jsonl
    prompt_file: <str >   # 默认为 tools/evaluator/gpt_eval/config/prompt.jsonl
    reviewer_file: <str>  # 默认为 tools/evaluator/gpt_eval/config/reviewer.jsonl
    answer_file: <str>    # 生成的回答文件的路径
    result_file: <str>    # 生成的评价文件的路径

```

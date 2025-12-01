# Auto Evaluation Toolkit

Automatically evaluate your model and monitor changes of metrics during the training process.

## Preparation

1. Multiple GPU machines (at least 2, one for evaluation, the other for training).

2. Mount a shared file system (e.g., NAS) to the same path (e.g., `/mnt/shared`) on the above machines.

3. Install Data-Juicer in the shared file system (e.g., `/mnt/shared/code/data-juicer`).

4. Install thirdparty dependencies (Megatron-LM and HELM) accoroding to [thirdparty/README.md](../../thirdparty/README.md) on each machine.

5. Prepare your dataset and tokenizer, preprocess your dataset with Megatron-LM into mmap format (see [README](../../thirdparty/Megatron-LM/README.md) of Megatron-LM for more details) in the shared file system (e.g., `/mnt/shared/dataset`).

6. Run Megatron-LM on training machines and save the checkpoint in the shared file system (e.g., `/mnt/shared/checkpoints`).

## Usage

Use [evaluator.py](evaluator.py) to automatically evaluate your models with HELM and OpenAI API.

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

- `config`: a yaml file containing various settings required to run the evaluation (see [Configuration](#configuration) for details)
- `begin_iteration`: iteration of the first checkpoint to be evaluated
- `end_iteration`: iteration of the last checkpoint to be evaluated. If not set, continuously monitor the training process and evaluate the generated checkpoints.
- `iteration_interval`: iteration interval between two checkpoints, default is 1000 iterations
- `check_interval`: time interval between checks, default is 30 minutes
- `model_type`: type of your model, support `megatron` and `huggingface` for now
    - `megatron`: evaluate Megatron-LM checkpoints (default)
    - `huggingface`: evaluate HuggingFace model, only support gpt eval type
- `eval-type`: type of the evaluation to run, support `helm` and `gpt` for now
    - `helm`: evaluate your model with HELM (default), you can change the benchmarks to run by modifying the helm specific template file
    - `gpt`: evaluate your model with OpenAI API, more details can be found in [gpt_eval/README.md](gpt_eval/README.md).

> e.g.,
> ```shell
> python evaluator.py --config <config_file> --begin-iteration 2000 --iteration-interval 1000 --check-interval 10
> ```
> will use HELM to evaluate a Megatron-LM checkpoint every 1000 iterations starting from iteration 2000, and check whether there is a new checkpoint meets the condition every 10 minutes.

After running the [evaluator.py](evaluator.py), you can use [recorder/wandb_writer.py](recorder/wandb_writer.py) to visualize the evaluation results, more details can be found in [recorder/README.md](recorder/README.md).

## Configuration

The format of `config_file` is as follows:

```yaml
auto_eval:
  project_name: <str> # your project name
  model_name: <str>   # your model name
  cache_dir: <str>    # path of cache dir
  megatron:
    process_num: <int>     # number of process to run megatron
    megatron_home: <str>   # root dir of Megatron-LM
    checkpoint_path: <str> # path of checkpoint dir
    tokenizer_type: <str>  # support gpt2 or sentencepiece for now
    vocab_path: <str>      # configuration for gpt2 tokenizer type, path to vocab file
    merge_path: <str>      # configuration for gpt2 tokenizer type, path to merge file
    tokenizer_path: <str>  # configuration for sentencepiece tokenizer type, path to model file
    max_tokens: <int>      # max tokens to generate in inference
    token_per_iteration: <float> # billions tokens per iteration
  helm:
    helm_spec_template_path: <str> # path of helm spec template file, default is tools/evaluator/config/helm_spec_template.conf
    helm_output_path: <str>  # path of helm output dir
    helm_env_name: <str>     # helm conda env name
  gpt_evaluation:
    # openai config
    openai_api_key: <str>       # your api key
    openai_organization: <str>  # your organization
    # files config
    question_file: <str>  # default is tools/evaluator/gpt_eval/config/question.jsonl
    baseline_file: <str>  # default is tools/evaluator/gpt_eval/answer/openai/gpt-3.5-turbo.jsonl
    prompt_file: <str >   # default is tools/evaluator/gpt_eval/config/prompt.jsonl
    reviewer_file: <str>  # default is tools/evaluator/gpt_eval/config/reviewer.jsonl
    answer_file: <str>    # path to generated answer file
    result_file: <str>    # path to generated review file
```

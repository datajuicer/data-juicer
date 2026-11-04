## 自动化评测：HELM 评测及可视化

### 什么是自动化评测

这里的自动化评测是指对模型训练过程中得到的检查点自动使用评测数据集执行评测并记录评测结果。

### 为什么要自动化评测

在大模型训练过程中仅通过训练 loss 难以准确评估模型的实际性能，需要使用多种评测数据集从各维度评价模型的能力，实时持续监控各项指标随训练迭代的变化情况，并与其他基线模型做比较，从而判断模型是否还有继续训练的价值，节省训练开销。

但上述的评测流程重复繁琐且易于出错（例如：检查是否有新检查点、运行评测、记录评测结果、结果可视化等），而本自动化评测工具则能够提供一键式的解决方案，节省大量人力成本。

将自动化评测与数据预处理结合可及时通过评测结果判断数据预处理阶段配置的合理性，形成反馈循环，更快地找出更合理的数据预处理方法。

### 如何使用自动化评测：以 HELM 和 Megatron-LM 为例

> - HELM 是 Stanford 开源的一套评测框架，包含了丰富的测试数据集以及多种评测指标，现已评测了超过 50 种可公开访问的大模型
> - Megatron-LM 是 Nvidia 开源的 Transformer 训练框架，支持大规模分布式训练且性能极高，是多个知名大模型训练框架 (GPT-Neox, Megatron-Deepspeed等) 的基础

本节介绍如何使用本工具中的 HELM 评测框架自动化评测 Megatron-LM 训练得到的 GPT2 模型，运行该样例需要至少一张 V100 或其他更高规格的显卡，该样例在计算资源允许的情况下可以扩展支持更大的模型。

#### 1. 准备环境

由于 HELM 和 Megatron-LM 的依赖项繁杂，为了减少安装过程中遇到的依赖问题，推荐基于 NGC 的 Pytorch 容器 (`nvcr.io/nvidia/pytorch:22.12-py3`) 构建环境。

假设您的数据集 jsonl 文件路径为 `/dataset/dataset.jsonl`，Data-Juicer 的代码路径为 `/code/data-juicer`，只需执行如下指令:

```shell
docker pull nvcr.io/nvidia/pytorch:22.12-py3
docker run --gpus all --ipc=host --ulimit memlock=-1 -it --rm -v /dataset:/workspace/data -v /code/data-juicer:/worksapce/data-juicer nvcr.io/nvidia/pytorch:22.12-py3
```
docker 容器成功运行后在容器内运行安装脚本并登录 wandb：

```shell
cd /workspace/data-juicer/thirdparty/LLM_ecosystems
./setup_megatron.sh
./setup_helm.sh
wandb login
```

安装完成后在容器外运行如下指令将容器保存下来方便后续使用，其中的 `container_id` 可通过 `docker ps` 获取

```shell
docker commit <container_id> data-juicer-eval
```

#### 2. 将数据集预处理为 Megatron-LM 可识别的格式

进入 Megatron-LM 目录并执行数据预处理脚本，该脚本会将 data-juicer 处理好的 jsonline（假设路径为 `/workspace/data/dataset.jsonl`）文件转化为二进制格式，并保存为 `/workspace.data/dataset_text_document.bin` 和 `/workspace.data/dataset_text_document.idx` 两个文件。

```shell
cd /workspace/data-juicer/thirdparty/LLM_ecosystems/Megatron-LM
python tools/preprocess_data.py              \
       --input /workspace/data/dataset.jsonl \
       --output-prefix dataset \
       --vocab /workspace/data-juicer/demos/gpt2-vocab.json \
       --dataset-impl mmap \
       --tokenizer-type GPT2BPETokenizer \
       --split-sentences
```


#### 3. 启动 Megatron-LM 训练

进入 Megatron-LM 目录并执行如下指令

```shell
cd /workspace/data-juicer/thirdparty/LLM_ecosystems/Megatron-LM
nohup bash /workspace/data-juicer/demos/auto_eval_helm/pretrain_example.sh > train.log 2>&1 &
```

> `pretrain_example.sh` 会执行 GPT2 模型的训练，从 `/workspace/data` 路径获取 `.bin` 和 `.idx` 二进制数据集文件，训练 200000 个 iteration，并每隔 2000 个 iteration 将模型以检查点形式保存在 `/workspace/data/checkpoints/GPT2` 目录下。
> 可通过修改 `pretrain_example.sh` 来调整模型的规模、数据集路径、检查点路径等参数，更详细的配置信息请参考 [Megatron-LM 官方仓库](https://github.com/NVIDIA/Megatron-LM)。

#### 4. 启动自动化评测

进入 data-juicer 的自动评测工具库目录并执行如下指令：

```shell
cd /workspace/data-juicer/tools/eval
python evaluator.py --config /workspace/data-juicer/demos/evaluator.yaml --begin-iteration 2000 --end-iteration 200000 --interation-interval 2000 --check-interval 30
```

该脚本会每隔 30 分钟检测一次 `/workspace/data/checkpoints/GPT2` 目录，并从 2000 iteration 开始每隔 2000 iteration 对检查点执行一次 HELM 评测并将评测结果记录至 wandb，直到评测完 200000 iteration 对应的检查点为止，您可以在 wandb 上查看已完成的评测结果，下图展示了模型训练到 140000 iteration 时 wandb 上的可视化展示结果。

![训练过程中的评测结果展示](imgs/eval-02.png)

> 本示例运行的 HELM 测试集配置位于 `/workspace/data-juicer/demos/helm_spec_template.conf` 中，这里仅选用了 MMLU 的一个子集、boolq、narrative_qa 以及 hellaswag 作为样例，完整的测试集配置位于 `/workspace/data-juicer/tools/eval/config/helm_spec_template.conf` 中。


#### 5. 汇总评测结果

为了体现模型训练效果，可以借助 `wandb_writer.py` 将多个模型的评测结果汇总到同一个排行榜上进行比较。

首先，将基线模型的各项评测结果直接记录到 wandb：

```shell
cd /workspace/data-juicer/tools/eval/recorder
python wandb_writer.py --config /workspace/data-juicer/demos/baselines.yaml
```

在确保所有参与排行榜的模型的评测结果都已经记录到 wandb 之后，使用 `leaderboard.yaml` 构建排行榜：

```shell
cd /workspace/data-juicer/tools/eval/recorder
python wandb_writer.py --config /workspace/data-juicer/demos/leaderboard.yaml
```

![排行榜](imgs/eval-01.png)

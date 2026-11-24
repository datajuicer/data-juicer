# 大语言模型生态

本目录包含了 Auto Evaluation Toolkit 的第三方依赖项，更多细节请参考 `tools/evaluator/README_ZH.md`。

## 安装

Auto Evaluation Toolkit 依赖于定制化的 [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) 和 [HELM](https://github.com/stanford-crfm/helm)。
为了避免安装这些软件包时可能出现的依赖项问题，我们建议使用 NGC 的 Pytorch 容器(`nvcr.io/nvidia/pytorch:22.12-py3`)。
假设您共享文件系统的路径（即数据集和模型检查点的存储路径）为`/mnt/shared`，请使用如下指令启动 Docker 容器。

```shell
docker pull nvcr.io/nvidia/pytorch:22.12-py3
docker run --gpus all -it --rm -v /mnt/shared:/workspace
```

启动 Docker 容器后，在容器中运行以下脚本以安装 Megatron-LM 或 HELM。

训练机只需要安装 Megatron-LM：

```shell
./setup_megatron.sh
```

评测机需要同时安装 Megatron-LM 和 HELM

```shell
./setup_megatron.sh
./setup_helm.sh
```

工具包使用 [WandB](https://wandb.ai/) 来监视训练期间各指标的趋势。上面的步骤中已安装 wandb，您只需要运行 `wand login` 并输入 wandb API 密钥即可。如果您有自己的 wandb 实例，请运行以下脚本。

```shell
wandb login --host <URL of your wandb instance>
＃输入您的 API 密钥
```

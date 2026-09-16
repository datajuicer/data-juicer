# LLM Ecosystems

Dependencies of Auto Evaluation Toolkit, see [`tools/evaluator/README.md`](../tools/evaluator/README.md) for more details.

## Installation

The auto-evaluation toolkit requires customized [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) and [HELM](https://github.com/stanford-crfm/helm).
To avoid dependency problems when installing those packages, we recommend using NGC's PyTorch container (`nvcr.io/nvidia/pytorch:22.12-py3`).
Assuming the path to your shared file system (where your data and model checkpoints are saved) is `/mnt/shared`, start the docker container with following commands.

```shell
docker pull nvcr.io/nvidia/pytorch:22.12-py3
docker run --gpus all -it --rm -v /mnt/shared:/workspace
```

After starting the docker container, run the following scripts in the container to install Megatron-LM or HELM.

The training machines only need to install Megatron-LM:

```shell
./setup_megatron.sh
```

The evaluation machine needs to install both Megatron-LM and HELM

```shell
./setup_megatron.sh
./setup_helm.sh
```

The toolkit use [W&B](https://wandb.ai/) (wandb) to monitor the trend of metrics during training. Above steps have installed wandb, and you only need to run `wandb login` and enter your wandb API key. If you have your own instance of wandb, run the following script.

```shell
wandb login --host <URL of your wandb instance>
# enter your api key
```


# Scripts for Running on Multi Nodes


#### Running Using DLC(Deep Learing Containers)

Internally we use [DLC](https://www.alibabacloud.com/help/zh/pai/user-guide/container-training/) from [PAI](https://www.alibabacloud.com/zh/product/machine-learning) to process data on multiple nodes.

The scripts to run are in ./dlc folder.

#### Running Using Slurm

We provide scripts to support running on slurm, see ./run_slurm.sh.

You can also manually partition the data according to specific circumstances and then use Slurm to run it on multiple machines by yourself.

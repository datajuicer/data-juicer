# Hyper-parameter Optimization for Data Recipe

## Auto-HPO based on 3-Sigma principles
A simple automatic hyper-parameter optimization method for data recipes is to assume that outlier data is harmful to training.
We thus can introduce the 3-sigma principle to automatically determine the hyper-parameters and filter the data.

Specifically, assuming that a certain analysis dimension of the original data obeys a normal distribution and has random errors, we can set the upper and lower bounds of the filtering OP in this dimension to three times the standard deviation based on the statistics produced by the DataJuicer's Analyzer.

$$P(|x-\mu| > 3\sigma) \leq 0.003$$

To automate this process, we provide the tool which can be used as follows:
```shell
# cd tools/hpo
# usage 1: do not save the refined recipe
python execute_hpo_3sigma.py --config <data-process-cfg-file-path>
# usage 2: save the refined recipe at the given path
python execute_hpo_3sigma.py --config <data-process-cfg-file-path> --path_3sigma_recipe <data-process-cfg-file-after-refined-path>

# e.g., usage 1
python execute_hpo_3sigma.py --config configs/process.yaml
# e.g., usage 2
python execute_hpo_3sigma.py --config configs/process.yaml --path_3sigma_recipe configs/process_3sigma.yaml
```

## Auto-HPO with WandB

We incorporate an automated HPO tool, WandB [Sweep](https://docs.wandb.ai/guides/sweeps), into Data-Juicer to streamline the finding of good data processing hyper-parameters.
With this tool, users can investigate correlations and importance scores of
specific hyper-parameters of data recipes from the HPO view.

**Note**: this is an experimental feature. Auto-HPO for data recipes still has
a large room to explore. Feel free to provide more suggestions, discussion,
and contribution via new PRs!


### Prerequisite
You need to install data-juicer first.
Besides, the tool leverages WandB, install it via `pip install wandb`.
Before using this tool, you need to run
```wandb login``` and enter your WandB
API key.
If you have your own instance of WandB (e.g., [locally-hosted machine](https://docs.wandb.ai/guides/hosting/)), run the following script:

```shell
wandb login --host <URL of your wandb instance>
# enter your api key
```



### Usage and Customization

Given a data recipe, characterized by specified configuration file
`<data-process-cfg-file-path>`, you can use `execute_hpo_wandb.py` to search the
hyper-parameter space defined by `<hpo-cfg-file-path>`.
```shell
# cd tools/hpo
python execute_hpo_wandb.py --config <data-process-cfg-file-path> --hpo_config <hpo-cfg-file-path>

# e.g.,
python execute_hpo_wandb.py --config configs/process.yaml --hpo_config configs/quality_score_hpo.yaml
```

For the configuration for data recipe, i.e., `<data-process-cfg-file-path>`,
please see more details in our [guidance](https://github.com/datajuicer/data-juicer#build-up-config-files). As for the configuration
for HPO, i.e., `<hpo-cfg-file-path>`, please refer to sweep [guidance](https://docs.wandb.ai/guides/sweeps/define-sweep-configuration).

We provide an illustrative objective "quality_score" in `hpo/objects.py`,
which uses quality scorer to measure the processed data, and links the average scores to hyper-parameters of data recipes.
After running it, you will get the result similar to: ![img](https://img.alicdn.com/imgextra/i2/O1CN017fT4Al1bVldeuCmiI_!!6000000003471-2-tps-2506-1710.png)


You can implement your own HPO objective in `get_hpo_objective` function, e.g., linking the data
recipes to
- model_loss (by replacing the quality scorer into a training procedure),
- downstream_task (by replacing the quality scorer with training and evaluation procedures), or
- some synergy measures that combine metrics you are interested in, such that the trade-offs from different views can be explored.

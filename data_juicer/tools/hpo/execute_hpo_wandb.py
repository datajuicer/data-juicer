import sys

import wandb
import yaml
from jsonargparse import namespace_to_dict
from objects import get_hpo_objective

from data_juicer.config import init_configs, merge_config

# 1: load the defined search space
sweep_cfg_file_path = None
for i in range(len(sys.argv) - 1):
    if sys.argv[i] == "--hpo_config":
        sweep_cfg_file_path = sys.argv[i + 1]
        break
if not sweep_cfg_file_path:
    raise ValueError("Not found --hpo_config, you should specify your " "hpo cfg file path following `--hpo_config`")
with open(sweep_cfg_file_path) as f:
    sweep_configuration = yaml.safe_load(f)


def search():
    wandb.init(project=sweep_configuration["sweep_name"])

    # 2.1: Choose objective that links the hyper-parameters you want to search
    object_func = get_hpo_objective(sweep_configuration["metric"]["name"])

    dj_cfg = init_configs()
    # merge the new hyper-parameters selected by HPO scheduler
    dj_cfg = merge_config(dj_cfg, wandb.config)
    wandb.config = namespace_to_dict(dj_cfg)  # for configuration track

    # 2.2: calculate objective using new hyper-parameters, track the results
    score = float(object_func(dj_cfg))
    wandb.log({sweep_configuration["metric"]["name"]: score})


# 3: Start the sweep, iteratively search hyper-parameters
sweep_id = wandb.sweep(sweep=sweep_configuration, project=sweep_configuration["sweep_name"])

wandb.agent(
    sweep_id,
    function=search,
    count=sweep_configuration["sweep_max_count"] if "sweep_max_count" in sweep_configuration else None,
)

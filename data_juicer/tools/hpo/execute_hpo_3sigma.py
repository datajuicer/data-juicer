import copy
import json
import sys

import yaml
from jsonargparse import Namespace, namespace_to_dict
from loguru import logger

from data_juicer.config import init_configs, prepare_cfgs_for_export
from data_juicer.core import Analyzer, DefaultExecutor
from data_juicer.utils.constant import StatsKeys


@logger.catch(reraise=True)
def main():
    path_k_sigma_recipe = None
    for i in range(len(sys.argv) - 1):
        if sys.argv[i] == "--path_k_sigma_recipe":
            path_k_sigma_recipe = sys.argv[i + 1]

    # 1. analyze using the given initial recipe
    cfg = init_configs()
    logger.info("Begin to analyze data using the given initial recipe")

    analyzer = Analyzer(cfg)
    analyzer.run()
    df = analyzer.overall_result

    # 2. adjust the hyper-parameters of the given recipe with k-sigma rule
    modify_recipe_k_sigma(cfg, df, path_k_sigma_recipe)

    # 3. process the data using the refined recipe
    logger.info("Begin to process the data with refined recipe")
    if cfg.executor_type == "default":
        executor = DefaultExecutor(cfg)
    elif cfg.executor_type == "ray":
        from data_juicer.core.executor.ray_executor import RayExecutor

        executor = RayExecutor(cfg)
    executor.run()


def modify_recipe_k_sigma(cfg, df, path_k_sigma_recipe, k=3):
    # get the mapping from op_name to their mu and sigma
    mean_series = df[df.index == "mean"]
    stats_key_to_mean = mean_series.iloc[0, :].to_dict()
    std_series = df[df.index == "std"]
    stats_key_to_std = std_series.iloc[0, :].to_dict()
    op_name_to_stats_key = StatsKeys.get_access_log(dj_cfg=cfg)
    logger.info(f"Begin to modify the recipe with {k}-sigma rule")
    for i in range(len(cfg.process)):
        if isinstance(cfg.process[i], Namespace):
            cfg.process[i] = namespace_to_dict(cfg.process[i])
    for process in cfg.process:
        op_name, args = list(process.items())[0]
        temp_args = copy.deepcopy(args)
        if op_name not in op_name_to_stats_key:
            # skip the op such as `clean_email_mapper`
            continue
        stats_keys = op_name_to_stats_key[op_name]
        for stats_key in stats_keys:
            if stats_key in stats_key_to_mean:
                for arg_name in temp_args.keys():
                    new_val = None
                    if "min" in arg_name:
                        new_val = stats_key_to_mean[stats_key] - k * stats_key_to_std[stats_key]
                    if "max" in arg_name:
                        new_val = stats_key_to_mean[stats_key] + k * stats_key_to_std[stats_key]
                    if new_val is not None and str(new_val) != "nan":
                        logger.info(
                            f"Using {k}-sigma rule, for op {op_name}, "
                            f"changed its para "
                            f"{arg_name}={args[arg_name]} into "
                            f"{arg_name}={new_val}"
                        )
                        args[arg_name] = new_val
    if path_k_sigma_recipe:
        cfg = prepare_cfgs_for_export(cfg)
        if path_k_sigma_recipe.endswith(".yaml") or path_k_sigma_recipe.endswith(".yml"):
            with open(path_k_sigma_recipe, "w") as fout:
                yaml.safe_dump(cfg, fout)
        elif path_k_sigma_recipe.endswith(".json"):
            with open(path_k_sigma_recipe, "w") as fout:
                json.dump(cfg, fout)
        else:
            raise TypeError(
                f"Unrecognized output file type:"
                f" [{path_k_sigma_recipe}]. Should be one of the types"
                f' [".yaml", ".yml", ".json"].'
            )


if __name__ == "__main__":
    main()

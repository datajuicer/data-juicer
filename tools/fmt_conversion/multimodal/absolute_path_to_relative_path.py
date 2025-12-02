# This tool is used to convert the absolute paths or relative paths in
# dj dataset.
#
# Data-Juicer format with absolute path:
#   - multi-chunk interleaved image-text sequence
#   - in jsonl
# {'id': '000000033471',
#  'images': ['/home/data/coco/train2017/000000033471.jpg'],
#  'text': '[[human]]: <image>\n'
#          'What are the colors of the bus in the image?\n'
#          '[[gpt]]: The bus in the image is white and red.\n'
#          '[[human]]: What feature can be seen on the back of the bus?\n'
#          '[[gpt]]: The back of the bus features an advertisement.\n'
#          '[[human]]: Is the bus driving down the street or pulled off to'
#          'the side?\n'
#          '[[gpt]]: The bus is driving down the street, which is crowded '
#          'with people and other vehicles. <|__dj__eoc|>'}
#
# Corresponding Data-Juicer format with relative path:
# {'id': '000000033471',
#  'images': ['coco/train2017/000000033471.jpg'],
#  'text': '[[human]]: <image>\n'
#          'What are the colors of the bus in the image?\n'
#          '[[gpt]]: The bus in the image is white and red.\n'
#          '[[human]]: What feature can be seen on the back of the bus?\n'
#          '[[gpt]]: The back of the bus features an advertisement.\n'
#          '[[human]]: Is the bus driving down the street or pulled off to'
#          'the side?\n'
#          '[[gpt]]: The bus is driving down the street, which is crowded '
#          'with people and other vehicles. <|__dj__eoc|>',
#  '__dj__meta__': {
#           'abs_dir': {
#               'images': ['/home/data']
#                      }
#                  }
# }
#
import os
from pathlib import Path

import click
import jsonlines as jl
from loguru import logger
from tqdm import tqdm

from data_juicer.utils.constant import Fields
from data_juicer.utils.file_utils import copy_data


def match_prefix(candidates, string):
    for candidate in candidates:
        if candidate == string[: len(candidate)]:
            return candidate
    logger.warning(f"Match no corresponding absolute dir in {string}.")
    return "/"


def get_last_dict(d, nest_key):
    sub_keys = nest_key.split(".")
    final_key = sub_keys[-1]
    last_dict = d
    for key in sub_keys[:-1]:
        if key not in last_dict:
            last_dict[key] = {}
        last_dict = last_dict[key]
    return last_dict, final_key


@logger.catch
@click.command()
@click.option("--dj_ds_path", "dj_ds_path", type=str, required=True)
@click.option("--absolute_dir", "-d", "absolute_dirs", type=str, required=True, multiple=True)
@click.option("--path_key", "-k", "path_keys", type=str, required=True, multiple=True)
@click.option("--target_dj_ds_path", "target_dj_ds_path", type=str, required=False, default=None)
@click.option("--target_mt_dir", "target_mt_dir", type=str, required=False, default=None)
def convert_absolute_path_to_relative_path(
    dj_ds_path: str,
    absolute_dirs: list[str],
    path_keys: list[str],
    target_dj_ds_path: str = None,
    target_mt_dir: str = None,
):
    """
    Convert the absolute paths or relative paths in Data-Juicer dataset.

    :param dj_ds_path: path to the input Data-Juicer dataset with absolute
        paths to multimodal data.
    :param absolute_dirs: all possible absolute dirs in the absolute paths.
        A list param to support multi sources of multimodal data here.
    :param path_keys: the keys to the absolute paths. Multi-level field
        information in the keys need to be separated by '.'.
    :param target_dj_ds_path: path to store the converted dataset if it is
        not None.
    :param target_mt_dir: copy and reorganize all relative multimodal
        data from multi sources to one directory for data migration
        if it is not None.
    """
    if not os.path.exists(dj_ds_path):
        raise FileNotFoundError(f"Input dataset [{dj_ds_path}] can not be found.")
    if (
        target_dj_ds_path is not None
        and os.path.dirname(target_dj_ds_path)
        and not os.path.exists(os.path.dirname(target_dj_ds_path))
    ):
        logger.info(f"Create directory [{os.path.dirname(target_dj_ds_path)}] for " f"the target dataset.")
        os.makedirs(os.path.dirname(target_dj_ds_path))

    abs_dir_key = "abs_dir"

    # normalize the dirs
    absolute_dirs = [str(Path(d)) + "/" for d in absolute_dirs]
    # match the deeper paths first
    absolute_dirs = sorted(absolute_dirs, key=lambda p: -len(p))

    logger.info("Start to convert absolute path to relative path.")
    samples = []
    with jl.open(dj_ds_path, "r") as reader:
        for sample in tqdm(reader):
            if Fields.meta not in sample:
                sample[Fields.meta] = {}
            sample[Fields.meta][abs_dir_key] = {}
            for path_key in path_keys:
                last_dict, final_key = get_last_dict(sample, path_key)
                abs_dir_last_dict, _ = get_last_dict(sample[Fields.meta][abs_dir_key], path_key)
                absolute_paths = last_dict[final_key]
                if type(absolute_paths) is not list:
                    absolute_paths = [absolute_paths]

                # normalize the paths
                absolute_paths = [str(Path(p)) for p in absolute_paths]

                cur_dirs = [match_prefix(absolute_dirs, p) for p in absolute_paths]
                relative_paths = [str(os.path.relpath(p, d)) for d, p in zip(cur_dirs, absolute_paths)]

                if type(last_dict[final_key]) is not list:
                    last_dict[final_key] = relative_paths[0]
                    abs_dir_last_dict[final_key] = cur_dirs[0]
                else:
                    last_dict[final_key] = relative_paths
                    abs_dir_last_dict[final_key] = cur_dirs

                # copy and reorganize multimodal data
                if target_mt_dir is not None:
                    for d, p in zip(cur_dirs, relative_paths):
                        succeed = copy_data(d, target_mt_dir, p)
                        if not succeed:
                            logger.warning(f"{p} does not exists in {d}.")

            samples.append(sample)

    if target_dj_ds_path is not None:
        logger.info(f"Start to write the converted dataset to " f"[{target_dj_ds_path}]...")
        with jl.open(target_dj_ds_path, "w") as writer:
            for sample in samples:
                writer.write(sample)
    return samples


if __name__ == "__main__":
    convert_absolute_path_to_relative_path()

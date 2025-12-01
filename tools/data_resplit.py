import argparse
import os
from typing import List

import pandas as pd
import ray
from loguru import logger

DEFAULT_MAX_FILE_SIZE = 128  # 128 MB
DEFAULT_MIN_FILE_SIZE = 1  # 1 MB


def split_jsonl(file_path: str, max_size: float, output_dir: str):
    """Split a jsonl file into multiple sub files more efficiently.

    Args:
        file_path (`str`): path of the original jsonl file
        max_size (`float`): max size of each sub file (in MB)
        output_dir (`str`): directory to save the sub files

    Yields:
        str: path of each newly created sub file
    """
    os.makedirs(output_dir, exist_ok=True)
    file_index = 0
    max_byte_size = max_size * 1024**2
    base_file_name = os.path.basename(file_path)
    file_name = os.path.splitext(base_file_name)[0]
    current_size = 0
    buffer = []
    buffer_size = 0

    with open(file_path, "r", encoding="utf-8") as infile:
        while True:
            # Determine the output file name
            output_file_name = f"{file_name}_{file_index}.jsonl"
            output_file_path = os.path.join(output_dir, output_file_name)

            # Read lines until we reach the max buffer size
            while current_size + buffer_size < max_byte_size:
                line = infile.readline()
                if not line:
                    break
                buffer.append(line)
                buffer_size += len(line)

            # Write the buffered lines to the current output file
            if buffer:
                with open(output_file_path, "w", encoding="utf-8") as outfile:
                    outfile.writelines(buffer)
                buffer = []
                buffer_size = 0
                file_index += 1

            if not line:
                break


def get_jsonl_file_names(dataset_dir_path: str) -> List[str]:
    """Load all jsonl files in a directory.

    Args:
        dataset_dir_path (`str`): path of the directory containing jsonl files
        or the path of a single jsonl file

    Returns:
        List[str]: list of jsonl file paths
    """
    if os.path.isdir(dataset_dir_path):
        jsonl_files = [os.path.join(dataset_dir_path, f) for f in os.listdir(dataset_dir_path)]
    elif os.path.isfile(dataset_dir_path) and dataset_dir_path.endswith(".jsonl") or dataset_dir_path.endswith(".json"):
        jsonl_files = [dataset_dir_path]
    else:
        raise ValueError("Invalid path: it should be a directory containing jsonl files" " or a single jsonl file.")
    return jsonl_files


def main(args):
    ray.init(args.ray_address)

    data_dir = args.data_dir
    jsonl_files = get_jsonl_file_names(data_dir)
    df = pd.DataFrame({"jsonl_files": jsonl_files})
    data = ray.data.from_pandas(df)

    total_size = sum(os.path.getsize(f) for f in jsonl_files) / 1024 / 1024
    cpu_num = ray.cluster_resources().get("CPU", 1)
    max_size = max(DEFAULT_MIN_FILE_SIZE, min(DEFAULT_MAX_FILE_SIZE, total_size / cpu_num / 4))
    logger.info(f"Number of files: {len(jsonl_files)}, " f"Total size: {total_size} MB, max size: {max_size} MB")

    def split_jsonl_dataset(
        jsonl_paths: pd.DataFrame,
    ) -> List[str]:
        for jsonl_path in jsonl_paths["jsonl_files"]:
            split_jsonl(jsonl_path, max_size, args.resplit_dir)
        return jsonl_paths

    data.map_batches(split_jsonl_dataset).materialize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ray-address", type=str, default="auto", help="The address of the Ray cluster.")
    parser.add_argument("--data-dir", "-i", type=str, required=True, help="Path to your dataset directory.")
    parser.add_argument("--resplit-dir", "-o", type=str, required=True, help="Path to resplited dataset directory.")
    args = parser.parse_args()

    main(args)

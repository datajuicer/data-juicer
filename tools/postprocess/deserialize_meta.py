import json
import os
import pathlib
from multiprocessing import Pool

import fire
import jsonlines


def fp_iter(src_dir):
    """
    Find all jsonl files in the source directory.
    :param src_dir: path to source dataset directory
    :return: iterator over jsonl files
    """
    for fp in pathlib.Path(src_dir).glob("*.jsonl"):
        yield fp


def meta_deserialize(file_name, target_file, serialized_key):
    """
    Deserialize the specified field into dict.
    :param file_name: path to source jsonl files.
    :param target_file: path to store the converted jsonl files.
    :param serialized_key: the key corresponding to the field that will be
    deserialized.
    """
    with open(target_file, "w") as fw:
        with jsonlines.open(file_name, "r") as fr:
            for obj in fr:
                obj[serialized_key] = json.loads(obj[serialized_key])
                fw.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main(src_dir, target_dir, serialized_key="source_info", num_proc=1):
    """
    Deserialize the specified field in the jsonl file.
    :param src_dir: path that's stores jsonl files.
    :param target_dir: path to save the converted jsonl files.
    :param serialized_key: the key corresponding to the field that will be
    deserialized. Default it's 'source_info'.
    :param num_proc: number of process workers. Default it's 1.
    """

    # check if the source directory exists
    if not os.path.exists(src_dir):
        raise ValueError("The raw source data directory does not exist," " Please check and retry.")
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)

    pool = Pool(num_proc)
    for fp in fp_iter(src_dir):
        print(fp)
        jsonl_fp = os.path.join(target_dir, fp.name)
        pool.apply_async(meta_deserialize, args=(str(fp), jsonl_fp, serialized_key))

    pool.close()
    pool.join()


if __name__ == "__main__":
    fire.Fire(main)

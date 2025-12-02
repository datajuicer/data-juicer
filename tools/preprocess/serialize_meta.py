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


def meta_serialize(file_name, target_file, text_key, serialized_key):
    """
    Serialize all fields except the specified fields into strings.
    :param file_name: path to source jsonl files.
    :param target_file: path to store the converted jsonl files.
    :text_key: the key corresponding to the field that will not be serialized.
    :param serialized_key: the key corresponding to the field that the
        serialized info saved. Default it's 'source_info'.
    """
    with open(target_file, "w") as fw:
        target = {}
        with jsonlines.open(file_name, "r") as fr:
            for obj in fr:
                for key in text_key:
                    target[key] = obj.pop(key)
                target[serialized_key] = json.dumps(obj, ensure_ascii=False)
                fw.write(json.dumps(target, ensure_ascii=False) + "\n")


def main(src_dir, target_dir, text_key="text", serialized_key="source_info", num_proc=1):
    """
    Serialize all the fields in the jsonl file except the fields specified
    by users to ensure that the jsonl file with inconsistent text format
    for each line can also be load normally by the dataset.
    :param src_dir: path that's stores jsonl files.
    :param target_dir: path to save the converted jsonl files.
    :param text_key: the key corresponding to the field that will not be
    serialized. Default it's 'text'.
    :param serialized_key: the key corresponding to the field that the
        serialized info saved. Default it's 'source_info'.
    :param num_proc: number of process worker. Default it's 1.
    """

    assert (
        isinstance(text_key, str) or isinstance(text_key, list) or isinstance(text_key, tuple)
    ), "text_key must be a string, list or tuple."

    if isinstance(text_key, str):
        text_key = [text_key]

    for key in text_key:
        assert key != serialized_key, "text_key '{}' cannot be the same as " "serialized_key.".format(key)

    # check if the source directory exists
    if not os.path.exists(src_dir):
        raise ValueError("The raw source data directory does not exist," " Please check and retry.")
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)

    pool = Pool(num_proc)
    for fp in fp_iter(src_dir):
        print(fp)
        jsonl_fp = os.path.join(target_dir, fp.name)
        pool.apply_async(meta_serialize, args=(str(fp), jsonl_fp, text_key, serialized_key))

    pool.close()
    pool.join()


if __name__ == "__main__":
    fire.Fire(main)

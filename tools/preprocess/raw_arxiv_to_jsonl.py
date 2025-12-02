# Part of the code here has been modified from:
# https://github.com/togethercomputer/RedPajama-Data/blob/rp_v1/data_prep/arxiv/arxiv_cleaner.py
# --------------------------------------------------------
#
# This tool is used for converting the raw arxiv data downloaded from S3
# (ref: https://info.arxiv.org/help/bulk_data_s3.html) to several jsonl files.
#
# For downloading process, please refer to:
# https://github.com/togethercomputer/RedPajama-Data/tree/rp_v1/data_prep/arxiv
#
# Notice: before you downloading, converting or processing, you might make sure
# that your drive space is large enough to store the raw data (over 3TB),
# converted data (over 3TB), at least processed data (about 500-600GB), and
# even more cache data during processing.

import gzip
import os
import pathlib
import tarfile
import tempfile
from multiprocessing import Pool

import fire
import jsonlines as jl
from loguru import logger


@logger.catch(reraise=True)
def tex_proj_loader(file_or_dir_path: pathlib.Path):
    """
    Load the tex files from a tar file or a gzip file.
    :param file_or_dir_path: path to tar file or the gzip file
    :return: a list of content in tex files
    """

    files_and_content = []
    try:
        # if it is a directory, open it as a tarfile
        with tarfile.open(file_or_dir_path) as sub_tf:
            for member in sub_tf.getmembers():
                if member.name.endswith(".tex"):
                    file_content = sub_tf.extractfile(member).read()
                    try:
                        file_content = file_content.decode("utf-8")
                    except UnicodeDecodeError:
                        logger.error(f"UnicodeDecodeError: {file_or_dir_path}")
                        return None
                    files_and_content.append(file_content)
    except tarfile.ReadError:
        # otherwise we try opening it as a gzip file
        try:
            with gzip.open(file_or_dir_path, "rb") as gz:
                file_content = gz.read()
        except Exception as e:
            # all fails, we skip this file
            logger.error(f"{e}: {file_or_dir_path}")
            return None

        try:
            file_content = file_content.decode("utf-8")
        except UnicodeDecodeError:
            logger.error(f"UnicodeDecodeError: {file_or_dir_path}")
            return None
        files_and_content.append(file_content)
    except Exception as e:
        logger.error(f"{e}: {file_or_dir_path}")
        return None

    return files_and_content


@logger.catch(reraise=True)
def convert_tar_to_jsonl(tar_fp, jsonl_fp, tmp_dir):
    """
    Extract the contents of tex files from tar file, convert and
    save to jsonl file
    :param tar_fp: path to tar file
    :param jsonl_fp: path to save jsonl file
    :param tmp_dir: a temporary directory to save extracted files
    """
    failed = 0
    success = 0
    with tempfile.TemporaryDirectory(dir=tmp_dir, prefix=tar_fp.name) as td:
        with jl.open(jsonl_fp, mode="w") as writer:
            with tarfile.open(tar_fp) as tf:
                tf.extractall(members=tf.getmembers(), path=td)
                for proj_dir_or_file in pathlib.Path(td).rglob("*.gz"):
                    data = tex_proj_loader(proj_dir_or_file)
                    if data is None:
                        failed += 1
                        continue
                    success += 1
                    writer.write_all([{"text": txt} for txt in data])

    logger.info(f"{jsonl_fp} done. Fail: {failed}, success: {success}")


def tar_fp_iter(src_dir):
    """
    Find all tar files in the source directory.
    :param src_dir: path to source dataset directory
    :return: iterator over tar files
    """
    for tar_fp in pathlib.Path(src_dir).glob("*.tar"):
        yield tar_fp


def main(arxiv_src_dir, target_dir, work_dir="./tmp/", num_proc=1):
    """
    :param arxiv_src_dir: if you download raw arXiv data as Redpajama did,
           you will get a directory src which includes thousands of tar
           files whose filenames are like "arXiv_src_yymm_xxx.tar". You
           just need to set this argument to the path of this dir.
    :param target_dir: result directory to store the converted jsonl files.
    :param work_dir: directory to store intermediate files, and they will
           be removed once the conversion ends. Default it's "./tmp"
    :param num_proc: number of process workers. Default it's 1.
    """
    # check if the source directory exists.
    if not os.path.exists(arxiv_src_dir):
        raise ValueError("The raw arXiv source data directory does not exist," " Please check and retry.")
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)

    if not os.path.exists(work_dir):
        os.makedirs(work_dir, exist_ok=True)

    # convert in multiprocess
    pool = Pool(num_proc)
    for tar_fp in tar_fp_iter(arxiv_src_dir):
        logger.info(f"Start to process {tar_fp}")
        jsonl_fp = os.path.join(target_dir, tar_fp.name.replace(".tar", ".jsonl"))
        pool.apply_async(
            convert_tar_to_jsonl,
            args=(
                tar_fp,
                jsonl_fp,
                work_dir,
            ),
        )
    pool.close()
    pool.join()


if __name__ == "__main__":
    fire.Fire(main)

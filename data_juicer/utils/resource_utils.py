import os
import subprocess
from typing import List

import psutil
from loguru import logger

from data_juicer.utils.availability_utils import _is_package_available
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.ray_utils import (
    check_and_initialize_ray,
    ray_available_gpu_memories,
    ray_available_memories,
    ray_cpu_count,
    ray_gpu_count,
)

torch = LazyLoader("torch")

ray = LazyLoader("ray")

NVSMI_REPORT = True


def query_cuda_info(query_key):
    global NVSMI_REPORT
    # get cuda info using "nvidia-smi" command in MB
    try:
        nvidia_smi_output = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={query_key}", "--format=csv,noheader,nounits"]
        ).decode("utf-8")
    except Exception as e:
        if "non-zero exit status 2" in str(e):
            err_msg = (
                f"The specified query_key [{query_key}] might not be "
                f"supported by command nvidia-smi. Please check and "
                f"retry!"
            )
        elif "No such file or directory" in str(e):
            err_msg = "Command nvidia-smi is not found. There might be no " "GPUs on this machine."
        else:
            err_msg = str(e)
        if NVSMI_REPORT:
            logger.warning(err_msg)
            NVSMI_REPORT = False
        return None
    cuda_info_list = []
    for line in nvidia_smi_output.strip().split("\n"):
        cuda_info_list.append(int(line))
    return cuda_info_list


def get_cpu_utilization():
    return psutil.cpu_percent()


def query_mem_info(query_key):
    mem = psutil.virtual_memory()
    if query_key not in mem._fields:
        logger.warning(f"No such query key [{query_key}] for memory info. " f"Should be one of {mem._fields}")
        return None
    val = round(mem.__getattribute__(query_key) / (2**20), 2)  # in MB
    return val


def _cuda_device_count(cfg=None):
    _torch_available = _is_package_available("torch")

    if check_and_initialize_ray(cfg):
        return int(ray_gpu_count())

    if _torch_available:
        return torch.cuda.device_count()

    try:
        nvidia_smi_output = subprocess.check_output(["nvidia-smi", "-L"], text=True)
        all_devices = nvidia_smi_output.strip().split("\n")

        cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
        if cuda_visible_devices is not None:
            logger.warning(
                "CUDA_VISIBLE_DEVICES is ignored when torch is unavailable. " "All detected GPUs will be used."
            )

        return len(all_devices)
    except Exception:
        # nvidia-smi not found or other error
        return 0


def cuda_device_count(cfg=None):
    return _cuda_device_count(cfg)


def is_cuda_available():
    return cuda_device_count() > 0


def cpu_count(cfg=None):
    if check_and_initialize_ray(cfg):
        return int(ray_cpu_count())

    return psutil.cpu_count()


def available_memories(cfg=None) -> List[int]:
    """Available memory for each node in MB."""
    if check_and_initialize_ray(cfg):
        return ray_available_memories()

    return [int(psutil.virtual_memory().available / (1024**2))]


def available_gpu_memories(cfg=None) -> List[int]:
    """Available gpu memory of each gpu card for each alive node in MB."""
    if check_and_initialize_ray(cfg):
        return ray_available_gpu_memories()

    try:
        return query_cuda_info("memory.free")
    except Exception:
        return []

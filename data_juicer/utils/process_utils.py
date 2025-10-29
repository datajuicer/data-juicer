import itertools
import math
import os
import subprocess
import sys

import multiprocess as mp
from loguru import logger

from data_juicer.utils.resource_utils import (
    available_gpu_memories,
    available_memories,
    cpu_count,
    cuda_device_count,
)

# A safety fraction to avoid OOM by not allocating all available memory to operators.
# This leaves some memory for Ray's overhead and other system processes.
_OPS_MEMORY_LIMIT_FRACTION = 1.0


def setup_mp(method=None):
    if mp.current_process().name != "MainProcess":
        return

    if method is None:
        method = ["fork", "forkserver", "spawn"]
    if not isinstance(method, (list, tuple)):
        method = [method]
    method = [m.lower() for m in method]

    env_method = os.getenv("MP_START_METHOD", "").lower()
    if env_method in method:
        method = [env_method]

    available_methods = mp.get_all_start_methods()
    for m in method:
        if m in available_methods:
            try:
                logger.debug(f"Setting multiprocess start method to '{m}'")
                mp.set_start_method(m, force=True)
            except RuntimeError as e:
                logger.warning(f"Error setting multiprocess start method: {e}")
            break


def get_min_cuda_memory():
    # get cuda memory info using "nvidia-smi" command
    import torch

    min_cuda_memory = torch.cuda.get_device_properties(0).total_memory / 1024**2
    nvidia_smi_output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"]
    ).decode("utf-8")
    for line in nvidia_smi_output.strip().split("\n"):
        free_memory = int(line)
        min_cuda_memory = min(min_cuda_memory, free_memory)
    return min_cuda_memory


def calculate_np(name, mem_required, cpu_required, use_cuda=False, gpu_required=0):
    """Calculate the optimum number of processes for the given OP automatically。"""

    if not use_cuda and gpu_required:
        raise ValueError(
            f"Op[{name}] attempted to request GPU resources (gpu_required={gpu_required}), "
            "but appears to lack GPU support. If you have verified this operator support GPU acceleration, "
            'please explicitly set its property: `_accelerator = "cuda"`.'
        )

    cpu_num = cpu_count()
    auto_proc_from_mem = auto_proc_from_gpu = auto_proc_from_cpu = sys.maxsize

    if use_cuda:
        cuda_mems_available = [m / 1024 for m in available_gpu_memories()]  # GB
        gpu_count = cuda_device_count()
        if not mem_required and not gpu_required:
            auto_num_proc = gpu_count
            logger.warning(
                f"The required cuda memory and gpu of Op[{name}] "
                f"has not been specified. "
                f"Please specify the mem_required field or gpu_required field in the "
                f"config file. You can reference the config_all.yaml file."
                f"Set the auto `num_proc` to number of GPUs {auto_num_proc}."
            )
        else:
            if mem_required:
                auto_proc_from_mem = sum(
                    [math.floor(mem_available / mem_required) for mem_available in cuda_mems_available]
                )
            if gpu_required:
                auto_proc_from_gpu = math.floor(gpu_count / gpu_required)
            if cpu_required:
                auto_proc_from_cpu = math.floor(cpu_num / cpu_required)

            auto_num_proc = min(auto_proc_from_mem, auto_proc_from_gpu, auto_proc_from_cpu)
            if auto_num_proc < 1:
                auto_num_proc = len(available_memories())  # set to the number of available nodes

            logger.info(
                f"Set the auto `num_proc` to {auto_num_proc} of Op[{name}] based on the "
                f"required cuda memory: {mem_required}GB "
                f"required gpu: {gpu_required} and required cpu: {cpu_required}."
            )
        return auto_num_proc
    else:
        mems_available = [m / 1024 for m in available_memories()]  # GB

        if mem_required:
            auto_proc_from_mem = sum([math.floor(mem_available / mem_required) for mem_available in mems_available])
        if cpu_required:
            auto_proc_from_cpu = math.floor(cpu_num / cpu_required)

        auto_num_proc = min(cpu_num, auto_proc_from_mem, auto_proc_from_cpu)

        if auto_num_proc < 1.0:
            auto_num_proc = len(available_memories())  # number of processes is equal to the number of nodes
            logger.warning(
                f"The required CPU number: {cpu_required} "
                f"and memory: {mem_required}GB might "
                f"be more than the available CPU: {cpu_num} "
                f"and memory: {mems_available}GB."
                f"This Op [{name}] might "
                f"require more resource to run. "
                f"Set the auto `num_proc` to available nodes number {auto_num_proc}."
            )
        else:
            logger.info(
                f"Set the auto `num_proc` to {auto_num_proc} of Op[{name}] based on the "
                f"required memory: {mem_required}GB "
                f"and required cpu: {cpu_required}."
            )
        return auto_num_proc


def _find_optimal_concurrency(resource_ratios, total_resource):
    """
    Search for the optimal concurrency allocation to achieve the
    highest total resource utilization and the most balanced processing capacity.

    Args:
        resource_ratios (list[float]): List of single-process resource ratios for each operator
        total_resource (float): Total resource

    Return:
        tuple: (list of optimal concurrency, total resource usage, standard deviation of processing capacity)
        If there is no valid combination, return (None, 0, 0)
    """
    n = len(resource_ratios)
    if n == 0:
        return (None, 0, 0)

    sum_r_squared = sum(r * r for r in resource_ratios)
    if sum_r_squared == 0:
        return (None, 0, 0)

    c_floats = [(total_resource * r) / sum_r_squared for r in resource_ratios]

    # generate candidate concurrency
    candidates = []
    for cf in c_floats:
        floor_cf = math.floor(cf)
        ceil_cf = math.ceil(cf)
        possible = set()
        if floor_cf >= 1:
            possible.add(floor_cf)
        possible.add(ceil_cf)
        possible = [max(1, v) for v in possible]
        candidates.append(sorted(list(set(possible))))

    # traverse all combinations
    best_combination = None
    max_resource_usage = 0
    min_std = float("inf")

    for combo in itertools.product(*candidates):
        total_used = sum(c * r for c, r in zip(combo, resource_ratios))
        if total_used > total_resource:
            continue

        # calculate the standard deviation of processing capacity
        processing_powers = [c / r for c, r in zip(combo, resource_ratios)]
        mean = sum(processing_powers) / n
        variance = sum((x - mean) ** 2 for x in processing_powers) / n
        std = math.sqrt(variance)

        # update the optimal solution (priority resource utilization, suboptimal standard deviation)
        if total_used > max_resource_usage:
            max_resource_usage = total_used
            best_combination = combo
            min_std = std
        elif total_used == max_resource_usage and std < min_std:
            best_combination = combo
            min_std = std

    return (
        list(best_combination) if best_combination else None,
        max_resource_usage,
        min_std if best_combination else 0,
    )


def calculate_ray_np(operators):
    """
    Automatically calculates optimal concurrency for Ray Data operator.
    This function handles both task and actor based operators, considering
    resource requirements and user specifications. The computation follows Ray Data's
    concurrency semantics while optimizing resource utilization.

    Key Concepts:
    - Resource Ratio: Individual operator's resource requirement (GPU/CPU/memory)
        compared to total cluster resources, using max(cpu_ratio, gpu_ratio, adjusted_mem_ratio)
    - Fixed Allocation: Portion of resources reserved by operators with user-specified num_proc
    - Dynamic Allocation: Remaining resources distributed among auto-scaling operators

    Design Logic:
    1. User Specification Priority:
        - If user provides concurrency setting, directly return it
        - Applies to both task and actor based operators
    2. Task Operators (equivalent to a cpu operator in dj):
        a. When unspecified: Return None to let Ray determine implicitly
        b. Auto-calculation: Returns maximum concurrency based on available
            resources and operator requirements
    3. Actor Operators (equivalent to a gpu operator in dj):
        a. Mandatory concurrency - set required gpus to 1 if unspecified, and then refer to the following `b`
            to calculate automatically based on this setting
        b. Auto-calculation returns tuple (min_concurrency, max_concurrency):
            i. Minimum: Ensures baseline resource allocation in remaining resources
                when all operators are active simultaneously (proportionally)
            ii. Maximum: Allows full utilization of remaining resources by single
                operator when others are idle
    """
    from data_juicer.utils.ray_utils import (
        ray_available_gpu_memories,
        ray_available_memories,
        ray_cpu_count,
        ray_gpu_count,
    )
    from data_juicer.utils.resource_utils import is_cuda_available

    cuda_available = is_cuda_available()
    total_cpu = ray_cpu_count()
    total_gpu = ray_gpu_count()
    available_mem = sum(ray_available_memories()) * _OPS_MEMORY_LIMIT_FRACTION / 1024  # Convert MB to GB
    available_gpu_mem = sum(ray_available_gpu_memories()) * _OPS_MEMORY_LIMIT_FRACTION / 1024  # Convert MB to GB
    resource_configs = {}

    for op in operators:
        cpu_req = op.cpu_required
        mem_req = op.mem_required
        gpu_req = 0
        gpu_mem_req = 0

        if op.gpu_required:
            if not op.use_cuda():
                raise ValueError(
                    f"Op[{op._name}] attempted to request GPU resources (gpu_required={op.gpu_required}), "
                    "but appears to lack GPU support. If you have verified this operator support GPU acceleration, "
                    'please explicitly set its property: `_accelerator = "cuda"`.'
                )
            if not cuda_available:
                raise ValueError(
                    f"Op[{op._name}] attempted to request GPU resources (gpu_required={op.gpu_required}), "
                    "but the gpu is unavailable. Please check whether your environment is installed correctly"
                    " and whether there is a gpu in the resource pool."
                )
        # if it is a cuda operator, mem_required will be calculated as gpu memory;
        # if it is a cpu, it will be calculated as memory.
        cpu_required_frac, gpu_required_frac = 0, 0
        # GPU operator calculations
        if op.use_cuda():
            gpu_req = op.gpu_required
            gpu_mem_req = op.mem_required
            if not gpu_req and not gpu_mem_req:
                logger.warning(
                    f"Neither the required cuda memory nor gpu of Op[{op._name}] is specified. "
                    f"We recommend specifying the `mem_required` field or `gpu_required` field in the "
                    f"config file. You can reference the `config_all.yaml` file."
                    f"Set the `gpu_required` to 1 now."
                )
                gpu_req = 1

            # if no cpu is specified, ray will apply for 1 cpu by default
            cpu_required_frac = cpu_req / total_cpu if cpu_req else 1 / total_cpu
            gpu_required_frac = max(
                gpu_req / total_gpu if gpu_req else 0,
                gpu_mem_req / available_gpu_mem if gpu_mem_req else 0,
            )

            if not gpu_req:
                gpu_req = math.ceil(gpu_required_frac * total_gpu * 100) / 100

        # CPU operator calculations
        else:
            if cpu_req or mem_req:
                cpu_required_frac = max(
                    cpu_req / total_cpu if cpu_req else 0, mem_req / available_mem if mem_req else 0
                )
            else:
                if op.use_auto_proc():
                    logger.warning(
                        f"Neither the required memory nor cpu of Op[{op._name}] is specified. "
                        f"We recommend specifying the `cpu_required` field in the "
                        f"config file. You can reference the `config_all.yaml` file."
                    )
                # if no cpu is specified, ray will apply for 1 cpu by default
                cpu_required_frac = 1 / total_cpu
            if op.num_proc:
                if not isinstance(op.num_proc, int):
                    raise ValueError(
                        f"Op[{op._name}] is running with cpu resource, ``num_proc`` is expected to be set as an integer. "
                        f"Use ``concurrency=n`` to control maximum number of workers to use,  but got: {op.num_proc}."
                    )
            # set concurrency to none, using the default autoscaler of ray to ensure performance
            if op.num_proc == -1:
                op.num_proc = None

        resource_configs[op._name] = {
            "cpu_required": cpu_req,
            "gpu_required": gpu_req,
            "mem_required": mem_req,
            "gpu_mem_required": gpu_mem_req,
            "cpu_required_frac": cpu_required_frac,
            "gpu_required_frac": gpu_required_frac,
            "num_proc": tuple(op.num_proc) if isinstance(op.num_proc, list) else op.num_proc,
            "auto_proc": op.use_auto_proc(),
            "is_actor": op.use_cuda(),
        }

    fixed_min_cpu = 0
    fixed_max_cpu = 0
    fixed_min_gpu = 0
    fixed_max_gpu = 0
    auto_resource_frac_map = {}
    for op_name, cfg in resource_configs.items():
        if cfg["auto_proc"]:
            auto_resource_frac_map[op_name] = (cfg["cpu_required_frac"], cfg["gpu_required_frac"])
        else:
            num_proc = cfg["num_proc"]
            if cfg["is_actor"]:
                min_proc = num_proc[0] if isinstance(num_proc, (tuple, list)) else num_proc
            else:
                min_proc = 1  # when ``fn`` is a function, , only the maximum concurrency can be specified
            max_proc = num_proc[1] if isinstance(num_proc, (tuple, list)) else num_proc
            fixed_min_cpu += cfg["cpu_required_frac"] * min_proc
            fixed_min_gpu += cfg["gpu_required_frac"] * min_proc
            if not max_proc:  # when num_proc is none, at least one process will be started
                max_proc = min_proc  # 1
            fixed_max_cpu += cfg["cpu_required_frac"] * max_proc
            fixed_max_gpu += cfg["gpu_required_frac"] * max_proc

    # Validate resource availability
    total_auto_base_cpu = sum([i[0] for i in list(auto_resource_frac_map.values())])
    total_auto_base_gpu = sum([i[1] for i in list(auto_resource_frac_map.values())])
    total_required_min_cpu = fixed_min_cpu + total_auto_base_cpu
    if total_required_min_cpu > 1:
        raise ValueError(
            f"Insufficient cpu resources: "
            f"At least {total_required_min_cpu * total_cpu} cpus are required,  but only {total_cpu} are available. "
            f"Please add resources to ray cluster or reduce operator requirements."
        )
    total_required_min_gpu = fixed_min_gpu + total_auto_base_gpu
    if total_required_min_gpu > 1:
        raise ValueError(
            f"Insufficient gpu resources: "
            f"At least {total_required_min_gpu * total_gpu} cpus are required,  but only {total_gpu} are available. "
            f"Please add resources to ray cluster or reduce operator requirements."
        )
    if len(auto_resource_frac_map) > 0:
        remaining_min_frac_cpu = 1 - fixed_max_cpu
        remaining_max_frac_cpu = 1 - fixed_min_cpu
        remaining_min_frac_gpu = 1 - fixed_max_gpu
        remaining_max_frac_gpu = 1 - fixed_min_gpu

        op_resources_cpu, op_resources_gpu = {}, {}
        # if both cpu and gpu are required, the allocation will be prioritized based on the gpu fraction
        for k, v in auto_resource_frac_map.items():
            if v[1] > 0:  # (cpu, gpu)
                op_resources_gpu[k] = v[1]
            elif v[0] > 0:
                op_resources_cpu[k] = v[0]

        best_combination_cpu, best_combination_gpu = {}, {}
        if len(op_resources_gpu) > 0:
            _gpu_names, _gpu_resources = [], []
            for k, v in op_resources_gpu.items():
                _gpu_names.append(k)
                _gpu_resources.append(v)
            _best_combination_gpu, _, _ = _find_optimal_concurrency(_gpu_resources, remaining_min_frac_gpu)
            best_combination_gpu = dict(zip(_gpu_names, _best_combination_gpu))
        if len(op_resources_cpu) > 0:
            _cpu_names, _cpu_resources = [], []
            for k, v in op_resources_cpu.items():
                _cpu_names.append(k)
                _cpu_resources.append(v)
            _best_combination_cpu, _, _ = _find_optimal_concurrency(_cpu_resources, remaining_min_frac_cpu)
            best_combination_cpu = dict(zip(_cpu_names, _best_combination_cpu))

        best_combination = {}
        for k in list(auto_resource_frac_map.keys()):
            best_combination[k] = min(
                best_combination_gpu.get(k, sys.maxsize), best_combination_cpu.get(k, sys.maxsize)
            )
        for op_name, cfg in resource_configs.items():
            if cfg["auto_proc"]:
                # TODO:
                # issue: https://github.com/ray-project/ray/issues/55307
                # or min_proc = 1 ?
                # or max_proc = int(max(1, 1 / cfg["base_resource_frac"])) ? use all resources

                # max_frac_cpu, max_frac_gpu = 1, 1
                max_frac_cpu, max_frac_gpu = remaining_max_frac_cpu, remaining_max_frac_gpu
                min_proc = best_combination[op_name]

                if cfg["cpu_required_frac"] and cfg["gpu_required_frac"]:
                    max_proc = min(
                        int(max(1, max_frac_cpu / cfg["cpu_required_frac"])),
                        int(max(1, max_frac_gpu / cfg["gpu_required_frac"])),
                    )
                elif cfg["gpu_required_frac"]:
                    max_proc = int(max(1, max_frac_gpu / cfg["gpu_required_frac"]))
                else:
                    max_proc = int(max(1, max_frac_cpu / cfg["cpu_required_frac"]))

                cfg["num_proc"] = min_proc if min_proc == max_proc else (min_proc, max_proc)

    for op in operators:
        cfg = resource_configs[op._name]
        auto_proc, num_proc = cfg["auto_proc"], cfg["num_proc"]
        if cfg["is_actor"]:
            op.cpu_required = cfg["cpu_required"]
            op.gpu_required = cfg["gpu_required"]
            op.num_proc = num_proc
        else:
            # * If ``fn`` is a function and ``concurrency`` is an  int ``n``, Ray Data
            # launches *at most* ``n`` concurrent tasks.
            op.cpu_required = cfg["cpu_required"]
            op.gpu_required = None
            # if concurrency left to None, the automatic concurrency of ray may be slightly higher, which could lead to OOM
            op.num_proc = num_proc[1] if (auto_proc and isinstance(num_proc, (tuple, list))) else num_proc

        logger.info(
            f"Op[{op._name}] will be executed with the following resources: "
            f"num_cpus: {op.cpu_required}, "
            f"num_gpus: {op.gpu_required}, "
            f"concurrency: {op.num_proc}, "
        )
    return operators

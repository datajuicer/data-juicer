import importlib.metadata
import importlib.util
import os
import platform
import sys
from typing import Tuple, Union

from loguru import logger

CHECK_SYSTEM_INFO_ONCE = True


def _is_package_available(pkg_name: str, return_version: bool = False) -> Union[Tuple[bool, str], bool]:
    # Check we're not importing a "pkg_name" directory somewhere
    # but the actual library by trying to grab the version
    package_exists = importlib.util.find_spec(pkg_name) is not None
    package_version = "N/A"
    if package_exists:
        try:
            package_version = importlib.metadata.version(pkg_name)
            package_exists = True
        except importlib.metadata.PackageNotFoundError:
            package_exists = False
        logger.debug(f"Detected {pkg_name} version {package_version}")
    if return_version:
        return package_exists, package_version
    else:
        return package_exists


def _torch_check_and_set():
    # only for python3.8 on mac
    global CHECK_SYSTEM_INFO_ONCE
    if CHECK_SYSTEM_INFO_ONCE and importlib.util.find_spec("torch") is not None:
        major, minor = sys.version_info[:2]
        system = platform.system()
        if major == 3 and minor == 8 and system == "Darwin":
            logger.warning(
                "The torch.set_num_threads function does not "
                "work in python3.8 version on Mac systems. We will set "
                "OMP_NUM_THREADS to 1 manually before importing torch"
            )

            os.environ["OMP_NUM_THREADS"] = str(1)
            CHECK_SYSTEM_INFO_ONCE = False
        import torch

        # avoid hanging when calling clip in multiprocessing
        torch.set_num_threads(1)

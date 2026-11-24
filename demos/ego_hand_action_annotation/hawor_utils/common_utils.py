import os
import sys
import subprocess

from data_juicer.utils.cache_utils import DATA_JUICER_ASSETS_CACHE


def prepare_hawor_repo():
    hawor_home = os.path.join(DATA_JUICER_ASSETS_CACHE, "HaWoR")
    if not os.path.exists(hawor_home):
        subprocess.run(["git", "clone", "https://github.com/ThunderVVV/HaWoR.git", hawor_home], check=True)

    return hawor_home


def prepare_hawor_and_add_to_path():
    hawor_home = prepare_hawor_repo()

    if hawor_home not in sys.path:
        sys.path.insert(0, hawor_home)

    return hawor_home

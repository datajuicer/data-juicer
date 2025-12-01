# Some code here has been modified from:
# https://github.com/Megvii-BaseDetection/YOLOX
# --------------------------------------------------------

# This file is used for package installation. Script of tools will be
# available.

import sys
from importlib import abc, util
from pathlib import Path

_TOOLS_PATH = Path(__file__).resolve().parent.parent.parent / "tools"

if _TOOLS_PATH.is_dir():
    # This is true only for in-place installation
    # (pip install -e, setup.py develop),
    # where setup(package_dir=) does not work:
    # https://github.com/pypa/setuptools/issues/230

    class _PathFinder(abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if not name.startswith("data_juicer.tools."):
                return
            project_name = name.split(".")[-1] + ".py"
            target_file = _TOOLS_PATH / project_name
            if not target_file.is_file():
                return
            return util.spec_from_file_location(name, target_file)

    sys.meta_path.append(_PathFinder())

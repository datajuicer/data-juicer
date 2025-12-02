import datetime
import os
import traceback
from typing import Dict

from loguru import logger

from data_juicer.config import get_init_configs
from data_juicer.core.executor import DefaultExecutor

DEFAULT_OUTPUT_DIR = "./outputs"


def add_extra_cfg(dj_cfg: Dict) -> Dict:
    """Add extra dj config."""
    if not dj_cfg.get("export_path"):
        logger.info("export_path is not set, use default export_path")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dj_cfg["export_path"] = os.path.join(DEFAULT_OUTPUT_DIR, timestamp, "processed_data.jsonl")

    if not dj_cfg.get("np") or os.getenv("SERVER_TRANSPORT", "sse") == "stdio":
        dj_cfg["np"] = 1  # set num proc to be 1
        logger.warning(
            "Multiprocessing has been disabled.  "
            "It is only supported when the MCP server transport is "
            "not 'stdio' and 'np' is explicitly set. Setting 'np' to 1."
        )
    dj_cfg["open_monitor"] = False  # unable monitor to avoid multi proc

    return dj_cfg


def execute_op(dj_cfg: Dict):

    try:
        dj_cfg = add_extra_cfg(dj_cfg)
        logger.info(f"DJ config in MCP server: {str(dj_cfg)}")
        dj_cfg = get_init_configs(dj_cfg, load_configs_only=False)
        executor = DefaultExecutor(dj_cfg)
        executor.run()
        return f"Result dataset is saved in: {dj_cfg['export_path']}"
    except Exception:
        error_msg = traceback.format_exc()
        return f"Occur error when executing Data-Juicer: {error_msg}"
